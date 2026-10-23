#!/usr/bin/env python3
"""Build the papers evaluation sets from arXiv PDFs.

Two sets of research papers, chosen by rule so nobody picked them by hand:
for each of six categories, the first papers submitted in a fixed week that
pass the filters below. "papers" (test) uses one week, "papers-dev" an earlier
one, so knobs are tuned on papers that are never reported.

The reference summary is the paper's abstract (from the arXiv API). The input
is the PDF's text from the first "Introduction" heading on, so the title,
authors and abstract are not in it. Everything after, references included, is
left in: handling that is distill's job.

    python3 eval/fetch_papers.py              # download what is missing, verify, write jsonl
    python3 eval/fetch_papers.py --extractor distill   # text from `distill --extract-text`
    python3 eval/fetch_papers.py --select     # (re)choose the papers, write eval/papers.tsv

eval/papers.tsv pins each paper's version and the SHA-256 of its PDF, so a
changed download fails instead of quietly changing the scores. The PDFs are not
committed. Standard library only, plus `pdftotext` (poppler) for the default
extractor.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
PDFS = DATA / "papers"
TABLE = HERE / "papers.tsv"
API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"

CATEGORIES = ["cs.CL", "q-bio.QM", "cond-mat.mtrl-sci", "astro-ph.EP", "econ.GN", "stat.ME"]
# set: (first day, last day) of the submission window, and papers per category
WINDOWS = {"papers": ("20250303", "20250309", 5), "papers-dev": ("20250203", "20250209", 5)}
MIN_WORDS = 2000  # of body text, after the cut
MAX_WORDS = 20000  # longer is usually a thesis or a review with a huge appendix

# "1 Introduction", "1.Introduction", "I. INTRODUCTION", and small capitals as
# pdftotext renders them ("I NTRODUCTION"). Journals without an introduction
# heading often open with "Background" instead. Papers with neither are skipped.
# Capitalised, so a table cell reading "background" is not taken for a heading.
INTRO = re.compile(
    r"^\s*(?:(?:\d+|[IVX]+)\s*\.?\s*)?"
    r"(?:I\s?(?:ntroduction|NTRODUCTION)|B\s?(?:ackground|ACKGROUND)(?:\s*&\s*Summary)?)\s*$",
    re.M,
)

CACHE = HERE / ".cache" / "arxiv"
_last_request = 0.0


def get(url, timeout=120):
    """GET with arXiv's requested pacing (one request every 3 s) and retries.
    API answers are cached in eval/.cache/, so an interrupted run resumes."""
    global _last_request
    cached = None
    if url.startswith(API):
        cached = CACHE / (hashlib.sha256(url.encode()).hexdigest() + ".xml")
        if cached.exists():
            return cached.read_bytes()
    for attempt in range(12):
        wait = 3.0 - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "distill-eval/1 (research evaluation)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            if body.startswith(b"Rate exceeded"):
                raise urllib.error.HTTPError(url, 429, "rate exceeded", {}, None)
            if cached:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(body)
            return body
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            code = getattr(e, "code", 0)
            if code not in (0, 429, 500, 502, 503, 504) or attempt == 11:
                raise
            delay = min(300, 5 * 2**attempt)
            print(f"  {code or e} from arXiv, retrying in {delay} s", file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def entries(query, max_results):
    q = urllib.parse.urlencode(
        {"search_query": query, "sortBy": "submittedDate", "sortOrder": "ascending", "max_results": max_results}
    )
    root = ET.fromstring(get(f"{API}?{q}"))
    out = []
    for e in root.findall(f"{ATOM}entry"):
        url = e.find(f"{ATOM}id").text.strip()  # http://arxiv.org/abs/2503.01234v2
        out.append(
            {
                "id": url.rsplit("/abs/", 1)[1],
                "abstract": " ".join(e.find(f"{ATOM}summary").text.split()),
                "primary": e.find("{http://arxiv.org/schemas/atom}primary_category").get("term"),
            }
        )
    return out


def abstracts(ids):
    """Abstract per versioned id, fetched in batches."""
    out = {}
    for i in range(0, len(ids), 50):
        batch = ids[i : i + 50]
        q = urllib.parse.urlencode({"id_list": ",".join(batch), "max_results": len(batch)})
        root = ET.fromstring(get(f"{API}?{q}"))
        for e in root.findall(f"{ATOM}entry"):
            vid = e.find(f"{ATOM}id").text.strip().rsplit("/abs/", 1)[1]
            out[vid] = " ".join(e.find(f"{ATOM}summary").text.split())
    return out


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def pdf_path(vid):
    return PDFS / (vid.replace("/", "_") + ".pdf")


def download(vid):
    path = pdf_path(vid)
    if not path.exists():
        PDFS.mkdir(parents=True, exist_ok=True)
        data = get(f"https://arxiv.org/pdf/{vid}")
        if not data.startswith(b"%PDF"):
            raise ValueError(f"{vid}: not a PDF")
        path.write_bytes(data)
    return path.read_bytes()


def extract(path, extractor):
    if extractor == "pdftotext":
        cmd = ["pdftotext", "-enc", "UTF-8", str(path), "-"]
    else:
        cmd = [str(ROOT / "target" / "release" / "distill"), "--extract-text", str(path)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {r.stderr.decode(errors='replace')[:200]}")
    text = r.stdout.decode("utf-8", errors="replace")
    if extractor == "pdftotext":
        # What `distill --dehyphenate` does, so every method sees the same text.
        text = re.sub(r"([a-z])-\n *([a-z])", r"\1\2", text)
    return text


def body(text):
    """The text from the first Introduction heading on, or None."""
    m = INTRO.search(text)
    return text[m.start() :] if m else None


def select():
    rows = []
    for name, (first, last, per) in WINDOWS.items():
        for cat in CATEGORIES:
            got = 0
            query = f"cat:{cat} AND submittedDate:[{first}0000 TO {last}2359]"
            for e in entries(query, 200):
                if got == per:
                    break
                # Primary category only, so a cross-list does not stand in for the field.
                if e["primary"] != cat:
                    continue
                try:
                    data = download(e["id"])
                    text = body(extract(pdf_path(e["id"]), "pdftotext"))
                except Exception as err:  # noqa: BLE001 - any failure just skips the paper
                    print(f"  skip {e['id']}: {err}", file=sys.stderr)
                    continue
                words = len(text.split()) if text else 0
                if not (MIN_WORDS <= words <= MAX_WORDS):
                    print(f"  skip {e['id']}: {words} body words", file=sys.stderr)
                    continue
                if len(e["abstract"].split()) < 80:
                    print(f"  skip {e['id']}: abstract under 80 words", file=sys.stderr)
                    continue
                rows.append((name, cat, e["id"], sha256(data), sha256(e["abstract"].encode())))
                got += 1
            print(f"{name} {cat}: {got} papers")
            if got < per:
                sys.exit(f"{name} {cat}: only {got} papers passed; widen the window")
    lines = ["# set\tcategory\tarxiv id (versioned)\tsha256(pdf)\tsha256(abstract)"]
    lines += ["\t".join(r) for r in rows]
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLE.relative_to(ROOT)}")


def build(extractor):
    rows = [line.split("\t") for line in TABLE.read_text().splitlines() if line and not line.startswith("#")]
    refs = abstracts([r[2] for r in rows])
    bad = 0
    out = {name: [] for name in WINDOWS}
    for name, cat, vid, pdf_hash, abs_hash in rows:
        data = download(vid)
        ref = refs.get(vid, "")
        if sha256(data) != pdf_hash:
            print(f"{vid}: PDF changed", file=sys.stderr)
            bad += 1
            continue
        if sha256(ref.encode()) != abs_hash:
            print(f"{vid}: abstract changed", file=sys.stderr)
            bad += 1
            continue
        text = body(extract(pdf_path(vid), extractor))
        if text is None:
            print(f"{vid}: no Introduction heading in the {extractor} text", file=sys.stderr)
            bad += 1
            continue
        out[name].append({"id": vid, "category": cat, "text": text, "reference": ref})
    for name, docs in out.items():
        with open(DATA / f"{name}.jsonl", "w") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"{name}: {len(docs)} papers in eval/data/{name}.jsonl (text from {extractor})")
    if bad:
        sys.exit(f"{bad} papers did not match eval/papers.tsv; scores would not be comparable")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", action="store_true", help="choose the papers and write eval/papers.tsv")
    ap.add_argument("--extractor", choices=["pdftotext", "distill"], default="distill")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    if args.select:
        select()
    build(args.extractor)


if __name__ == "__main__":
    main()
