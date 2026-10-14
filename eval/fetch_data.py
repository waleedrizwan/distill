#!/usr/bin/env python3
"""Download the frozen evaluation sample into eval/data/ and verify it.

The sample is a fixed slice of three public test sets. The text itself is not
committed (the news articles are copyrighted); instead every document's SHA-256
is in eval/manifest.tsv, so a run that scores different text fails loudly.

    python3 eval/fetch_data.py                   # download + verify
    python3 eval/fetch_data.py --write-manifest  # (re)create the manifest

Standard library only.
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MANIFEST = HERE / "manifest.tsv"
API = "https://datasets-server.huggingface.co/rows"

# name: (dataset, config, split, first row, count, document field, reference field, id field)
SETS = {
    "cnndm": ("abisee/cnn_dailymail", "3.0.0", "test", 0, 100, "article", "highlights", "id"),
    "billsum": ("FiscalNote/billsum", "default", "test", 0, 60, "text", "summary", None),
    "arxiv": ("ccdv/arxiv-summarization", "section", "test", 0, 10, "article", "abstract", None),
}


def get_rows(dataset, config, split, offset, length):
    q = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
                body = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise
            time.sleep(2**attempt)
    rows = []
    for item in body["rows"]:
        if item.get("truncated_cells"):
            sys.exit(f"{dataset}: row {item['row_idx']} came back truncated; cannot freeze it")
        rows.append((item["row_idx"], item["row"]))
    return rows


def digest(doc):
    h = hashlib.sha256()
    h.update(doc["text"].encode())
    h.update(b"\0")
    h.update(doc["reference"].encode())
    return h.hexdigest()


def fetch(name):
    dataset, config, split, first, count, text_f, ref_f, id_f = SETS[name]
    docs = []
    offset = first
    while len(docs) < count:
        batch = get_rows(dataset, config, split, offset, min(100, count - len(docs)))
        if not batch:
            break
        for idx, row in batch:
            ref = row[ref_f]
            if name == "cnndm":
                # Highlights are one sentence per line without final periods on some lines.
                ref = " ".join(line.strip() for line in ref.splitlines() if line.strip())
            docs.append(
                {
                    "id": row[id_f] if id_f else f"{name}-{split}-{idx:05d}",
                    "text": row[text_f],
                    "reference": ref,
                }
            )
        offset += len(batch)
    return docs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-manifest", action="store_true", help="write eval/manifest.tsv from what was downloaded")
    ap.add_argument("--sets", default=",".join(SETS), help="comma-separated subset of: " + ", ".join(SETS))
    args = ap.parse_args()
    names = [s for s in args.sets.split(",") if s]

    expected = {}
    if not args.write_manifest:
        if not MANIFEST.exists():
            sys.exit("eval/manifest.tsv is missing; run with --write-manifest to create it")
        for line in MANIFEST.read_text().splitlines():
            if line and not line.startswith("#"):
                s, i, h = line.split("\t")
                expected.setdefault(s, {})[i] = h

    DATA.mkdir(exist_ok=True)
    manifest_lines = ["# set\tid\tsha256(text NUL reference)"]
    bad = 0
    for name in names:
        docs = fetch(name)
        if not args.write_manifest:
            want = expected.get(name, {})
            got = {d["id"]: digest(d) for d in docs}
            missing = sorted(set(want) - set(got))
            changed = sorted(i for i in want if i in got and got[i] != want[i])
            extra = sorted(set(got) - set(want))
            for label, ids in (("missing", missing), ("changed", changed), ("not in manifest", extra)):
                if ids:
                    bad += len(ids)
                    print(f"{name}: {len(ids)} {label}, e.g. {ids[0]}", file=sys.stderr)
        with open(DATA / f"{name}.jsonl", "w") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                manifest_lines.append(f"{name}\t{d['id']}\t{digest(d)}")
        print(f"{name}: {len(docs)} documents -> eval/data/{name}.jsonl")

    if args.write_manifest:
        if set(names) != set(SETS):
            sys.exit("--write-manifest needs all sets, so the manifest stays complete")
        MANIFEST.write_text("\n".join(manifest_lines) + "\n")
        print(f"wrote {MANIFEST.relative_to(HERE.parent)}")
    elif bad:
        sys.exit("the downloaded sample does not match eval/manifest.tsv; scores would not be comparable")
    else:
        print("all documents match eval/manifest.tsv")


if __name__ == "__main__":
    main()
