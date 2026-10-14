#!/usr/bin/env python3
"""Build src/lexicon.tsv, the substitution table behind `distill --simplify`.

    python3 eval/build_lexicon.py --simpleppdb path/to/SimplePPDB.gz

Source: SimplePPDB (Pavlick and Callison-Burch, ACL 2016), licensed CC BY 3.0
US. The original download link is dead; the archive used here is the Internet
Archive's 2023-05-18 capture of
https://www.seas.upenn.edu/~nlp/resources/simple-ppdb.tgz
(SHA-256 03a32b1e5f1dd30571737a5a1ca55dfe3ca9fd06e82105a318927066ad49d505).

Filtering, all deterministic:
- single words, lowercase letters only, input at least 5 letters
- part of speech folded to noun / verb / adjective / adverb; other tags dropped
- paraphrase score >= 3.5 (the SimplePPDB authors' suggested cut-off for
  single words) and simplification score >= 0.75
- in the training documents (eval/data/cnndm-train and billsum-train): the
  input must not be among the 3,000 most frequent words, and the output must
  be at least twice as frequent and seen at least 50 times
- one output per (input, part of speech): the highest simplification score,
  then the highest paraphrase score, then the most frequent output
- only pairs listed in eval/lexicon-approved.txt are kept; every pair was read
  by hand, and rejected ones are recorded in eval/lexicon-exclude.txt ("word *"
  rejects a word entirely). Candidates in neither file are reported as
  unreviewed and left out.

Standard library only.
"""

import argparse
import collections
import gzip
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "src" / "lexicon.tsv"
EXCLUDE = HERE / "lexicon-exclude.txt"
APPROVED = HERE / "lexicon-approved.txt"

POS = {
    "NN": "n", "NNS": "n", "NNP": "n", "NNPS": "n",
    "VB": "v", "VBD": "v", "VBG": "v", "VBN": "v", "VBZ": "v", "VBP": "v",
    "JJ": "a",
    "RB": "r",
}
WORD = re.compile(r"^[a-z]+$")

HEADER = """# Word substitutions for `distill --simplify`: part of speech (n/v/a/r),
# word, simpler word, paraphrase score, simplification score.
#
# Derived from SimplePPDB by Ellie Pavlick and Chris Callison-Burch, "Simple
# PPDB: A Paraphrase Database for Simplification", ACL 2016, licensed under
# Creative Commons Attribution 3.0 United States
# (https://creativecommons.org/licenses/by/3.0/us/). This is a modified
# version: filtered by eval/build_lexicon.py and reviewed by hand (removed
# pairs are in eval/lexicon-exclude.txt). Regenerate rather than edit, or add
# a pair to the exclude list.
"""


def corpus_counts():
    counts = collections.Counter()
    for name in ("cnndm-train", "billsum-train"):
        for line in (HERE / "data" / f"{name}.jsonl").read_text().splitlines():
            if line.strip():
                counts.update(re.findall(r"[a-z]+", json.loads(line)["text"].lower()))
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simpleppdb", required=True, help="SimplePPDB.gz from the SimplePPDB archive")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    def pairs(path):
        out = set()
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.split("#", 1)[0].strip()
                if line:
                    a, b = line.split()
                    out.add((a, b))
        return out

    excluded, approved = pairs(EXCLUDE), pairs(APPROVED)
    unreviewed = set()

    freq = corpus_counts()
    common = {w for w, _ in freq.most_common(3000)}

    best = {}
    with gzip.open(args.simpleppdb, "rt", encoding="utf-8") as f:
        for line in f:
            para, simp, cat, src, dst = line.rstrip("\n").split("\t")
            pos = POS.get(cat.strip("[]"))
            if pos is None or not WORD.match(src) or not WORD.match(dst) or src == dst:
                continue
            para, simp = float(para), float(simp)
            if len(src) < 5 or para < 3.5 or simp < 0.75:
                continue
            if src in common or freq[dst] < 50 or freq[dst] < 2 * freq[src]:
                continue
            if dst.startswith(src) or src.startswith(dst) and len(dst) < 4:
                continue
            if (src, dst) not in approved:
                if (src, dst) not in excluded and (src, "*") not in excluded:
                    unreviewed.add((src, dst))
                continue
            key = (pos, src)
            cand = (simp, para, freq[dst], dst)
            if key not in best or cand > best[key]:
                best[key] = cand

    rows = sorted((pos, src, c[3], c[1], c[0]) for (pos, src), c in best.items())
    with open(args.out, "w") as f:
        f.write(HEADER)
        for pos, src, dst, para, simp in rows:
            f.write(f"{pos}\t{src}\t{dst}\t{para:.3f}\t{simp:.3f}\n")
    print(f"wrote {len(rows)} substitutions to {args.out}")
    if unreviewed:
        todo = HERE / ".cache" / "lexicon-unreviewed.txt"
        todo.parent.mkdir(exist_ok=True)
        ranked = sorted(unreviewed, key=lambda p: (-freq[p[0]], p))
        todo.write_text("".join(f"{a} {b}\t# {freq[a]} occurrences\n" for a, b in ranked))
        print(f"{len(unreviewed)} candidate pairs are unreviewed and were left out; the most "
              f"frequent first are in {todo.relative_to(ROOT)}. Read them and add each to "
              "lexicon-approved.txt or lexicon-exclude.txt.")


if __name__ == "__main__":
    main()
