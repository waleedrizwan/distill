#!/usr/bin/env python3
"""Score distill and two baselines against reference summaries.

    python3 eval/fetch_data.py         # once
    cargo build --release
    python3 eval/score.py              # both budget modes, all sets
    python3 eval/score.py --budget words --sets cnndm --save before.json
    python3 eval/score.py --compare before.json   # change per cell, with a
                                                  # paired bootstrap interval

Methods
  lead      the first sentences of the document, as distill segments it
  centroid  similarity to the document's mean vector, no lead bonus, penalties,
            repair or diversity (the plain unsupervised baseline)
  distill   distill's defaults, plus any --extra flags

Budgets
  sentences  a fixed number of sentences per set (cnndm 3, billsum 5, arxiv 6),
             as in the README's original table. Longer picks score higher here.
  words      each document's budget is its reference summary's word count, so
             every method is compared at about the same length. Lead takes
             sentences in document order, skipping any that would overflow,
             which is the same rule distill applies. Context sentences that
             distill adds for readability are extra, so check the Words column.

ROUGE is computed here with no stemming or stopword removal: lowercase
alphanumeric tokens, F1 of unigram and bigram overlap, and F1 of the longest
common subsequence over the whole summary. Numbers are comparable with each
other and with the README, not with published papers. Standard library only.
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SENTENCES = {"cnndm": 3, "billsum": 5, "arxiv": 6}
DEV_SETS = "cnndm-dev,billsum-dev"
METHODS = ["lead", "centroid", "distill"]
CENTROID_FLAGS = [
    "--centrality", "centroid", "--lambda", "1", "--lead", "0", "--penalty", "0",
    "--no-repair", "--section-weight", "0", "--boost", "1", "--salience", "0",
]


def base(set_name):
    return set_name.split("-")[0]


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(toks, n):
    return Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def f1(overlap, n_sys, n_ref):
    if overlap == 0 or n_sys == 0 or n_ref == 0:
        return 0.0
    p, r = overlap / n_sys, overlap / n_ref
    return 2 * p * r / (p + r)


def lcs(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge(system, reference):
    s, r = tokens(system), tokens(reference)
    out = {}
    for n in (1, 2):
        sn, rn = ngrams(s, n), ngrams(r, n)
        out[f"r{n}"] = f1(sum((sn & rn).values()), sum(sn.values()), sum(rn.values()))
    out["rl"] = f1(lcs(s, r), len(s), len(r))
    return out


def run(binary, args, text, env):
    p = subprocess.run([binary, *args], input=text, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(args)}: {p.stderr.strip()}")
    return p.stdout


SEGMENT_LINE = re.compile(r"^\s*\d+ p(\d+)")


def segments(binary, text, env):
    """Sentences in document order, from --show-segments."""
    out = []
    for line in run(binary, ["--show-segments"], text, env).splitlines():
        m = SEGMENT_LINE.match(line)
        if not m:
            continue  # section header
        # "{index:>5} p{paragraph:<4} {flag:2} {text}"
        start = m.start(1) + max(4, len(m.group(1)))
        out.append(line[start + 4 :])
    return out


def lead(sentences, n_sentences, n_words):
    if n_words is None:
        return sentences[:n_sentences]
    picked, total = [], 0
    for s in sentences:
        w = len(s.split())
        if picked and total + w > n_words:
            continue  # same rule as distill's word budget: skip what does not fit
        picked.append(s)
        total += w
    return picked


def summarize(binary, flags, text, env):
    out = json.loads(run(binary, ["--json", *flags], text, env))
    return [s["text"] for s in out["sentences"]]


def score_doc(job):
    binary, set_name, doc, budget, extra, env = job
    ref_words = len(doc["reference"].split())
    if budget == "words":
        n_words, n_sent = ref_words, None
        flags = ["--words", str(ref_words)]
    else:
        n_words, n_sent = None, SENTENCES[base(set_name)]
        flags = ["-n", str(n_sent)]
    picks = {
        "lead": lead(segments(binary, doc["text"], env), n_sent, n_words),
        "centroid": summarize(binary, flags + CENTROID_FLAGS, doc["text"], env),
        "distill": summarize(binary, flags + extra, doc["text"], env),
    }
    res = {}
    for m, sents in picks.items():
        summary = " ".join(sents)
        res[m] = {**rouge(summary, doc["reference"]), "words": len(summary.split())}
    res["reference"] = {"words": ref_words}
    return res


def load(set_name, limit):
    path = HERE / "data" / f"{set_name}.jsonl"
    if not path.exists():
        sys.exit(f"{path.relative_to(ROOT)} is missing; run: python3 eval/fetch_data.py")
    docs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return docs[:limit] if limit else docs


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap(diffs, n=2000, seed=0):
    """95% interval for the mean of paired per-document differences (fixed seed)."""
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=str(ROOT / "target" / "release" / "distill"))
    ap.add_argument("--sets", default="cnndm,billsum,arxiv", help=f"comma-separated; tune on {DEV_SETS}")
    ap.add_argument("--budget", choices=["sentences", "words", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N documents per set")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--model", help="model folder for distill and centroid (default: distill's own lookup)")
    ap.add_argument("--extra", default="", help="extra flags for the distill method, e.g. \"--lambda 0.6\"")
    ap.add_argument("--save", help="write per-cell averages to this JSON file")
    ap.add_argument("--compare", help="JSON from an earlier --save; prints the difference in each cell")
    args = ap.parse_args()

    if not Path(args.bin).is_file():
        sys.exit(f"{args.bin} not found; run: cargo build --release")
    env = dict(os.environ)
    env.setdefault("DISTILL_CACHE", str(HERE / ".cache"))  # embeddings are reused across methods and runs
    env.setdefault("RAYON_NUM_THREADS", "2")
    if args.model:
        if not (Path(args.model) / "model.safetensors").is_file():
            sys.exit(f"{args.model} has no model.safetensors; run: ./fetch-model.sh NAME")
        env["DISTILL_MODEL"] = str(Path(args.model).resolve())
    extra = args.extra.split()
    budgets = ["sentences", "words"] if args.budget == "both" else [args.budget]
    before = json.loads(Path(args.compare).read_text()) if args.compare else {}

    table = {}
    per_doc = {}
    rows = ["| Set | Budget | Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Words |", "|---|---|---|---|---|---|---|"]
    for set_name in [s for s in args.sets.split(",") if s]:
        docs = load(set_name, args.limit)
        for budget in budgets:
            jobs = [(args.bin, set_name, d, budget, extra, env) for d in docs]
            with ThreadPoolExecutor(args.jobs) as pool:
                results = list(pool.map(score_doc, jobs))
            ref_words = mean([r["reference"]["words"] for r in results])
            label = f"{SENTENCES[base(set_name)]} sentences" if budget == "sentences" else f"ref length ({ref_words:.0f} w)"
            for i, m in enumerate(METHODS):
                key = f"{set_name}/{budget}/{m}"
                cell = {k: mean([r[m][k] for r in results]) for k in ("r1", "r2", "rl", "words")}
                cell = {k: v * 100 if k != "words" else v for k, v in cell.items()}
                table[key] = cell
                per_doc[key] = [round(r[m]["r2"] * 100, 3) for r in results]
                old = before.get(key)

                def fmt(k, digits=1):
                    v = f"{cell[k]:.{digits}f}"
                    if old is not None:
                        v += f" ({cell[k] - old[k]:+.{digits}f})"
                    return f"**{v}**" if m == "distill" else v

                head = f"{set_name} (n={len(docs)})" if (i == 0 and budget == budgets[0]) else ""
                rows.append(
                    f"| {head} | {label if i == 0 else ''} | {m} | {fmt('r1')} | {fmt('r2')} | {fmt('rl')} | {fmt('words', 0)} |"
                )
            print(f"scored {set_name} / {budget}", file=sys.stderr)

    print("\n".join(rows))
    old_docs = before.get("per_doc", {})
    lines = []
    for key, now in per_doc.items():
        was = old_docs.get(key)
        if key.endswith("/distill") and was and len(was) == len(now):
            lo, hi = bootstrap([a - b for a, b in zip(now, was)])
            verdict = "better" if lo > 0 else "worse" if hi < 0 else "no clear change"
            lines.append(f"{key.rsplit('/', 1)[0]}: ROUGE-2 {mean(now) - mean(was):+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}], {verdict}")
    if lines:
        print("\nPaired bootstrap, distill vs --compare file:\n" + "\n".join(lines))
    if args.save:
        out = {"extra": args.extra, "limit": args.limit, **table, "per_doc": per_doc}
        Path(args.save).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
