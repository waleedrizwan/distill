#!/usr/bin/env python3
"""Train the salience head compiled into distill (src/salience.json).

    python3 eval/fetch_data.py
    cargo build --release
    python3 eval/train_salience.py          # writes src/salience.json
    cargo build --release                   # compiles the new weights in

1. Every training document is run through `distill --dump-features`, which
   prints each candidate sentence with exactly the features the binary uses at
   run time: centrality, position, length, dependence flags and the frozen
   sentence embedding.
2. Oracle labels: for each document, sentences are added greedily while they
   raise the mean of ROUGE-1 and ROUGE-2 F1 against the reference summary.
   Those sentences are positives, everything else negative.
3. An L2-regularised logistic regression is fitted by Newton's method on
   standardised features. The regularisation strength is chosen by log-loss on
   10% of the training documents held out, then the model is refitted on all.

Training data is the cnndm-train and billsum-train sets from fetch_data.py.
None of it overlaps the development or test sets. Needs numpy; distill itself
does not.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# numpy 2.0 with Apple's Accelerate warns about divide-by-zero inside matmul even
# for finite inputs. The fitted weights are checked for finiteness instead.
np.seterr(all="ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "src" / "salience.json"

_spec = importlib.util.spec_from_file_location("score", HERE / "score.py")
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)


def dump(binary, text, env):
    p = subprocess.run([binary, "--dump-features"], input=text, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        return None  # e.g. no usable sentences
    return json.loads(p.stdout)


def oracle(sentences, reference):
    """Indices chosen greedily to maximise mean(ROUGE-1 F1, ROUGE-2 F1)."""
    ref = score.tokens(reference)
    ref1, ref2 = score.ngrams(ref, 1), score.ngrams(ref, 2)
    toks = [score.tokens(s) for s in sentences]

    def value(chosen):
        sys_t = [t for i in sorted(chosen) for t in toks[i]]
        s1, s2 = score.ngrams(sys_t, 1), score.ngrams(sys_t, 2)
        r1 = score.f1(sum((s1 & ref1).values()), sum(s1.values()), sum(ref1.values()))
        r2 = score.f1(sum((s2 & ref2).values()), sum(s2.values()), sum(ref2.values()))
        return (r1 + r2) / 2

    chosen, best = set(), 0.0
    while True:
        gains = [(value(chosen | {i}), -i) for i in range(len(sentences)) if i not in chosen]
        if not gains:
            break
        v, neg_i = max(gains)
        if v <= best:
            break
        chosen.add(-neg_i)
        best = v
    return chosen


def load_set(name, binary, env, jobs):
    """Features and oracle labels for one training set, cached by data and binary."""
    data = HERE / "data" / f"{name}.jsonl"
    docs = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    key = hashlib.sha256(data.read_bytes() + Path(binary).read_bytes()).hexdigest()[:16]
    cache = HERE / ".cache" / f"features-{name}-{key}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return z["X"], z["y"], z["doc"], json.loads(str(z["encoder"]))

    def one(d):
        out = dump(binary, d["text"], env)
        if out is None or not out["candidates"]:
            return None
        cands = out["candidates"]
        labels = oracle([c["text"] for c in cands], d["reference"])
        X = np.array([c["features"] for c in cands], dtype=np.float32)
        y = np.array([1.0 if i in labels else 0.0 for i in range(len(cands))], dtype=np.float32)
        return X, y, out["encoder"]

    with ThreadPoolExecutor(jobs) as pool:
        results = list(pool.map(one, docs))
    Xs, ys, ds, encoder = [], [], [], None
    for k, r in enumerate(results):
        if r is None:
            continue
        Xs.append(r[0])
        ys.append(r[1])
        ds.append(np.full(len(r[1]), k, dtype=np.int32))
        encoder = r[2]
    X, y, doc = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ds)
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, X=X, y=y, doc=doc, encoder=json.dumps(encoder))
    return X, y, doc, encoder


def fit(X, y, l2, iters=25):
    """Logistic regression by Newton's method; X is standardised, bias unpenalised."""
    n, d = X.shape
    A = np.hstack([X, np.ones((n, 1))]).astype(np.float64)
    w = np.zeros(d + 1)
    reg = np.full(d + 1, l2)
    reg[-1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(A @ w)))
        grad = A.T @ (p - y) / n + reg * w
        H = (A * (p * (1 - p))[:, None]).T @ A / n + np.diag(reg)
        step = np.linalg.solve(H, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return w[:-1], w[-1]


def log_loss(X, y, w, b):
    z = X.astype(np.float64) @ w + b
    return float(np.mean(np.logaddexp(0, z) - y * z))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=str(ROOT / "target" / "release" / "distill"))
    ap.add_argument("--sets", default="cnndm-train,billsum-train")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--l2", default="1e-4,1e-3,1e-2,1e-1", help="regularisation strengths to try")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    env = dict(os.environ)
    env.setdefault("DISTILL_CACHE", str(HERE / ".cache"))
    env.setdefault("RAYON_NUM_THREADS", "2")

    parts, encoder = [], None
    for set_index, name in enumerate(s for s in args.sets.split(",") if s):
        X, y, doc, enc = load_set(name, args.bin, env, args.jobs)
        if encoder is not None and enc != encoder:
            sys.exit("training sets were dumped with different encoders")
        encoder = enc
        n_docs = len(np.unique(doc))
        print(f"{name}: {n_docs} documents, {len(y)} sentences, {y.mean():.1%} oracle positives", file=sys.stderr)
        parts.append((X, y, doc + set_index * 1_000_000))
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts]).astype(np.float64)
    doc = np.concatenate([p[2] for p in parts])

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    Z = (X - mean) / std

    # Hold out every tenth document (a fixed, document-level split) to choose l2.
    held = (doc % 10) == 0
    best = None
    for l2 in [float(v) for v in args.l2.split(",")]:
        w, b = fit(Z[~held], y[~held], l2)
        loss = log_loss(Z[held], y[held], w, b)
        print(f"l2 {l2:g}: held-out log-loss {loss:.4f}", file=sys.stderr)
        if best is None or loss < best[0]:
            best = (loss, l2)
    l2 = best[1]
    w, b = fit(Z, y, l2)
    if not (np.isfinite(w).all() and np.isfinite(b)):
        sys.exit("fit diverged: non-finite weights")

    n_hand = X.shape[1] - encoder["hidden_size"]
    names = [
        "doc_centrality", "section_centrality", "position", "log_position", "section_position",
        "section_first", "log_words", "dangling", "points_elsewhere", "lead_in", "list_item",
        "summary_section",
    ]
    if n_hand != len(names):
        sys.exit(f"expected {len(names)} hand features, the dump has {n_hand}; update this list")

    def r(v):
        return [float(f"{x:.6g}") for x in v]

    head = {
        "comment": "Generated by eval/train_salience.py. Do not edit by hand.",
        "trained_on": args.sets,
        "l2": l2,
        "held_out_log_loss": round(best[0], 5),
        "encoder": encoder,
        "features": names,
        "mean": r(mean),
        "std": r(std),
        "weights": r(w),
        "bias": float(f"{b:.6g}"),
    }
    Path(args.out).write_text(json.dumps(head) + "\n")
    top = sorted(zip(names, w[:n_hand]), key=lambda t: -abs(t[1]))
    print("hand-feature weights (standardised): " + ", ".join(f"{k} {v:+.2f}" for k, v in top), file=sys.stderr)
    print(f"chose l2 {l2:g}; wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
