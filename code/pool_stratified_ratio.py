"""Pool-stratified headline TOT:DK ratio.

Reports the headline standard TOT:DK ratio separately for
the NORMED item pool vs the BEYOND-CORPUS (hard) pool, per language, pooled
over the primary six models (PRIMARY6). Conventions (item keying, classify(),
PRIMARY6) come from analyze_pilot / analysis_common as everywhere else in this
repo. Adds item-cluster bootstrap 95% CIs on the pooled ratio within each
pool stratum, reusing the resampling logic in ratio_ci.py (resample the
ITEM, the independent unit within a stratum; a single item contributes up
to six correlated rows, one per model).

  python -u pool_stratified_ratio.py --reps 1000 --out ../results/pool_stratified_ratio.txt
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from analysis_common import PRIMARY6, ROOT, classify, load_jsonl

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


NORMED = {"en_normed", "zh_v2"}
BEYOND = {"en_hard", "zh_hard"}


def pool_bucket(source_pool):
    if source_pool in NORMED:
        return "normed"
    if source_pool in BEYOND:
        return "beyond-corpus"
    return None


def build(lang):
    """Per-model, per-item TOT/DK indicator arrays, plus each item's pool
    bucket ('normed' / 'beyond-corpus')."""
    items = load_jsonl(os.path.join(ROOT, "data", f"items_full_{lang}_final.jsonl"))
    ids = list(items)
    idx = {iid: k for k, iid in enumerate(ids)}
    n = len(ids)
    bucket = np.array([pool_bucket(items[iid]["source_pool"]) for iid in ids], dtype=object)

    M = {}
    for tag in PRIMARY6:
        base = os.path.join(ROOT, "results", f"full_{lang}_{tag}")
        prod = load_jsonl(os.path.join(base, "production.jsonl"))
        rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
        tot = np.zeros(n)
        dk = np.zeros(n)
        for iid in ids:
            i = idx[iid]
            st = classify(prod.get(iid), rec.get(iid))
            if st == "TOT":
                tot[i] = 1
            elif st == "DK":
                dk[i] = 1
        M[tag] = dict(tot=tot, dk=dk)
    return ids, bucket, M


def pooled_counts(M, mask, w):
    T = D = 0.0
    for tag, d in M.items():
        T += (d["tot"] * mask) @ w
        D += (d["dk"] * mask) @ w
    return T, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "pool_stratified_ratio.txt"))
    args = ap.parse_args()

    say("=" * 92)
    say("POOL-STRATIFIED HEADLINE TOT:DK RATIO (NORMED vs BEYOND-CORPUS)")
    say("=" * 92)
    say(f"PRIMARY6 models pooled per language; classify() from analyze_pilot.py; "
        f"item-cluster bootstrap reps={args.reps} seed={args.seed} (resample unit "
        f"= item, within stratum).")
    say("sanity targets: EN 501 normed / 1,002 beyond-corpus;  "
        "ZH 402 normed / 501 beyond-corpus")

    rng = np.random.default_rng(args.seed)
    for lang in ("en", "zh"):
        ids, bucket, M = build(lang)
        n = len(ids)
        say(f"\n--- {lang.upper()}  (bank n={n} items, {len(M)} primary models) ---")

        cnt = Counter(bucket.tolist())
        say(f"  item counts by pool: " + ", ".join(f"{k}={v}" for k, v in
                                                    sorted(cnt.items(), key=str)))
        if None in cnt:
            say(f"  WARNING: {cnt[None]} items had an unrecognized source_pool value")

        say(f"  {'pool':>14s} {'n_items':>9s} {'TOT':>9s} {'DK':>9s} "
            f"{'ratio':>8s} {'95% CI':>16s} {'P(TOT|classified)':>20s}")
        for pool_name in ("normed", "beyond-corpus"):
            mask = (bucket == pool_name).astype(float)
            n_items = int(mask.sum())
            w1 = np.ones(n)
            T, D = pooled_counts(M, mask, w1)
            ratio = T / D if D else float("nan")
            p_tot = T / (T + D) if (T + D) else float("nan")

            reps = []
            for _ in range(args.reps):
                draw = rng.integers(0, n, n)
                w = np.bincount(draw, minlength=n).astype(float)
                t, d = pooled_counts(M, mask, w)
                reps.append(t / d if d else float("nan"))
            a = np.array([x for x in reps if np.isfinite(x)])
            lo, hi = (np.percentile(a, [2.5, 97.5]) if len(a) else (float("nan"),) * 2)

            say(f"  {pool_name:>14s} {n_items:9d} {T:9.0f} {D:9.0f} "
                f"{ratio:8.2f} [{lo:5.2f},{hi:5.2f}] {p_tot:20.3f}")

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(("\n".join(OUT) + "\n").rstrip("\n") + "\n")
    print("wrote", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
