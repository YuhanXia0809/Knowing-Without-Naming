"""Propagate answer-key / certificate uncertainty into a printed interval.

Every interval in the paper resamples ITEMS with each item's state label held
fixed, so the measured label perturbation (13.6% item-level disagreement
between the original and the paraphrased gloss) reaches no confidence
interval.

Two nested sources of uncertainty:

  (1) SAMPLING   -- resample items with replacement (the existing
                    item-cluster bootstrap; the bank is shared, so one item
                    contributes up to six correlated rows).

  (2) CERTIFICATE -- each item carries a uniqueness certificate re-issued by
                    the primary judge on the prompt (paraphrased) gloss.
                    Items whose certificate is lost under the paraphrase
                    (unique_orig and not unique) are the measured
                    perturbation.  The certificate itself is a model
                    judgment with a measured error rate: on a stratified
                    400-item English sample a second-family judge
                    (Llama-3.1-70B) agreed with the primary judge on 90.0%
                    of verdicts.  In each replicate we therefore re-draw
                    every item's certificate, flipping it independently with
                    probability e = 1 - 0.900, and drop the items whose
                    certificate is lost.

The point estimate is unchanged (it is the no-drop, no-flip configuration).
What changes is the interval, which now covers judge disagreement as well as
item sampling.

  python -u label_bootstrap.py --out ../results/label_bootstrap.txt   # reps=1000
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from analysis_common import PRIMARY6, ROOT, classify, load_jsonl

OUT = []

# second-family judge agreement on the stratified 400-item English
# re-certification sample (Llama-3.1-70B vs Qwen2.5-72B)
JUDGE_AGREEMENT = 0.900


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def cert_lost(lang):
    """Item ids whose uniqueness certificate is lost under the prompt gloss."""
    path = os.path.join(ROOT, "data", f"items_recert_{lang}_judged.jsonl")
    seen = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                seen[r["item_id"]] = r
    lost = {iid for iid, r in seen.items()
            if r.get("unique_orig") and not r.get("unique")}
    return lost, set(seen)


def build(lang):
    """Per-item TOT / DK indicator vectors, summed over the primary six."""
    items = list(load_jsonl(os.path.join(
        ROOT, "data", f"items_full_{lang}_final.jsonl")))
    idx = {iid: k for k, iid in enumerate(items)}
    n = len(items)
    tot = np.zeros(n)
    dk = np.zeros(n)
    for tag in PRIMARY6:
        base = os.path.join(ROOT, "results", f"full_{lang}_{tag}")
        prod = load_jsonl(os.path.join(base, "production.jsonl"))
        rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
        for iid in items:
            st = classify(prod.get(iid), rec.get(iid))
            if st == "TOT":
                tot[idx[iid]] += 1
            elif st == "DK":
                dk[idx[iid]] += 1
    return items, idx, tot, dk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=18)
    ap.add_argument("--item-seed", type=int, default=17,
                    help="mirror ratio_ci.py's item-resampling stream")
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "label_bootstrap.txt"))
    args = ap.parse_args()

    say("=" * 92)
    say("LABEL-PERTURBATION BOOTSTRAP: judge uncertainty inside the "
        "interval, not beside it")
    say("=" * 92)
    say(f"reps={args.reps}  seed={args.seed}  percentile intervals.")
    say("Source (1) item resampling; source (2) certificate re-draw at a "
        f"judge error rate of {1 - JUDGE_AGREEMENT:.3f} "
        f"(= 1 - {JUDGE_AGREEMENT:.3f} two-judge agreement, "
        "stratified 400-item English sample).")

    # two independent streams: the item draw mirrors ratio_ci.py exactly
    # (same seed, same order), so the item-only leg reproduces the published
    # interval and the widening is attributable to the certificate re-draw alone.
    rng = np.random.default_rng(args.item_seed)
    rng_cert = np.random.default_rng(args.seed)
    for lang in ("en", "zh"):
        items, idx, tot, dk = build(lang)
        n = len(items)
        lost, judged = cert_lost(lang)
        lost_v = np.zeros(n, dtype=bool)
        judged_v = np.zeros(n, dtype=bool)
        for iid in items:
            if iid in lost:
                lost_v[idx[iid]] = True
            if iid in judged:
                judged_v[idx[iid]] = True

        point = tot.sum() / dk.sum()
        keep_pt = ~lost_v
        drop_pt = (tot[keep_pt].sum() / dk[keep_pt].sum())

        say("")
        say(f"--- {lang.upper()}  (n items {n}; re-certified {judged_v.sum()}; "
            f"certificate lost {int(lost_v.sum())})")
        say(f"  point ratio, all items                : {point:6.2f}:1")
        say(f"  point ratio, certificate-losers dropped: {drop_pt:6.2f}:1")

        e = 1.0 - JUDGE_AGREEMENT
        both = np.empty(args.reps)
        itemonly = np.empty(args.reps)
        for b in range(args.reps):
            w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            itemonly[b] = (tot @ w) / (dk @ w)
            # re-draw each judged item's certificate at the measured error rate
            flip = rng_cert.random(n) < e
            lost_b = np.where(judged_v, lost_v ^ flip, lost_v)
            wk = w * (~lost_b)
            d = dk @ wk
            both[b] = (tot @ wk) / d if d else np.nan

        def ci(x):
            x = x[np.isfinite(x)]
            return np.percentile(x, 2.5), np.percentile(x, 97.5)

        lo1, hi1 = ci(itemonly)
        lo2, hi2 = ci(both)
        say(f"  item-cluster only        95% CI [{lo1:5.2f}, {hi1:5.2f}]  "
            f"width {hi1 - lo1:5.2f}")
        say(f"  + certificate re-draw    95% CI [{lo2:5.2f}, {hi2:5.2f}]  "
            f"width {hi2 - lo2:5.2f}   "
            f"({(hi2 - lo2) / (hi1 - lo1):.2f}x wider)")
        say(f"  both intervals exclude parity (1:1): "
            f"{'yes' if lo1 > 1 and lo2 > 1 else 'NO'}")

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(("\n".join(OUT) + "\n").rstrip("\n") + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
