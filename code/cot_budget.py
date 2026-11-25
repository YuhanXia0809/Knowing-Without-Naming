"""Is the cue-minus-retry increment a sampling-budget artifact?

The cue arm draws greedy + 20 samples; the no-cue CoT retry draws greedy + 5.
Any-of-21 versus any-of-6 favours the cue arm before any cue is delivered.

The cue arm records k_cue = the number of the 20 samples that hit the target,
so the arm can be downsampled to 5 draws EXACTLY, with no refit and no
independence assumption: drawing 5 of the 20 without replacement,

    P(no hit in 5 | k hits in 20) = C(20-k, 5) / C(20, 5)

is the exact sampling distribution of the smaller budget. That yields a true
greedy+5 versus greedy+5 comparison on the same items.

The reverse direction is also reported as a sensitivity: extrapolating the CoT
arm UP to 20 draws needs an iid assumption (1-(1-p)^20 with p = k_cot/5) and is
anticonservative for us, so it bounds the increment from below.

  python -u cot_budget.py --out ../results/cot_budget.txt
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from math import comb

import numpy as np

from analysis_common import PRIMARY6, ROOT, classify, load_jsonl

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def any_of_m(k, n, m):
    """Exact P(>=1 hit) when m of the n stored draws are kept, k of which hit."""
    if k <= 0:
        return 0.0
    if n - k < m:
        return 1.0
    return 1.0 - comb(n - k, m) / comb(n, m)


def build():
    """One row per (dataset, item) on the matched cue/CoT population."""
    rows = []
    for lang in ("en", "zh"):
        for tag in PRIMARY6:
            base = os.path.join(ROOT, "results", f"full_{lang}_{tag}")
            cp = os.path.join(base, "cue.jsonl")
            if not os.path.exists(cp):
                continue
            prod = load_jsonl(os.path.join(base, "production.jsonl"))
            rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
            cot = load_jsonl(os.path.join(base, "cot.jsonl"))
            cues = defaultdict(dict)
            with open(cp, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        cues[r["item_id"]][r["condition"]] = r
            for iid, conds in cues.items():
                p = prod.get(iid)
                cc, ct = conds.get("correct"), cot.get(iid)
                if not p or cc is None or ct is None:
                    continue
                if classify(p, rec.get(iid)) not in ("TOT", "DK"):
                    continue
                nq = cc.get("n_cue", cc.get("n", 20)) or 20
                nt = ct.get("n_cot", ct.get("n", 5)) or 5
                kq, kt = cc.get("k_cue", 0), ct.get("k_cot", 0)
                gq, gt = float(cc["greedy_correct"]), float(ct["greedy_correct"])
                rows.append({
                    "lang": lang, "ds": f"{lang}/{tag}", "item": f"{lang}_{iid}",
                    # A: main convention, greedy + all stored samples (21 vs 6)
                    "cue_full": 1.0 if (gq or kq > 0) else 0.0,
                    "cot_full": 1.0 if (gt or kt > 0) else 0.0,
                    # B: greedy only, matched 1-to-1
                    "cue_g": gq, "cot_g": gt,
                    # C: cue downsampled to the CoT budget, exact (greedy + 5)
                    "cue_dn": 1.0 if gq else any_of_m(kq, nq, min(nt, nq)),
                    # D: BOTH arms extrapolated to 20 draws from their own
                    #    per-sample rate (iid). Coherent only if both sides are
                    #    modelled: pitting a modelled CoT against the cue arm's
                    #    observed indicator would shrink CoT, not grow it.
                    "cue_20": 1.0 if gq else 1.0 - (1.0 - kq / nq) ** nq,
                    "cot_20": 1.0 if gt else 1.0 - (1.0 - kt / nt) ** nq,
                    "nq": nq, "nt": nt})
    return rows


def pooled(rows, a, b, lang, boot=4000, seed=0):
    """Paired difference a-b, item-block bootstrap over shared item ids."""
    R = [r for r in rows if r["lang"] == lang]
    d = np.array([r[a] - r[b] for r in R])
    items = np.array([r["item"] for r in R])
    uniq, inv = np.unique(items, return_inverse=True)
    by = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    bs = np.empty(boot)
    for i in range(boot):
        pick = rng.integers(0, len(by), len(by))
        bs[i] = d[np.concatenate([by[j] for j in pick])].mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return (np.mean([r[a] for r in R]), np.mean([r[b] for r in R]),
            d.mean(), lo, hi, len(R), len(uniq))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "cot_budget.txt"))
    args = ap.parse_args()
    rows = build()
    say("=" * 92)
    say("IS THE CUE-MINUS-RETRY INCREMENT A SAMPLING-BUDGET ARTIFACT?")
    say("=" * 92)
    say(f"matched rows (correct cue AND CoT retry present) = {len(rows)}")
    say(f"cue budgets n_cue = {sorted({r['nq'] for r in rows})}; "
        f"cot budgets n_cot = {sorted({r['nt'] for r in rows})}")

    specs = [
        ("A published    greedy+20 vs greedy+5 ", "cue_full", "cot_full"),
        ("B greedy only  1 vs 1  (matched)     ", "cue_g", "cot_g"),
        ("C cue DOWN to greedy+5 (exact)       ", "cue_dn", "cot_full"),
        ("D both UP to greedy+20 (iid)         ", "cue_20", "cot_20"),
    ]
    for lang in ("en", "zh"):
        say(f"\n--- {lang.upper()} " + "-" * 78)
        for label, a, b in specs:
            ra, rb, d, lo, hi, n, ni = pooled(rows, a, b, lang)
            say(f"  {label} cue {ra:.4f}  retry {rb:.4f} | "
                f"increment {d:+.4f} [{lo:+.4f},{hi:+.4f}]  "
                f"n={n} rows / {ni} items")

    say("\nper-dataset, spec C (the budget-matched increment), "
        "item-block bootstrap:")
    nsig = 0
    for ds in sorted({r["ds"] for r in rows}):
        R = [r for r in rows if r["ds"] == ds]
        lang = R[0]["lang"]
        _, _, d, lo, hi, n, _ = pooled([r for r in rows if r["ds"] == ds],
                                       "cue_dn", "cot_full", lang)
        sig = lo > 0
        nsig += sig
        say(f"  {ds:14s} n={n:4d} | cue@5 "
            f"{np.mean([r['cue_dn'] for r in R]):.3f}  CoT@5 "
            f"{np.mean([r['cot_full'] for r in R]):.3f} | "
            f"{d:+.4f} [{lo:+.4f},{hi:+.4f}] {'*' if sig else ''}")
    say(f"  individually positive with CI excluding 0: {nsig}/12 datasets")

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(("\n".join(OUT) + "\n").rstrip("\n") + "\n")
    say(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
