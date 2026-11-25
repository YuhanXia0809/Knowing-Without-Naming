"""The fully matched no-cue control for the cue arm.

The cue prompt is the production prompt with one line inserted, and the cue
stage already inherits PROD_SYSTEM, MAX_TOKENS_WORD, N_PROD and T_PROD. The
`retest` stage replays that prompt on the cue stage's own item pool with an
independent seed, so cue-minus-retest differs in the first unit and nothing
else: same template, same system prompt, same 20 draws at T=1, same single turn.

Why this is the valid contrast. The pool is selected on k_prod<=2 out of 20, so
production's own rate on it is ~0 by construction and cannot serve as a
baseline. Both the cue arm and the retest arm are fresh measurements on the
same selected items, so the selection acts identically on both and cancels in
the paired difference. The CoT retry also escapes the selection problem, but it
changes the instruction, the output format and the sample budget at the same
time as it removes the cue.

Greedy is excluded from the matched contrast on purpose: temperature 0 on the
production prompt reproduces production's own greedy answer, which is wrong for
every item in the pool. Only the sampled arms are informative, so the matched
comparison is any-of-20 against any-of-20.

  python -u retest_contrast.py --out ../results/retest_contrast.txt
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from analysis_common import PRIMARY6, ROOT, classify, load_jsonl

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def build(strict):
    rows = []
    for lang in ("en", "zh"):
        for tag in PRIMARY6:
            base = os.path.join(ROOT, "results", f"full_{lang}_{tag}")
            cp = os.path.join(base, "cue.jsonl")
            rp = os.path.join(base, "retest.jsonl")
            if not (os.path.exists(cp) and os.path.exists(rp)):
                continue
            prod = load_jsonl(os.path.join(base, "production.jsonl"))
            rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
            cot = load_jsonl(os.path.join(base, "cot.jsonl"))
            ret = load_jsonl(rp)
            cues = defaultdict(dict)
            with open(cp, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        cues[r["item_id"]][r["condition"]] = r
            for iid, conds in cues.items():
                p, cc, rr = prod.get(iid), conds.get("correct"), ret.get(iid)
                ww, ct = conds.get("wrong"), cot.get(iid)
                if not p or cc is None or rr is None:
                    continue
                if classify(p, rec.get(iid)) not in ("TOT", "DK"):
                    continue
                if strict and p.get("k_prod", 0) != 0:
                    continue
                rows.append({
                    "lang": lang, "ds": f"{lang}/{tag}", "item": f"{lang}_{iid}",
                    # matched pair: 20 sampled draws each, same template
                    "cue_s": 1.0 if cc.get("k_cue", 0) > 0 else 0.0,
                    "ret_s": 1.0 if rr.get("k_retest", 0) > 0 else 0.0,
                    "plac_s": 0.0 if ww is None else (
                        1.0 if ww.get("k_cue", 0) > 0 else 0.0),
                    # main convention, for reference only
                    "cue_full": 1.0 if (cc["greedy_correct"]
                                        or cc.get("k_cue", 0) > 0) else 0.0,
                    "cot_full": None if ct is None else (
                        1.0 if (ct["greedy_correct"]
                                or ct.get("k_cot", 0) > 0) else 0.0),
                    "ret_g": float(rr["greedy_correct"])})
    return rows


def pooled(rows, a, b, lang, boot=4000, seed=0):
    R = [r for r in rows if r["lang"] == lang
         and r[a] is not None and r[b] is not None]
    if not R:
        return None
    d = np.array([r[a] - r[b] for r in R])
    uniq, inv = np.unique(np.array([r["item"] for r in R]), return_inverse=True)
    by = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    bs = np.empty(boot)
    for i in range(boot):
        pick = rng.integers(0, len(by), len(by))
        bs[i] = d[np.concatenate([by[j] for j in pick])].mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return (np.mean([r[a] for r in R]), np.mean([r[b] for r in R]),
            d.mean(), lo, hi, len(R), len(uniq))


def report(rows, title):
    say("\n" + "=" * 92)
    say(title)
    say("=" * 92)
    if not rows:
        say("  no rows -- retest.jsonl not present yet")
        return
    say(f"rows = {len(rows)}; datasets = {len(set(r['ds'] for r in rows))}")
    g = np.mean([r["ret_g"] for r in rows])
    say(f"retest greedy correct = {g:.4f}  (expected ~0: temperature 0 on the "
        f"production prompt replays production's own greedy answer)")

    specs = [
        ("MATCHED  cue@20 vs retest@20 (the causal estimate)", "cue_s", "ret_s"),
        ("published cue@21 vs CoT@6 (reference)             ", "cue_full",
         "cot_full"),
        ("CoT@6 vs retest@20  (what CoT buys beyond redraw) ", "cot_full",
         "ret_s"),
        ("placebo@20 vs retest@20 (suppression, matched)    ", "plac_s",
         "ret_s"),
    ]
    for lang in ("en", "zh"):
        say(f"\n--- {lang.upper()} " + "-" * 78)
        for label, a, b in specs:
            r = pooled(rows, a, b, lang)
            if r is None:
                continue
            ra, rb, d, lo, hi, n, ni = r
            say(f"  {label} {ra:.4f} vs {rb:.4f} | "
                f"{d:+.4f} [{lo:+.4f},{hi:+.4f}]  n={n}/{ni} items")

    say("\nper-dataset, the matched contrast:")
    nsig = 0
    for ds in sorted({r["ds"] for r in rows}):
        R = [r for r in rows if r["ds"] == ds]
        res = pooled(R, "cue_s", "ret_s", R[0]["lang"])
        ra, rb, d, lo, hi, n, _ = res
        nsig += lo > 0
        say(f"  {ds:14s} n={n:4d} | cue {ra:.3f}  retest {rb:.3f} | "
            f"{d:+.4f} [{lo:+.4f},{hi:+.4f}] {'*' if lo > 0 else ''}")
    say(f"  positive with CI excluding 0: {nsig}/"
        f"{len(set(r['ds'] for r in rows))} datasets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "retest_contrast.txt"))
    args = ap.parse_args()
    report(build(strict=True),
           "MATCHED NO-CUE CONTROL -- strict failures (k_prod = 0/20), "
           "as in the paper")
    report(build(strict=False),
           "same, on the full cue pool (k_prod <= 2/20)")
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(("\n".join(OUT) + "\n").rstrip("\n") + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
