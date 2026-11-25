"""Item-cluster bootstrap confidence intervals on every headline
TOT:DK ratio.

Every ratio in the paper is a pooled count ratio over datasets that share
one item bank, so the naive binomial interval is wrong twice: the numerator
and denominator are dependent, and a single item contributes up to six
correlated rows.  We resample ITEMS (the independent unit), recompute each
model's TOT/DK counts inside the replicate, pool, and take the percentile
interval of the pooled ratio.

Ratios covered:
  * standard        TOT:DK on the criterial 4AFC, six primary models
  * adversarial     TOT:DK with all three distractors rebuilt from the
                    highest gloss-overlap neighbours, six primary models
  * joint           adversarial recognition AND reversal, four core models,
                    against the comparability denominator DK_adv
  * joint corrected the same with the reversal leg divided by each model's
                    own known-item ceiling (capped at the ungated TOT count)

  python -u ratio_ci.py --reps 1000 --out ../results/ratio_ci.txt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from analysis_common import CORE, PRIMARY6, PRETTY, ROOT, classify, load_jsonl

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def build(lang):
    """Per-model, per-item indicator matrices for one language."""
    items = list(load_jsonl(os.path.join(
        ROOT, "data", f"items_full_{lang}_final.jsonl")))
    idx = {iid: k for k, iid in enumerate(items)}
    n = len(items)
    M = {}
    for tag in PRIMARY6:
        base = os.path.join(ROOT, "results", f"full_{lang}_{tag}")
        advdir = os.path.join(ROOT, "results", f"adv_{lang}_{tag}")
        prod = load_jsonl(os.path.join(base, "production.jsonl"))
        rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
        arec = load_jsonl(os.path.join(advdir, "recognition.jsonl")) \
            if os.path.isdir(advdir) else {}
        rev = load_jsonl(os.path.join(base, "reverse_judged.jsonl"))
        z = lambda: np.zeros(n)                                   # noqa: E731
        d = dict(tot=z(), dk=z(), atot=z(), adk=z(),
                 jtot=z(), jdk=z(), kpass=z(), kn=z())
        for iid in items:
            i = idx[iid]
            p = prod.get(iid)
            st = classify(p, rec.get(iid))
            if st == "TOT":
                d["tot"][i] = 1
            elif st == "DK":
                d["dk"][i] = 1
            if arec:
                sa = classify(p, arec.get(iid))
                if sa == "TOT":
                    d["atot"][i] = 1
                elif sa == "DK":
                    d["adk"][i] = 1
                if sa in ("TOT", "DK", "ambiguous_rec"):
                    rv = rev.get(iid)
                    dc = bool(rv.get("def_correct")) if rv else False
                    if sa == "TOT" and dc:
                        d["jtot"][i] = 1
                    if sa == "DK":
                        d["jdk"][i] = 1
            if st == "known":
                rv = rev.get(iid)
                if rv is not None and rv.get("def_correct") is not None:
                    d["kn"][i] = 1
                    d["kpass"][i] = 1 if rv["def_correct"] else 0
        M[tag] = d
    return items, M


def ratios(M, w, core_only_joint=True):
    """Pooled ratios under item weights w (counts from one bootstrap draw)."""
    T = D = AT = AD = 0.0
    JT = JTC = JD = 0.0
    for tag, d in M.items():
        T += d["tot"] @ w
        D += d["dk"] @ w
        AT += d["atot"] @ w
        AD += d["adk"] @ w
        if tag in CORE:
            jt = d["jtot"] @ w
            jd = d["jdk"] @ w
            kp, kn = d["kpass"] @ w, d["kn"] @ w
            pK = kp / kn if kn else float("nan")
            JT += jt
            JD += jd
            JTC += min(jt / pK, d["tot"] @ w) if pK else float("nan")
    f = lambda a, b: (a / b) if b else float("nan")              # noqa: E731
    return dict(std=f(T, D), adv=f(AT, AD), jnt=f(JT, JD), jntc=f(JTC, JD))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "ratio_ci.txt"))
    args = ap.parse_args()
    say("=" * 92)
    say("ITEM-CLUSTER BOOTSTRAP 95% CIs ON THE POOLED TOT:DK RATIOS")
    say("=" * 92)
    say(f"reps={args.reps}  seed={args.seed}  percentile intervals; the "
        "resampling unit is the ITEM (the bank is shared across models, so a "
        "single item contributes up to six correlated rows).")

    rng = np.random.default_rng(args.seed)
    for lang in ("en", "zh"):
        items, M = build(lang)
        n = len(items)
        w1 = np.ones(n)
        point = ratios(M, w1)
        reps = {k: [] for k in point}
        for _ in range(args.reps):
            draw = rng.integers(0, n, n)
            w = np.bincount(draw, minlength=n).astype(float)
            r = ratios(M, w)
            for k, v in r.items():
                reps[k].append(v)
        say(f"\n--- {lang.upper()}  (bank n={n} items, "
            f"{len(M)} primary models) ---")
        say(f"{'ratio':>26s} {'point':>8s} {'95% CI':>18s}")
        names = dict(std="standard TOT:DK (6 models)",
                     adv="adversarial TOT:DK (6 models)",
                     jnt="joint adv+reversal (4 core)",
                     jntc="joint, instrument-corr.")
        for k in ("std", "adv", "jnt", "jntc"):
            a = np.array([x for x in reps[k] if np.isfinite(x)])
            lo, hi = np.percentile(a, [2.5, 97.5])
            say(f"{names[k]:>26s} {point[k]:8.2f} "
                f"[{lo:6.2f},{hi:6.2f}]")

        # per-cell joint ratios (zh/Llama-8B)
        say(f"  per-cell joint (adv+reversal), literal and corrected:")
        for tag in CORE:
            if tag not in M:
                continue
            d = M[tag]
            cell = {"jnt": [], "jntc": []}
            pt = ratios({tag: d}, w1)
            for _ in range(args.reps):
                draw = rng.integers(0, n, n)
                w = np.bincount(draw, minlength=n).astype(float)
                r = ratios({tag: d}, w)
                cell["jnt"].append(r["jnt"])
                cell["jntc"].append(r["jntc"])
            s = []
            for k in ("jnt", "jntc"):
                a = np.array([x for x in cell[k] if np.isfinite(x)])
                lo, hi = np.percentile(a, [2.5, 97.5])
                s.append(f"{pt[k]:5.2f} [{lo:5.2f},{hi:5.2f}]")
            say(f"    {lang}/{PRETTY[tag]:15s} literal {s[0]}   "
                f"corrected {s[1]}")

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(("\n".join(OUT) + "\n").rstrip("\n") + "\n")
    print("wrote", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
