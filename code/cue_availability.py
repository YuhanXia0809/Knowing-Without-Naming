"""Cue-unlocked PRODUCTION as generative evidence of availability.

The construct critique says the TOT cell is inferred from 4AFC
recognition, which is a forced-choice judgement and could reflect
gloss-to-option inference rather than lexical availability.  The cue
intervention is immune to that: every item here is a STRICT production
failure (k_prod = 0 of 20 and greedy wrong), so the no-cue production
rate is exactly 0 by construction; anything produced after a one-unit
form cue is the target itself, generated, not chosen from a menu.

Per model x language, among strict failures classified TOT vs DK:
  * production rate after the CORRECT first-unit cue (any of 20 samples,
    and the stricter greedy-only variant), with Wilson CIs;
  * the same under the WRONG-letter PLACEBO cue;
  * NET cue-unlocked = produced under the correct cue and NOT under the
    placebo (within-item, so it nets out cue-independent recovery);
  * TOT-vs-DK differences with Newcombe CIs and Fisher p;
  * the pooled contrast: fixed-effect logistic (dataset FE, item-
    clustered SEs) and a DerSimonian-Laird random-effects pool of the
    per-dataset log-ORs.

  python -u cue_availability.py --out ../results/cue_availability.txt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import pandas as pd
import statsmodels.api as sm

from paths import prop_test
from analysis_common import (CORE, PRIMARY6, PRETTY, ROOT, classify, dl_yv,
                        fmt_meta, load_jsonl, logor_2x2, wilson, zscore)

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def zipf_of(it, lang):
    fpm = it.get("freq_per_million")
    if lang == "zh" and fpm and fpm > 0 and "log_freq" in it:
        return math.log10(fpm) + 3.0
    return it.get("zipf")


def build(models):
    """One row per (dataset, item): both cue conditions side by side."""
    rows = []
    for lang in ("en", "zh"):
        items = load_jsonl(os.path.join(ROOT, "data",
                                        f"items_full_{lang}_final.jsonl"))
        len_col = "n_letters" if lang == "en" else "n_chars"
        for tag in models:
            base = os.path.join(ROOT, "results", f"full_{lang}_{tag}")
            cp = os.path.join(base, "cue.jsonl")
            if not os.path.exists(cp):
                continue
            prod = load_jsonl(os.path.join(base, "production.jsonl"))
            rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
            tok_col = "n_tokens_qwen" if "qwen" in tag else "n_tokens_llama"
            cues = defaultdict(dict)
            with open(cp, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        cues[r["item_id"]][r["condition"]] = r
            for iid, conds in cues.items():
                it = items.get(iid)
                p = prod.get(iid)
                if not it or not p:
                    continue
                st = classify(p, rec.get(iid))
                if st not in ("TOT", "DK"):
                    continue
                cc, ww = conds.get("correct"), conds.get("wrong")
                if cc is None:
                    continue

                def res(r):
                    return None if r is None else (
                        1.0 if (r["greedy_correct"] or r["k_cue"] > 0) else 0.0)

                def gre(r):
                    return None if r is None else (
                        1.0 if r["greedy_correct"] else 0.0)
                rows.append({
                    "lang": lang, "model": tag, "ds": f"{lang}/{tag}",
                    "item": iid, "state": st,
                    "is_TOT": 1.0 if st == "TOT" else 0.0,
                    "pre_cue_k_prod": p["k_prod"],
                    "pre_cue_greedy": 1.0 if p["greedy_correct"] else 0.0,
                    "corr": res(cc), "corr_greedy": gre(cc),
                    "plac": res(ww), "plac_greedy": gre(ww),
                    "zipf": zipf_of(it, lang), "length": it.get(len_col),
                    "tokens": it.get(tok_col)})
    df = pd.DataFrame(rows)
    df["net"] = df.apply(
        lambda r: (1.0 if (r["corr"] == 1.0 and r["plac"] == 0.0) else 0.0)
        if (r["corr"] is not None and r["plac"] is not None) else None, axis=1)
    return df


def cell(sub, col):
    s = sub[col].dropna()
    k, n = int(s.sum()), int(len(s))
    lo, hi = wilson(k, n)
    return k, n, (k / n if n else float("nan")), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(PRIMARY6))
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "cue_availability.txt"))
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    df = build(models)

    say("=" * 100)
    say("CUE-UNLOCKED PRODUCTION AS GENERATIVE EVIDENCE OF AVAILABILITY")
    say("=" * 100)
    say(f"rows (dataset x item) = {len(df)}; datasets = "
        f"{sorted(df.ds.unique())}")
    say(f"SANITY -- every analysed item is a strict production failure: "
        f"max pre-cue k_prod = {df.pre_cue_k_prod.max():.0f} of 20, "
        f"pre-cue greedy-correct rate = {df.pre_cue_greedy.mean():.4f} "
        f"(both must be 0: the uncued production rate is 0/n by "
        f"construction, so any post-cue production is newly generated)")

    say("\n" + "-" * 100)
    say("PER-DATASET PRODUCTION RATES AFTER THE CORRECT FIRST-UNIT CUE "
        "(any of 20 samples)")
    say("-" * 100)
    say(f"{'dataset':16s} {'state':5s} {'k/n':>10s} {'rate':>6s} "
        f"{'95% Wilson':>16s} | {'placebo':>14s} | {'net unlocked':>16s} | "
        f"{'greedy-only':>14s}")
    for ds in sorted(df.ds.unique()):
        for st in ("TOT", "DK"):
            s = df[(df.ds == ds) & (df.state == st)]
            if not len(s):
                continue
            k, n, r, lo, hi = cell(s, "corr")
            pk, pn, pr, plo, phi = cell(s, "plac")
            nk, nn, nr, nlo, nhi = cell(s, "net")
            gk, gn, gr, glo, ghi = cell(s, "corr_greedy")
            say(f"{ds:16s} {st:5s} {k:4d}/{n:<5d} {r:6.3f} "
                f"[{lo:.3f},{hi:.3f}]".rjust(16) +
                f" | {pk:3d}/{pn:<4d} {pr:.3f}".rjust(15) +
                f" | {nk:3d}/{nn:<4d} {nr:.3f}".rjust(17) +
                f" | {gk:3d}/{gn:<4d} {gr:.3f}".rjust(15))
        s = df[df.ds == ds]
        t = s[s.state == "TOT"]["corr"].dropna()
        d = s[s.state == "DK"]["corr"].dropna()
        if len(t) and len(d):
            pt = prop_test(int(t.sum()), len(t), int(d.sum()), len(d))
            say(f"{'':16s} {'TOT-DK':5s} diff={pt['diff']:+.3f} "
                f"95%CI[{pt['lo']:+.3f},{pt['hi']:+.3f}] "
                f"Fisher p={pt['p']:.4f}")

    # ---------------------------------------------------------- pooled by lang
    for lang in ("en", "zh", "both"):
        s = df if lang == "both" else df[df.lang == lang]
        say("\n" + "-" * 100)
        say(f"POOLED [{lang}] -- raw cell counts (items pooled across "
            f"datasets; datasets share items, so these are descriptive)")
        for col, name in (("corr", "correct cue"), ("plac", "placebo cue"),
                          ("net", "net unlocked"),
                          ("corr_greedy", "correct cue, greedy only")):
            for st in ("TOT", "DK"):
                k, n, r, lo, hi = cell(s[s.state == st], col)
                say(f"  {name:26s} {st:3s}  {k:5d}/{n:<6d} = {r:.3f} "
                    f"[{lo:.3f},{hi:.3f}]")
            t = s[s.state == "TOT"][col].dropna()
            d = s[s.state == "DK"][col].dropna()
            if len(t) and len(d):
                pt = prop_test(int(t.sum()), len(t), int(d.sum()), len(d))
                say(f"  {name:26s} TOT-DK diff={pt['diff']:+.3f} "
                    f"95%CI[{pt['lo']:+.3f},{pt['hi']:+.3f}] "
                    f"Fisher p={pt['p']:.4g}")

    # ------------------------------------------------------- pooled contrasts
    say("\n" + "=" * 100)
    say("POOLED TOT-vs-DK CONTRAST ON CUE-UNLOCKED PRODUCTION")
    say("=" * 100)
    for col, name in (("corr", "correct-cue production"),
                      ("net", "net cue-unlocked production"),
                      ("plac", "placebo-cue production")):
        d0 = df.dropna(subset=[col, "zipf", "length", "tokens"]).copy()
        X = pd.DataFrame({"const": 1.0, "is_TOT": d0["is_TOT"],
                          "zipf": zscore(d0["zipf"]),
                          "length": zscore(d0["length"]),
                          "tokens": zscore(d0["tokens"])}, index=d0.index)
        for m in sorted(d0.ds.unique())[1:]:
            X[f"d_{m}"] = (d0["ds"] == m).astype(float)
        try:
            fit = sm.GLM(d0[col], X, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": d0["item"]})
            ci = fit.conf_int()
            say(f"\n  {name}: FE logistic (dataset FE, item-clustered) "
                f"b_isTOT={fit.params['is_TOT']:+.3f} "
                f"SE={fit.bse['is_TOT']:.3f} "
                f"CI[{ci.loc['is_TOT', 0]:+.3f},{ci.loc['is_TOT', 1]:+.3f}] "
                f"p={fit.pvalues['is_TOT']:.5f}  n={len(d0)}")
        except Exception as e:
            say(f"\n  {name}: FE logistic failed: {str(e)[:80]}")
        ys, vs, labs = [], [], []
        for ds in sorted(d0.ds.unique()):
            s = d0[d0.ds == ds]
            t, k = s[s.is_TOT == 1], s[s.is_TOT == 0]
            if not len(t) or not len(k):
                continue
            y, v = logor_2x2(int(t[col].sum()), int(len(t) - t[col].sum()),
                             int(k[col].sum()), int(len(k) - k[col].sum()))
            ys.append(y)
            vs.append(v)
            labs.append(ds)
        if len(ys) >= 2:
            say(fmt_meta(dl_yv(ys, vs, labs), f"{name}: TOT-vs-DK log-OR"))

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(("\n".join(OUT) + "\n").rstrip("\n") + "\n")
    print("wrote", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
