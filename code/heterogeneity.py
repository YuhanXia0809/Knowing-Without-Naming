"""Heterogeneity machinery applied to the OTHER pooled headline claims.

The random-effects battery (Q, tau^2, I^2, HKSJ, PI) is
applied to the 28-dataset self-report claim but the other pooled headline
numbers are reported as fixed-effect pools with visible between-dataset
spread.  This script applies the identical DerSimonian-Laird machinery to:

  FAMILY 1  partial access -- the confirmatory pooled first-unit logistic
            Two scales: (a) the per-dataset covariate-adjusted logistic
            coefficient, (b) the unadjusted Haldane log-OR of the 2x2,
            which is always defined even when a cell is empty.
            The exact pooled FE fit (model FE + lang FE, item-clustered)
            is recomputed here so FE and RE sit on the same data.
            The length and sound channels are carried along.

  FAMILY 2  the pooled TOT:DK ratios (paper: 11.2:1 En, 4.9:1 Zh over the
            six primary models).  Effect = log(n_TOT / n_DK), variance
            1/n_TOT + 1/n_DK.  The paper's ratio is an aggregate
            sum(TOT)/sum(DK), which is reported alongside.

  FAMILY 3  the cue-rescue estimates: (a) correct-vs-placebo resolution
            and (b) TOT-vs-DK resolution under the correct cue.

  python -u heterogeneity.py --out ../results/heterogeneity.txt
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

from analyze_pilot import score_partial
from analysis_common import (CORE, PRIMARY6, PRETTY, ROOT, classify, dl_yv,
                        fmt_meta, load_jsonl, logor_2x2, log_ratio, zscore)

_fmt_meta = fmt_meta


def fmt_meta(*a, **k):
    return _fmt_meta(*a, **k).replace(
        "paper's published fixed-effect value:", "reference value:")

ALIGN = {"initial_unit": {"en": "first_letter", "zh": "first_char"},
         "length_unit": {"en": "n_letters", "zh": "n_chars"},
         "sound_unit": {"en": "sounds_like", "zh": "pinyin_initial"}}

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


# ---------------------------------------------------------------- FAMILY 1
def partial_rows(models):
    items_by_lang = {l: load_jsonl(os.path.join(ROOT, "data",
                                                f"items_full_{l}_final.jsonl"))
                     for l in ("en", "zh")}
    rows = []
    for lang in ("en", "zh"):
        items = items_by_lang[lang]
        len_col = "n_letters" if lang == "en" else "n_chars"
        for tag in models:
            base = os.path.join(ROOT, "results", f"full_{lang}_{tag}")
            prod = load_jsonl(os.path.join(base, "production.jsonl"))
            rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
            epis = load_jsonl(os.path.join(base, "partial_episode.jsonl"))
            tok_col = "n_tokens_qwen" if "qwen" in tag else "n_tokens_llama"
            for iid, it in items.items():
                st = classify(prod.get(iid), rec.get(iid))
                if st not in ("TOT", "DK"):
                    continue
                ep = epis.get(iid)
                if not ep or ep.get("group") != "failed_greedy":
                    continue
                for aligned, chmap in ALIGN.items():
                    ch = chmap[lang]
                    sc = score_partial(ch, ep.get(ch, {}).get("greedy"), it,
                                       lang, "target")
                    if sc is None:
                        continue
                    y = float(sc >= 0.5) if isinstance(sc, float) else float(sc)
                    rows.append({"aligned": aligned, "y": y,
                                 "is_TOT": 1.0 if st == "TOT" else 0.0,
                                 "zipf": zipf_of(it, lang),
                                 "length": it.get(len_col),
                                 "tokens": it.get(tok_col),
                                 "model": tag, "lang": lang,
                                 "ds": f"{lang}/{tag}",
                                 "item": f"{lang}_{iid}"})
    return pd.DataFrame(rows).dropna()


def fit_one(sub, cluster=None, fes=()):
    X = pd.DataFrame({"const": 1.0, "is_TOT": sub["is_TOT"],
                      "zipf": zscore(sub["zipf"]),
                      "length": zscore(sub["length"]),
                      "tokens": zscore(sub["tokens"])}, index=sub.index)
    for c in fes:
        X[c] = sub[c]
    try:
        if cluster is not None:
            m = sm.GLM(sub["y"], X, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": cluster})
        else:
            m = sm.GLM(sub["y"], X, family=sm.families.Binomial()).fit(
                cov_type="HC3")
        return (float(m.params["is_TOT"]), float(m.bse["is_TOT"]),
                float(m.pvalues["is_TOT"]))
    except Exception as e:
        return (float("nan"), float("nan"), str(e)[:40])


def family1(models):
    df = partial_rows(models)
    say(f"\nFAMILY 1 rows: {len(df)} over {df.ds.nunique()} datasets")
    for aligned in ALIGN:
        sub = df[df.aligned == aligned].copy()
        if not len(sub):
            continue
        say("\n" + "=" * 78)
        say(f"FAMILY 1 -- {aligned}  (confirmatory channel: initial_unit)")
        say("=" * 78)
        # exact pooled fixed-effect fit, as in pooled_contrast.py
        X = pd.DataFrame({"const": 1.0, "is_TOT": sub["is_TOT"],
                          "zipf": zscore(sub["zipf"]),
                          "length": zscore(sub["length"]),
                          "tokens": zscore(sub["tokens"])}, index=sub.index)
        for m in sorted(sub.model.unique())[1:]:
            X[f"m_{m}"] = (sub["model"] == m).astype(float)
        X["lang_zh"] = (sub["lang"] == "zh").astype(float)
        fit = sm.GLM(sub["y"], X, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": sub["item"]})
        say(f"  exact pooled FE fit (model+lang FE, item-clustered): "
            f"b={fit.params['is_TOT']:+.4f} SE={fit.bse['is_TOT']:.4f} "
            f"p={fit.pvalues['is_TOT']:.5f}  n={len(sub)}")

        ys_a, vs_a, lab_a, ys_o, vs_o, lab_o = [], [], [], [], [], []
        say(f"  {'dataset':16s} {'nTOT':>5s} {'nDK':>5s} {'pTOT':>6s} "
            f"{'pDK':>6s} {'adj b (SE)':>18s} {'logOR (SE)':>18s}")
        for ds in sorted(sub.ds.unique()):
            s2 = sub[sub.ds == ds]
            t = s2[s2.is_TOT == 1]
            d = s2[s2.is_TOT == 0]
            a, b = int(t.y.sum()), int(len(t) - t.y.sum())
            c, dd = int(d.y.sum()), int(len(d) - d.y.sum())
            lo, vo = logor_2x2(a, b, c, dd)
            ys_o.append(lo)
            vs_o.append(vo)
            lab_o.append(ds)
            bb, se, p = fit_one(s2)
            adj = "n/e"
            if bb == bb and se == se and 0 < se < 12 and abs(bb) < 12:
                ys_a.append(bb)
                vs_a.append(se ** 2)
                lab_a.append(ds)
                adj = f"{bb:+.3f} ({se:.3f})"
            say(f"  {ds:16s} {len(t):5d} {len(d):5d} {t.y.mean():6.3f} "
                f"{d.y.mean():6.3f} {adj:>18s} "
                f"{lo:+.3f} ({math.sqrt(vo):.3f})".rjust(18))
        if len(ys_a) >= 2:
            say(fmt_meta(dl_yv(ys_a, vs_a, lab_a),
                         f"{aligned}: covariate-adjusted logistic coefficients",
                         ref=None))
        if len(ys_o) >= 2:
            say(fmt_meta(dl_yv(ys_o, vs_o, lab_o),
                         f"{aligned}: unadjusted Haldane log-OR (2x2)"))


# ---------------------------------------------------------------- FAMILY 2
def family2(models):
    say("\n" + "=" * 78)
    say("FAMILY 2 -- pooled TOT:DK ratio (paper: 11.2:1 En, 4.9:1 Zh)")
    say("=" * 78)
    per = {}
    for lang in ("en", "zh"):
        items = load_jsonl(os.path.join(ROOT, "data",
                                        f"items_full_{lang}_final.jsonl"))
        for tag in models:
            base = os.path.join(ROOT, "results", f"full_{lang}_{tag}")
            prod = load_jsonl(os.path.join(base, "production.jsonl"))
            rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
            if not prod:
                continue
            nt = nd = 0
            for iid in items:
                st = classify(prod.get(iid), rec.get(iid))
                nt += st == "TOT"
                nd += st == "DK"
            per[f"{lang}/{tag}"] = (nt, nd)
    say(f"  {'dataset':18s} {'nTOT':>6s} {'nDK':>5s} {'ratio':>7s}")
    for k, (nt, nd) in per.items():
        say(f"  {k:18s} {nt:6d} {nd:5d} "
            f"{(nt / nd if nd else float('inf')):7.2f}")
    for lang in ("en", "zh", "both"):
        keys = [k for k in per if lang == "both" or k.startswith(lang + "/")]
        ys, vs = zip(*[log_ratio(*per[k]) for k in keys])
        agg_t = sum(per[k][0] for k in keys)
        agg_d = sum(per[k][1] for k in keys)
        r = dl_yv(ys, vs, keys)
        say(fmt_meta(r, f"TOT:DK log-ratio [{lang}]", expo=True,
                     ref=f"aggregate sum(TOT)/sum(DK) = "
                         f"{agg_t}/{agg_d} = {agg_t / agg_d:.2f}:1"))


# ---------------------------------------------------------------- FAMILY 3
def cue_frame(models):
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
                if not it:
                    continue
                st = classify(prod.get(iid), rec.get(iid))
                if st not in ("TOT", "DK"):
                    continue
                for cond, r in conds.items():
                    rows.append({
                        "lang": lang, "model": tag, "ds": f"{lang}/{tag}",
                        "item": iid, "state": st,
                        "is_TOT": 1.0 if st == "TOT" else 0.0,
                        "cond": cond,
                        "correct_cue": 1.0 if cond == "correct" else 0.0,
                        "resolved": 1.0 if (r["greedy_correct"] or
                                            r["k_cue"] > 0) else 0.0,
                        "greedy_only": 1.0 if r["greedy_correct"] else 0.0,
                        "zipf": zipf_of(it, lang),
                        "length": it.get(len_col),
                        "tokens": it.get(tok_col)})
    return pd.DataFrame(rows).dropna()


def family3(models):
    df = cue_frame(models)
    say("\n" + "=" * 78)
    say("FAMILY 3 -- cue rescue")
    say("=" * 78)
    say(f"  rows={len(df)} datasets={sorted(df.ds.unique())}")

    # (a) correct vs placebo, within dataset
    ys, vs, labs = [], [], []
    say(f"\n  (a) correct-cue vs placebo resolution")
    say(f"  {'dataset':16s} {'corr k/n':>12s} {'wrong k/n':>12s} "
        f"{'logOR (SE)':>18s}")
    for ds in sorted(df.ds.unique()):
        s = df[df.ds == ds]
        c = s[s.cond == "correct"]
        w = s[s.cond == "wrong"]
        a, b = int(c.resolved.sum()), int(len(c) - c.resolved.sum())
        cc, d = int(w.resolved.sum()), int(len(w) - w.resolved.sum())
        y, v = logor_2x2(a, b, cc, d)
        ys.append(y)
        vs.append(v)
        labs.append(ds)
        say(f"  {ds:16s} {a:5d}/{len(c):<6d} {cc:5d}/{len(w):<6d} "
            f"{y:+.3f} ({math.sqrt(v):.3f})".rjust(18))
    say(fmt_meta(dl_yv(ys, vs, labs), "cue specificity (correct vs placebo)",
                 ref=None))

    # (b) TOT vs DK under the correct cue
    for cond in ("correct", "wrong"):
        ys, vs, labs = [], [], []
        say(f"\n  (b) TOT vs DK resolution under the {cond} cue")
        say(f"  {'dataset':16s} {'TOT k/n':>12s} {'DK k/n':>12s} "
            f"{'logOR (SE)':>18s}")
        for ds in sorted(df.ds.unique()):
            s = df[(df.ds == ds) & (df.cond == cond)]
            t = s[s.is_TOT == 1]
            k = s[s.is_TOT == 0]
            if not len(t) or not len(k):
                continue
            a, b = int(t.resolved.sum()), int(len(t) - t.resolved.sum())
            c, d = int(k.resolved.sum()), int(len(k) - k.resolved.sum())
            y, v = logor_2x2(a, b, c, d)
            ys.append(y)
            vs.append(v)
            labs.append(ds)
            say(f"  {ds:16s} {a:5d}/{len(t):<6d} {c:5d}/{len(k):<6d} "
                f"{y:+.3f} ({math.sqrt(v):.3f})".rjust(18))
        if len(ys) >= 2:
            say(fmt_meta(dl_yv(ys, vs, labs),
                         f"state effect on rescue ({cond} cue)",
                         ref=None))

    # exact pooled FE fit for reference (cue_analysis.py specification)
    X = pd.DataFrame({"const": 1.0, "is_TOT": df["is_TOT"],
                      "correct_cue": df["correct_cue"],
                      "TOTxCue": df["is_TOT"] * df["correct_cue"],
                      "zipf": zscore(df["zipf"]),
                      "length": zscore(df["length"]),
                      "tokens": zscore(df["tokens"])}, index=df.index)
    for m in sorted(df.ds.unique())[1:]:
        X[f"d_{m}"] = (df["ds"] == m).astype(float)
    fit = sm.GLM(df["resolved"], X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": df["item"]})
    say("\n  exact pooled FE fit (dataset FE, item-clustered), all langs:")
    for k in ("is_TOT", "correct_cue", "TOTxCue"):
        say(f"    {k:12s} b={fit.params[k]:+.3f} SE={fit.bse[k]:.3f} "
            f"p={fit.pvalues[k]:.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "heterogeneity.txt"))
    ap.add_argument("--core", default=",".join(CORE))
    ap.add_argument("--primary", default=",".join(PRIMARY6))
    args = ap.parse_args()
    core = [m.strip() for m in args.core.split(",")]
    prim = [m.strip() for m in args.primary.split(",")]

    say("=" * 78)
    say("HETEROGENEITY CONSISTENCY ACROSS THE POOLED HEADLINE CLAIMS")
    say("       DerSimonian-Laird tau^2, Cochran Q, I^2, H^2, HKSJ CI, 95% PI")
    say("=" * 78)
    family1(core)
    family2(prim)
    say("\n[FAMILY 2 sensitivity: confirmatory four models only]")
    family2(core)
    family3(core)
    say("\n[FAMILY 3 sensitivity: all six primary models with cue data]")
    family3(prim)

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(("\n".join(OUT) + "\n").rstrip("\n") + "\n")
    print("wrote", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
