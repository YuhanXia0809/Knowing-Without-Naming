"""Cue-arm coefficient reconciliation + no-cue CoT retry
baseline for the cue increment.

Three jobs:
  A. Re-run
     cue_analysis.py's own joint model over every plausible roster/language
     slice, and then report ONE canonical joint model (12 datasets, dataset
     FE, item-clustered) giving all three quantities from a single fit with
     the reference level stated, plus the two arm-wise marginal contrasts
     derived from it by linear combination.
  B. The scoping of the pooled cue-unlock rates: cue-unlock rate over the TOT cell,
     over the DK cell, and over ALL cleanly classified strict failures.
  C. Cue-unlock vs no-cue CoT retry on the SAME items: the honest increment
     the correct letter buys beyond simply retrying, and whether the
     wrong-letter placebo suppresses below the no-cue retry baseline.

  python -u cue_state_model.py --out ../results/cue_state_model.txt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm

from paths import prop_test
from analysis_common import (CORE, PRIMARY6, ROOT, classify, dl_yv, fmt_meta,
                        load_jsonl, logor_2x2, wilson, zscore)

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
    """One row per (dataset, item): cue arms, CoT retry, covariates."""
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
            rev = load_jsonl(os.path.join(base, "reverse_judged.jsonl"))
            cot = load_jsonl(os.path.join(base, "cot.jsonl"))
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
                ct = cot.get(iid)
                dc = rev.get(iid, {}).get("def_correct")

                def res(r, kk):
                    return None if r is None else (
                        1.0 if (r["greedy_correct"] or r.get(kk, 0) > 0) else 0.0)
                rows.append({
                    "lang": lang, "model": tag, "ds": f"{lang}/{tag}",
                    "item": f"{lang}_{iid}", "state": st,
                    "is_TOT": 1.0 if st == "TOT" else 0.0,
                    "corr": res(cc, "k_cue"), "plac": res(ww, "k_cue"),
                    "cot": res(ct, "k_cot"),
                    "corr_g": None if cc is None else float(cc["greedy_correct"]),
                    "plac_g": None if ww is None else float(ww["greedy_correct"]),
                    "cot_g": None if ct is None else float(ct["greedy_correct"]),
                    "n_cue": cc.get("n_cue", cc.get("n", None)),
                    "n_cot": (ct or {}).get("n_cot", (ct or {}).get("n", None)),
                    "def_correct": (1.0 if dc else 0.0) if dc is not None else None,
                    "zipf": zipf_of(it, lang), "length": it.get(len_col),
                    "tokens": it.get(tok_col)})
    return pd.DataFrame(rows)


def joint_fit(d, cov=(), fe="ds", label=""):
    """cue_analysis.py's joint model: long format, is_TOT + cue + TOTxCue."""
    lg = []
    for _, r in d.iterrows():
        for cond, y in (("wrong", r["plac"]), ("correct", r["corr"])):
            if y is None or (isinstance(y, float) and np.isnan(y)):
                continue
            lg.append({**r.to_dict(), "cond": cond, "resolved": y,
                       "correct_cue": 1.0 if cond == "correct" else 0.0})
    L = pd.DataFrame(lg)
    X = pd.DataFrame({"const": 1.0, "is_TOT": L["is_TOT"],
                      "correct_cue": L["correct_cue"],
                      "TOTxCue": L["is_TOT"] * L["correct_cue"],
                      "zipf": zscore(L["zipf"]), "length": zscore(L["length"]),
                      "tokens": zscore(L["tokens"])}, index=L.index)
    for c in cov:
        X[c] = L[c]
    for m in sorted(L[fe].unique())[1:]:
        X[f"fe_{m}"] = (L[fe] == m).astype(float)
    fit = sm.GLM(L["resolved"], X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": L["item"]})
    ci = fit.conf_int()
    say(f"\n  [{label}]  n_rows={len(L)}  items={L.item.nunique()}  "
        f"datasets={L[fe].nunique()}")
    for k in ("is_TOT", "correct_cue", "TOTxCue"):
        say(f"    {k:12s} b={fit.params[k]:+.3f} SE={fit.bse[k]:.3f} "
            f"CI[{ci.loc[k, 0]:+.3f},{ci.loc[k, 1]:+.3f}] p={fit.pvalues[k]:.4f}")
    # linear combination: state effect ON THE CORRECT-CUE ARM
    idx = [list(X.columns).index("is_TOT"), list(X.columns).index("TOTxCue")]
    V = np.asarray(fit.cov_params())
    b = fit.params["is_TOT"] + fit.params["TOTxCue"]
    se = math.sqrt(V[idx[0], idx[0]] + V[idx[1], idx[1]] + 2 * V[idx[0], idx[1]])
    from scipy.stats import norm
    say(f"    {'TOT@correct':12s} b={b:+.3f} SE={se:.3f} "
        f"CI[{b - 1.96 * se:+.3f},{b + 1.96 * se:+.3f}] "
        f"p={2 * (1 - norm.cdf(abs(b / se))):.4f}   (= is_TOT + TOTxCue)")
    return fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cue_state_model.txt"))
    args = ap.parse_args()
    df = build(PRIMARY6)

    say("=" * 100)
    say("CUE-ARM COEFFICIENT RECONCILIATION + NO-CUE RETRY BASELINE")
    say("=" * 100)
    say(f"rows (dataset x item) = {len(df)}; datasets = {sorted(df.ds.unique())}")
    say(f"cue sample sizes n_cue = {sorted(df.n_cue.dropna().unique())[:5]}; "
        f"cot sample sizes n_cot = {sorted(df.n_cot.dropna().unique())[:5]}")
    say(f"CoT records available for {int(df.cot.notna().sum())} of {len(df)} rows")

    # ---------------------------------------------------------------- PART A
    say("\n" + "=" * 100)
    say("A. cue_analysis.py joint model, "
        "every roster/language slice")
    say("=" * 100)
    say("(spec: resolved ~ is_TOT + correct_cue + TOTxCue + z(zipf,length,"
        "tokens) + model FE, item-clustered; is_TOT is the state effect AT "
        "THE PLACEBO REFERENCE)")
    for langs, lname in ((("en",), "EN"), (("zh",), "ZH"), (("en", "zh"), "BOTH")):
        for roster, rname in ((CORE, "core4"), (PRIMARY6, "primary6")):
            d = df[(df.lang.isin(langs)) & (df.model.isin(roster))]
            if not len(d):
                continue
            fe = "model" if len(langs) == 1 else "ds"
            # (i) exactly cue_analysis.py: drops rows with missing def_correct
            dd = d.dropna(subset=["corr", "plac", "zipf", "length", "tokens",
                                  "def_correct"])
            if len(dd):
                joint_fit(dd, label=f"{lname}/{rname} dropna-incl-def_correct "
                                    f"(= cue_analysis.py as written)", fe=fe)
            # (ii) same but keeping rows without a reversal judgement
            dk = d.dropna(subset=["corr", "plac", "zipf", "length", "tokens"])
            if len(dk):
                joint_fit(dk, label=f"{lname}/{rname} no-def_correct-filter", fe=fe)

    say("\n" + "-" * 100)
    say("A2. CANONICAL SINGLE JOINT MODEL: all 12 "
        "datasets, dataset FE, item-clustered, reference level = DK state on "
        "the WRONG-UNIT (placebo) arm")
    say("-" * 100)
    dall = df.dropna(subset=["corr", "plac", "zipf", "length", "tokens"])
    joint_fit(dall, label="CANONICAL 12-dataset joint model", fe="ds")
    say("\n  reading: is_TOT = placebo-arm state contrast; TOTxCue = state x cue "
        "interaction; TOT@correct = correct-arm state contrast. All three from "
        "ONE fit, so their signs are guaranteed mutually consistent.")

    # ---------------------------------------------------------------- PART B
    say("\n" + "=" * 100)
    say("B. SCOPING OF THE POOLED CUE-UNLOCK RATES")
    say("=" * 100)
    for lang in ("en", "zh"):
        s = df[df.lang == lang]
        for lab, sub in (("TOT cell", s[s.state == "TOT"]),
                         ("DK cell", s[s.state == "DK"]),
                         ("ALL cleanly classified strict failures (TOT+DK)", s)):
            for col, cn in (("corr", "correct cue"), ("plac", "placebo")):
                v = sub[col].dropna()
                k, n = int(v.sum()), len(v)
                lo, hi = wilson(k, n)
                say(f"  {lang} {lab:48s} {cn:12s} {k:5d}/{n:<6d} = "
                    f"{k / n:.4f} [{lo:.3f},{hi:.3f}]")

    # ---------------------------------------------------------------- PART C
    say("\n" + "=" * 100)
    say("C. CUE-UNLOCK vs NO-CUE CoT RETRY ON THE MATCHED ITEM POPULATION")
    say("=" * 100)
    m = df.dropna(subset=["corr", "plac", "cot"]).copy()
    say(f"matched rows (correct cue AND placebo AND CoT retry all present) = "
        f"{len(m)} of {len(df)}")
    say("\nper dataset x state: rates on the identical items, and the paired "
        "increments (correct-cue minus CoT, placebo minus CoT); paired CI = "
        "Wald on the within-item difference, McNemar exact p")
    from scipy.stats import binomtest
    inc_rows = []
    for ds in sorted(m.ds.unique()):
        for st in ("TOT", "DK", "ALL"):
            s = m[m.ds == ds] if st == "ALL" else m[(m.ds == ds) & (m.state == st)]
            n = len(s)
            if n < 5:
                continue
            rc, rp, rt = s["corr"].mean(), s["plac"].mean(), s["cot"].mean()
            d1 = s["corr"] - s["cot"]
            se1 = d1.std(ddof=1) / math.sqrt(n)
            b01 = int(((s["corr"] == 1) & (s["cot"] == 0)).sum())
            b10 = int(((s["corr"] == 0) & (s["cot"] == 1)).sum())
            pm = binomtest(b01, b01 + b10, 0.5).pvalue if (b01 + b10) else 1.0
            d2 = s["plac"] - s["cot"]
            se2 = d2.std(ddof=1) / math.sqrt(n)
            c01 = int(((s["plac"] == 1) & (s["cot"] == 0)).sum())
            c10 = int(((s["plac"] == 0) & (s["cot"] == 1)).sum())
            pm2 = binomtest(c01, c01 + c10, 0.5).pvalue if (c01 + c10) else 1.0
            say(f"  {ds:14s} {st:3s} n={n:4d} | cue {rc:.3f}  placebo {rp:.3f}  "
                f"CoT {rt:.3f} | cue-CoT {d1.mean():+.3f} "
                f"[{d1.mean() - 1.96 * se1:+.3f},{d1.mean() + 1.96 * se1:+.3f}] "
                f"McNemar p={pm:.2g} | plac-CoT {d2.mean():+.3f} "
                f"[{d2.mean() - 1.96 * se2:+.3f},{d2.mean() + 1.96 * se2:+.3f}] "
                f"p={pm2:.2g}")
            if st == "ALL":
                inc_rows.append((ds, d1.mean(), se1 ** 2, d2.mean(), se2 ** 2))
    say("\npooled increments over the 12 datasets (DL random effects on the "
        "paired risk difference; datasets share items, so see cue_meta.py for "
        "the item-clustered version):")
    say(fmt_meta(dl_yv([r[1] for r in inc_rows], [r[2] for r in inc_rows],
                       [r[0] for r in inc_rows]), "correct-cue MINUS CoT-retry"))
    say(fmt_meta(dl_yv([r[3] for r in inc_rows], [r[4] for r in inc_rows],
                       [r[0] for r in inc_rows]), "placebo MINUS CoT-retry"))
    for lang in ("en", "zh"):
        s = m[m.lang == lang]
        for st in ("TOT", "DK", "ALL"):
            ss = s if st == "ALL" else s[s.state == st]
            a1, a2, a3 = ss["corr"].mean(), ss["plac"].mean(), ss["cot"].mean()
            say(f"  POOLED {lang} {st:3s} n={len(ss):5d} | cue {a1:.4f}"
                f"  placebo {a2:.4f}  CoT {a3:.4f} | "
                f"cue-CoT {a1 - a3:+.4f} | plac-CoT {a2 - a3:+.4f}")

    say("\nSAMPLING-MATCHED ARMS.  The cue arm draws 20 samples per item, the "
        "CoT arm 5, so the any-of-k comparison above favours the cue arm on "
        "sampling budget alone.  The single greedy decode is matched 1-to-1 "
        "across all three arms:")
    g = df.dropna(subset=["corr_g", "plac_g", "cot_g"]).copy()
    say(f"  greedy-matched rows = {len(g)}")
    for ds in sorted(g.ds.unique()):
        for st in ("TOT", "DK", "ALL"):
            s = g[g.ds == ds] if st == "ALL" else g[(g.ds == ds) & (g.state == st)]
            if len(s) < 5:
                continue
            d1 = s["corr_g"] - s["cot_g"]
            se1 = d1.std(ddof=1) / math.sqrt(len(s))
            g1, g2, g3 = s["corr_g"].mean(), s["plac_g"].mean(), s["cot_g"].mean()
            say(f"  {ds:14s} {st:3s} n={len(s):4d} | greedy: cue "
                f"{g1:.3f}  placebo {g2:.3f}  CoT "
                f"{g3:.3f} | cue-CoT {d1.mean():+.3f} "
                f"[{d1.mean() - 1.96 * se1:+.3f},{d1.mean() + 1.96 * se1:+.3f}]")
    for lang in ("en", "zh"):
        for st in ("TOT", "DK", "ALL"):
            s = g[g.lang == lang] if st == "ALL" else \
                g[(g.lang == lang) & (g.state == st)]
            h1, h2, h3 = s["corr_g"].mean(), s["plac_g"].mean(), s["cot_g"].mean()
            say(f"  POOLED-GREEDY {lang} {st:3s} n={len(s):5d} | cue "
                f"{h1:.4f}  placebo {h2:.4f}  CoT "
                f"{h3:.4f} | cue-CoT {h1 - h3:+.4f}"
                f" | plac-CoT {h2 - h3:+.4f}")

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(("\n".join(OUT) + "\n").rstrip("\n") + "\n")
    print("wrote", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
