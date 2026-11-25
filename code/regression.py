"""Norms regressions + FOK calibration stats on full-run results.

Per model:
  A. production strict-failure ~ z(zipf) + z(aoa) + z(concreteness) + z(n_letters)
     + z(n_tokens) + z(ortho_N)   [full covariates -> normed subset only]
  B. same minus aoa/concreteness on ALL items (hard pool lacks norms)
  C. TOT-vs-DK (recognition success on strict-fail subset) ~ B-covariates
  D. FOK -> recognition: AUC + Goodman-Kruskal gamma (+ bootstrap CI)
Pooled fits use GLM with cluster-robust SEs clustered on item.

  python -u regression.py --items ../data/items_full_en_final.jsonl \
      --results-root ../results --models full_en_qwen7b,full_en_llama8b \
      --lang en --out ../results/regression_en.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_jsonl(path, key="item_id"):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    r = json.loads(line)
                    out[r[key]] = r
                except Exception:
                    pass
    return out


def strict_fail(p):
    return p["k_prod"] == 0 and not p["greedy_correct"]


def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


def gamma_gk(pairs):
    """Goodman-Kruskal gamma over (score, binary outcome) pairs."""
    pairs = [(s, y) for s, y in pairs if s is not None]
    conc = disc = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            (s1, y1), (s2, y2) = pairs[i], pairs[j]
            if y1 == y2 or s1 == s2:
                continue
            if (s1 > s2) == (y1 > y2):
                conc += 1
            else:
                disc += 1
    return (conc - disc) / (conc + disc) if conc + disc else None


def auc(pairs):
    pairs = [(s, y) for s, y in pairs if s is not None]
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def boot_ci(fn, pairs, n_boot=1000, seed=1):
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        samp = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        v = fn(samp)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]]


def fit_logit(df, outcome, predictors, cluster=None):
    df = df.dropna(subset=[outcome] + predictors).copy()
    if len(df) < 30 or df[outcome].nunique() < 2:
        return {"n": len(df), "error": "insufficient data"}
    X = sm.add_constant(df[predictors].apply(zscore))
    y = df[outcome].astype(float)
    try:
        if cluster:
            m = sm.GLM(y, X, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": df[cluster]})
        else:
            m = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="HC3")
        out = {"n": int(len(df)), "coef": {}, "vif": {}}
        # (quasi-)separation: some fitted probabilities reach 0 or 1, the
        # likelihood has no finite maximum and the standard errors are
        # not estimable (NaN, or values that change between runs)
        mu = np.asarray(m.fittedvalues, dtype=float)
        n_ext = int(((mu < 1e-6) | (mu > 1 - 1e-6)).sum())
        sep = n_ext > 0
        if sep:
            out["separation"] = (
                "fitted probability within 1e-6 of 0 or 1 for %d of %d "
                "rows; se and p not estimable (null)" % (n_ext, len(df)))
        for name in X.columns:
            out["coef"][name] = {
                "b": round(float(m.params[name]), 4),
                "se": None if sep else round(float(m.bse[name]), 4),
                "p": None if sep else round(float(m.pvalues[name]), 5)}
        # VIF via R^2 of each predictor on the others
        for pcol in predictors:
            others = [q for q in predictors if q != pcol]
            if others:
                r2 = sm.OLS(df[pcol], sm.add_constant(df[others])).fit().rsquared
                out["vif"][pcol] = round(1.0 / max(1e-9, 1 - r2), 2)
        return out
    except Exception as e:
        return {"n": int(len(df)), "error": str(e)[:200]}


def classify(prod, rec):
    if prod is None:
        return None
    sfail = strict_fail(prod)
    if prod["k_prod"] >= 6:
        return "known"
    if not sfail:
        return "weak" if prod["k_prod"] <= 2 else "ambiguous_prod"
    if rec is None or rec.get("n_rec", 0) < 4:
        return "fail_norec"
    return "TOT" if rec["k_rec"] >= 3 else ("DK" if rec["k_rec"] <= 1 else "ambiguous_rec")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--models", required=True, help="comma list of result dir names")
    ap.add_argument("--lang", choices=["en", "zh"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = load_jsonl(args.items)
    freq_col = "zipf" if args.lang == "en" else None
    len_col = "n_letters" if args.lang == "en" else "n_chars"

    report = {"lang": args.lang, "models": {}}
    pooled_rows = []
    for mdir in args.models.split(","):
        mdir = mdir.strip()
        base = os.path.join(args.results_root, mdir)
        prod = load_jsonl(os.path.join(base, "production.jsonl"))
        rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
        fok = load_jsonl(os.path.join(base, "fok.jsonl"))
        tok_col = "n_tokens_qwen" if "qwen" in mdir else "n_tokens_llama"
        rows = []
        for iid, it in items.items():
            p = prod.get(iid)
            if not p:
                continue
            if args.lang == "en":
                zipf = it.get("zipf")
            else:
                # harmonize the two ZH pools onto one Zipf scale:
                # v2 pool has freq_per_million (SUBTLEX-CH); hard pool has wordfreq zipf
                fpm = it.get("freq_per_million")
                if fpm and fpm > 0 and "log_freq" in it:
                    zipf = math.log10(fpm) + 3.0
                else:
                    zipf = it.get("zipf")
            state = classify(p, rec.get(iid))
            rows.append({
                "item_id": iid, "model": mdir,
                "fail": 1 if strict_fail(p) else 0,
                "k_prod": p["k_prod"],
                "state": state,
                "rec_ok": (1 if state == "TOT" else 0) if state in ("TOT", "DK") else None,
                "zipf": zipf, "aoa": it.get("aoa"),
                "concreteness": it.get("concreteness"),
                "length": it.get(len_col), "tokens": it.get(tok_col),
                "ortho_N": it.get("ortho_N"),
                "fok_verbal": fok.get(iid, {}).get("fok_verbal"),
                "fok_p_yes": fok.get(iid, {}).get("fok_p_yes"),
                "source_pool": it.get("source_pool"), "unique": it.get("unique")})
        df = pd.DataFrame(rows)
        pooled_rows.extend(rows)
        mrep = {"n_items": len(df), "fail_rate": round(float(df["fail"].mean()), 4),
                "states": dict(Counter(df["state"]))}
        preds_full = (["zipf", "aoa", "concreteness", "length", "tokens", "ortho_N"]
                      if args.lang == "en" else ["zipf", "aoa", "length", "tokens"])
        preds_red = ["zipf", "length", "tokens"] + (
            ["ortho_N"] if args.lang == "en" else [])
        mrep["A_fail_full_covars"] = fit_logit(df, "fail", preds_full)
        mrep["B_fail_reduced_all_items"] = fit_logit(df, "fail", preds_red)
        dfc = df[df["rec_ok"].notna()]
        mrep["C_totdk"] = fit_logit(dfc, "rec_ok", preds_red)
        # Gollan & Brown step two — the norm
        # signature must be tested on TOT-vs-DK among failures, not on
        # strict failure itself.
        mrep["C_totdk_full_norms"] = fit_logit(dfc, "rec_ok", preds_full)
        mrep["C_n_dk"] = int((dfc["rec_ok"] == 0).sum())
        fokpairs_v = [(r["fok_verbal"], int(r["rec_ok"]))
                      for r in rows if r["rec_ok"] is not None]
        fokpairs_y = [(r["fok_p_yes"], int(r["rec_ok"]))
                      for r in rows if r["rec_ok"] is not None]
        mrep["D_fok"] = {
            "n_fail": len(fokpairs_v),
            "verbal_auc": auc(fokpairs_v), "verbal_gamma": gamma_gk(fokpairs_v),
            "verbal_gamma_ci": boot_ci(gamma_gk, fokpairs_v) if fokpairs_v else None,
            "pyes_auc": auc(fokpairs_y), "pyes_gamma": gamma_gk(fokpairs_y)}
        report["models"][mdir] = mrep
        print(f"== {mdir}: n={len(df)} fail={mrep['fail_rate']:.1%} "
              f"states={mrep['states']}")

    dfp = pd.DataFrame(pooled_rows)
    preds_red = ["zipf", "length", "tokens"] + (["ortho_N"] if args.lang == "en" else [])
    report["pooled_B"] = fit_logit(dfp, "fail", preds_red, cluster="item_id")
    # The all-item pool mixes a normed pool and a
    # beyond-corpus "hard" pool that differ in more than frequency. Report the
    # frequency coefficient with a pool indicator, and within each pool.
    dfp["hard_pool"] = dfp["source_pool"].astype(str).str.contains(
        "hard").astype(float)
    report["pooled_B_poolctl"] = fit_logit(
        dfp, "fail", preds_red + ["hard_pool"], cluster="item_id")
    report["pooled_B_normedpool"] = fit_logit(
        dfp[dfp["hard_pool"] == 0.0], "fail", preds_red, cluster="item_id")
    report["pooled_B_hardpool"] = fit_logit(
        dfp[dfp["hard_pool"] == 1.0], "fail", preds_red, cluster="item_id")
    dfpc = dfp[dfp["rec_ok"].notna()]
    report["pooled_C_totdk"] = fit_logit(dfpc, "rec_ok", preds_red, cluster="item_id")
    preds_full_p = (["zipf", "aoa", "concreteness", "length", "tokens",
                     "ortho_N"] if args.lang == "en"
                    else ["zipf", "aoa", "length", "tokens"])
    report["pooled_C_totdk_full_norms"] = fit_logit(
        dfpc, "rec_ok", preds_full_p, cluster="item_id")
    report["pooled_C_n_dk"] = int((dfpc["rec_ok"] == 0).sum())

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1, ensure_ascii=False, default=float,
                  allow_nan=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
