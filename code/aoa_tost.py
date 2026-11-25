"""TOST equivalence test on the AoA coefficient in the confirmatory
production-failure regression (regression.py analysis A), for the four
confirmatory EN models.

For each of full_en_{qwen7b,llama8b,qwen72b,llama70b}:
  fit  fail ~ z(zipf) + z(aoa) + z(concreteness) + z(length) + z(tokens)
             + z(ortho_N)         [HC3 robust SEs, English normed subset —
                                    same rows regression.py analysis A uses]
  TOST the standardized AoA coefficient against equivalence bounds +-0.2:
    p_lower = P(b > -0.2)  [H0: b <= -0.2]
    p_upper = P(b <  0.2)  [H0: b >=  0.2]
    p_TOST  = max(p_lower, p_upper); equivalence iff p_TOST < alpha
  Also a freq-residualized variant: aoa first residualized on zipf (OLS
  residuals on the analysis subset), then z-scored and used in place of aoa.

Also pools the 4 per-model estimates via inverse-variance weighted
fixed-effect meta-analysis and TOSTs the pooled estimate (raw and
freq-residualized variants).

  python -u aoa_tost.py

Writes results/aoa_tost.json and prints a summary table.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from regression import fit_logit, load_jsonl, strict_fail, zscore

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODELS = ["qwen7b", "llama8b", "qwen72b", "llama70b"]
PREDS_FULL = ["zipf", "aoa", "concreteness", "length", "tokens", "ortho_N"]
BOUND = 0.2
ALPHA = 0.05


def build_model_df(mdir, items):
    base = os.path.join(ROOT, "results", f"full_en_{mdir}")
    prod = load_jsonl(os.path.join(base, "production.jsonl"))
    tok_col = "n_tokens_qwen" if "qwen" in mdir else "n_tokens_llama"
    rows = []
    for iid, it in items.items():
        p = prod.get(iid)
        if not p:
            continue
        rows.append({
            "item_id": iid,
            "fail": 1 if strict_fail(p) else 0,
            "zipf": it.get("zipf"), "aoa": it.get("aoa"),
            "concreteness": it.get("concreteness"),
            "length": it.get("n_letters"), "tokens": it.get(tok_col),
            "ortho_N": it.get("ortho_N"),
        })
    return pd.DataFrame(rows)


def residualize_aoa(df):
    """OLS-residualize aoa on zipf within the analysis subset (same subset
    fit_logit would use for PREDS_FULL: non-null fail + all 6 predictors)."""
    sub = df.dropna(subset=["fail"] + PREDS_FULL).copy()
    X = sm.add_constant(sub[["zipf"]].astype(float))
    ols = sm.OLS(sub["aoa"].astype(float), X).fit()
    sub["aoa_resid"] = ols.resid.values
    return sub


def tost(b, se, bound=BOUND, alpha=ALPHA):
    z_lower = (b - (-bound)) / se
    p_lower = float(1 - norm.cdf(z_lower))          # H0: b <= -bound
    z_upper = (b - bound) / se
    p_upper = float(norm.cdf(z_upper))               # H0: b >=  bound
    p_tost = max(p_lower, p_upper)
    return {"p_lower": round(p_lower, 6), "p_upper": round(p_upper, 6),
            "p_tost": round(p_tost, 6), "equivalent": bool(p_tost < alpha)}


def meta_pool(ests):
    """Inverse-variance weighted fixed-effect meta-analysis."""
    w = np.array([1.0 / e["se"] ** 2 for e in ests])
    b = np.array([e["b"] for e in ests])
    b_pool = float(np.sum(w * b) / np.sum(w))
    se_pool = float(np.sqrt(1.0 / np.sum(w)))
    return b_pool, se_pool


def main():
    items = load_jsonl(os.path.join(ROOT, "data", "items_full_en_final.jsonl"))
    report = {"models": {}}
    raw_ests, resid_ests = [], []

    print(f"{'model':10s} {'variant':10s} {'n':>5s} {'b':>8s} {'se':>7s} "
          f"{'p_tost':>8s}  equivalent")
    for mdir in MODELS:
        df = build_model_df(mdir, items)

        raw = fit_logit(df, "fail", PREDS_FULL)
        b, se = raw["coef"]["aoa"]["b"], raw["coef"]["aoa"]["se"]
        raw_tost = tost(b, se)
        raw_out = {"n": raw["n"], "b": b, "se": se,
                   "p_model": raw["coef"]["aoa"]["p"], **raw_tost}
        raw_ests.append({"b": b, "se": se})

        sub = residualize_aoa(df)
        preds_resid = ["zipf", "aoa_resid", "concreteness", "length", "tokens", "ortho_N"]
        resid = fit_logit(sub, "fail", preds_resid)
        rb, rse = resid["coef"]["aoa_resid"]["b"], resid["coef"]["aoa_resid"]["se"]
        resid_tost = tost(rb, rse)
        resid_out = {"n": resid["n"], "b": rb, "se": rse,
                     "p_model": resid["coef"]["aoa_resid"]["p"], **resid_tost}
        resid_ests.append({"b": rb, "se": rse})

        report["models"][mdir] = {"raw": raw_out, "resid": resid_out}
        print(f"{mdir:10s} {'raw':10s} {raw_out['n']:5d} {b:+8.4f} {se:7.4f} "
              f"{raw_tost['p_tost']:8.5f}  {raw_tost['equivalent']}")
        print(f"{mdir:10s} {'resid':10s} {resid_out['n']:5d} {rb:+8.4f} {rse:7.4f} "
              f"{resid_tost['p_tost']:8.5f}  {resid_tost['equivalent']}")

    b_pool_raw, se_pool_raw = meta_pool(raw_ests)
    pool_raw_tost = tost(b_pool_raw, se_pool_raw)
    b_pool_resid, se_pool_resid = meta_pool(resid_ests)
    pool_resid_tost = tost(b_pool_resid, se_pool_resid)

    report["pooled"] = {
        "raw": {"b": round(b_pool_raw, 4), "se": round(se_pool_raw, 4), **pool_raw_tost},
        "resid": {"b": round(b_pool_resid, 4), "se": round(se_pool_resid, 4), **pool_resid_tost},
    }
    print(f"\n{'POOLED':10s} {'raw':10s} {'':>5s} {b_pool_raw:+8.4f} {se_pool_raw:7.4f} "
          f"{pool_raw_tost['p_tost']:8.5f}  {pool_raw_tost['equivalent']}")
    print(f"{'POOLED':10s} {'resid':10s} {'':>5s} {b_pool_resid:+8.4f} {se_pool_resid:7.4f} "
          f"{pool_resid_tost['p_tost']:8.5f}  {pool_resid_tost['equivalent']}")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "aoa_tost.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1, ensure_ascii=False, default=float)
    print("\nwrote results/aoa_tost.json")


if __name__ == "__main__":
    main()
