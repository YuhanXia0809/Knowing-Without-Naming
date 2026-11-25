"""Shared helpers for the released reanalysis scripts.

Provides the DerSimonian-Laird random-effects machinery in a generic
(effect, variance) form -- selfreport_meta3.dl_pool only accepts 2x2
tables -- plus Haldane log-ORs, Wilson intervals and a zero-variance-safe
logistic fitter.  Conventions (item keying, classify()) come from
analyze_pilot as everywhere else in this repo.
"""
from __future__ import annotations

import math
import os

import numpy as np
from scipy import stats as sps

from analyze_pilot import classify, load_jsonl  # noqa: F401  (re-export)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

CORE = ["qwen7b", "llama8b", "qwen72b", "llama70b"]
PRIMARY6 = ["qwen7b", "qwen14b", "qwen32b", "qwen72b", "llama8b", "llama70b"]
PRETTY = {"qwen7b": "Qwen2.5-7B", "qwen14b": "Qwen2.5-14B",
          "qwen32b": "Qwen2.5-32B", "qwen72b": "Qwen2.5-72B",
          "llama8b": "Llama-3.1-8B", "llama70b": "Llama-3.1-70B"}
NAN = float("nan")


# ------------------------------------------------------------------ effects
def logor_2x2(a, b, c, d, corr=0.5):
    """Haldane-corrected log-OR of a 2x2 [[a,b],[c,d]] and its variance."""
    A, B, C, D = (x + corr for x in (a, b, c, d))
    return math.log(A * D / (B * C)), 1 / A + 1 / B + 1 / C + 1 / D


def log_ratio(n1, n2):
    """log(n1/n2) with the delta-method variance for two independent counts."""
    if n1 <= 0 or n2 <= 0:
        n1, n2 = n1 + 0.5, n2 + 0.5
    return math.log(n1 / n2), 1.0 / n1 + 1.0 / n2


def wilson(k, n, z=1.96):
    if n == 0:
        return NAN, NAN
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return ctr - half, ctr + half


# -------------------------------------------------------------- meta-analysis
def dl_yv(ys, vs, labels=None):
    """DerSimonian-Laird RE pool + full heterogeneity battery on (y, var).

    Identical math to selfreport_meta3.dl_pool, but taking effects and
    variances directly so it can be used on regression coefficients as
    well as on log-ORs.
    """
    ys = np.asarray([float(y) for y in ys], dtype=float)
    vs = np.asarray([float(v) for v in vs], dtype=float)
    k = len(ys)
    w = 1.0 / vs
    mu_f = float((w * ys).sum() / w.sum())
    se_f = float(math.sqrt(1.0 / w.sum()))
    q = float((w * (ys - mu_f) ** 2).sum())
    df = k - 1
    c = w.sum() - (w * w).sum() / w.sum()
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 else 0.0
    h2 = q / df if df > 0 else NAN
    q_p = float(sps.chi2.sf(q, df)) if df > 0 else NAN

    wr = 1.0 / (vs + tau2)
    mu = float((wr * ys).sum() / wr.sum())
    se = float(math.sqrt(1.0 / wr.sum()))
    if k > 1:
        qr = float((wr * (ys - mu) ** 2).sum() / (k - 1))
        se_hk = se * math.sqrt(max(qr, 1e-12))
        tcrit = float(sps.t.ppf(0.975, k - 1))
        hk = (mu - tcrit * se_hk, mu + tcrit * se_hk)
        sepi = math.sqrt(se ** 2 + tau2)
        pi = (mu - tcrit * sepi, mu + tcrit * sepi)
    else:
        hk = pi = (NAN, NAN)
    z = mu / se
    return {"k": k, "labels": list(labels) if labels else None,
            "ys": [round(float(y), 4) for y in ys],
            "ses": [round(float(math.sqrt(v)), 4) for v in vs],
            "mu": mu, "se": se, "lo": mu - 1.96 * se, "hi": mu + 1.96 * se,
            "z": z, "p": float(2 * sps.norm.sf(abs(z))),
            "hk_lo": hk[0], "hk_hi": hk[1], "pi_lo": pi[0], "pi_hi": pi[1],
            "Q": q, "df": df, "Q_p": q_p, "tau2": tau2,
            "tau": math.sqrt(tau2), "I2": i2, "H2": h2,
            "fixed_mu": mu_f, "fixed_se": se_f,
            "fixed_lo": mu_f - 1.96 * se_f, "fixed_hi": mu_f + 1.96 * se_f,
            "fixed_p": float(2 * sps.norm.sf(abs(mu_f / se_f)))}


def fmt_meta(r, name, expo=False, ref=None):
    """One-block printout of a dl_yv result."""
    def tr(x):
        return math.exp(x) if expo else x
    lab = "OR" if expo else "b"
    out = [f"-- {name}  (k={r['k']})"]
    if r["labels"]:
        out.append("   per-dataset: " + ", ".join(
            f"{l}={y:+.3f}({s:.3f})" for l, y, s in
            zip(r["labels"], r["ys"], r["ses"])))
    out.append(f"   FE  {lab}={tr(r['fixed_mu']):+.3f} "
               f"[{tr(r['fixed_lo']):+.3f},{tr(r['fixed_hi']):+.3f}] "
               f"p={r['fixed_p']:.4g}")
    out.append(f"   RE  {lab}={tr(r['mu']):+.3f} "
               f"[{tr(r['lo']):+.3f},{tr(r['hi']):+.3f}] p={r['p']:.4g}"
               f"   HKSJ [{tr(r['hk_lo']):+.3f},{tr(r['hk_hi']):+.3f}]"
               f"   PI [{tr(r['pi_lo']):+.3f},{tr(r['pi_hi']):+.3f}]")
    out.append(f"   HET Q({r['df']})={r['Q']:.2f} p={r['Q_p']:.4g}  "
               f"I2={r['I2']:.1f}%  tau2={r['tau2']:.4f} (tau={r['tau']:.3f})  "
               f"H2={r['H2']:.2f}")
    if ref is not None:
        out.append(f"   paper's published fixed-effect value: {ref}")
    return "\n".join(out)


# ------------------------------------------------------------------- fitting
def zscore(s):
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def fit_logit(df, outcome, predictors, cluster=None, extra=None, zed=True):
    """Cluster-robust binomial GLM; drops zero-variance predictors safely.

    `extra` columns are entered unstandardized (dummies / fixed effects).
    Returns {"n", "coef": {name: {b, se, p}}, "dropped": [...]} or
    {"error": ...}.
    """
    import statsmodels.api as sm
    import pandas as pd
    cols = [outcome] + list(predictors) + list(extra or [])
    if cluster:
        cols.append(cluster)
    d = df.dropna(subset=[c for c in cols if c in df.columns]).copy()
    dropped = []
    keep = []
    for p in predictors:
        if d[p].nunique(dropna=True) < 2:
            dropped.append(p)
        else:
            keep.append(p)
    if len(d) < 30 or d[outcome].nunique() < 2 or not keep:
        return {"n": int(len(d)), "dropped": dropped,
                "error": "insufficient data / no varying predictor"}
    X = pd.DataFrame({"const": 1.0}, index=d.index)
    for p in keep:
        X[p] = zscore(d[p]) if zed else d[p]
    for e in (extra or []):
        X[e] = d[e].astype(float)
    y = d[outcome].astype(float)
    try:
        if cluster:
            m = sm.GLM(y, X, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": d[cluster]})
        else:
            m = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="HC3")
        out = {"n": int(len(d)), "dropped": dropped, "coef": {}}
        for name in X.columns:
            out["coef"][name] = {"b": float(m.params[name]),
                                 "se": float(m.bse[name]),
                                 "p": float(m.pvalues[name])}
        # separation / non-convergence smell test
        if any(abs(v["b"]) > 12 or v["se"] > 12 for v in out["coef"].values()):
            out["warn"] = "possible separation (|b|>12 or se>12)"
        return out
    except Exception as e:  # pragma: no cover
        return {"n": int(len(d)), "dropped": dropped, "error": str(e)[:200]}


def cf(fit, name):
    """(b, se, p) for a coefficient, or (nan, nan, nan)."""
    if "coef" not in fit or name not in fit["coef"]:
        return NAN, NAN, NAN
    c = fit["coef"][name]
    return c["b"], c["se"], c["p"]


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else " "
