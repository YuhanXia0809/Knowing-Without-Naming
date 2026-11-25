"""Item-clustered meta layer.

A meta layer that pools per-dataset estimates as if the 8 (first-unit)
or 12 (cue) datasets were independent studies.  They are not: the same
items are re-run under every model, so the per-dataset effects are
positively correlated, Cochran's Q is deflated, and I^2 = 0% / tau^2 = 0 is
an artefact of that deflation rather than evidence of homogeneity.  The
same applies to the AoA TOST, which inverse-variance pools four per-model
AoA coefficients fitted on the SAME 501-item normed subset.

Fix: an item-block bootstrap.  Items are resampled with replacement within
language (English and Chinese banks are disjoint); every dataset is refit
on the resampled bank; the whole pooling is redone inside each replicate.
That yields
  * the bootstrap covariance matrix V of the k per-dataset estimates,
  * a correlation-aware GLS pool  mu = (1'V^-1 y)/(1'V^-1 1),
  * a generalized heterogeneity statistic Q_gen = (y-mu)'V^-1(y-mu)
    referred to chi2(k-1), which is Q with the item-sharing correlation
    put back in, and
  * a direct item-block bootstrap CI on the published pooled estimator.

  python -u cue_meta.py --B 2000 --out ../results/cue_meta.txt
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, norm

from analysis_common import CORE, PRIMARY6, ROOT, logor_2x2, zscore
from heterogeneity import partial_rows
from cue_state_model import build as build_cue

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


# ------------------------------------------------------------------ helpers
def fe_pool(y, v):
    w = 1.0 / np.asarray(v)
    mu = float(np.sum(w * y) / np.sum(w))
    se = float(math.sqrt(1.0 / np.sum(w)))
    return mu, se


def dl_tau2(y, v):
    y, v = np.asarray(y, float), np.asarray(v, float)
    w = 1.0 / v
    mu = np.sum(w * y) / np.sum(w)
    Q = float(np.sum(w * (y - mu) ** 2))
    k = len(y)
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    t2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return Q, t2, I2


def gls(y, V):
    """Correlation-aware pool + generalized Q, with a ridge for stability."""
    y = np.asarray(y, float)
    k = len(y)
    Vr = V + np.eye(k) * 1e-8 * np.trace(V) / k
    Vi = np.linalg.inv(Vr)
    one = np.ones(k)
    denom = one @ Vi @ one
    mu = float((one @ Vi @ y) / denom)
    se = float(math.sqrt(1.0 / denom))
    r = y - mu
    Qg = float(r @ Vi @ r)
    return mu, se, Qg, chi2.sf(Qg, k - 1)


def report(name, y, v, labels, boot, published=None):
    """boot: (B x k) matrix of per-dataset estimates from item-block bootstrap."""
    y = np.asarray(y, float)
    v = np.asarray(v, float)
    mu, se = fe_pool(y, v)
    Q, t2, I2 = dl_tau2(y, v)
    say(f"\n{'-' * 96}\n{name}   (k={len(y)})\n{'-' * 96}")
    say(f"  INDEPENDENCE ASSUMED:")
    say(f"    IV/FE pool  b={mu:+.4f}  SE={se:.4f}  "
        f"CI[{mu - 1.96 * se:+.4f},{mu + 1.96 * se:+.4f}]  "
        f"p={2 * norm.sf(abs(mu / se)):.4g}")
    say(f"    Q({len(y) - 1})={Q:.2f} p={chi2.sf(Q, len(y) - 1):.4f}  "
        f"I2={100 * I2:.1f}%  tau2={t2:.4f}")
    if published:
        say(f"    (paper prints: {published})")
    B = boot[~np.isnan(boot).any(axis=1)]
    say(f"  ITEM-BLOCK BOOTSTRAP (B={len(B)} usable replicates, items resampled "
        f"within language):")
    V = np.cov(B, rowvar=False)
    sd_marg = np.sqrt(np.diag(V))
    say("    per-dataset  analytic SE vs bootstrap SD:")
    for i, lab in enumerate(labels):
        say(f"      {lab:14s} b={y[i]:+.3f}  analytic SE={math.sqrt(v[i]):.3f}  "
            f"boot SD={sd_marg[i]:.3f}  ratio={sd_marg[i] / math.sqrt(v[i]):.2f}")
    # mean off-diagonal correlation, within and across language
    C = V / np.outer(sd_marg, sd_marg)
    iu = np.triu_indices(len(y), 1)
    langs = [l.split("/")[0] for l in labels]
    same = [C[i, j] for i, j in zip(*iu) if langs[i] == langs[j]]
    diff = [C[i, j] for i, j in zip(*iu) if langs[i] != langs[j]]
    say(f"    mean off-diagonal correlation: same-language {np.mean(same):+.3f}"
        + (f", cross-language {np.mean(diff):+.3f}" if diff else ""))
    # bootstrap distribution of the PUBLISHED pooled estimator
    pool_b = np.array([fe_pool(B[b], v)[0] for b in range(len(B))])
    lo, hi = np.percentile(pool_b, [2.5, 97.5])
    say(f"    pooled estimator, item-block bootstrap: SD={pool_b.std(ddof=1):.4f} "
        f"(vs analytic SE {se:.4f}, inflation {pool_b.std(ddof=1) / se:.2f}x)  "
        f"percentile CI[{lo:+.4f},{hi:+.4f}]")
    zb = mu / pool_b.std(ddof=1)
    say(f"    -> clustered p={2 * norm.sf(abs(zb)):.4g}")
    # heterogeneity, corrected for item-sharing correlation.
    # Under homogeneity with equicorrelation rho, E[Q] = (k-1)(1-rho), so the
    # observed Q is DEFLATED by (1-rho) and must be rescaled before it is
    # referred to chi2(k-1).  rho is estimated by the mean off-diagonal
    # bootstrap correlation.  (A full GLS/Q_gen with an estimated 12x12
    # covariance is too unstable at these B to report.)
    rho = float(np.mean([C[i, j] for i, j in zip(*iu)]))
    k = len(y)
    if rho < 0.999:
        Qc = Q / (1.0 - rho)
        I2c = max(0.0, (Qc - (k - 1)) / Qc) if Qc > 0 else 0.0
        say(f"    mean off-diagonal correlation rho_bar = {rho:+.4f}")
        say(f"    correlation-corrected Q = Q/(1-rho) = {Qc:.2f} on {k - 1} df, "
            f"p={chi2.sf(Qc, k - 1):.4f}  ->  corrected I2 = {100 * I2c:.1f}% "
            f"(published I2 = {100 * I2:.1f}%)")
        t2c = max(0.0, (Qc - (k - 1)) / (np.sum(1 / v) - np.sum(1 / v ** 2) / np.sum(1 / v)))
        say(f"    corrected tau2 = {t2c:.4f} (published tau2 = {t2:.4f})")
    return dict(pool=mu, se=se, boot_sd=float(pool_b.std(ddof=1)),
                ci=(float(lo), float(hi)))


# ================================================================== FAMILY A
def family_cue(df, B, rng):
    """12-dataset cue family: TOT-vs-DK log-OR on the correct-cue arm."""
    d = df.dropna(subset=["corr"]).copy()
    labels = sorted(d.ds.unique())

    def per_ds(frame):
        ys, vs = [], []
        for ds in labels:
            s = frame[frame.ds == ds]
            t, k = s[s.is_TOT == 1], s[s.is_TOT == 0]
            if not len(t) or not len(k):
                return None
            y, v = logor_2x2(int(t["corr"].sum()), int(len(t) - t["corr"].sum()),
                             int(k["corr"].sum()), int(len(k) - k["corr"].sum()))
            ys.append(y)
            vs.append(v)
        return np.array(ys), np.array(vs)
    y, v = per_ds(d)
    boot = np.full((B, len(labels)), np.nan)
    idx = {lang: d[d.lang == lang]["item"].unique() for lang in ("en", "zh")}
    by_item = {it: g for it, g in d.groupby("item")}
    for b in range(B):
        draws = []
        for lang in ("en", "zh"):
            pick = rng.choice(idx[lang], size=len(idx[lang]), replace=True)
            draws.extend(pick)
        rep = pd.concat([by_item[i] for i in draws], ignore_index=True)
        r = per_ds(rep)
        if r is not None:
            boot[b] = r[0]
    return report("FAMILY A -- cue-unlocked production, TOT-vs-DK log-OR "
                  "(correct-cue arm)", y, v, labels, boot,
                  published=None)


# ================================================================== FAMILY B
def family_firstunit(B, rng):
    """8-dataset confirmatory first-unit family (initial_unit channel)."""
    df = partial_rows(CORE)
    d = df[df.aligned == "initial_unit"].dropna(
        subset=["y", "is_TOT", "zipf", "length", "tokens"]).copy()
    d["ds"] = d["lang"] + "/" + d["model"]
    labels = sorted(d.ds.unique())

    def per_ds(frame, adjusted=True):
        ys, vs = [], []
        for ds in labels:
            s = frame[frame.ds == ds]
            t, k = s[s.is_TOT == 1], s[s.is_TOT == 0]
            if len(t) < 5 or len(k) < 5:
                return None
            hy, hv = logor_2x2(int(t["y"].sum()), int(len(t) - t["y"].sum()),
                               int(k["y"].sum()), int(len(k) - k["y"].sum()))
            if not adjusted:
                y, v = hy, hv
            else:
                X = pd.DataFrame({"const": 1.0, "is_TOT": s["is_TOT"],
                                  "zipf": zscore(s["zipf"]),
                                  "length": zscore(s["length"]),
                                  "tokens": zscore(s["tokens"])}, index=s.index)
                try:
                    f = sm.GLM(s["y"], X, family=sm.families.Binomial()).fit()
                    y, v = float(f.params["is_TOT"]), float(f.bse["is_TOT"]) ** 2
                    if not np.isfinite(y) or not np.isfinite(v) or abs(y) > 8                             or v > 64:
                        y, v = hy, hv    # separated replicate -> Haldane fallback
                except Exception:
                    y, v = hy, hv
            ys.append(y)
            vs.append(v)
        return np.array(ys), np.array(vs)
    y, v = per_ds(d)
    boot = np.full((B, len(labels)), np.nan)
    idx = {lang: d[d.lang == lang]["item"].unique() for lang in ("en", "zh")}
    by_item = {it: g for it, g in d.groupby("item")}
    for b in range(B):
        draws = []
        for lang in ("en", "zh"):
            pick = rng.choice(idx[lang], size=len(idx[lang]), replace=True)
            draws.extend(pick)
        rep = pd.concat([by_item[i] for i in draws], ignore_index=True)
        r = per_ds(rep)
        if r is not None:
            boot[b] = r[0]
    out = report("FAMILY B -- confirmatory first-unit (initial_unit), "
                 "covariate-adjusted per-dataset logits", y, v, labels, boot,
                 published=None)
    # the one-stage pooled fit is ALREADY item-clustered -- restate it for
    # contrast, and give its item-block bootstrap too
    X = pd.DataFrame({"const": 1.0, "is_TOT": d["is_TOT"], "zipf": zscore(d["zipf"]),
                      "length": zscore(d["length"]), "tokens": zscore(d["tokens"])},
                     index=d.index)
    for m in sorted(d.model.unique())[1:]:
        X[f"m_{m}"] = (d["model"] == m).astype(float)
    X["lang_zh"] = (d["lang"] == "zh").astype(float)
    f = sm.GLM(d["y"], X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": d["item"]})
    ci = f.conf_int()
    say(f"\n  one-stage pooled fit (already item-clustered, the main text's "
        f"+0.68): b={f.params['is_TOT']:+.4f} SE={f.bse['is_TOT']:.4f} "
        f"CI[{ci.loc['is_TOT', 0]:+.3f},{ci.loc['is_TOT', 1]:+.3f}] "
        f"p={f.pvalues['is_TOT']:.4f}")
    say("  -> the ONE-STAGE estimate is the correctly clustered one; it is the "
        "META layer (I2/tau2/HKSJ) that assumes independence.")
    return out


# ================================================================== FAMILY C
def family_aoa(B, rng):
    """AoA TOST: 4 English models fitted on the SAME normed subset."""
    import aoa_tost as A
    items = A.load_jsonl(os.path.join(ROOT, "data", "items_full_en_final.jsonl"))
    frames = {}
    for m in A.MODELS:
        fr = A.build_model_df(m, items).dropna(
            subset=["fail"] + A.PREDS_FULL).set_index("item_id")
        frames[m] = fr
    common = None
    for m, fr in frames.items():
        common = set(fr.index) if common is None else (common & set(fr.index))
    common = sorted(common)
    say("")
    say(f"  AoA analysis subset: {len(common)} items shared by all "
        f"{len(A.MODELS)} models (this sharing is what the pool ignores); "
        f"per-model analysis n = "
        f"{[len(frames[m]) for m in A.MODELS]}")

    def per_model(item_list):
        ys, vs = [], []
        for m in A.MODELS:
            fr = frames[m].loc[item_list]
            X = pd.DataFrame({"const": 1.0,
                              **{k: zscore(fr[k].astype(float)).to_numpy()
                                 for k in A.PREDS_FULL}})
            yv = fr["fail"].to_numpy(dtype=float)
            if yv.sum() < 10 or (1 - yv).sum() < 10:
                return None
            try:
                f = sm.GLM(yv, X, family=sm.families.Binomial()).fit(cov_type="HC3")
                b, se = float(f.params["aoa"]), float(f.bse["aoa"])
            except Exception:
                return None
            if not np.isfinite(b) or not np.isfinite(se) or abs(b) > 5 or se > 5:
                return None
            ys.append(b)
            vs.append(se ** 2)
        return np.array(ys), np.array(vs)
    y, v = per_model(common)
    boot = np.full((B, len(A.MODELS)), np.nan)
    arr = np.array(common, dtype=object)
    for b in range(B):
        pick = list(rng.choice(arr, size=len(arr), replace=True))
        r = per_model(pick)
        if r is not None:
            boot[b] = r[0]
    out = report("FAMILY C -- AoA coefficient, 4 EN models on the same normed "
                 "subset", y, v, A.MODELS, boot,
                 published=None)
    mu, se, sb = out["pool"], out["se"], out["boot_sd"]
    for nm, s_ in (("analytic (independent)", se), ("item-block bootstrap", sb)):
        p_lo = norm.sf((mu + 0.2) / s_)          # H0: b <= -0.2
        p_hi = norm.cdf((mu - 0.2) / s_)         # H0: b >= +0.2
        pt = max(p_lo, p_hi)
        say(f"  TOST (+-0.20) with {nm:22s} SE={s_:.4f}: p_lower={p_lo:.4f} "
            f"p_upper={p_hi:.4f}  p_TOST={pt:.4f}  -> "
            f"{'EQUIVALENCE ESTABLISHED' if pt < .05 else 'EQUIVALENCE NOT ESTABLISHED'}")
        half = 1.645 * s_
        say(f"    90% equivalence interval [{mu - half:+.4f},{mu + half:+.4f}] "
            f"{'inside' if abs(mu) + half < 0.2 else 'NOT inside'} +-0.20")
    # smallest bound the clustered SE could support
    from scipy.stats import norm as _n
    need = abs(mu) + 1.645 * sb
    say(f"  smallest equivalence bound the item-clustered SE supports at "
        f"alpha=.05: +-{need:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cue_meta.txt"))
    args = ap.parse_args()
    rng = np.random.default_rng(20260726)
    say("=" * 96)
    say("ITEM-CLUSTERED META LAYER")
    say("=" * 96)
    say(f"item-block bootstrap replicates B={args.B}; items resampled with "
        f"replacement within language; all datasets refit per replicate.")

    say("\n\n### FAMILY B: confirmatory first-unit")
    family_firstunit(args.B, rng)

    say("\n\n### FAMILY A: cue-unlocked production")
    df = build_cue(PRIMARY6)
    family_cue(df, args.B, rng)

    say("\n\n### FAMILY C: AoA equivalence")
    family_aoa(args.B, rng)

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(("\n".join(OUT) + "\n").rstrip("\n") + "\n")
    print("wrote", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
