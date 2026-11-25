"""Heterogeneity-, degeneracy- and live-variable-aware meta-analysis of
the 28-dataset self-report contrast.

  Heterogeneity: Cochran Q, tau^2 (DL), I^2, H^2, 95% prediction
  interval, plus HKSJ (Knapp-Hartung) CIs.

  Degeneracy: datasets where the self-report variable is effectively
  constant (0% or ~100% report rate) are separated from those where it
  varies ("live" restriction; meta-regression of log-OR on report rate).

  Zero-report datasets: with a+c=0 the Haldane-corrected OR collapses
  algebraically to (d+.5)/(b+.5) = the DK:TOT base-rate ratio and
  carries no self-report information; such ORs are n/a and excluded
  from pooling.

Definitions (stated once, used everywhere):
  2x2 is the failure-conditional table used throughout the paper:
      rows  = behavioral state in {TOT, DK}
      cols  = self-report in {TOT, not-TOT}
      a=hit (behav TOT & self TOT), b=miss, c=fa (behav DK & self TOT),
      d=cr.
  report rate (primary, rr_fail) = (a+c)/(a+b+c+d), i.e. the report rate
      *within the analysis set*.  This is the rate that governs whether
      the self-report variable is degenerate in the 2x2 actually pooled.
  report rate (secondary, rr_all) = n_self_tot / n_classified, the
      all-item rate the paper prints; used as a sensitivity check.
  LIVE  <=> rr_fail in [0.05, 0.90]  (a fixed rule, applied to the
      full-sample rates before any bootstrap).
  ZERO-REPORT <=> a+c == 0  -> OR is n/a, dataset carries no information,
      excluded from every pooled estimate and from the meta-regression.

Estimators:
  DL   inverse-variance random effects on Haldane-corrected log-OR.
  MH   Mantel-Haenszel common OR, no continuity correction (zero-report
       tables contribute 0 to both numerator and denominator, so MH is
       already immune to the zero-report problem -- verified numerically).
  MH CI by item-cluster bootstrap: items are resampled *within language*
       and shared across all models of that language, which is the
       dependence structure of the design.

  python selfreport_meta3.py --out ../results --boot 2000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np
from scipy import stats as sps

from analyze_pilot import classify, load_jsonl

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

LIVE_LO, LIVE_HI = 0.05, 0.90

MODELS = ["qwen7b", "qwen14b", "qwen32b", "qwen72b", "llama8b", "llama70b",
          "gemma12b", "gemma27b", "glm9b", "mistral7b", "olmo13b", "olmo32b",
          "phi4", "r1d14b"]

PRETTY = {"qwen7b": "Qwen2.5-7B", "qwen14b": "Qwen2.5-14B",
          "qwen32b": "Qwen2.5-32B", "qwen72b": "Qwen2.5-72B",
          "llama8b": "Llama-3.1-8B", "llama70b": "Llama-3.3-70B",
          "gemma12b": "Gemma-3-12B", "gemma27b": "Gemma-3-27B",
          "glm9b": "GLM-4-9B", "mistral7b": "Mistral-7B",
          "olmo13b": "OLMo-2-13B", "olmo32b": "OLMo-2-32B",
          "phi4": "Phi-4", "r1d14b": "R1-Distill-14B"}


# ---------------------------------------------------------------- effects
def logor(t, corr=0.5):
    a, b, c, d = (x + corr for x in t)
    return math.log(a * d / (b * c)), 1 / a + 1 / b + 1 / c + 1 / d


def dl_pool(tables, labels=None):
    """DerSimonian-Laird RE + full heterogeneity battery."""
    ys, vs = [], []
    for t in tables:
        y, v = logor(t)
        ys.append(y)
        vs.append(v)
    ys, vs = np.array(ys), np.array(vs)
    k = len(ys)
    w = 1.0 / vs
    mu_f = float((w * ys).sum() / w.sum())
    se_f = float(math.sqrt(1.0 / w.sum()))
    q = float((w * (ys - mu_f) ** 2).sum())
    df = k - 1
    c = w.sum() - (w * w).sum() / w.sum()
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 else 0.0
    h2 = q / df if df > 0 else float("nan")
    q_p = float(sps.chi2.sf(q, df)) if df > 0 else float("nan")

    wr = 1.0 / (vs + tau2)
    mu = float((wr * ys).sum() / wr.sum())
    se = float(math.sqrt(1.0 / wr.sum()))
    lo, hi = mu - 1.96 * se, mu + 1.96 * se

    # Knapp-Hartung / Sidik-Jonkman variance inflation + t reference
    if k > 1:
        qr = float((wr * (ys - mu) ** 2).sum() / (k - 1))
        se_hk = se * math.sqrt(max(qr, 1e-12))
        tcrit = float(sps.t.ppf(0.975, k - 1))
        hk = (mu - tcrit * se_hk, mu + tcrit * se_hk)
        # 95% prediction interval (Higgins-Thompson-Spiegelhalter)
        sepi = math.sqrt(se ** 2 + tau2)
        pi = (mu - tcrit * sepi, mu + tcrit * sepi)
    else:
        hk = pi = (float("nan"), float("nan"))

    z = mu / se
    return {
        "k": k, "logor": mu, "se": se,
        "or": math.exp(mu), "lo": math.exp(lo), "hi": math.exp(hi),
        "z": z, "p": float(2 * sps.norm.sf(abs(z))),
        "or_hk_lo": math.exp(hk[0]), "or_hk_hi": math.exp(hk[1]),
        "pi_lo": math.exp(pi[0]), "pi_hi": math.exp(pi[1]),
        "Q": q, "df": df, "Q_p": q_p, "tau2": tau2,
        "tau": math.sqrt(tau2), "I2": i2, "H2": h2,
        "fixed_or": math.exp(mu_f), "fixed_se": se_f,
    }


def mh_or(tables):
    num = den = 0.0
    for (a, b, c, d) in tables:
        n = a + b + c + d
        if n == 0:
            continue
        num += a * d / n
        den += b * c / n
    return num / den if den > 0 else float("nan")


def metareg(tables, xs, quad=False):
    """(Random-effects) meta-regression of log-OR on report rate.

    Reports both the plain inverse-variance WLS
    and the residual-tau^2 / Knapp-Hartung version, which is the honest
    one under this much heterogeneity.
    """
    ys, vs = zip(*(logor(t) for t in tables))
    ys, vs = np.array(ys), np.array(vs)
    xs = np.array(xs, dtype=float)
    cols = [np.ones_like(xs), xs] + ([xs ** 2] if quad else [])
    X = np.column_stack(cols)
    k, p = X.shape

    def fit(v):
        W = np.diag(1.0 / v)
        XtW = X.T @ W
        cov = np.linalg.inv(XtW @ X)
        beta = cov @ (XtW @ ys)
        resid = ys - X @ beta
        qe = float(resid @ W @ resid)
        return beta, cov, qe, resid

    beta, cov, qe, _ = fit(vs)
    # residual tau^2 (DL-style, generalised)
    W = np.diag(1.0 / vs)
    P = W - W @ X @ np.linalg.inv(X.T @ W @ X) @ X.T @ W
    trp = float(np.trace(P))
    tau2r = max(0.0, (qe - (k - p)) / trp) if trp > 0 else 0.0
    beta_r, cov_r, _, _ = fit(vs + tau2r)

    se_r = np.sqrt(np.diag(cov_r))
    tcrit = float(sps.t.ppf(0.975, k - p))
    out = {"k": k, "Qe": qe, "Qe_df": k - p,
           "Qe_p": float(sps.chi2.sf(qe, k - p)), "tau2_resid": tau2r,
           "wls": [], "re": []}
    se_f = np.sqrt(np.diag(cov))
    for j in range(p):
        out["wls"].append({
            "beta": float(beta[j]), "se": float(se_f[j]),
            "lo": float(beta[j] - 1.96 * se_f[j]),
            "hi": float(beta[j] + 1.96 * se_f[j]),
            "p": float(2 * sps.norm.sf(abs(beta[j] / se_f[j])))})
        out["re"].append({
            "beta": float(beta_r[j]), "se": float(se_r[j]),
            "lo": float(beta_r[j] - tcrit * se_r[j]),
            "hi": float(beta_r[j] + tcrit * se_r[j]),
            "p": float(2 * sps.t.sf(abs(beta_r[j] / se_r[j]), k - p))})
    return out


# ------------------------------------------------------------------ data
def load_cells():
    """Rebuild the per-item TOT/DK x report assignment and the 2x2s."""
    per_lang, tables, rr_all = {}, {}, {}
    sdt = json.load(open(os.path.join(ROOT, "results", "selfreport_sdt.json")))
    for lang in ("en", "zh"):
        items = list(load_jsonl(os.path.join(
            ROOT, "data", f"items_full_{lang}_final.jsonl")).keys())
        idx = {iid: i for i, iid in enumerate(items)}
        data = {}
        for m in MODELS:
            name = f"full_{lang}_{m}"
            base = os.path.join(ROOT, "results", name)
            if not os.path.exists(os.path.join(base, "selfreport.jsonl")):
                continue
            prod = load_jsonl(os.path.join(base, "production.jsonl"))
            rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
            selfr = load_jsonl(os.path.join(base, "selfreport.jsonl"))
            # one-hot over (a,b,c,d) per item; all-zero row = not in 2x2
            oh = np.zeros((len(items), 4), dtype=np.int32)
            n_self_tot = n_class = 0
            for iid in items:
                if iid not in selfr:
                    continue
                st = classify(prod.get(iid), rec.get(iid))
                if not st:
                    continue
                rep = selfr[iid].get("label") == "TOT"
                n_class += 1
                n_self_tot += rep
                if st == "TOT":
                    oh[idx[iid], 0 if rep else 1] = 1
                elif st == "DK":
                    oh[idx[iid], 2 if rep else 3] = 1
            data[name] = oh
            tables[name] = tuple(int(x) for x in oh.sum(0))
            rr_all[name] = n_self_tot / n_class if n_class else float("nan")
            # verify against the published SDT json
            ref = sdt[name]["tot_vs_dk"]["counts"]
            got = tables[name]
            assert got == (ref["hit"], ref["miss"], ref["fa"], ref["cr"]), \
                f"{name}: rebuilt {got} != published {ref}"
        per_lang[lang] = (items, data)
    return per_lang, tables, rr_all


def boot_mh(per_lang, names, reps, seed=0):
    """Item-cluster bootstrap of the MH OR over a fixed dataset subset."""
    rng = np.random.default_rng(seed)
    sel = {lang: [n for n in names if n.startswith(f"full_{lang}_")]
           for lang in per_lang}
    out = []
    for _ in range(reps):
        tt = []
        for lang, (items, data) in per_lang.items():
            if not sel[lang]:
                continue
            n = len(items)
            cnt = np.bincount(rng.integers(0, n, n), minlength=n)
            for name in sel[lang]:
                tt.append(tuple(int(x) for x in cnt @ data[name]))
        v = mh_or(tt)
        if v == v and math.isfinite(v):
            out.append(v)
    out.sort()
    return out


# ------------------------------------------------------------------ main
def fmt(r):
    return (f"OR={r['or']:.3f} [{r['lo']:.3f},{r['hi']:.3f}] "
            f"(HKSJ [{r['or_hk_lo']:.3f},{r['or_hk_hi']:.3f}]) k={r['k']} "
            f"Q={r['Q']:.1f} df={r['df']} p={r['Q_p']:.3g} "
            f"tau2={r['tau2']:.4f} I2={r['I2']:.1f}% "
            f"PI [{r['pi_lo']:.2f},{r['pi_hi']:.2f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="output folder (results/ holds the stored files)")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    per_lang, tables, rr_all = load_cells()
    print(f"rebuilt {len(tables)} datasets, all 2x2s match "
          f"results/selfreport_sdt.json\n")

    names = sorted(tables)
    rr = {n: (tables[n][0] + tables[n][2]) / sum(tables[n]) for n in names}
    zero = [n for n in names if tables[n][0] + tables[n][2] == 0]
    info = [n for n in names if n not in zero]
    live = [n for n in info if LIVE_LO <= rr[n] <= LIVE_HI]
    dead = [n for n in names if n not in live]
    live_all = [n for n in info if LIVE_LO <= rr_all[n] <= LIVE_HI]

    # ---- per-dataset table / CSV -------------------------------------
    print("=" * 108)
    print("PER-DATASET 2x2 (failure-conditional: behav TOT/DK x self-report)")
    print(f"{'dataset':<22}{'a':>5}{'b':>6}{'c':>5}{'d':>6}"
          f"{'rr_fail':>9}{'rr_all':>8}{'OR_raw':>9}{'OR_hal':>8}"
          f"{'ci_lo':>8}{'ci_hi':>8}  status")
    rows = []
    for n in names:
        a, b, c, d = tables[n]
        lang, mdl = n.split("_")[1], n.split("_")[2]
        is_zero = (a + c) == 0
        anyzero = min(a, b, c, d) == 0
        if is_zero:
            status, or_raw = "zero_report_na", None
        elif anyzero:
            status, or_raw = "single_zero_cell", None
        else:
            status, or_raw = "ok", (a * d) / (b * c)
        y, v = logor((a, b, c, d))
        orh, se = math.exp(y), math.sqrt(v)
        cl, ch = math.exp(y - 1.96 * se), math.exp(y + 1.96 * se)
        p = float(sps.fisher_exact([[a, b], [c, d]])[1])
        print(f"{n:<22}{a:>5}{b:>6}{c:>5}{d:>6}{rr[n]:>9.3f}"
              f"{rr_all[n]:>8.3f}"
              f"{('n/a' if or_raw is None else f'{or_raw:.3f}'):>9}"
              f"{orh:>8.3f}{cl:>8.3f}{ch:>8.3f}  {status}"
              f"{'' if n in live else '   [not live]'}")
        rows.append({
            "dataset": n, "language": lang, "model": PRETTY.get(mdl, mdl),
            "a_totTOT_selfTOT": a, "b_totTOT_selfNo": b,
            "c_dk_selfTOT": c, "d_dk_selfNo": d,
            "n_2x2": a + b + c + d,
            "n_reports_in_2x2": a + c,
            "report_rate_failureset": round(rr[n], 4),
            "report_rate_allitems": round(rr_all[n], 4),
            "or_uncorrected": "" if or_raw is None else round(or_raw, 4),
            "or_haldane": round(orh, 4),
            "ci_lo": round(cl, 4), "ci_hi": round(ch, 4),
            "log_or": round(y, 4), "se_log_or": round(se, 4),
            "fisher_p": round(p, 5),
            "status": status,
            "live": int(n in live),
            "pooled": int(n in info),
        })
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "selfreport_cells.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")

    # ---- zero-report algebra ------------------------------------
    print("\n" + "=" * 108)
    print("ZERO-REPORT DATASETS (a+c=0 -> OR must be n/a)")
    for n in zero:
        a, b, c, d = tables[n]
        printed = math.exp(logor((a, b, c, d))[0])
        base = (d + .5) / (b + .5)
        print(f"  {n:<22} cells=({a},{b},{c},{d})  printed OR={printed:.3f}"
              f"  ==  (d+.5)/(b+.5)={base:.3f}  = DK:TOT base-rate ratio,"
              f" 0 bits of self-report information")
    print(f"  -> {len(zero)} such datasets, not 2 as currently reported "
          f"({', '.join(zero)})")

    # ---- pooled estimates --------------------------------------------
    print("\n" + "=" * 108)
    print("POOLED ESTIMATES  (DL random effects, Haldane-corrected log-OR)")
    subsets = [("A. all 28 (main analysis)", names),
               ("B. informative only (drop zero-report)", info),
               ("C. LIVE: rr_fail in [5%,90%]", live),
               ("C'. live by all-item rate (sensitivity)", live_all)]
    dl = {}
    for lab, ns in subsets:
        dl[lab] = dl_pool([tables[n] for n in ns])
        dl[lab]["members"] = ns
        print(f"  {lab:<40} {fmt(dl[lab])}")

    print("\nMANTEL-HAENSZEL common OR (no continuity correction)")
    mh = {}
    for lab, ns in subsets:
        mh[lab] = mh_or([tables[n] for n in ns])
        print(f"  {lab:<40} MH OR={mh[lab]:.4f}")
    print("  (A and B are identical by construction: a zero-report table "
          "contributes 0 to both the MH numerator and denominator.)")

    print(f"\nITEM-CLUSTER BOOTSTRAP of MH OR ({args.boot} reps, items "
          f"resampled within language, shared across models)")
    bmh = {}
    for lab, ns in subsets[1:]:
        bs = boot_mh(per_lang, ns, args.boot, seed=0)
        lo, hi = bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]
        bmh[lab] = {"point": mh[lab], "lo": lo, "hi": hi, "n": len(bs)}
        print(f"  {lab:<40} MH OR={mh[lab]:.3f} 95% CI [{lo:.3f},{hi:.3f}]")

    print("\nEXCLUDED BY THE LIVE RULE (report rate outside [5%,90%]):")
    for n in sorted(dead, key=lambda x: rr[x]):
        a, b, c, d = tables[n]
        why = ("zero reports" if (a + c) == 0 else
               "near-silent" if rr[n] < LIVE_LO else "near-ceiling")
        print(f"  {n:<22} rr_fail={rr[n]*100:6.2f}%  rr_all={rr_all[n]*100:6.2f}%"
              f"  reports={a+c:>4}/{a+b+c+d:<5} {why}")

    # ---- meta-regression -----------------------------------------
    print("\n" + "=" * 108)
    print("META-REGRESSION of log-OR on report rate")
    reg = {}
    for lab, ns, quad in [
            ("all 28 (incl. zero-report artifacts)", names, False),
            ("informative (25), linear", info, False),
            ("informative (25), + quadratic", info, True),
            ("live (16), linear", live, False),
            ("live (16), + quadratic", live, True)]:
        r = metareg([tables[n] for n in ns], [rr[n] for n in ns], quad=quad)
        reg[lab] = r
        terms = ["intercept", "report_rate"] + (["report_rate^2"] if quad else [])
        print(f"\n  [{lab}]  k={r['k']}  residual tau^2={r['tau2_resid']:.4f}"
              f"  Q_E={r['Qe']:.1f} (df={r['Qe_df']}, p={r['Qe_p']:.3g})")
        for t, wl, re_ in zip(terms, r["wls"], r["re"]):
            print(f"    {t:<15} WLS b={wl['beta']:+.4f} "
                  f"[{wl['lo']:+.4f},{wl['hi']:+.4f}] p={wl['p']:.3g}"
                  f"   |  RE/HK b={re_['beta']:+.4f} "
                  f"[{re_['lo']:+.4f},{re_['hi']:+.4f}] p={re_['p']:.3g}")
        if not quad:
            b = r["wls"][1]
            print(f"    -> per +10pp report rate: OR x {math.exp(b['beta']*.1):.3f} "
                  f"[{math.exp(b['lo']*.1):.3f},{math.exp(b['hi']*.1):.3f}] (WLS)")

    print("\n  where do OR>=1 sit? (informative datasets, by report rate)")
    for n in sorted(info, key=lambda x: rr[x]):
        orh = math.exp(logor(tables[n])[0])
        print(f"    {n:<22} rr={rr[n]*100:6.2f}%  OR={orh:.3f}"
              f"  {'<<<< OR>=1' if orh >= 1 else ''}")
    pos = [n for n in info if math.exp(logor(tables[n])[0]) >= 1]
    if pos:
        print(f"    OR>=1 report-rate range: "
              f"{min(rr[n] for n in pos)*100:.1f}%-{max(rr[n] for n in pos)*100:.1f}%"
              f" (n={len(pos)}/{len(info)})")

    # ---- equivalence bound -------------------------------------------
    # ---- is the report-rate slope a continuity-correction artifact?
    # With a+c small, log OR_hal -> log((d+.5)/(b+.5)) = -log(TOT:DK odds),
    # and with b+d small it -> log((a+.5)/(c+.5)) = +log(TOT:DK odds).  Since
    # behavioural TOT outnumbers DK everywhere, the Haldane correction alone
    # manufactures a POSITIVE log-OR-on-report-rate slope.  Two checks.
    print("\n" + "=" * 108)
    print("IS THE SLOPE A CONTINUITY-CORRECTION ARTIFACT?")
    print("  (i) null calibration: expected corrected log-OR when the 2x2 is")
    print("      exactly independent, using each dataset's own margins")
    y_null = {}
    for n in names:
        a, b, c, d = tables[n]
        N = a + b + c + d
        r1, r2, k1 = a + b, c + d, a + c
        e = (r1 * k1 / N, r1 * (N - k1) / N, r2 * k1 / N, r2 * (N - k1) / N)
        y_null[n] = logor(e)[0]
    for lab, ns in [("all 28", names), ("informative (25)", info),
                    ("live (16)", live)]:
        yr = [logor(tables[n])[0] - y_null[n] for n in ns]
        vv = [logor(tables[n])[1] for n in ns]
        X = np.column_stack([np.ones(len(ns)), np.array([rr[n] for n in ns])])
        W = np.diag(1.0 / np.array(vv))
        cov = np.linalg.inv(X.T @ W @ X)
        bet = cov @ (X.T @ W @ np.array(yr))
        se = np.sqrt(np.diag(cov))
        raw = metareg([tables[n] for n in ns], [rr[n] for n in ns])["wls"][1]
        print(f"    {lab:<18} raw slope={raw['beta']:+.4f} "
              f"[{raw['lo']:+.4f},{raw['hi']:+.4f}]  ->  "
              f"null-calibrated slope={bet[1]:+.4f} "
              f"[{bet[1]-1.96*se[1]:+.4f},{bet[1]+1.96*se[1]:+.4f}] "
              f"p={2*sps.norm.sf(abs(bet[1]/se[1])):.3g}")
        print(f"    {'':<18} artifact accounts for "
              f"{100*(raw['beta']-bet[1])/raw['beta']:.0f}% of the raw slope")

    # (ii) correction-free: cell-level logistic with dataset fixed effects,
    #      P(behav TOT | failure) ~ dataset + report + report x (rr - mean rr).
    #      Firth-penalised so the zero cells do not need any ad-hoc +0.5.
    print("\n  (ii) correction-free Firth logistic, dataset fixed effects:")
    print("       logit P(behav TOT | failure) = alpha_j + beta*report"
          " + gamma*report*(rate - mean rate)")

    def firth_logit(ns, use_gamma=True, iters=200):
        idx = {n: i for i, n in enumerate(ns)}
        rows, yv, nv = [], [], []
        rbar = float(np.mean([rr[n] for n in ns]))
        for n in ns:
            a, b, c, d = tables[n]
            for s, (succ, fail) in ((1, (a, c)), (0, (b, d))):
                if succ + fail == 0:
                    continue
                z = np.zeros(len(ns) + (2 if use_gamma else 1))
                z[idx[n]] = 1.0
                z[len(ns)] = s
                if use_gamma:
                    z[len(ns) + 1] = s * (rr[n] - rbar)
                rows.append(z)
                yv.append(succ)
                nv.append(succ + fail)
        X = np.array(rows)
        y = np.array(yv, float)
        nn = np.array(nv, float)
        beta = np.zeros(X.shape[1])
        for _ in range(iters):
            eta = X @ beta
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
            w = nn * p * (1 - p)
            XtW = X.T * w
            F = XtW @ X
            Fi = np.linalg.pinv(F)
            h = np.einsum("ij,jk,ik->i", X, Fi, X) * w
            ya, na = y + h / 2.0, nn + h
            pa = np.clip(ya / np.maximum(na, 1e-12), 1e-9, 1 - 1e-9)
            wa = na * pa * (1 - pa)
            zz = (X @ beta) + (ya - na * p) / np.maximum(na * p * (1 - p), 1e-12)
            XtWa = X.T * wa
            step = np.linalg.pinv(XtWa @ X) @ (XtWa @ zz) - beta
            beta = beta + np.clip(step, -2, 2)
            if np.max(np.abs(step)) < 1e-9:
                break
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        cov = np.linalg.pinv((X.T * (nn * p * (1 - p))) @ X)
        se = np.sqrt(np.diag(cov))
        return beta, se, rbar

    for lab, ns in [("all 28", names), ("informative (25)", info),
                    ("live (16)", live)]:
        bt, se, rbar = firth_logit(ns)
        for j, tag in ((len(ns), "beta (log-OR at mean rate)"),
                       (len(ns) + 1, "gamma (slope on report rate)")):
            print(f"    {lab:<18} {tag:<30} {bt[j]:+.4f} "
                  f"[{bt[j]-1.96*se[j]:+.4f},{bt[j]+1.96*se[j]:+.4f}] "
                  f"p={2*sps.norm.sf(abs(bt[j]/se[j])):.3g}"
                  + (f"   -> OR={math.exp(bt[j]):.3f} "
                     f"[{math.exp(bt[j]-1.96*se[j]):.3f},"
                     f"{math.exp(bt[j]+1.96*se[j]):.3f}]"
                     if "beta" in tag else ""))
        print(f"    {'':<18} (mean report rate = {rbar:.3f})")

    # (iii) the meta-regression SEs above ignore the shared-item dependence.
    #       Re-derive slope + fitted values by the same item-cluster bootstrap
    #       used for MH (subset membership frozen at the full-sample rates).
    print("\n  (iii) item-cluster bootstrap of the meta-regression "
          f"({args.boot} reps)")
    grid = [0.10, 0.30, 0.50, 0.70, 0.90]

    def wls_slope(tt, xs):
        ys, vs = zip(*(logor(t) for t in tt))
        X = np.column_stack([np.ones(len(xs)), np.array(xs)])
        W = np.diag(1.0 / np.array(vs))
        return np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ np.array(ys))

    for lab, ns in [("informative (25)", info), ("live (16)", live)]:
        rng = np.random.default_rng(1)
        sel = {lg: [n for n in ns if n.startswith(f"full_{lg}_")]
               for lg in per_lang}
        bs, fits = [], []
        for _ in range(args.boot):
            tt, xs = [], []
            for lg, (items, data) in per_lang.items():
                if not sel[lg]:
                    continue
                nI = len(items)
                cnt = np.bincount(rng.integers(0, nI, nI), minlength=nI)
                for nm in sel[lg]:
                    cell = tuple(int(x) for x in cnt @ data[nm])
                    tt.append(cell)
                    xs.append((cell[0] + cell[2]) / max(sum(cell), 1))
            b = wls_slope(tt, xs)
            bs.append(b[1])
            fits.append([b[0] + b[1] * g for g in grid])
        bs = np.sort(np.array(bs))
        fits = np.sort(np.array(fits), axis=0)
        lo, hi = bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]
        pt = wls_slope([tables[n] for n in ns], [rr[n] for n in ns])
        print(f"    {lab:<18} slope={pt[1]:+.4f} cluster-CI "
              f"[{lo:+.4f},{hi:+.4f}]  (naive CI was "
              f"[{metareg([tables[n] for n in ns], [rr[n] for n in ns])['wls'][1]['lo']:+.4f},"
              f"{metareg([tables[n] for n in ns], [rr[n] for n in ns])['wls'][1]['hi']:+.4f}])")
        print(f"    {'':<18} fitted OR at report rate:")
        for j, g in enumerate(grid):
            f_lo = fits[int(.025 * len(fits)), j]
            f_hi = fits[int(.975 * len(fits)), j]
            print(f"    {'':<18}   {g*100:>5.0f}%  OR={math.exp(pt[0]+pt[1]*g):.3f}"
                  f"  [{math.exp(f_lo):.3f},{math.exp(f_hi):.3f}]")

    print("\n" + "=" * 108)
    print("EQUIVALENCE / LICENSED BOUND")
    for lab in ("C. LIVE: rr_fail in [5%,90%]",):
        r = dl[lab]
        for tag, hi_ in (("normal-z", r["hi"]), ("HKSJ", r["or_hk_hi"])):
            print(f"  live DL upper bound ({tag}): OR<={hi_:.3f}"
                  f"  <=>  d'<={math.log(hi_)*math.sqrt(3)/math.pi:.3f}")
        b = bmh[lab]
        print(f"  live item-clustered MH: OR={b['point']:.3f} "
              f"[{b['lo']:.3f},{b['hi']:.3f}]  <=>  d' upper "
              f"{math.log(b['hi'])*math.sqrt(3)/math.pi:.3f}")
        print(f"  live 95% prediction interval: [{r['pi_lo']:.2f},{r['pi_hi']:.2f}]"
              f"  (a NEW model is NOT bounded as tightly as the mean)")

    json.dump({
        "definitions": {"live_rule": [LIVE_LO, LIVE_HI],
                        "report_rate": "(a+c)/(a+b+c+d) on the TOT-vs-DK set"},
        "tables": {n: list(tables[n]) for n in names},
        "report_rate_failureset": rr, "report_rate_allitems": rr_all,
        "zero_report": zero, "live": live, "excluded": sorted(dead),
        "dl": {k: {kk: vv for kk, vv in v.items()} for k, v in dl.items()},
        "mh": mh, "mh_bootstrap": bmh, "metareg": reg,
    }, open(os.path.join(args.out, "selfreport_meta3.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(args.out, 'selfreport_meta3.json')}")


if __name__ == "__main__":
    main()
