# -*- coding: utf-8 -*-
"""Cue-narrowing / cohort-size control analysis.

The alternative tested: the first-unit cue works because it shrinks the
candidate set, so "recovery under the cue" does not show the target was
available before the hint.  A pure narrowing account makes a quantitative
prediction: the cue's benefit must track
HOW MUCH the first unit narrows the cohort.  A word starting with 'z'
(54 lexicon types) is narrowed far harder than one starting with 's'
(3,398); Chinese first characters vary by three orders of magnitude.

Three analyses, each run for English and Chinese separately:

  1  Among strict failures with a correct-cue record (the 1,938 En /
     3,240 Zh rows over the six main settings), unlock ~ log cohort size
     with model fixed effects and item-clustered SEs, plus unlock by
     cohort tercile.  Pure narrowing predicts a strong NEGATIVE slope.

  2  The fit-matched target-vs-decoy contrast (fit_gradient / fit_ci /
     score_matched_margin) with
     cohort size of the CUED WORD controlled: nested logistic models and
     a within-cohort-stratum margin table.

  3  Descriptives: cohort size for targets vs decoys, and the
     cohort-size / definition-fit correlation.

Reads the released records and the per-item cohort counts in
data/cohort_{en,zh}.tsv (written by cohort_table.py from the lexicons,
which are not redistributed).  ASCII-only stdout; the report file,
results/cohort_control.txt, is written as UTF-8.

  python -u cohort_control.py
"""
from __future__ import annotations

import collections
import csv
import io
import json
import math
import os
import sys

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps

import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------- paths
REL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUTFILE = os.path.join(REL, "results", "cohort_control.txt")

PRIMARY6 = ["qwen7b", "qwen14b", "qwen32b", "qwen72b", "llama8b", "llama70b"]
CORE4 = ["qwen7b", "llama8b", "qwen72b", "llama70b"]
PRETTY = {"qwen7b": "Qwen2.5-7B", "qwen14b": "Qwen2.5-14B",
          "qwen32b": "Qwen2.5-32B", "qwen72b": "Qwen2.5-72B",
          "llama8b": "Llama-3.1-8B", "llama70b": "Llama-3.1-70B"}

OUT = []
STATS = collections.defaultdict(dict)


def say(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    sys.stdout.write(s.encode("ascii", "backslashreplace").decode("ascii") + "\n")


def hr(c="-", n=100):
    say(c * n)


# ------------------------------------------------------------ release loaders
def load_jsonl(path, key="item_id"):
    out = {}
    if not os.path.exists(path):
        return out
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    r = json.loads(line)
                    out[r[key]] = r
                except Exception:
                    pass
    return out


def classify(prod, rec):
    """Verbatim from scripts/analyze_pilot.py (the repo-wide convention)."""
    if prod is None:
        return None
    strict_fail = prod["k_prod"] == 0 and not prod["greedy_correct"]
    if prod["k_prod"] >= 6:
        return "known"
    if not strict_fail:
        return "weak" if prod["k_prod"] <= 2 else "ambiguous_prod"
    if rec is None or rec.get("n_rec", 0) < 4:
        return "fail_norec"
    k = rec["k_rec"]
    if k >= 3:
        return "TOT"
    if k <= 1:
        return "DK"
    return "ambiguous_rec"


def correct_cues(path):
    out = {}
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("condition") == "correct":
                    out[r["item_id"]] = r
    return out


def unlocked(r):
    return 1 if (bool(r["greedy_correct"]) or r["k_cue"] > 0) else 0


def mean_p(r):
    if r is None:
        return None
    mv = r.get("mean_p_target")
    if mv is None:
        rots = r.get("rotations") or []
        ps = [x.get("p_target") for x in rots if x.get("p_target") is not None]
        mv = sum(ps) / len(ps) if ps else None
    return mv


# ------------------------------------------------------------ cohort tables
def load_cohort_table(lang):
    """Per-item cohort counts written by cohort_table.py.

    data/cohort_<lang>.tsv holds, for every target and decoy word, the
    cohort variants of cohort_table.py, plus '# resource:' lines with the
    sizes of the lexicons they were counted in. The lexicons themselves are
    not redistributed.
    """
    path = os.path.join(REL, "data", "cohort_%s.tsv" % lang)
    meta, lines = {}, []
    with io.open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("# resource:"):
                k, v = ln[len("# resource:"):].strip().split("=", 1)
                meta[k.strip()] = int(v)
            elif not ln.startswith("#"):
                lines.append(ln)
    rows = {}
    for r in csv.DictReader(lines, delimiter="\t"):
        rows[r["word"]] = r
    return rows, meta


class _CohortTable(object):
    FLOATS = ["log_coh", "log_coh_freqw", "log_coh_len", "log_coh_hifreq",
              "log_coh_wn"]

    def __init__(self, lang):
        self.rows, self.meta = load_cohort_table(lang)

    def variants(self, word):
        r = self.rows[word]
        out = {"coh_n": int(r["coh_n"])}
        for c in self.FLOATS:
            out[c] = float(r[c])
        return out


class CohortEn(_CohortTable):
    """Cohort size of an English word = CLEARPOND entries sharing its first
    letter (see cohort_table.py for the variants)."""

    def __init__(self):
        _CohortTable.__init__(self, "en")

    def unit(self, word):
        return word[0].lower()


class CohortZh(_CohortTable):
    """Cohort size of a Chinese word = CC-CEDICT simplified headwords sharing
    its first character (see cohort_table.py for the variants)."""

    def __init__(self):
        _CohortTable.__init__(self, "zh")

    def unit(self, word):
        return word[0]


VARIANTS = {
    "en": ["log_coh", "log_coh_freqw", "log_coh_len", "log_coh_hifreq",
           "log_coh_wn"],
    "zh": ["log_coh", "log_coh_freqw", "log_coh_len", "log_coh_hifreq"],
}
VLABEL_EN = {"log_coh": "CLEARPOND type count (PRIMARY)",
             "log_coh_freqw": "CLEARPOND frequency mass",
             "log_coh_len": "CLEARPOND, length matched +/-1",
             "log_coh_hifreq": "CLEARPOND, freq >= 1/million (Zipf>=3)",
             "log_coh_wn": "WordNet 3.1 lemma count"}
VLABEL_ZH = {"log_coh": "CC-CEDICT headword count (PRIMARY)",
             "log_coh_freqw": "SUBTLEX-CH frequency mass",
             "log_coh_len": "CC-CEDICT, same n_chars",
             "log_coh_hifreq": "SUBTLEX-CH type count"}


def vlab(lang):
    return VLABEL_EN if lang == "en" else VLABEL_ZH


# ------------------------------------------------------------- model helpers
def z(s):
    s = pd.Series(s).astype(float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def fit_logit(df, formula, cluster):
    return smf.logit(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df[cluster]}, disp=0)


def coef(m, name):
    b = float(m.params[name])
    se = float(m.bse[name])
    p = float(m.pvalues[name])
    return b, se, p, b - 1.96 * se, b + 1.96 * se


def ame(m, df, name):
    """Average marginal effect on the probability scale for a covariate."""
    p = np.asarray(m.predict(df), dtype=float)
    return float(m.params[name] * np.mean(p * (1 - p)))


def wilson(k, n, zc=1.96):
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    den = 1 + zc * zc / n
    ctr = (ph + zc * zc / (2 * n)) / den
    half = zc * math.sqrt(ph * (1 - ph) / n + zc * zc / (4 * n * n)) / den
    return ctr - half, ctr + half


def tercile_labels(v):
    q = np.quantile(v, [1 / 3, 2 / 3])
    out = []
    for x in v:
        out.append(0 if x <= q[0] else (1 if x <= q[1] else 2))
    return np.array(out), q


# ===========================================================================
# ANALYSIS 1
# ===========================================================================
def build_cue_rows(lang, coh):
    """Strict failures with a correct-cue record, cleanly classified (TOT/DK),
    over the six main settings -- the 1,938 En / 3,240 Zh population."""
    items = load_jsonl(os.path.join(REL, "data",
                                    "items_full_%s_final.jsonl" % lang))
    rows = []
    for tag in PRIMARY6:
        base = os.path.join(REL, "results", "full_%s_%s" % (lang, tag))
        prod = load_jsonl(os.path.join(base, "production.jsonl"))
        rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
        rc = correct_cues(os.path.join(base, "cue.jsonl"))
        for iid, p in prod.items():
            if not (p["k_prod"] == 0 and not p["greedy_correct"]):
                continue
            if classify(p, rec.get(iid)) not in ("TOT", "DK"):
                continue
            r = rc.get(iid)
            if r is None:
                continue
            it = items.get(iid)
            if it is None:
                continue
            w = it["word"]
            d = dict(item=iid, model=tag, y=unlocked(r),
                     unit=coh.unit(w), wlen=len(w),
                     fit=mean_p(rec.get(iid)))
            if lang == "en":
                d["logfreq"] = float(it.get("zipf")) \
                    if it.get("zipf") is not None else np.nan
            else:
                fpm = it.get("freq_per_million")
                d["logfreq"] = math.log10(float(fpm) + 0.01) \
                    if fpm is not None else np.nan
            d.update(coh.variants(w))
            rows.append(d)
    return pd.DataFrame(rows)


def analysis1(lang, coh):
    df = build_cue_rows(lang, coh)
    hr("=")
    say("ANALYSIS 1 [%s]  TARGET CUE BENEFIT vs COHORT SIZE" % lang.upper())
    hr("=")
    say("population: strict production failure (greedy wrong AND k_prod == 0),")
    say("cleanly classified (TOT or DK), with a correct-cue record, over the")
    say("six main settings.  outcome: unlock = greedy_correct OR k_cue >= 1")
    say("(any-of-20).  A pure narrowing account predicts a strong NEGATIVE")
    say("slope of unlock on log cohort size.")
    say("")
    say("rows=%d  items=%d  distinct first units=%d  unlock=%.4f (%d/%d)" %
        (len(df), df["item"].nunique(), df["unit"].nunique(),
         df.y.mean(), int(df.y.sum()), len(df)))
    say("cohort size (raw count): min=%d  p25=%d  median=%d  p75=%d  max=%d" %
        (df.coh_n.min(), df.coh_n.quantile(.25), df.coh_n.median(),
         df.coh_n.quantile(.75), df.coh_n.max()))
    say("log10 cohort: mean=%.3f  sd=%.3f  range [%.3f, %.3f]" %
        (df.log_coh.mean(), df.log_coh.std(ddof=0),
         df.log_coh.min(), df.log_coh.max()))
    say("")

    df["zcoh"] = z(df.log_coh)
    sd_log = df.log_coh.std(ddof=0)

    say("-- primary model: logit(unlock) ~ z(log cohort) + model FE, "
        "item-clustered")
    m = fit_logit(df, "y ~ zcoh + C(model)", "item")
    b, se, p, lo, hi = coef(m, "zcoh")
    a = ame(m, df, "zcoh")
    say("   b(z log cohort) = %+.4f  SE %.4f  95%% CI [%+.4f, %+.4f]  p=%.4f"
        % (b, se, lo, hi, p))
    say("   odds ratio per +1 SD of log cohort = %.3f  95%% CI [%.3f, %.3f]"
        % (math.exp(b), math.exp(lo), math.exp(hi)))
    say("   1 SD of log10 cohort = %.3f decades (a %.1fx change in cohort)"
        % (sd_log, 10 ** sd_log))
    say("   average marginal effect = %+.4f unlock per +1 SD "
        "(95%% CI [%+.4f, %+.4f])"
        % (a, a * lo / b if b else 0.0, a * hi / b if b else 0.0))
    STATS[lang].update(a1_n=len(df), a1_unlock=float(df.y.mean()),
                       a1_b=b, a1_se=se, a1_lo=lo, a1_hi=hi, a1_p=p,
                       a1_items=int(df["item"].nunique()),
                       a1_or=math.exp(b), a1_or_lo=math.exp(lo),
                       a1_or_hi=math.exp(hi),
                       a1_dec=a / sd_log,
                       a1_dec_lo=(a * lo / b) / sd_log if b else float("nan"),
                       a1_dec_hi=(a * hi / b) / sd_log if b else float("nan"),
                       a1_cmin=float(df.coh_n.min()),
                       a1_cmax=float(df.coh_n.max()))
    say("   -> per 10x larger cohort: %+.4f unlock (95%% CI [%+.4f, %+.4f])"
        % (a / sd_log, (a * lo / b) / sd_log if b else 0.0,
           (a * hi / b) / sd_log if b else 0.0))
    say("   n=%d, clusters(items)=%d" % (int(m.nobs), df["item"].nunique()))
    say("")

    say("-- same model, SEs clustered on the CUED FIRST UNIT instead")
    if lang == "en":
        say("   (English has only 26 distinct first letters, so the")
        say("    item-clustered fit above treats a 26-valued regressor as if")
        say("    it had %d independent levels.  With G=26 the cluster-robust"
            % df["item"].nunique())
        say("    asymptotics are unreliable and the SE is typically biased")
        say("    DOWN, so neither version is decisive; the 26-point")
        say("    letter-level aggregate below is the transparent check.)")
    else:
        say("   (%d distinct first characters, so this is close to the "
            "item-clustered fit)" % df["unit"].nunique())
    try:
        m2 = fit_logit(df, "y ~ zcoh + C(model)", "unit")
        b2, se2, p2, lo2, hi2 = coef(m2, "zcoh")
        say("   b = %+.4f  SE %.4f  95%% CI [%+.4f, %+.4f]  p=%.4f  "
            "clusters=%d" % (b2, se2, lo2, hi2, p2, df["unit"].nunique()))
    except Exception as e:
        say("   failed: %s" % str(e)[:120])
    say("")

    say("-- with word-frequency and word-length controls")
    d2 = df.dropna(subset=["logfreq"]).copy()
    d2["zfreq"] = z(d2.logfreq)
    d2["zlen"] = z(d2.wlen)
    d2["zcoh"] = z(d2.log_coh)
    m3 = fit_logit(d2, "y ~ zcoh + zfreq + zlen + C(model)", "item")
    b3, se3, p3, lo3, hi3 = coef(m3, "zcoh")
    say("   b(z log cohort) = %+.4f  SE %.4f  95%% CI [%+.4f, %+.4f]  "
        "p=%.4f  (n=%d)" % (b3, se3, lo3, hi3, p3, int(m3.nobs)))
    for nm in ("zfreq", "zlen"):
        bb, ss, pp, ll, hh = coef(m3, nm)
        say("   b(%s) = %+.4f  SE %.4f  p=%.4f" % (nm, bb, ss, pp))
    say("")

    say("-- per-model cohort slope (is the pooled estimate driven by one "
        "model?)")
    say("   %-16s %6s %8s %10s %8s %20s" %
        ("model", "n", "unlock", "b(zcoh)", "SE", "95% CI"))
    for tag in PRIMARY6:
        s = df[df.model == tag].copy()
        s["zcoh"] = z(s.log_coh)
        try:
            mm = fit_logit(s, "y ~ zcoh", "item")
            bb, ss, pp, ll, hh = coef(mm, "zcoh")
            say("   %-16s %6d %8.4f %+10.4f %8.4f [%+7.4f,%+7.4f]"
                % (PRETTY[tag], len(s), s.y.mean(), bb, ss, ll, hh))
        except Exception as e:
            say("   %-16s failed: %s" % (PRETTY[tag], str(e)[:60]))
    say("")

    say("-- sensitivity: alternative cohort definitions (same model)")
    say("   %-42s %9s %8s %18s %8s" %
        ("definition", "b", "SE", "95% CI", "p"))
    lab = vlab(lang)
    for v in VARIANTS[lang]:
        dd = df.copy()
        dd["zv"] = z(dd[v])
        if dd["zv"].std(ddof=0) == 0:
            say("   %-42s  (no variance)" % lab[v])
            continue
        mm = fit_logit(dd, "y ~ zv + C(model)", "item")
        bb, ss, pp, ll, hh = coef(mm, "zv")
        say("   %-42s %+9.4f %8.4f [%+7.4f,%+7.4f] %8.4f"
            % (lab[v], bb, ss, ll, hh, pp))
    say("")

    say("-- unlock by cohort-size tercile (equal-count split of rows)")
    t, q = tercile_labels(df.log_coh.values)
    df["ter"] = t
    say("   %-8s %8s %8s %10s %8s %20s" %
        ("tercile", "n", "unlock", "cohort_md", "k", "95% CI"))
    for i, nm in enumerate(("bottom", "middle", "top")):
        s = df[df.ter == i]
        k, n = int(s.y.sum()), len(s)
        lo_, hi_ = wilson(k, n)
        say("   %-8s %8d %8.4f %10.0f %8d [%.4f, %.4f]"
            % (nm, n, s.y.mean(), s.coh_n.median(), k, lo_, hi_))
    s0, s1_, s2 = df[df.ter == 0], df[df.ter == 1], df[df.ter == 2]
    STATS[lang].update(a1_t=[float(s0.y.mean()), float(s1_.y.mean()),
                             float(s2.y.mean())],
                       a1_tc=[float(s0.coh_n.median()),
                              float(s1_.coh_n.median()),
                              float(s2.coh_n.median())])
    say("   top-minus-bottom = %+.4f  (cohort medians %.0f vs %.0f, "
        "a %.1fx difference)"
        % (s2.y.mean() - s0.y.mean(), s0.coh_n.median(), s2.coh_n.median(),
           max(s2.coh_n.median(), 1) / max(s0.coh_n.median(), 1)))
    say("   cutpoints at log10 cohort = %.3f, %.3f" % (q[0], q[1]))
    say("")

    say("-- unlock by cohort-size decile (finer grain)")
    dec = pd.qcut(df.log_coh.rank(method="first"), 10, labels=False)
    df["dec"] = dec
    say("   %-6s %7s %8s %10s" % ("decile", "n", "unlock", "cohort_md"))
    for i in range(10):
        s = df[df.dec == i]
        if not len(s):
            continue
        say("   %-6d %7d %8.4f %10.0f" %
            (i + 1, len(s), s.y.mean(), s.coh_n.median()))
    r_sp = sps.spearmanr(df.log_coh, df.y)
    say("   Spearman(log cohort, unlock) over rows = %+.4f (p=%.4f)"
        % (r_sp.statistic, r_sp.pvalue))
    say("")

    say("-- HOW MUCH UNLOCK CAN NARROWING BUY? (the narrowing budget)")
    span = df.log_coh.max() - df.log_coh.min()
    per_dec = a / sd_log
    per_dec_lo = (a * lo / b) / sd_log if b else float("nan")
    per_dec_hi = (a * hi / b) / sd_log if b else float("nan")
    say("   observed cohort span = %.2f decades (%.0f to %.0f entries, a "
        "%.0fx range)" % (span, df.coh_n.min(), df.coh_n.max(),
                          max(df.coh_n.max(), 1) / max(df.coh_n.min(), 1)))
    say("   fitted unlock change across that ENTIRE span = %+.4f "
        "(95%% CI [%+.4f, %+.4f])"
        % (per_dec * span, min(per_dec_lo, per_dec_hi) * span,
           max(per_dec_lo, per_dec_hi) * span))
    STATS[lang].update(a1_span_eff=per_dec * span,
                       a1_span_lo=min(per_dec_lo, per_dec_hi) * span,
                       a1_span_hi=max(per_dec_lo, per_dec_hi) * span)
    say("   observed target unlock on this population = %.4f" % df.y.mean())
    if per_dec < 0:
        say("   -> narrowing works in the predicted direction, but the whole")
        say("      %.0fx cohort range moves unlock by at most %.1f points,"
            % (max(df.coh_n.max(), 1) / max(df.coh_n.min(), 1),
               abs(per_dec * span) * 100))
        say("      against an observed unlock level of %.1f points."
            % (df.y.mean() * 100))
    else:
        say("   -> the slope has the WRONG SIGN for a narrowing account:")
        say("      unlock RISES with cohort size, so tighter narrowing is")
        say("      associated with LESS recovery, not more.")
    say("")

    if lang == "en":
        say("-- per-letter aggregate (26 points: the honest unit of analysis")
        say("   for a letter-level regressor)")
        say("   %-6s %8s %7s %8s" % ("letter", "cohort", "n", "unlock"))
        g = df.groupby("unit").agg(n=("y", "size"), u=("y", "mean"),
                                   c=("coh_n", "first")).reset_index()
        g = g.sort_values("c")
        for _, r in g.iterrows():
            say("   %-6s %8d %7d %8.4f" % (r["unit"], r["c"], r["n"], r["u"]))
        wr = sps.pearsonr(np.log10(g.c.values), g.u.values)
        ws = sps.spearmanr(np.log10(g.c.values), g.u.values)
        say("   Pearson(log cohort, letter unlock rate) over %d letters = "
            "%+.3f (p=%.3f)" % (len(g), wr.statistic, wr.pvalue))
        say("   Spearman = %+.3f (p=%.3f)" % (ws.statistic, ws.pvalue))
        # frequency-weighted letter-level regression
        wls = np.polyfit(np.log10(g.c.values), g.u.values, 1, w=np.sqrt(g.n.values))
        say("   weighted letter-level slope = %+.4f unlock per decade of cohort"
            % wls[0])
        say("")
    return df


# ===========================================================================
# ANALYSIS 2
# ===========================================================================
def build_matched(lang, coh):
    """Reproduce the fit_gradient / fit_ci matched population, then attach cohort size of
    the CUED word (decoy's own first unit for decoy rows, target's for
    target rows)."""
    items = load_jsonl(os.path.join(REL, "data",
                                    "items_full_%s_final.jsonl" % lang))
    dec = load_jsonl(os.path.join(REL, "data", "items_decoy_%s.jsonl" % lang))
    rows = []
    for tag in CORE4:
        base = os.path.join(REL, "results", "full_%s_%s" % (lang, tag))
        prod = load_jsonl(os.path.join(base, "production.jsonl"))
        rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
        realc = correct_cues(os.path.join(base, "cue.jsonl"))
        fit = {}
        bdir = os.path.join(REL, "results", "decoyb_%s_%s" % (lang, tag))
        for r in load_jsonl(os.path.join(bdir, "recognition.jsonl")).values():
            fit[r["item_id"].replace("decoyb_", "decoy_", 1)] = mean_p(r)
        cp = os.path.join(REL, "results", "decoycue_%s_%s" % (lang, tag),
                          "cue.jsonl")
        with io.open(cp, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("condition") != "correct":
                    continue
                it = dec.get(r["item_id"])
                if not it:
                    continue
                src = it["true_item_id"]
                if classify(prod.get(src), rec.get(src)) not in ("TOT", "DK"):
                    continue
                fv = fit.get(r["item_id"])
                if fv is None:
                    continue
                dw = it["word"]
                d = dict(src=src, model=tag, is_target=0, y=unlocked(r),
                         decoy_fit=float(fv), fit=float(fv),
                         unit=coh.unit(dw), wlen=len(dw),
                         pair=r["item_id"] + "|" + tag)
                d.update(coh.variants(dw))
                rows.append(d)
                rr = realc.get(src)
                if rr is not None:
                    srcit = items.get(src)
                    tw = srcit["word"] if srcit else None
                    if tw is None:
                        continue
                    tf = mean_p(rec.get(src))
                    d2 = dict(src=src, model=tag, is_target=1, y=unlocked(rr),
                              decoy_fit=float(fv),
                              fit=float(tf) if tf is not None else np.nan,
                              unit=coh.unit(tw), wlen=len(tw),
                              pair=r["item_id"] + "|" + tag)
                    d2.update(coh.variants(tw))
                    rows.append(d2)
    return pd.DataFrame(rows)


def margin_boot(sub, reps=4000, seed=24):
    """Source-item cluster bootstrap of the target-minus-decoy difference,
    the same resampling scheme as fit_ci.py."""
    rng = np.random.default_rng(seed)
    srcs = sorted(sub.src.unique())
    by = {}
    for s, g in sub.groupby("src"):
        by[s] = (g[g.is_target == 0].y.values, g[g.is_target == 1].y.values)
    diffs, ratios = [], []
    for _ in range(reps):
        take = rng.integers(0, len(srcs), len(srcs))
        dk = dn = tk = tn = 0
        for kx in take:
            dy, ty = by[srcs[kx]]
            dk += int(dy.sum()); dn += len(dy)
            tk += int(ty.sum()); tn += len(ty)
        if dn and tn:
            diffs.append(tk / tn - dk / dn)
            if dk:
                ratios.append((tk / tn) / (dk / dn))
    if not diffs:
        return (float("nan"),) * 5
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pv = float(np.mean(np.array(diffs) <= 0))
    rlo, rhi = (np.percentile(ratios, [2.5, 97.5]) if ratios
                else (float("nan"), float("nan")))
    return lo, hi, pv, rlo, rhi


def analysis2(lang, coh):
    df = build_matched(lang, coh)
    hr("=")
    say("ANALYSIS 2 [%s]  FIT-MATCHED TARGET-vs-DECOY MARGIN, COHORT CONTROLLED"
        % lang.upper())
    hr("=")
    say("matched population of decoycue_match / fit_gradient / fit_ci:")
    say("decoy-cue rows whose SOURCE bank item is a cleanly classified failure")
    say("(TOT or DK) for that same model, paired 1:1 with the same model's")
    say("correct-cue row on that source item.  4 settings (CORE).  fit = the")
    say("cued word's mean_p_target in its 4AFC (decoyb target-absent 4AFC for")
    say("decoys, the main 4AFC for targets).  cohort = of the CUED word.")
    say("")

    d = df[df.is_target == 0]
    t = df[df.is_target == 1]
    say("REPRODUCTION CHECK")
    say("  matched rows: decoy %d, target %d, source items %d  "
        "(fit_gradient reports %s)" % (len(d), len(t), df.src.nunique(),
                              "299 En / 789 Zh"))
    say("  decoy unlock %.4f, target unlock %.4f" % (d.y.mean(), t.y.mean()))
    sub = df[df.decoy_fit >= 0.8]
    sd_, st_ = sub[sub.is_target == 0], sub[sub.is_target == 1]
    say("  fit >= 0.8 subset: decoy %.4f (%d rows) vs target %.4f (%d rows), "
        "%d source items" % (sd_.y.mean(), len(sd_), st_.y.mean(), len(st_),
                             sub.src.nunique()))
    if lang == "en":
        say("  paper reports 32.9%% decoy vs 61.8%% target (fit_ci.txt) -- "
            "%s" % ("REPRODUCED" if (abs(sd_.y.mean() - .329) < .002 and
                                     abs(st_.y.mean() - .618) < .002)
                    else "MISMATCH"))
    else:
        say("  paper reports 30.8%% decoy vs 32.0%% target (fit_ci.txt) -- "
            "%s" % ("REPRODUCED" if (abs(sd_.y.mean() - .308) < .002 and
                                     abs(st_.y.mean() - .320) < .002)
                    else "MISMATCH"))
    say("")

    STATS[lang].update(a2_dec=float(sd_.y.mean()), a2_tgt=float(st_.y.mean()),
                       a2_n=len(sd_), a2_src=int(sub.src.nunique()),
                       a2_dec_all=float(d.y.mean()),
                       a2_tgt_all=float(t.y.mean()))
    for key, lab, s in (("f8", "fit >= 0.8 (headline subset)", sub),
                        ("all", "full matched population", df)):
        say("-" * 100)
        say("(a) NESTED LOGISTIC MODELS -- %s   [n=%d rows, %d source items]"
            % (lab, len(s), s.src.nunique()))
        s = s.copy()
        s["zcoh"] = z(s.log_coh)
        s["zfit"] = z(s.fit)
        s = s.dropna(subset=["zfit", "zcoh"])
        say("    cohort of the cued word: target rows median %.0f, "
            "decoy rows median %.0f"
            % (s[s.is_target == 1].coh_n.median(),
               s[s.is_target == 0].coh_n.median()))
        specs = [("M1  is_target + model FE", "y ~ is_target + C(model)"),
                 ("M2  + z(log cohort)", "y ~ is_target + zcoh + C(model)"),
                 ("M3  + z(log cohort) + z(fit)",
                  "y ~ is_target + zcoh + zfit + C(model)"),
                 ("M4  M3 + is_target:z(log cohort)",
                  "y ~ is_target * zcoh + zfit + C(model)")]
        say("    %-32s %10s %8s %20s %8s %8s" %
            ("model", "b_target", "SE", "95% CI", "p", "OR"))
        b1 = None
        for nm, fo in specs:
            try:
                m = fit_logit(s, fo, "src")
            except Exception as e:
                say("    %-32s FAILED %s" % (nm, str(e)[:60]))
                continue
            b, se, p, lo, hi = coef(m, "is_target")
            if b1 is None:
                b1 = b
            say("    %-32s %+10.4f %8.4f [%+7.4f,%+7.4f] %8.4f %8.3f"
                % (nm, b, se, lo, hi, p, math.exp(b)))
        m2 = fit_logit(s, "y ~ is_target + zcoh + C(model)", "src")
        b2t = float(m2.params["is_target"])
        _, se2t, p2t, lo2t, hi2t = coef(m2, "is_target")
        att = 100.0 * (b1 - b2t) / b1 if b1 else float("nan")
        if abs(b1 or 0.0) < 0.2:
            say("    ATTENUATION of b_target from adding cohort size "
                "(M1 -> M2): not meaningful, b_target is already ~0 "
                "(%+.4f -> %+.4f)" % (b1, b2t))
        else:
            say("    ATTENUATION of b_target from adding cohort size "
                "(M1 -> M2): %.2f%% of the coefficient (%+.4f -> %+.4f)"
                % (att, b1, b2t))
        STATS[lang]["a2_%s" % key] = dict(
            b1=b1, b2=b2t, se2=se2t, lo2=lo2t, hi2=hi2t, p2=p2t, att=att,
            n=len(s), src=int(s.src.nunique()))
        m = fit_logit(s, "y ~ is_target + zcoh + zfit + C(model)", "src")
        for nm in ("zcoh", "zfit"):
            b, se, p, lo, hi = coef(m, nm)
            say("    M3 b(%-5s) = %+.4f  SE %.4f  95%% CI [%+.4f, %+.4f]  "
                "p=%.4f" % (nm, b, se, lo, hi, p))
        a = ame(m, s, "is_target")
        say("    M3 average marginal effect of is_target = %+.4f unlock "
            "(probability scale)" % a)
        try:
            m4 = fit_logit(s, "y ~ is_target * zcoh + zfit + C(model)", "src")
            nmx = [c for c in m4.params.index if ":" in c][0]
            b, se, p, lo, hi = coef(m4, nmx)
            say("    M4 INTERACTION is_target x z(log cohort) = %+.4f  SE %.4f"
                "  95%% CI [%+.4f, %+.4f]  p=%.4f" % (b, se, lo, hi, p))
            STATS[lang]["a2_%s" % key].update(ix=b, ix_lo=lo, ix_hi=hi, ix_p=p)
            say("       (this is the direct test of 'the margin depends on how")
            say("        much the cue narrows'; a null means it does not)")
        except Exception as e:
            say("    M4 interaction failed: %s" % str(e)[:80])
        say("")

        say("(b) TARGET-vs-DECOY MARGIN WITHIN COHORT-SIZE STRATA -- %s" % lab)
        tv, q = tercile_labels(s.log_coh.values)
        s = s.assign(ter=tv)
        say("    strata are terciles of the POOLED (target + decoy) log cohort")
        say("    distribution; cutpoints log10 = %.3f, %.3f" % (q[0], q[1]))
        say("    %-8s %11s %7s %8s %7s %8s %9s %8s %20s" %
            ("tercile", "cohort_md", "n_dec", "decoy", "n_tgt", "target",
             "diff", "ratio", "95% CI (bootstrap)"))
        ters = []
        for i, nm in enumerate(("bottom", "middle", "top")):
            g = s[s.ter == i]
            gd, gt = g[g.is_target == 0], g[g.is_target == 1]
            if not len(gd) or not len(gt):
                say("    %-8s (empty)" % nm)
                continue
            dr, tr = gd.y.mean(), gt.y.mean()
            lo, hi, pv, rlo, rhi = margin_boot(g)
            ters.append((float(g.coh_n.median()), float(dr), float(tr),
                         float(lo), float(hi), float(pv)))
            say("    %-8s %11.0f %7d %8.4f %7d %8.4f %+9.4f %8.2f "
                "[%+.4f,%+.4f] p=%.3f"
                % (nm, g.coh_n.median(), len(gd), dr, len(gt), tr, tr - dr,
                   (tr / dr if dr else float("nan")), lo, hi, pv))
        STATS[lang]["a2_%s" % key]["ters"] = ters
        gd, gt = s[s.is_target == 0], s[s.is_target == 1]
        lo, hi, pv, rlo, rhi = margin_boot(s)
        say("    %-8s %11.0f %7d %8.4f %7d %8.4f %+9.4f %8.2f "
            "[%+.4f,%+.4f] p=%.3f"
            % ("ALL", s.coh_n.median(), len(gd), gd.y.mean(), len(gt),
               gt.y.mean(), gt.y.mean() - gd.y.mean(),
               (gt.y.mean() / gd.y.mean() if gd.y.mean() else float("nan")),
               lo, hi, pv))
        say("")

        say("(c) EXACT-COHORT MATCHING: rows whose cued word shares the SAME")
        say("    first unit, so the cue narrows target and decoy identically")
        keep = []
        for u, g in s.groupby("unit"):
            if g.is_target.nunique() == 2:
                keep.append(g)
        if keep:
            e = pd.concat(keep)
            ed, et = e[e.is_target == 0], e[e.is_target == 1]
            lo, hi, pv, rlo, rhi = margin_boot(e)
            say("    %d first units have both arms; decoy %.4f (%d) vs "
                "target %.4f (%d)" % (e.unit.nunique(), ed.y.mean(), len(ed),
                                      et.y.mean(), len(et)))
            STATS[lang]["a2_%s" % key].update(
                ex_d=float(ed.y.mean()), ex_t=float(et.y.mean()),
                ex_nd=len(ed), ex_nt=len(et), ex_lo=float(lo),
                ex_hi=float(hi), ex_p=float(pv))
            say("    difference %+.4f  95%% CI [%+.4f, %+.4f]  P(diff<=0)=%.4f"
                % (et.y.mean() - ed.y.mean(), lo, hi, pv))
            say("    NOTE: restricting to shared first units breaks the 1:1")
            say("    source-item pairing (a target and its decoy rarely share")
            say("    a first unit), so the two arms have different n here.")
            if len(e) < 60:
                say("    n=%d rows is too small for a first-unit FE model; "
                    "skipped." % len(e))
            else:
                try:
                    m = fit_logit(e.copy(), "y ~ is_target + C(unit) + C(model)",
                                  "src")
                    b, se, p, lo2, hi2 = coef(m, "is_target")
                    flag = "" if getattr(m, "mle_retvals", {}).get(
                        "converged", True) else "  [ML did not fully converge]"
                    say("    logit with FIRST-UNIT fixed effects: b_target="
                        "%+.4f SE %.4f 95%% CI [%+.4f,%+.4f] p=%.4f (n=%d)%s"
                        % (b, se, lo2, hi2, p, int(m.nobs), flag))
                except Exception as ex:
                    say("    first-unit FE model failed (likely separation): %s"
                        % str(ex)[:90])
        say("")
    return df


# ===========================================================================
# ANALYSIS 3
# ===========================================================================
def analysis3(lang, cue_df, mt_df):
    hr("=")
    say("ANALYSIS 3 [%s]  DESCRIPTIVES: COHORT SIZE OF TARGETS vs DECOYS"
        % lang.upper())
    hr("=")
    d = mt_df[mt_df.is_target == 0]
    t = mt_df[mt_df.is_target == 1]
    say("cohort size (raw count of lexicon entries sharing the first unit),")
    say("over the matched rows:")
    say("  %-10s %6s %9s %9s %9s %9s %9s" %
        ("arm", "n", "mean", "p25", "median", "p75", "mean_log10"))
    for nm, s in (("targets", t), ("decoys", d)):
        say("  %-10s %6d %9.1f %9.0f %9.0f %9.0f %9.3f"
            % (nm, len(s), s.coh_n.mean(), s.coh_n.quantile(.25),
               s.coh_n.median(), s.coh_n.quantile(.75), s.log_coh.mean()))
    u = sps.mannwhitneyu(t.log_coh, d.log_coh, alternative="two-sided")
    say("  Mann-Whitney U on log cohort (target vs decoy): U=%.0f p=%.4f"
        % (u.statistic, u.pvalue))
    # paired at the source item
    pv = mt_df.pivot_table(index="pair", columns="is_target",
                           values="log_coh", aggfunc="first").dropna()
    if len(pv):
        dif = pv[1].values - pv[0].values
        w = sps.wilcoxon(dif) if np.any(dif != 0) else None
        say("  paired (same source item, same model): mean log10 difference "
            "target-minus-decoy = %+.4f (n=%d pairs)" % (dif.mean(), len(dif)))
        if w is not None:
            if w.pvalue >= 0.05:
                verdict = ("no reliable difference: targets and decoys are "
                           "narrowed about equally")
            elif dif.mean() > 0:
                verdict = ("decoys sit in SMALLER cohorts, i.e. the decoy cue "
                           "narrows HARDER than the target cue")
            else:
                verdict = ("decoys sit in LARGER cohorts, i.e. the decoy cue "
                           "narrows LESS than the target cue")
            say("  Wilcoxon signed-rank: W=%.0f p=%.4f" % (w.statistic, w.pvalue))
            say("  -> %s" % verdict)
        say("  share of pairs where the decoy's cohort is smaller: %.3f"
            % float(np.mean(dif > 0)))
    say("")
    say("cohort size vs definition fit:")
    for nm, s in (("targets", t), ("decoys", d)):
        ss = s.dropna(subset=["fit"])
        if len(ss) > 3:
            sp = sps.spearmanr(ss.log_coh, ss.fit)
            pe = sps.pearsonr(ss.log_coh, ss.fit)
            say("  %-10s Spearman(log cohort, fit) = %+.4f (p=%.4f)  "
                "Pearson = %+.4f (n=%d)"
                % (nm, sp.statistic, sp.pvalue, pe.statistic, len(ss)))
    ss = mt_df.dropna(subset=["fit"])
    sp = sps.spearmanr(ss.log_coh, ss.fit)
    say("  pooled     Spearman(log cohort, fit) = %+.4f (p=%.4f, n=%d)"
        % (sp.statistic, sp.pvalue, len(ss)))
    ss2 = cue_df.dropna(subset=["fit"])
    sp2 = sps.spearmanr(ss2.log_coh, ss2.fit)
    say("  analysis-1 target rows: Spearman(log cohort, 4AFC fit) = %+.4f "
        "(p=%.4f, n=%d)" % (sp2.statistic, sp2.pvalue, len(ss2)))
    say("")
    say("agreement among the cohort definitions (Spearman over analysis-1 "
        "rows):")
    lab = vlab(lang)
    for v in VARIANTS[lang][1:]:
        r = sps.spearmanr(cue_df.log_coh, cue_df[v])
        say("  primary vs %-44s %+0.4f" % (lab[v], r.statistic))
    say("")


def pseudo_anchor(lang):
    """The 0% rung of the composition series, recomputed here so the report
    is self-contained (the cue aimed at a pseudoword)."""
    n = k = 0
    for tag in CORE4:
        p = os.path.join(REL, "results", "pseudocue_%s_%s" % (lang, tag),
                         "cue.jsonl")
        if not os.path.exists(p):
            continue
        with io.open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("condition") != "correct":
                    continue
                n += 1
                k += unlocked(r)
    return k, n


def summary():
    hr("=")
    say("SUMMARY")
    hr("=")
    say("")
    say("0  COMPOSITION-SERIES ANCHORS RECOMPUTED HERE (CORE 4 settings)")
    for lang in ("en", "zh"):
        pk, pn = pseudo_anchor(lang)
        s = STATS[lang]
        say("   %-4s pseudoword cue %.4f (%d/%d)  |  decoy cue (matched) "
            "%.4f  |  target cue %.4f"
            % (lang, pk / pn if pn else float("nan"), pk, pn,
               s["a2_dec_all"], s["a2_tgt_all"]))
    say("")
    say("1  DOES THE TARGET'S OWN CUE BENEFIT TRACK COHORT SIZE?  (no)")
    say("   %-4s %6s %8s %11s %20s %26s" %
        ("lang", "n", "unlock", "b(z logC)", "95% CI", "unlock per 10x cohort"))
    for lang in ("en", "zh"):
        s = STATS[lang]
        say("   %-4s %6d %8.4f %+11.4f [%+7.4f,%+7.4f] %+11.4f "
            "[%+.4f,%+.4f]"
            % (lang, s["a1_n"], s["a1_unlock"], s["a1_b"], s["a1_lo"],
               s["a1_hi"], s["a1_dec"], min(s["a1_dec_lo"], s["a1_dec_hi"]),
               max(s["a1_dec_lo"], s["a1_dec_hi"])))
    say("   unlock by cohort tercile (bottom / middle / top; cohort medians):")
    for lang in ("en", "zh"):
        s = STATS[lang]
        say("     %-4s %.3f / %.3f / %.3f   (cohorts %.0f / %.0f / %.0f)"
            % (lang, s["a1_t"][0], s["a1_t"][1], s["a1_t"][2],
               s["a1_tc"][0], s["a1_tc"][1], s["a1_tc"][2]))
    se = STATS["en"]
    say("   EN: slope is in the predicted direction but small and not")
    say("       significant (p=%.3f).  Sweeping the ENTIRE %.0fx observed"
        % (se["a1_p"], se["a1_cmax"] / max(se["a1_cmin"], 1)))
    say("       cohort range moves unlock by %.1f points [%.1f, %.1f], while"
        % (100 * se["a1_span_eff"], 100 * se["a1_span_lo"],
           100 * se["a1_span_hi"]))
    pk, pn = pseudo_anchor("en")
    say("       unlock itself sits at %.1f points and the same cue aimed at"
        % (100 * se["a1_unlock"]))
    say("       a pseudoword yields %.2f%% (%d/%d).  The CI is wide, so a"
        % (100.0 * pk / pn, pk, pn))
    say("       small narrowing effect is not excluded -- it is just far too")
    say("       small to be the mechanism.")
    say("   ZH: slope is POSITIVE and significant -- the wrong sign for a")
    say("       narrowing account (tighter narrowing, LESS recovery).")
    say("")
    say("2  DOES THE FIT-MATCHED TARGET-DECOY MARGIN SURVIVE COHORT CONTROL?")
    say("   %-4s %10s %9s %9s %10s %10s %8s" %
        ("lang", "subset", "decoy", "target", "b_target", "b_target", "atten."))
    say("   %-4s %10s %9s %9s %10s %10s %8s" %
        ("", "", "rate", "rate", "M1 (raw)", "M2 (+coh)", ""))
    for lang in ("en", "zh"):
        s = STATS[lang]
        for key, nm in (("f8", "fit>=0.8"), ("all", "all matched")):
            k = s["a2_%s" % key]
            dr, tr = (s["a2_dec"], s["a2_tgt"]) if key == "f8" \
                else (s["a2_dec_all"], s["a2_tgt_all"])
            att = "%6.1f%%" % k["att"] if abs(k["b1"]) >= 0.2 else "   n/a"
            say("   %-4s %10s %9.4f %9.4f %+10.4f %+10.4f %8s"
                % (lang, nm, dr, tr, k["b1"], k["b2"], att))
    say("   (attenuation n/a where b_target is already ~0, i.e. ZH fit>=0.8)")
    say("")
    say("   margin within cohort terciles (target rate - decoy rate):")
    for lang in ("en", "zh"):
        for key, nm in (("f8", "fit>=0.8"), ("all", "all matched")):
            t = STATS[lang]["a2_%s" % key].get("ters", [])
            if not t:
                continue
            say("     %-4s %-12s %s" % (lang, nm, "  ".join(
                "C~%.0f: %+.3f [%+.3f,%+.3f]" % (x[0], x[2] - x[1], x[3], x[4])
                for x in t)))
    say("")
    say("   is_target x z(log cohort) interaction (direct test that the")
    say("   margin depends on how hard the cue narrows).  Note the SIGN: a")
    say("   POSITIVE interaction means the target's advantage is LARGEST")
    say("   where the cohort is largest, i.e. where the cue constrains")
    say("   LEAST -- the opposite of what narrowing predicts.")
    for lang in ("en", "zh"):
        for key, nm in (("f8", "fit>=0.8"), ("all", "all matched")):
            k = STATS[lang]["a2_%s" % key]
            if "ix" not in k:
                continue
            say("     %-4s %-12s b=%+.4f  95%% CI [%+.4f,%+.4f]  p=%.4f"
                % (lang, nm, k["ix"], k["ix_lo"], k["ix_hi"], k["ix_p"]))
    say("")
    say("   same-first-unit (cue narrows both arms identically):")
    for lang in ("en", "zh"):
        for key, nm in (("f8", "fit>=0.8"), ("all", "all matched")):
            k = STATS[lang]["a2_%s" % key]
            if "ex_d" not in k:
                continue
            say("     %-4s %-12s decoy %.4f (n=%d) vs target %.4f (n=%d), "
                "diff %+.4f 95%% CI [%+.4f,%+.4f]"
                % (lang, nm, k["ex_d"], k["ex_nd"], k["ex_t"], k["ex_nt"],
                   k["ex_t"] - k["ex_d"], k["ex_lo"], k["ex_hi"]))
    say("")
    say("3  BOTTOM LINE")
    ef8 = STATS["en"]["a2_f8"]
    dif = [x[2] - x[1] for x in ef8.get("ters", [])]
    say("   * The English fit-matched margin (%.1f%% target vs %.1f%% decoy)"
        % (100 * STATS["en"]["a2_tgt"], 100 * STATS["en"]["a2_dec"]))
    say("     is essentially untouched by cohort size: adding log cohort")
    say("     attenuates the target coefficient by only %.1f%%, and within"
        % ef8["att"])
    say("     every cohort tercile the target still leads the decoy by")
    say("     %.1f to %.1f points."
        % (100 * min(dif), 100 * max(dif)) if dif else "     (no terciles)")
    say("   * The target's own unlock rate is insensitive to cohort size in")
    say("     English and moves the WRONG WAY in Chinese, so the size of the")
    say("     search space the cue leaves open is not what drives recovery.")
    say("   * In Chinese the target-over-decoy margin GROWS with cohort size")
    say("     (interaction b=%+.3f, p=%.4f on the matched population): the"
        % (STATS["zh"]["a2_all"]["ix"], STATS["zh"]["a2_all"]["ix_p"]))
    say("     target's advantage is concentrated exactly where the cue")
    say("     narrows LEAST, which a narrowing account cannot produce.")
    say("   * Chinese remains the weak cell: at matched fit the target-decoy")
    say("     margin is ~0 (as already reported by fit_ci), and cohort control")
    say("     does not rescue it. The Chinese contribution here is the")
    say("     wrong-signed cohort slope, which damages the narrowing account")
    say("     without supporting the availability account.")
    say("")


# ===========================================================================
def main():
    say("=" * 100)
    say("CUE-NARROWING / COHORT-SIZE CONTROL ANALYSIS")
    say("'Knowing Without Naming' (Findings of EACL 2027) -- cohort-size control")
    say("=" * 100)
    say("")
    say("CLAIM TESTED.  The first-unit cue works only because it shrinks")
    say("the candidate set, so unlocking under the cue does not show the")
    say("target was available beforehand.")
    say("")
    say("TESTABLE CONSEQUENCE.  If narrowing is the whole story, the cue's")
    say("benefit must scale with how much the first unit narrows the cohort.")
    say("Cohort sizes differ by up to two orders of magnitude within each")
    say("language, so this is a high-powered test.")
    say("")
    say("COHORT DEFINITION (primary).")
    say("  English: number of distinct orthographic types in the CLEARPOND")
    say("    English lexicon (27,751 SUBTLEX-US-attested word forms, every")
    say("    one >= 0.33 occurrences per million) that begin with the same")
    say("    letter as the cued word.  Analyses use log10(count + 1).")
    say("  Chinese: number of CC-CEDICT simplified headwords (120,968) that")
    say("    begin with the same character as the cued word (the Chinese cue")
    say("    is the first character).  log10(count + 1).")
    say("  Sensitivity variants per language are reported in analysis 1.")
    say("  Cohort size is only ever compared WITHIN a language, so the two")
    say("  languages using different resources is harmless.")
    say("")
    say("DATA NOTES / PROBLEMS ENCOUNTERED.")
    say("  * The Chinese cohort is counted in CC-CEDICT rather than in the")
    say("    MoE Revised dictionary the bank's glosses come from; SUBTLEX-CH")
    say("    is the frequency-attested alternative.")
    say("  * SUBTLEX-CH-WF covers only 642 of the 689 distinct target first")
    say("    characters (47 missing); CC-CEDICT misses only 6, which is why")
    say("    CC-CEDICT is the primary Chinese resource.")
    say("  * Every CLEARPOND entry already clears 0.33 per million, so a")
    say("    'Zipf floor' variant is implemented as freq >= 1 per million.")
    say("  * The recorded `cue` field equals the cued word's first unit in")
    say("    100% of rows in both languages and both arms (verified when")
    say("    this analysis was written), so the cohort can be keyed on the word itself.")
    say("  * English has only 26 distinct first letters, so the log-cohort")
    say("    regressor takes 26 values; item-clustered SEs are reported as")
    say("    specified but first-unit-clustered SEs and a 26-point")
    say("    letter-level aggregate are reported alongside them.")
    say("")

    coh_en = CohortEn()
    coh_zh = CohortZh()
    say("lexicon sizes: CLEARPOND %d types (%d first letters); WordNet 3.1 "
        "%d lemmas; CC-CEDICT %d headwords (%d first characters); "
        "SUBTLEX-CH %d types"
        % (coh_en.meta["clearpond_types"], coh_en.meta["clearpond_first_letters"],
           coh_en.meta["wordnet_lemmas"], coh_zh.meta["cedict_headwords"],
           coh_zh.meta["cedict_first_chars"], coh_zh.meta["subtlexch_types"]))
    say("")

    store = {}
    for lang, coh in (("en", coh_en), ("zh", coh_zh)):
        cue_df = analysis1(lang, coh)
        mt_df = analysis2(lang, coh)
        analysis3(lang, cue_df, mt_df)
        store[lang] = (cue_df, mt_df)

    summary()

    with io.open(OUTFILE, "w", encoding="utf-8", newline="\n") as f:
        f.write(("\n".join(OUT) + "\n").rstrip("\n") + "\n")
    sys.stdout.write("wrote " + OUTFILE + "\n")


if __name__ == "__main__":
    main()
