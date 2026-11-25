"""Path and convention helpers shared by the released analyses.

Resolves the repository root and the item files, and keeps the shared
conventions in one place: state relabelling under an alternative
answer key, two-proportion inference, and z-scoring.
"""
from __future__ import annotations

import math
import os

from analyze_pilot import classify, load_jsonl  # noqa: F401  (re-export)
from scoring import score_production

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

CONFIRMATORY = ["qwen7b", "llama8b", "qwen72b", "llama70b"]
PRIMARY = [("qwen7b", "Qwen2.5-7B"), ("qwen14b", "Qwen2.5-14B"),
           ("qwen32b", "Qwen2.5-32B"), ("qwen72b", "Qwen2.5-72B"),
           ("llama8b", "Llama-3.1-8B"), ("llama70b", "Llama-3.1-70B")]
EXTENDED = [("gemma12b", "Gemma-3-12B"), ("gemma27b", "Gemma-3-27B"),
            ("glm9b", "GLM-4-9B"), ("mistral7b", "Mistral-7B"),
            ("olmo13b", "OLMo-2-13B"), ("olmo32b", "OLMo-2-32B"),
            ("phi4", "Phi-4"), ("r1d14b", "R1-Distill-14B")]
ALL_MODELS = PRIMARY + EXTENDED

INITIAL_CH = {"en": "first_letter", "zh": "first_char"}
CHANNELS = {"en": ["first_letter", "n_letters", "n_syllables", "sounds_like"],
            "zh": ["first_char", "n_chars", "pinyin_initial", "tone1"]}


def rdir(lang, tag):
    return os.path.join(ROOT, "results", f"full_{lang}_{tag}")


def items_path(lang):
    return os.path.join(ROOT, "data", f"items_full_{lang}_final.jsonl")


def rejudge_path(lang):
    return os.path.join(ROOT, "data", f"items_{lang}_rejudge_llama.jsonl")


def tok_col(tag):
    return "n_tokens_qwen" if "qwen" in tag else "n_tokens_llama"


def zipf_of(it, lang):
    fpm = it.get("freq_per_million")
    if lang == "zh" and fpm and fpm > 0 and "log_freq" in it:
        return math.log10(fpm) + 3.0
    return it.get("zipf")


def rescore(prod_rec, answers, lang):
    """(greedy_correct, k_prod) recomputed against an alternative answer set."""
    g = bool(score_production(prod_rec.get("greedy", ""), answers, lang))
    k = sum(1 for t in prod_rec.get("samples", [])
            if score_production(t, answers, lang))
    return g, k


def classify_key(prod_rec, rec_rec, answers, lang):
    """analyze_pilot.classify, but with production scored against `answers`."""
    if prod_rec is None:
        return None
    g, k = rescore(prod_rec, answers, lang)
    strict_fail = (k == 0 and not g)
    if k >= 6:
        return "known"
    if not strict_fail:
        return "weak" if k <= 2 else "ambiguous_prod"
    if rec_rec is None or rec_rec.get("n_rec", 0) < 4:
        return "fail_norec"
    kr = rec_rec["k_rec"]
    if kr >= 3:
        return "TOT"
    if kr <= 1:
        return "DK"
    return "ambiguous_rec"


def zscore(s):
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def prop_test(c1, n1, c2, n2):
    """Group1 vs group2 proportions: rates, difference, 95% CI, Fisher p.

    Returns dict with p1, p2, diff (p1-p2), lo, hi (Newcombe hybrid-score CI
    on the difference), and two-sided Fisher exact p.
    """
    from scipy.stats import fisher_exact
    out = {"c1": c1, "n1": n1, "c2": c2, "n2": n2,
           "p1": c1 / n1 if n1 else float("nan"),
           "p2": c2 / n2 if n2 else float("nan")}
    out["diff"] = out["p1"] - out["p2"]
    out["lo"] = out["hi"] = float("nan")
    if n1 and n2:
        try:
            from statsmodels.stats.proportion import confint_proportions_2indep
            lo, hi = confint_proportions_2indep(c1, n1, c2, n2,
                                                method="newcomb",
                                                compare="diff")
            out["lo"], out["hi"] = lo, hi
        except Exception as e:  # pragma: no cover
            out["ci_err"] = str(e)[:60]
        try:
            out["p"] = fisher_exact([[c1, n1 - c1], [c2, n2 - c2]])[1]
        except Exception:
            out["p"] = float("nan")
    else:
        out["p"] = float("nan")
    return out


def fmt_prop(t, lab1="A", lab2="B"):
    return (f"{lab1}={t['c1']}/{t['n1']} ({t['p1']:.1%}) vs "
            f"{lab2}={t['c2']}/{t['n2']} ({t['p2']:.1%}); "
            f"diff={t['diff']:+.1%} 95%CI[{t['lo']:+.1%},{t['hi']:+.1%}] "
            f"Fisher p={t['p']:.4f}")
