"""Pooled TOT-vs-DK partial-access test across all models and languages.

Aligns channels cross-linguistically (initial unit = first_letter/first_char;
length unit = n_letters/n_chars; sound = sounds_like/pinyin_initial), pools all
8 datasets, and fits per aligned channel:
    correct ~ is_TOT + z(zipf) + z(length) + z(tokens) + model FE + lang FE
with item-clustered robust SEs. Also a sign test over the 8 per-dataset raw
TOT-DK differences for the initial-unit channel.

  python -u pooled_contrast.py
"""
from __future__ import annotations

import os
from itertools import product

import pandas as pd
import statsmodels.api as sm

from analyze_pilot import classify, load_jsonl, score_partial
from scoring import extract_candidate


def _lev_sim(a, b):
    """1 - normalized Levenshtein distance."""
    if not a or not b:
        return None
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return 1.0 - prev[lb] / max(la, lb)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODELS = ["qwen7b", "llama8b", "qwen72b", "llama70b"]
ALIGN = {
    "initial_unit": {"en": "first_letter", "zh": "first_char"},
    "length_unit": {"en": "n_letters", "zh": "n_chars"},
    "sound_unit": {"en": "sounds_like", "zh": "pinyin_initial"},
}


def zscore(s):
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def classify_lenient(prod, rec):
    """Sensitivity variant: production-impaired = k_prod<=2 & greedy wrong."""
    if prod is None:
        return None
    if not (prod["k_prod"] <= 2 and not prod["greedy_correct"]):
        return "other"
    if rec is None or rec.get("n_rec", 0) < 4:
        return None
    k = rec["k_rec"]
    return "TOT" if k >= 3 else ("DK" if k <= 1 else None)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lenient", action="store_true",
                    help="fail band <=2/20 instead of strict 0/20")
    ap.add_argument("--unique-only", action="store_true",
                    help="restrict to judge-verified unique-answer items")
    ap.add_argument("--include-ext", action="store_true",
                    help="also include the EN ultrarare extension datasets")
    ap.add_argument("--include-dk2", action="store_true",
                    help="also include the EN opaque-word (DK2) enrichment datasets")
    ap.add_argument("--dk2-only", action="store_true",
                    help="fit on the DK2 datasets alone (pooled over models)")
    ap.add_argument("--langs", default="en,zh",
                    help="comma list of languages to pool (per-language "
                         "decomposition of the confirmatory estimate)")
    ap.add_argument("--cand-covar", action="store_true",
                    help="add candidate-anchoring covariates: first-unit "
                         "match and edit-distance similarity between the "
                         "model's rejected greedy candidate and the target")
    args = ap.parse_args()
    langs = tuple(l.strip() for l in args.langs.split(",") if l.strip())
    items_by_lang = {
        lang: load_jsonl(os.path.join(ROOT, "data", f"items_full_{lang}_final.jsonl"))
        for lang in ("en", "zh")}
    ap_dk2_only = getattr(args, "dk2_only", False)
    if ap_dk2_only:
        dk2_items = load_jsonl(os.path.join(ROOT, "data",
                                            "items_dk2_en_final.jsonl"))
        datasets = [("en", m, f"dk2_en_{m}", dk2_items) for m in MODELS]
        langs = ("en",)
        rows = []
        _skip_default = True
    else:
        _skip_default = False
    if not _skip_default:
        datasets = [(lang, model, f"full_{lang}_{model}", None)
                    for lang, model in product(langs, MODELS)]
    if args.include_ext and "en" in langs:
        ext_items = load_jsonl(os.path.join(ROOT, "data",
                                            "items_ext_en_final.jsonl"))
        datasets += [("en", m, f"ext_en_{m}", ext_items) for m in MODELS]
    if args.include_dk2 and "en" in langs:
        dk2_items = load_jsonl(os.path.join(ROOT, "data",
                                            "items_dk2_en_final.jsonl"))
        datasets += [("en", m, f"dk2_en_{m}", dk2_items) for m in MODELS]
    rows = []
    for lang, model, dirname, items_override in datasets:
        base = os.path.join(ROOT, "results", dirname)
        prod = load_jsonl(os.path.join(base, "production.jsonl"))
        rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
        epis = load_jsonl(os.path.join(base, "partial_episode.jsonl"))
        tok_col = "n_tokens_qwen" if "qwen" in model else "n_tokens_llama"
        len_col = "n_letters" if lang == "en" else "n_chars"
        items = items_override if items_override is not None else items_by_lang[lang]
        for iid, it in items.items():
            if args.unique_only and not it.get("unique"):
                continue
            if args.lenient:
                st = classify_lenient(prod.get(iid), rec.get(iid))
            else:
                st = classify(prod.get(iid), rec.get(iid))
            if st not in ("TOT", "DK"):
                continue
            ep = epis.get(iid)
            if not ep or ep.get("group") != "failed_greedy":
                continue
            import math
            fpm = it.get("freq_per_million")
            if lang == "zh" and fpm and fpm > 0 and "log_freq" in it:
                zipf = math.log10(fpm) + 3.0
            else:
                zipf = it.get("zipf")
            cand_match = cand_sim = None
            if args.cand_covar:
                cand = extract_candidate(prod[iid].get("greedy", ""), lang)
                tgt = it["word"][0].lower() if lang == "en" else it["word"][0]
                word = it["word"].lower() if lang == "en" else it["word"]
                if cand:
                    c = cand.lower() if lang == "en" else cand
                    cand_match = 1.0 if c[:1] == tgt else 0.0
                    cand_sim = _lev_sim(c, word)
            for aligned, chmap in ALIGN.items():
                ch = chmap[lang]
                sc = score_partial(ch, ep.get(ch, {}).get("greedy"), it, lang, "target")
                if sc is None:
                    continue
                y = float(sc >= 0.5) if isinstance(sc, float) else float(sc)
                rows.append({"aligned": aligned, "y": y,
                             "is_TOT": 1.0 if st == "TOT" else 0.0,
                             "zipf": zipf, "length": it.get(len_col),
                             "tokens": it.get(tok_col), "model": model,
                             "lang": lang, "item": f"{lang}_{iid}",
                             "cand_match": cand_match, "cand_sim": cand_sim})
    df = pd.DataFrame(rows)
    if not args.cand_covar:
        df = df.drop(columns=["cand_match", "cand_sim"])
    df = df.dropna()
    print(f"pooled rows: {len(df)} (langs={langs}, cand_covar={args.cand_covar})")

    for aligned in ALIGN:
        sub = df[df.aligned == aligned].copy()
        if not len(sub):
            continue
        X = pd.DataFrame({"const": 1.0, "is_TOT": sub["is_TOT"],
                          "zipf": zscore(sub["zipf"]),
                          "length": zscore(sub["length"]),
                          "tokens": zscore(sub["tokens"])})
        if args.cand_covar:
            X["cand_match"] = sub["cand_match"]
            X["cand_sim"] = zscore(sub["cand_sim"])
        for m in MODELS[1:]:
            X[f"m_{m}"] = (sub["model"] == m).astype(float)
        if len(langs) > 1:
            X["lang_zh"] = (sub["lang"] == "zh").astype(float)
        m = sm.GLM(sub["y"], X, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": sub["item"]})
        print(f"\n== {aligned}: n={len(sub)} "
              f"(TOT={int(sub.is_TOT.sum())}, DK={int((1 - sub.is_TOT).sum())})")
        print(f"   b_is_TOT = {m.params['is_TOT']:+.3f}  se={m.bse['is_TOT']:.3f}  "
              f"p={m.pvalues['is_TOT']:.5f}")
        if args.cand_covar:
            for k in ("cand_match", "cand_sim"):
                print(f"   {k}: b={m.params[k]:+.3f} p={m.pvalues[k]:.5f}")
        # per-dataset raw differences + sign test
        diffs = []
        for lang, model in product(langs, MODELS):
            s2 = sub[(sub.lang == lang) & (sub.model == model)]
            if not len(s2):
                continue
            d = s2[s2.is_TOT == 1]["y"].mean() - s2[s2.is_TOT == 0]["y"].mean()
            diffs.append((f"{lang}/{model}", round(d, 3)))
        pos = sum(1 for _, d in diffs if d > 0)
        from scipy.stats import binomtest
        p_sign = binomtest(pos, len(diffs), 0.5, alternative="greater").pvalue
        print(f"   per-dataset TOT-DK diffs: {diffs}")
        print(f"   sign test: {pos}/{len(diffs)} positive, p={p_sign:.4f}")


if __name__ == "__main__":
    main()
