"""Item-level (not just ratio-level) check of the cross-family
answer-key rebuild.

On the stratified rebuild samples (299 En / 300 Zh) every item is RELABELLED
under the Llama-3.1-70B key (production re-scored, recognition unchanged), and
the two flagship item-level contrasts are re-run on the relabelled states:

  (a) in-episode first-unit partial access, TOT vs DK  (the flagship contrast)
  (b) cue-rescue: correct-first-unit-cue resolution rate, TOT vs DK

Each is reported three ways so the sample-restriction effect and the
relabelling effect are separable:
    FULL      all items, original key            (published estimate)
    SAMPLE-0  rebuild-sample items, original key (restriction only)
    SAMPLE-1  rebuild-sample items, Llama key    (restriction + relabelling)

  python -u key_relabel.py            # both languages
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import pandas as pd
import statsmodels.api as sm

from analyze_pilot import classify, load_jsonl, score_partial
from paths import (CHANNELS, CONFIRMATORY, INITIAL_CH, classify_key,
                        fmt_prop, items_path, prop_test, rdir, rejudge_path,
                        rescore, tok_col, zipf_of, zscore)


def load_cue(base):
    """cue.jsonl -> {item_id: {condition: record}}"""
    out = defaultdict(dict)
    p = os.path.join(base, "cue.jsonl")
    if not os.path.exists(p):
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["item_id"]][r["condition"]] = r
    return out


def pooled_glm(rows, label, extra=()):
    df = pd.DataFrame(rows).dropna()
    if len(df) < 40 or df["y"].nunique() < 2 or df["is_TOT"].nunique() < 2:
        print(f"    [{label}] insufficient data (n={len(df)})")
        return
    X = pd.DataFrame({"const": 1.0, "is_TOT": df["is_TOT"],
                      "zipf": zscore(df["zipf"]), "length": zscore(df["length"]),
                      "tokens": zscore(df["tokens"])})
    for k in extra:
        X[k] = df[k]
    for m in sorted(df["model"].unique())[1:]:
        X[f"m_{m}"] = (df["model"] == m).astype(float)
    # a zero cell in the TOT/DK x success table separates the data: the
    # maximum-likelihood b_TOT is infinite and its SE is not estimable
    cells = {(s, v): int(((df["is_TOT"] == s) & (df["y"] == v)).sum())
             for s in (0, 1) for v in (0, 1)}
    if min(cells.values()) == 0:
        n_t = int(df.is_TOT.sum())
        print(f"    [{label}] n={len(df)} (TOT={n_t}, DK={len(df) - n_t}) "
              f"successes TOT={cells[(1, 1)]}/{n_t} "
              f"DK={cells[(0, 1)]}/{len(df) - n_t} | b_TOT not estimable "
              f"(separation: zero cell)")
        return
    try:
        fit = sm.GLM(df["y"], X, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": df["item"]})
    except Exception as e:
        print(f"    [{label}] fit error: {str(e)[:80]}")
        return
    ci = fit.conf_int()
    n_t = int(df.is_TOT.sum())
    raw_t = df[df.is_TOT == 1]["y"].mean()
    raw_d = df[df.is_TOT == 0]["y"].mean()
    print(f"    [{label}] n={len(df)} (TOT={n_t}, DK={len(df) - n_t}) "
          f"raw TOT={raw_t:.3f} DK={raw_d:.3f} diff={raw_t - raw_d:+.3f} | "
          f"b_TOT={fit.params['is_TOT']:+.3f} se={fit.bse['is_TOT']:.3f} "
          f"CI[{ci.loc['is_TOT', 0]:+.3f},{ci.loc['is_TOT', 1]:+.3f}] "
          f"p={fit.pvalues['is_TOT']:.5f}")


def run_lang(lang):
    print(f"\n{'=' * 78}\n=== {lang.upper()} ===\n{'=' * 78}")
    items = load_jsonl(items_path(lang))
    rej = load_jsonl(rejudge_path(lang))
    shared = [i for i in rej if i in items]
    print(f"rebuild sample: {len(shared)} items (of {len(items)})")

    ich = INITIAL_CH[lang]
    len_col = "n_letters" if lang == "en" else "n_chars"

    # accumulators for pooled fits
    pool = {k: [] for k in ("full", "s0", "s1")}
    cue_pool = {k: [] for k in ("full", "s0", "s1")}

    print("\n-- state counts on the rebuild sample --")
    for tag in CONFIRMATORY:
        base = rdir(lang, tag)
        prod = load_jsonl(os.path.join(base, "production.jsonl"))
        rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
        # sanity: does re-scoring under the ORIGINAL key reproduce stored k_prod?
        mism = 0
        for iid in shared:
            p = prod.get(iid)
            if not p:
                continue
            g0, k0 = rescore(p, items[iid].get("answers", []), lang)
            if k0 != p["k_prod"] or g0 != p["greedy_correct"]:
                mism += 1
        c0 = Counter()
        c1 = Counter()
        for iid in shared:
            p = prod.get(iid)
            if not p:
                continue
            c0[classify(p, rec.get(iid))] += 1
            c1[classify_key(p, rec.get(iid), rej[iid].get("answers", []), lang)] += 1
        r0 = c0["TOT"] / max(1, c0["DK"])
        r1 = c1["TOT"] / max(1, c1["DK"])
        print(f"  {tag}: orig TOT={c0['TOT']} DK={c0['DK']} ({r0:.1f}:1) | "
              f"llama-key TOT={c1['TOT']} DK={c1['DK']} ({r1:.1f}:1) | "
              f"re-score/stored mismatches={mism}")

    # ---------------- (a) first-unit in-episode contrast ----------------
    print(f"\n-- (a) in-episode FIRST-UNIT partial access ({ich}) --")
    for tag in CONFIRMATORY:
        base = rdir(lang, tag)
        prod = load_jsonl(os.path.join(base, "production.jsonl"))
        rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
        epis = load_jsonl(os.path.join(base, "partial_episode.jsonl"))
        tc = tok_col(tag)
        cnt = {k: {"TOT": [0, 0], "DK": [0, 0]} for k in ("full", "s0", "s1")}
        for iid, it in items.items():
            p = prod.get(iid)
            if not p:
                continue
            ep = epis.get(iid)
            if not ep or ep.get("group") != "failed_greedy":
                continue
            sc = score_partial(ich, ep.get(ich, {}).get("greedy"), it, lang, "target")
            if sc is None:
                continue
            y = float(sc >= 0.5) if isinstance(sc, float) else float(sc)
            st0 = classify(p, rec.get(iid))
            variants = [("full", st0)]
            if iid in rej:
                variants.append(("s0", st0))
                variants.append(("s1", classify_key(
                    p, rec.get(iid), rej[iid].get("answers", []), lang)))
            for key, st in variants:
                if st not in ("TOT", "DK"):
                    continue
                cnt[key][st][0] += y
                cnt[key][st][1] += 1
                pool[key].append({"y": y, "is_TOT": 1.0 if st == "TOT" else 0.0,
                                  "zipf": zipf_of(it, lang),
                                  "length": it.get(len_col), "tokens": it.get(tc),
                                  "model": tag, "item": iid})
        for key, name in (("full", "FULL    "), ("s0", "SAMPLE-0"),
                          ("s1", "SAMPLE-1")):
            t = cnt[key]["TOT"]
            d = cnt[key]["DK"]
            st = prop_test(int(t[0]), t[1], int(d[0]), d[1])
            print(f"  {tag:9s} {name}: {fmt_prop(st, 'TOT', 'DK')}")
    print("  pooled (model FE, item-clustered SE):")
    for key, name in (("full", "FULL"), ("s0", "SAMPLE-0"), ("s1", "SAMPLE-1")):
        pooled_glm(pool[key], name)

    # ---------------- (b) cue-rescue contrast ----------------
    print("\n-- (b) cue-rescue: correct-first-unit-cue resolution --")
    print("   (NB resolution is scored under the ORIGINAL key: cue.jsonl stores "
          "only k_cue/greedy_correct, no raw text; only the STATE is relabelled)")
    for tag in CONFIRMATORY:
        base = rdir(lang, tag)
        prod = load_jsonl(os.path.join(base, "production.jsonl"))
        rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
        cues = load_cue(base)
        tc = tok_col(tag)
        if not cues:
            print(f"  {tag}: no cue.jsonl")
            continue
        cnt = {k: {"TOT": [0, 0], "DK": [0, 0]} for k in ("full", "s0", "s1")}
        for iid, conds in cues.items():
            it = items.get(iid)
            p = prod.get(iid)
            if not it or not p:
                continue
            r = conds.get("correct")
            if not r:
                continue
            y = 1.0 if (r["greedy_correct"] or r["k_cue"] > 0) else 0.0
            st0 = classify(p, rec.get(iid))
            variants = [("full", st0)]
            if iid in rej:
                variants.append(("s0", st0))
                variants.append(("s1", classify_key(
                    p, rec.get(iid), rej[iid].get("answers", []), lang)))
            for key, st in variants:
                if st not in ("TOT", "DK"):
                    continue
                cnt[key][st][0] += y
                cnt[key][st][1] += 1
                cue_pool[key].append({"y": y, "is_TOT": 1.0 if st == "TOT" else 0.0,
                                      "zipf": zipf_of(it, lang),
                                      "length": it.get(len_col),
                                      "tokens": it.get(tc),
                                      "model": tag, "item": iid})
        for key, name in (("full", "FULL    "), ("s0", "SAMPLE-0"),
                          ("s1", "SAMPLE-1")):
            t = cnt[key]["TOT"]
            d = cnt[key]["DK"]
            st = prop_test(int(t[0]), t[1], int(d[0]), d[1])
            print(f"  {tag:9s} {name}: {fmt_prop(st, 'TOT', 'DK')}")
    print("  pooled (model FE, item-clustered SE):")
    for key, name in (("full", "FULL"), ("s0", "SAMPLE-0"), ("s1", "SAMPLE-1")):
        pooled_glm(cue_pool[key], name)

    # ---------------- other channels, raw only ----------------
    print("\n-- (a-extra) remaining in-episode channels, raw TOT-DK diffs --")
    for ch in CHANNELS[lang]:
        if ch == ich:
            continue
        for tag in CONFIRMATORY:
            base = rdir(lang, tag)
            prod = load_jsonl(os.path.join(base, "production.jsonl"))
            rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
            epis = load_jsonl(os.path.join(base, "partial_episode.jsonl"))
            acc = {k: {"TOT": [0, 0], "DK": [0, 0]} for k in ("s0", "s1")}
            for iid in rej:
                it = items.get(iid)
                p = prod.get(iid)
                ep = epis.get(iid)
                if not it or not p or not ep or ep.get("group") != "failed_greedy":
                    continue
                sc = score_partial(ch, ep.get(ch, {}).get("greedy"), it, lang, "target")
                if sc is None:
                    continue
                y = float(sc >= 0.5) if isinstance(sc, float) else float(sc)
                for key, st in (("s0", classify(p, rec.get(iid))),
                                ("s1", classify_key(p, rec.get(iid),
                                                    rej[iid].get("answers", []), lang))):
                    if st in ("TOT", "DK"):
                        acc[key][st][0] += y
                        acc[key][st][1] += 1
            def r(a):
                return (f"{a['TOT'][0] / a['TOT'][1]:.3f}(n={a['TOT'][1]})"
                        if a["TOT"][1] else "na")
            def rd(a):
                return (f"{a['DK'][0] / a['DK'][1]:.3f}(n={a['DK'][1]})"
                        if a["DK"][1] else "na")
            print(f"  {ch:15s} {tag:9s} S0 TOT={r(acc['s0'])} DK={rd(acc['s0'])} | "
                  f"S1 TOT={r(acc['s1'])} DK={rd(acc['s1'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="en,zh")
    args = ap.parse_args()
    for lang in args.langs.split(","):
        run_lang(lang.strip())


if __name__ == "__main__":
    main()
