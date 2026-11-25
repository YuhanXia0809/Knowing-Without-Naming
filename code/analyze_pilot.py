"""Pilot analysis: state taxonomy, partial-access contrasts (fresh + in-episode),
feature ceilings, FOK calibration, self-report overlap.

  python -u analyze_pilot.py --items ../data/items_full_en_final.jsonl \
      --results ../results/full_en_qwen7b --lang en
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

from make_pilot_items import edit_distance_le
from scoring import normalize_en, normalize_zh

ZH_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
          "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


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


def classify(prod, rec):
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


def parse_int_any(s, lo=0, hi=100):
    """Arabic digit or Chinese numeral."""
    if not s:
        return None
    m = re.search(r"\d+", s)
    if m:
        v = int(m.group())
        return v if lo <= v <= hi else None
    for ch, v in ZH_NUM.items():
        if ch in s:
            return v
    return None


def answers_of(item):
    return [a for a in item.get("answers", [item["word"]]) if a]


def contains_answer(resp, item, lang):
    """Did the model surface an acceptable answer inside a probe response?"""
    if not resp:
        return False
    if lang == "zh":
        r = normalize_zh(resp)
        return any(a in r for a in answers_of(item))
    r = normalize_en(resp)
    toks = set(r.split())
    return any(normalize_en(a) in toks or normalize_en(a) == r for a in answers_of(item))


def score_partial(ch, g, item, lang, vs="target"):
    """Score one probe response. vs='target' scores against the target word;
    vs='any' scores against the best-matching acceptable answer."""
    if g is None:
        return None
    g = g.strip()
    words = [item["word"]] if vs == "target" else answers_of(item)
    if lang == "en":
        if ch == "first_letter":
            m = re.search(r"[a-zA-Z]", g)
            return (m.group().lower() in {w[0].lower() for w in words}) if m else None
        if ch == "n_letters":
            v = parse_int_any(g)
            lens = {len(w) for w in words} if vs == "any" else {item["n_letters"]}
            return (v in lens) if v is not None else None
        if ch == "n_syllables":
            v = parse_int_any(g)
            return (v == item["n_syllables"]) if v is not None else None  # target only
        if ch == "sounds_like":
            cand = normalize_en(g)
            if not cand:
                return None
            best = min(edit_distance_le(cand, w, 10) / max(len(cand), len(w))
                       for w in words)
            return 1.0 - best
    else:
        if ch == "first_char":
            c = g[:1]
            return (c in {w[0] for w in words}) if c else None
        if ch == "n_chars":
            v = parse_int_any(g)
            lens = {len(w) for w in words} if vs == "any" else {item["n_chars"]}
            return (v in lens) if v is not None else None
        if ch == "pinyin_initial":
            cand = re.sub(r"[^a-zü]", "", g.lower())
            return (cand == item["initial"]) if cand else None  # target only
        if ch == "tone1":
            v = parse_int_any(g, 1, 5)
            return (v == item["tone1"]) if v is not None else None  # target only
    return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def auc(pairs):
    pairs = [(s, y) for s, y in pairs if s is not None]
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def fmt(v):
    return f"{v:>9.3f}" if isinstance(v, float) else f"{v:>9}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--lang", choices=["en", "zh"], required=True)
    args = ap.parse_args()
    lang = args.lang

    items = load_jsonl(args.items)
    S = lambda name: load_jsonl(os.path.join(args.results, name + ".jsonl"))
    prod, fok, part, rec = S("production"), S("fok"), S("partial"), S("recognition")
    epis, feat, selfr = S("partial_episode"), S("feature_ceiling"), S("selfreport")
    print(f"items={len(items)} prod={len(prod)} fok={len(fok)} partial={len(part)} "
          f"episode={len(epis)} featceil={len(feat)} rec={len(rec)} selfrep={len(selfr)}")

    states = {iid: classify(prod.get(iid), rec.get(iid)) for iid in items}
    print("\n=== State taxonomy ===")
    for s, n in Counter(states.values()).most_common():
        print(f"  {s}: {n}")
    sf = mean([1.0 if (prod[i]["k_prod"] == 0 and not prod[i]["greedy_correct"]) else 0.0
               for i in items if i in prod])
    print(f"  strict production-fail rate: {sf:.1%}")

    channels = (["first_char", "n_chars", "pinyin_initial", "tone1"] if lang == "zh"
                else ["first_letter", "n_letters", "n_syllables", "sounds_like"])
    groups = ["known", "TOT", "DK"]

    if part:
        print("\n=== FRESH-CONTEXT partial access (greedy; target-scored | any-answer-scored) ===")
        print(f"  {'channel':16s}" + "".join(f"{g + sfx:>14s}" for g in groups for sfx in ("", "*")))
        for ch in channels:
            row = f"  {ch:16s}"
            for g in groups:
                for vs in ("target", "any"):
                    vals = [score_partial(ch, part.get(i, {}).get(ch, {}).get("greedy"),
                                          items[i], lang, vs)
                            for i in items if states[i] == g and i in part]
                    row += fmt(mean(vals)) + ("     " if vs == "target" else "")
            print(row)
        print("\n  resolved-during-probe rate (any probe response contains an answer):")
        for g in groups:
            ids = [i for i in items if states[i] == g and i in part]
            res = [any(contains_answer(part[i].get(ch, {}).get("greedy"), items[i], lang)
                       for ch in channels) for i in ids]
            print(f"    {g}: {mean([1.0 if r else 0.0 for r in res]):.1%} (n={len(ids)})")

    if epis:
        print("\n=== IN-EPISODE partial access (referent pinned; target | any-answer) ===")
        print(f"  {'channel':16s}" + "".join(f"{g + sfx:>14s}" for g in groups for sfx in ("", "*")))
        for ch in channels:
            row = f"  {ch:16s}"
            for g in groups:
                for vs in ("target", "any"):
                    if g == "known":
                        # ceiling_retrieval: score vs the model's own produced word
                        vals = []
                        for i in items:
                            if states[i] != "known" or i not in epis:
                                continue
                            if epis[i].get("group") != "ceiling_retrieval":
                                continue
                            vals.append(score_partial(
                                ch, epis[i].get(ch, {}).get("greedy"), items[i], lang, "any"))
                    else:
                        vals = [score_partial(ch, epis[i].get(ch, {}).get("greedy"),
                                              items[i], lang, vs)
                                for i in items if states[i] == g and i in epis
                                and epis[i].get("group") == "failed_greedy"]
                    row += fmt(mean(vals)) + ("     " if vs == "target" else "")
            print(row)
        print("  resolved-during-episode-probe:")
        for g in ("TOT", "DK"):
            ids = [i for i in items if states[i] == g and i in epis
                   and epis[i].get("group") == "failed_greedy"]
            res = [any(contains_answer(epis[i].get(ch, {}).get("greedy"), items[i], lang)
                       for ch in channels) for i in ids]
            print(f"    {g}: {mean([1.0 if r else 0.0 for r in res]):.1%} (n={len(ids)})")

    if feat:
        print("\n=== FEATURE CEILING (word visible; channel competence) ===")
        for ch in channels:
            vals = [score_partial(ch, feat[i].get(ch, {}).get("greedy"), items[i], lang,
                                  "target") for i in items if i in feat]
            print(f"  {ch:16s}{fmt(mean(vals))}   (n={sum(1 for v in vals if v is not None)})")

    print("\n=== FOK calibration (production-fail items: predict recognition) ===")
    fail_ids = [i for i in items if states[i] in ("TOT", "DK")]
    for measure in ("fok_verbal", "fok_p_yes"):
        pairs = [(fok.get(i, {}).get(measure), 1 if states[i] == "TOT" else 0)
                 for i in fail_ids]
        a = auc(pairs)
        print(f"  {measure}: AUC={a if a is None else round(a, 3)}  (n={len(fail_ids)})")
    cons_pairs = []
    for i in fail_ids:
        cands = [c for c in prod.get(i, {}).get("candidates", []) if c]
        cons = (Counter(cands).most_common(1)[0][1] / len(cands)) if cands else None
        cons_pairs.append((cons, 1 if states[i] == "TOT" else 0))
    a = auc(cons_pairs)
    print(f"  sample-consistency: AUC={a if a is None else round(a, 3)}")

    print("\n=== Self-report vs behavioral state ===")
    if selfr:
        sr_tot = {i for i in items if selfr.get(i, {}).get("label") == "TOT"}
        bh_tot = {i for i in items if states[i] == "TOT"}
        print(f"  labels: {Counter(selfr[i]['label'] for i in items if i in selfr)}")
        inter, union = len(sr_tot & bh_tot), len(sr_tot | bh_tot)
        print(f"  behavioral TOT={len(bh_tot)}, self-report TOT={len(sr_tot)}, "
              f"Jaccard={inter / union if union else float('nan'):.3f}")

    print("\n=== Recognition sanity ===")
    kr = [rec[i]["k_rec"] for i in items if i in rec and rec[i].get("n_rec") == 4]
    print(f"  k_rec distribution: {Counter(kr)}")
    known_rec = [1.0 if rec[i]["k_rec"] >= 3 else 0.0 for i in items
                 if states[i] == "known" and i in rec and rec[i].get("n_rec") == 4]
    print(f"  recognition on KNOWN items (ceiling): {mean(known_rec):.1%}")


if __name__ == "__main__":
    main()
