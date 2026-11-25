"""Rebuild 4AFC distractor sets so ALL THREE distractors are maximally confusable
semantic neighbors of the target (the pilot's d1=orthographic/d2=semantic/d3=random
design uses only one semantic associate; this is the adversarial upper bound on
recognition difficulty).

For each item, candidates are drawn from the union of the language's two pool files
(the same pools make_pilot_items.py draws d1/d2/d3 from) and ranked by an IDF-weighted
content-token overlap between the candidate's gloss and the target's gloss:

    score(c) = sum_{t in toks(target) & toks(c)} idf(t)

  idf(t) = log(N / (1 + df(t))) + 1     (N = pool size, df = pool glosses containing t)

Top-3 by score become the new distractors. A candidate is excluded if:
  - it equals the target word, or is in the item's `answers` set (target + WN +
    judge-vetted synonyms — judge_synonyms is already folded into `answers` by
    merge_judged.py, so checking `answers` covers it);
  - its normalized form equals a morphological variant of any answer
    (scoring.en_variants for EN; exact normalized match for ZH);
  - it is within edit distance 1 of any answer (fast lexical guard — must differ by
    edit distance >= 2, via make_pilot_items.edit_distance_le);
  - (EN only) it shares a >=5-char prefix with the target word, or trips
    make_pilot_items.morph_leak against the target's gloss (derivational leakage
    like 'abstention'/'abstaining').
ZH gets none of the EN-specific prefix/morph_leak filtering — sharing characters
with the target is exactly what makes a distractor a strong semantic neighbor in
Chinese (compounds share morphemes), so only the answers/variant/edit-distance
guards apply.

Items where fewer than 3 candidates have score > 0 (i.e. we had to backfill with a
zero-overlap candidate to fill the 4AFC slot) are reported as low-confusability.

  python -u make_adversarial_items.py --items ../data/items_full_en_final.jsonl \
      --pools ../data/items_en_pool.jsonl,../data/items_en_hard_pool.jsonl \
      --out ../data/items_adv_en.jsonl --lang en

  python -u make_adversarial_items.py --items ../data/items_full_zh_final.jsonl \
      --pools ../data/items_zh_pool_v2.jsonl,../data/items_zh_hard_pool.jsonl \
      --out ../data/items_adv_zh.jsonl --lang zh
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_pilot_items import content_tokens, edit_distance_le, morph_leak
from scoring import en_variants, normalize_en, normalize_zh


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def norm(w, lang):
    return normalize_en(w) if lang == "en" else normalize_zh(w)


def toks_of(gloss, lang):
    """content_tokens(), with a ZH-specific fix: make_pilot_items.content_tokens
    for zh is `set(gloss) - {stopword chars}`, which lets punctuation (notably the
    sentence-final '。', present in nearly every gloss) through as a "content"
    character. That's harmless for its original use (picking a single d2 semantic
    associate, diluted by the d1/d3 slots) but here it's a real source of noise
    since ALL three slots are ranked by this overlap score. Restrict to actual CJK
    ideographs; leave the EN path and the shared helper itself untouched."""
    t = content_tokens(gloss, lang)
    if lang == "zh":
        t = {c for c in t if "一" <= c <= "鿿"}
    return t


def build_pool(pool_paths, lang):
    """Union of pool files, de-duped by word (first occurrence wins)."""
    pool, seen = [], set()
    for p in pool_paths:
        for it in load_jsonl(p):
            w = it["word"]
            if w in seen:
                continue
            seen.add(w)
            pool.append(it)
    for it in pool:
        it["_toks"] = toks_of(it["gloss"], lang)
    return pool


def build_idf(pool):
    df = defaultdict(int)
    for it in pool:
        for t in it["_toks"]:
            df[t] += 1
    n = len(pool)
    return {t: math.log(n / (1 + c)) + 1.0 for t, c in df.items()}


def build_index(pool):
    """token -> set of pool indices whose gloss contains it, for fast candidate
    lookup (only pool words sharing >=1 content token with the target can score
    > 0, so we never need to touch the full pool per item)."""
    idx = defaultdict(set)
    for i, it in enumerate(pool):
        for t in it["_toks"]:
            idx[t].add(i)
    return idx


def prefix5(a, b):
    p = 0
    for x, y in zip(a.lower(), b.lower()):
        if x != y:
            break
        p += 1
    return p >= 5


def forbidden_forms(answers, lang):
    """Normalized forms (incl. morphological variants) that a distractor must not
    collide with — this is the `answers`/judge_synonyms/variant-equality guard."""
    out = set()
    for a in answers:
        if lang == "en":
            out |= en_variants(a)
        else:
            out.add(normalize_zh(a))
    return out


def too_close(cand_word, answers, forbidden, lang):
    cand_norm = norm(cand_word, lang)
    if cand_norm in forbidden:
        return True
    for a in answers:
        a_norm = norm(a, lang)
        if edit_distance_le(cand_norm, a_norm, 1) <= 1:
            return True
    return False


def pick_distractors(item, pool, idx, idf, lang):
    target = item["word"]
    answers = list(dict.fromkeys(item.get("answers", [target]) or [target]))
    forbidden = forbidden_forms(answers, lang)
    target_toks = toks_of(item["gloss"], lang)
    gloss = item["gloss"]

    def eval_cand(c):
        cw = c["word"]
        if cw == target or cw in answers:
            return None
        if too_close(cw, answers, forbidden, lang):
            return None
        if lang == "en":
            if prefix5(cw, target) or morph_leak(cw, gloss):
                return None
        ov = target_toks & c["_toks"]
        return sum(idf.get(t, 1.0) for t in ov)

    # fast path: only candidates sharing >=1 content token with the target can
    # score > 0, so gather them via the inverted index instead of the full pool
    cand_idxs = set()
    for t in target_toks:
        cand_idxs |= idx.get(t, set())
    scored = []
    for i in cand_idxs:
        s = eval_cand(pool[i])
        if s is not None:
            scored.append((s, pool[i]["word"]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    n_confusable = sum(1 for s, _ in scored[:3] if s > 0)

    if len(scored) >= 3:
        return [w for _, w in scored[:3]], n_confusable, len(scored)

    # rare fallback: not enough token-overlap candidates at all (n_confusable will
    # be < 3 too) -- scan the full pool once to backfill the 4AFC slot
    have = {w for _, w in scored}
    for c in pool:
        if c["word"] in have:
            continue
        s = eval_cand(c)
        if s is not None:
            scored.append((s, c["word"]))
            have.add(c["word"])
    scored.sort(key=lambda x: (-x[0], x[1]))
    n_confusable = sum(1 for s, _ in scored[:3] if s > 0)
    return [w for _, w in scored[:3]], n_confusable, len(scored)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--pools", required=True, help="comma-separated pool files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", choices=["en", "zh"], required=True)
    args = ap.parse_args()

    pool_paths = [p.strip() for p in args.pools.split(",") if p.strip()]
    pool = build_pool(pool_paths, args.lang)
    idf = build_idf(pool)
    idx = build_index(pool)
    print(f"pool (union, deduped): {len(pool)} words from {pool_paths}")

    items = load_jsonl(args.items)
    print(f"items: {len(items)}")

    out = []
    low_confusability = []  # item_ids where <3 candidates had score > 0
    starved = []            # item_ids where fewer than 3 candidates survived at all
    for it in items:
        ds, n_conf, n_cand = pick_distractors(it, pool, idx, idf, args.lang)
        if n_cand < 3:
            starved.append((it["item_id"], n_cand))
        elif n_conf < 3:
            low_confusability.append((it["item_id"], n_conf))
        rec = dict(it)
        rec["distractors"] = ds
        rec["distractors_orig"] = it.get("distractors")
        out.append(rec)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} adversarial items -> {args.out}")
    print(f"low-confusability (found <3 candidates with overlap>0): "
          f"{len(low_confusability)}")
    for iid, n in low_confusability[:20]:
        print(f"  {iid}: only {n}/3 confusable")
    print(f"starved (fewer than 3 legal candidates at all, incl. score=0 backfill): "
          f"{len(starved)}")
    for iid, n in starved[:20]:
        print(f"  {iid}: only {n} legal candidates total")


if __name__ == "__main__":
    main()
