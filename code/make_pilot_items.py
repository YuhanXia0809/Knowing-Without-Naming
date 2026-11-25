"""Assemble the pilot item file from the candidate pool: stratified sample + 4AFC distractors.

Distractors (SELF-[IN]CORRECT defense: experimenter-built, matched):
  d1 orthographic/phonological neighbor — pool word minimizing edit distance (<=3),
     preferring same length and close frequency
  d2 semantic associate — pool word with max content-word overlap between glosses
  d3 frequency+length matched random word

All distractors are real pool words, never in the item's synonym set, distinct.

  python -u make_pilot_items.py --pool ../data/items_en_pool.jsonl \
      --out ../rebuild/items_en.jsonl --per-stratum 50 --lang en --seed 7
"""
from __future__ import annotations

import argparse
import json
import random
import re

STOP = set("a an the of or and to in for on with that which is are was were be being "
           "been by as at from it its this these those someone something who whose "
           "used use having has have not no other any some more most very".split())


def edit_distance_le(a: str, b: str, k: int = 3) -> int:
    """Levenshtein distance, early-exit if > k (returns k+1)."""
    if abs(len(a) - len(b)) > k:
        return k + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        lo = k + 1
        for j, cb in enumerate(b, 1):
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            cur.append(v)
            lo = min(lo, v)
        if lo > k:
            return k + 1
        prev = cur
    return prev[-1]


def morph_leak(word: str, gloss: str) -> bool:
    """True if any gloss token shares a >=5-char prefix (or >=4-char containment)
    with the target — derivational leakage like 'abstention'/'abstaining'."""
    w = word.lower()
    for t in re.findall(r"[a-z]+", gloss.lower()):
        if len(t) < 4:
            continue
        p = 0
        for a, b in zip(w, t):
            if a != b:
                break
            p += 1
        if (p >= 5 or (p >= 4 and min(len(w), len(t)) >= 7)
                or (len(t) >= 4 and (t in w or w in t))):
            return True
    return False


def content_tokens(gloss: str, lang: str) -> set:
    if lang == "zh":
        return set(gloss) - set("的了在是有和与或及等一个把被对从")
    toks = re.findall(r"[a-z]+", gloss.lower())
    return {t for t in toks if t not in STOP and len(t) > 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-stratum", type=int, default=50)
    ap.add_argument("--lang", choices=["en", "zh"], required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    with open(args.pool, encoding="utf-8") as f:
        pool = [json.loads(l) for l in f if l.strip()]
    print(f"pool: {len(pool)}")
    if args.lang == "en":
        n0 = len(pool)
        pool = [it for it in pool if not morph_leak(it["word"], it["gloss"])]
        print(f"morphological-leak filter: {n0} -> {len(pool)}")

    freq_key = "zipf" if args.lang == "en" else (
        "log_freq" if "log_freq" in json.loads(open(args.pool, encoding="utf-8")
                                               .readline()) else "zipf")
    for it in pool:
        it["_toks"] = content_tokens(it["gloss"], args.lang)

    # -------- stratified sample --------
    if args.lang == "en":
        strata = {}
        for it in pool:
            s = it.get("freq_stratum")
            if s:
                strata.setdefault(s, []).append(it)
    else:
        vals = sorted(it[freq_key] for it in pool)
        t1, t2 = vals[len(vals) // 3], vals[2 * len(vals) // 3]
        strata = {"low": [], "midlow": [], "mid": []}
        for it in pool:
            s = ("low" if it[freq_key] <= t1 else
                 "midlow" if it[freq_key] <= t2 else "mid")
            strata[s].append(it)

    chosen = []
    for name, bucket in strata.items():
        # prefer less polysemous items
        key_n = "n_senses_wn" if args.lang == "en" else "n_defs"
        bucket = sorted(bucket, key=lambda x: (x.get(key_n, 1), rng.random()))
        take = bucket[: args.per_stratum * 3]
        rng.shuffle(take)
        sel = take[: args.per_stratum]
        for it in sel:
            it["freq_stratum"] = name
        chosen.extend(sel)
        print(f"stratum {name}: {len(sel)} of {len(bucket)}")

    # -------- distractors --------
    by_len = {}
    for it in pool:
        by_len.setdefault(len(it["word"]), []).append(it)

    def synset(it):
        return {it["word"]} | set(it.get("wn_synonyms", []) or [])

    out = []
    for it in chosen:
        w, fr = it["word"], it[freq_key]
        taboo = synset(it)
        # d1: nearest orthographic neighbor
        cands = []
        for L in range(max(1, len(w) - 2), len(w) + 3):
            for c in by_len.get(L, []):
                if c["word"] in taboo or c["word"] == w:
                    continue
                d = edit_distance_le(w, c["word"], 3)
                if d <= 3:
                    cands.append((d, abs(c[freq_key] - fr), c["word"]))
        cands.sort()
        d1 = cands[0][2] if cands else None
        # d2: semantic associate by gloss overlap
        best, best_ov = None, 0
        for c in pool:
            if c["word"] in taboo or c["word"] == w or c["word"] == d1:
                continue
            ov = len(it["_toks"] & c["_toks"])
            if ov > best_ov or (ov == best_ov and best is not None and ov > 0
                                and abs(c[freq_key] - fr) < 0.1):
                best, best_ov = c["word"], ov
        d2 = best if best_ov >= 1 else None
        # d3: frequency+length matched random
        pool3 = [c["word"] for c in pool
                 if c["word"] not in taboo and c["word"] not in (w, d1, d2)
                 and abs(c[freq_key] - fr) < 0.35 and abs(len(c["word"]) - len(w)) <= 3]
        d3 = rng.choice(pool3) if pool3 else None
        ds = [d for d in (d1, d2, d3) if d]
        # backfill if any slot failed
        while len(ds) < 3:
            extra = rng.choice(pool)["word"]
            if extra not in taboo and extra != w and extra not in ds:
                ds.append(extra)
        rec = {k: v for k, v in it.items() if not k.startswith("_")}
        rec.update({"item_id": f"{args.lang}_{w}", "target_option": w,
                    "distractors": ds[:3],
                    "answers": sorted(synset(it))})
        out.append(rec)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in sorted(out, key=lambda x: x["item_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} pilot items -> {args.out}")


if __name__ == "__main__":
    main()
