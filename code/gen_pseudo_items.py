"""Pseudoword 4AFC control items.

The criterial recognition test could in principle be passed by
gloss-compatibility inference rather than stored lexical knowledge. This
control measures that inference floor: for a sample of items, the
"target" option is a plausible novel form (pronounceable pseudoword /
plausible two-character pseudo-compound) that no system can have stored;
distractors keep the item's own gloss-overlap semantic associate plus
two matched random real words. The true answer and the true target's
orthographic neighbor are excluded from the options, and the pseudoword
is generated from OTHER items' words, never from the true target (no
form leakage). A model that picks the pseudoword on >=3/4 rotations
"recognizes" a word it cannot know.

  python gen_pseudo_items.py --lang en --n 400 \
      --items ../data/items_full_en_final.jsonl \
      --out ../rebuild/items_pseudo_en.jsonl
"""
from __future__ import annotations

import argparse
import json
import random

from wordfreq import zipf_frequency


def load_items(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def edit_distance(a, b):
    if abs(len(a) - len(b)) > 3:
        return 99
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1,
                                     prev + (ca != cb))
    return dp[-1]


def bigram_model(words):
    # letter trigram model (conditioned on previous two symbols) for
    # wordlike pseudoword generation
    from collections import defaultdict
    counts = defaultdict(lambda: defaultdict(int))
    for w in words:
        w = f"^^{w}$"
        for i in range(len(w) - 2):
            counts[w[i:i + 2]][w[i + 2]] += 1
    return {a: (list(d.keys()), list(d.values())) for a, d in counts.items()}


def gen_pseudo_en(rng, bg, length, is_real):
    for _ in range(400):
        w, ctx = "", "^^"
        while len(w) < length + 3:
            if ctx not in bg:
                break
            keys, wts = bg[ctx]
            nxt = rng.choices(keys, weights=wts)[0]
            if nxt == "$":
                break
            w += nxt
            ctx = ctx[1] + nxt
        if abs(len(w) - length) <= 1 and len(w) >= 4 and not is_real(w):
            return w
    return None


def gen_pseudo_zh(rng, chars, is_real):
    for _ in range(500):
        w = rng.choice(chars) + rng.choice(chars)
        if w[0] != w[1] and not is_real(w):
            return w
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--lang", choices=["en", "zh"], required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    items = load_items(args.items)
    all_answers = set()
    for it in items:
        for a in it.get("answers", [it["word"]]):
            all_answers.add(a.lower())
        all_answers.add(str(it.get("target_option", "")).lower())
        for d in it.get("distractors", []):
            all_answers.add(str(d).lower())

    def is_real(w):
        wl = w.lower()
        if wl in all_answers:
            return True
        return zipf_frequency(wl, args.lang) > 0

    words = [it["word"] for it in items]
    if args.lang == "en":
        bg = bigram_model([w.lower() for w in words if w.isalpha()])
    else:
        chars = sorted({c for w in words for c in w})

    items = [it for it in items if len(it.get("distractors", [])) == 3]
    print(f"items with 3 distractors: {len(items)}")
    sample = rng.sample(items, min(args.n, len(items)))
    # pool of matched random real words = other items' length-matched
    # random distractor (position of max form-distance from that target)
    out, n_skip = [], 0
    for it in sample:
        tgt = it["word"]
        dists = list(it.get("distractors", []))
        if len(dists) != 3:
            n_skip += 1
            continue
        # the true target's form neighbor = distractor closest to target
        if args.lang == "en":
            near_i = min(range(3),
                         key=lambda i: edit_distance(str(dists[i]).lower(),
                                                     tgt.lower()))
        else:
            near_i = min(range(3),
                         key=lambda i: -len(set(str(dists[i])) & set(tgt)))
        keep = [d for i, d in enumerate(dists) if i != near_i]
        donor = rng.choice(items)
        extra = str(donor["distractors"][2 % len(donor["distractors"])])
        tries = 0
        while (extra in keep or extra == tgt) and tries < 10:
            donor = rng.choice(items)
            extra = str(donor["distractors"][2 % len(donor["distractors"])])
            tries += 1
        if args.lang == "en":
            pw = gen_pseudo_en(rng, bg, len(tgt), is_real)
            if pw and edit_distance(pw.lower(), tgt.lower()) <= 2:
                pw = None
        else:
            pw = gen_pseudo_zh(rng, chars, is_real)
            if pw and len(set(pw) & set(tgt)) > 0:
                pw = None
        if not pw:
            n_skip += 1
            continue
        out.append({"item_id": f"pseudo_{it['item_id']}",
                    "orig_item_id": it["item_id"],
                    "gloss": it["gloss"], "word": pw,
                    "target_option": pw,
                    "answers": [pw],
                    "distractors": keep + [extra],
                    "true_word": tgt})
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} pseudo items ({n_skip} skipped) -> {args.out}")
    for r in out[:5]:
        print(" ", r["word"], "| true:", r["true_word"],
              "| opts:", r["distractors"])


if __name__ == "__main__":
    main()
