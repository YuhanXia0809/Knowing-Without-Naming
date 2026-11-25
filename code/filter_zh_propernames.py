#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter proper names out of the existing Chinese item pool.

items_zh_pool_clean.jsonl (built by build_items_zh.py + clean_zh_pool.py) draws
its glosses from moedict dict-revised.json, which includes biographical
(person-name) and geographical (place-name) dictionary entries alongside
ordinary content words -- e.g. 崔浩 (a Northern Wei historian) or 布鲁塞尔
(Brussels). These are not answerable "guess the word from its definition"
items in the intended sense (the gloss just recites biographical/geographical
facts), so they are excluded here.

Two independent exclusion rules (an item is dropped if EITHER fires):

1. Gloss-marker rule. moedict tags biographical/geographical/bibliographic
   entries with a short category marker at the very start of the gloss, e.g.
   "人名。（西元...", "地名。位于...", "城市名。...", "职官名。...", "书名。...".
   We check only the first few characters of the gloss for these markers
   (not a search over the whole gloss) because some of the marker strings
   are common substrings of ordinary text elsewhere in a gloss -- e.g.
   the marker "人名" also occurs inside the ordinary phrase "他人名誉"
   (another person's reputation), which can appear well past the start of
   an unrelated gloss; that must NOT be treated as a "人名" marker. A
   prefix window avoids this.

   In addition to the marker rule, biographical glosses have a recognizable
   internal shape wherever they occur in the text: "字某某，" (courtesy name),
   "号某某，" (art name), or a "(西元...年)"/"(公元...年)" birth-death-year
   parenthetical. These are searched for anywhere in the gloss.

2. Transliterated-personal-name rule. If en_equiv looks like a capitalized
   English proper noun (e.g. "Washington", "Homer") AND the Chinese word is
   built from characters drawn from the standard personal-name transliteration
   syllable set, the item is a transliterated foreign name. Both conditions
   are required together -- capitalization alone would also flag ordinary
   loanwords whose CEDICT gloss happens to be capitalized for other reasons,
   and the transliteration-character condition alone would kill legitimate
   loanwords such as 沙发 (sofa) that use the same syllables.

Usage:
    python -u filter_zh_propernames.py \
        --pool ../data/items_zh_pool_clean.jsonl \
        --out ../rebuild/items_zh_pool_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re

# ---------------------------------------------------------------------------
# Rule 1: gloss-marker + biographical-shape patterns
# ---------------------------------------------------------------------------
CATEGORY_MARKERS = (
    "人名", "地名", "县名", "市名", "山名", "河川", "水名", "朝代", "国名",
    "书名", "官名", "姓氏", "复姓", "年号", "谥号", "庙号", "人物", "作者",
)
CATEGORY_MARKER_RE = re.compile("|".join(CATEGORY_MARKERS))
MARKER_PREFIX_WINDOW = 6  # chars of gloss checked for a leading category marker

BIO_SHAPE_RE = re.compile(
    r"字[一-龥]{1,2}，|号[一-龥]{1,2}，|(?:西元|公元).{0,15}年"
)


def has_proper_name_gloss(gloss: str) -> bool:
    prefix = gloss[:MARKER_PREFIX_WINDOW]
    if CATEGORY_MARKER_RE.search(prefix):
        return True
    if BIO_SHAPE_RE.search(gloss):
        return True
    return False


# ---------------------------------------------------------------------------
# Rule 2: transliterated proper-noun en_equiv + translit-syllable characters
# ---------------------------------------------------------------------------
TRANSLIT_CHARS = set("阿布德弗格赫杰克勒姆奈普奎瑞斯特维沃克斯泽尔伦娜莉丽尼库图")
CAP_PROPER_RE = re.compile(r"^[A-Z][a-z]")


def is_translit_name(word: str, en_equiv: str | None) -> bool:
    if not en_equiv:
        return False
    if not CAP_PROPER_RE.match(en_equiv):
        return False
    return any(ch in TRANSLIT_CHARS for ch in word)


def is_proper_name(item: dict) -> bool:
    if has_proper_name_gloss(item.get("gloss", "")):
        return True
    if is_translit_name(item.get("word", ""), item.get("en_equiv")):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    kept, dropped = [], []
    n_in = 0
    n_drop_gloss = 0
    n_drop_translit = 0
    with open(args.pool, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_in += 1
            it = json.loads(line)
            gloss_hit = has_proper_name_gloss(it.get("gloss", ""))
            translit_hit = is_translit_name(it.get("word", ""), it.get("en_equiv"))
            if gloss_hit or translit_hit:
                if gloss_hit:
                    n_drop_gloss += 1
                elif translit_hit:
                    n_drop_translit += 1
                dropped.append(it)
            else:
                kept.append(it)

    with open(args.out, "w", encoding="utf-8") as f:
        for it in kept:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"in={n_in} kept={len(kept)} dropped={len(dropped)}")
    print(f"  dropped by gloss-marker/bio-shape rule: {n_drop_gloss}")
    print(f"  dropped by translit-name rule (en_equiv + syllables), gloss rule didn't already catch: {n_drop_translit}")
    print(f"Wrote {len(kept)} items to {args.out}")

    random.seed(42)
    print("\n--- 10 random DROPPED spot-check ---")
    for it in random.sample(dropped, min(10, len(dropped))):
        print(f"  {it['word']!r:12s} en={it.get('en_equiv')!r:30s} gloss={it['gloss'][:60]}")
    print("\n--- 10 random KEPT spot-check ---")
    for it in random.sample(kept, min(10, len(kept))):
        print(f"  {it['word']!r:12s} en={it.get('en_equiv')!r:30s} gloss={it['gloss'][:60]}")


if __name__ == "__main__":
    main()
