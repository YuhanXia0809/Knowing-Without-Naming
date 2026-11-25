"""Clean the Chinese item pool: strip gloss segments that leak the answer.

moedict glosses often contain metalinguistic material that names the
English source of a loanword (a Latin-script word followed by the marker
音译, transliteration) or other Latin-script hints. Split each
gloss into segments on 。；！？, drop any segment containing Latin letters or
loan/translation metalanguage, and rejoin. Drop the item if the cleaned gloss
is shorter than 6 characters.

  python -u clean_zh_pool.py --pool ../data/items_zh_pool.jsonl \
      --out ../data/items_zh_pool_clean.jsonl
"""
from __future__ import annotations

import argparse
import json
import re

META = re.compile(r"[A-Za-z]|音译|义译|意译|译音|译名|译自|外来语|外来词|简称为|缩写")
SPLIT = re.compile(r"(?<=[。；！？])")


def clean_gloss(gloss: str) -> str:
    segs = [s for s in SPLIT.split(gloss) if s.strip()]
    kept = [s for s in segs if not META.search(s)]
    return "".join(kept).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n_in, n_out, n_modified, n_dropped = 0, 0, 0, 0
    with open(args.pool, encoding="utf-8") as f, \
         open(args.out, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            n_in += 1
            it = json.loads(line)
            cleaned = clean_gloss(it["gloss"])
            if cleaned != it["gloss"]:
                n_modified += 1
            # word-character containment could re-arise only by segment removal? no,
            # removal only deletes text; original pool already enforced containment.
            if len(cleaned) < 6:
                n_dropped += 1
                continue
            it["gloss_raw"] = it["gloss"]
            it["gloss"] = cleaned
            g.write(json.dumps(it, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"in={n_in} out={n_out} modified={n_modified} dropped_too_short={n_dropped}")


if __name__ == "__main__":
    main()
