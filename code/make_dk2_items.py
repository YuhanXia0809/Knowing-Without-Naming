"""Second DK-enrichment extension -- morphologically OPAQUE rare
words (the item class Limitations names: defeats recognition without
defeating the gloss), to grow English DK cells.

Opacity filter: alphabetic single tokens, length 5-12, and either no
common derivational suffix or a suffix-stripped stem that is not
itself a common word (stem Zipf < 1.7) -- i.e., the form is not
compositionally recoverable from familiar pieces.

  python make_dk2_items.py --pool ../data/items_en_hard_pool.jsonl \
      --exclude ../data/items_full_en.jsonl \
      --exclude2 ../data/items_ext_en.jsonl \
      --out ../data/items_dk2_en.jsonl --n 500
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys

from wordfreq import zipf_frequency

SUFFIXES = ["ations", "ation", "sions", "sion", "tion", "ness", "ments",
            "ment", "ities", "ity", "isms", "ism", "ists", "ist", "ings",
            "ing", "ers", "er", "ors", "or", "ous", "ives", "ive", "als",
            "al", "ics", "ic", "ly", "ful", "less", "ables", "able",
            "ibles", "ible", "ances", "ance", "ences", "ence", "aries",
            "ary", "eries", "ery", "ologies", "ology", "ographies",
            "ography", "es", "s"]


def opaque(word):
    w = word.lower()
    if not w.isalpha() or not (5 <= len(w) <= 12):
        return False
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            stem = w[: -len(suf)]
            for variant in (stem, stem + "e"):
                if zipf_frequency(variant, "en") >= 1.7:
                    return False
            break
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--exclude", required=True)
    ap.add_argument("--exclude2", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=47)
    args = ap.parse_args()

    used = set()
    for path in filter(None, [args.exclude, args.exclude2]):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    used.add(json.loads(line)["word"])

    with open(args.pool, encoding="utf-8") as f:
        pool = [json.loads(l) for l in f if l.strip()]
    cands = [it for it in pool
             if it["word"] not in used and opaque(it["word"])]
    print(f"opaque unused candidates: {len(cands)}")
    rng = random.Random(args.seed)
    rng.shuffle(cands)
    sel = cands[: args.n]
    print("sample:", [it["word"] for it in sel[:12]])

    tmp = args.out + ".pool"
    with open(tmp, "w", encoding="utf-8") as g:
        for it in sel:
            g.write(json.dumps(it, ensure_ascii=False) + "\n")
    r = subprocess.run([sys.executable, "make_pilot_items.py", "--pool", tmp,
                        "--out", args.out, "--per-stratum", str(args.n),
                        "--lang", "en", "--seed", str(args.seed)])
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
