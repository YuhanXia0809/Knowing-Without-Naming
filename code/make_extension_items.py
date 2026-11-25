"""DK-enrichment extension set: ultrarare-only EN items not already in the full run.

  python -u make_extension_items.py --pool ../data/items_en_hard_pool.jsonl \
      --exclude ../data/items_full_en.jsonl --out ../data/items_ext_en.jsonl --n 500
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--exclude", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()

    used = set()
    with open(args.exclude, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                used.add(json.loads(line)["word"])

    with open(args.pool, encoding="utf-8") as f:
        pool = [json.loads(l) for l in f if l.strip()]
    cands = [it for it in pool
             if it["freq_stratum"] == "ultrarare" and it["word"] not in used]
    print(f"ultrarare unused: {len(cands)}")
    rng = random.Random(args.seed)
    rng.shuffle(cands)
    sel = cands[: args.n]

    # temp pool file restricted to selected words + full pool as distractor source
    # -> easiest: write selected as its own pool and call make_pilot_items with
    #    per-stratum = n (single stratum present)
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
