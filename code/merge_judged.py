"""Merge judge_screen output back into an item file.

- answers <- judge-vetted set (target + WN synonyms + judge synonyms)
- gloss   <- paraphrase when valid (original kept as gloss_original)
- distractors colliding with the enlarged answers set are dropped and reported

  python -u merge_judged.py --items ../data/items_full_en.jsonl \
      --judged ../data/items_full_en_judged.jsonl \
      --out ../rebuild/items_full_en_final.jsonl [--use-paraphrase]
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--judged", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--use-paraphrase", action="store_true")
    args = ap.parse_args()

    judged = {}
    with open(args.judged, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                judged[r["word"]] = r

    n_out, n_para, n_dropped_d, n_nonunique = 0, 0, 0, 0
    with open(args.items, encoding="utf-8") as f, \
         open(args.out, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            it = json.loads(line)
            j = judged.get(it["word"])
            if not j:
                continue
            it["answers"] = sorted(set(it.get("answers", [])) | set(j["answers"]))
            it["unique"] = j["unique"]
            if not j["unique"]:
                n_nonunique += 1
            it["gloss_original"] = it["gloss"]
            if args.use_paraphrase and j.get("gloss_paraphrased"):
                it["gloss"] = j["gloss_paraphrased"]
                n_para += 1
            kept = [d for d in it["distractors"] if d not in it["answers"]]
            n_dropped_d += len(it["distractors"]) - len(kept)
            it["distractors"] = kept if len(kept) == 3 else None
            if it["distractors"] is None:
                del it["distractors"]  # recognition stage will skip it
            g.write(json.dumps(it, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"wrote {n_out} items; paraphrased {n_para}; "
          f"non-unique flagged {n_nonunique}; distractors dropped {n_dropped_d}")


if __name__ == "__main__":
    main()
