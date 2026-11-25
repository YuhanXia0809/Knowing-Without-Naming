"""Annotate an item file with per-model tokenizer token counts of the target word.

Token count is a key confound covariate (multi-token low-frequency words may fail
for tokenization reasons); it goes into every norms regression.

  python -u add_token_counts.py --items ../data/items_full_en_final.jsonl \
      --out ../rebuild/items_full_en_final.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",       # same tokenizer as 72B
    "llama": "meta-llama/Llama-3.1-8B-Instruct",  # same tokenizer as 70B
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    toks = {k: AutoTokenizer.from_pretrained(v) for k, v in MODELS.items()}

    out_tmp = args.out + ".tmp"
    n = 0
    with open(args.items, encoding="utf-8") as f, \
         open(out_tmp, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            it = json.loads(line)
            w = it["word"]
            is_en = w[0].isascii()
            for k, tk in toks.items():
                # English: leading space = how the word appears mid-text; Chinese: no space
                form = (" " + w) if is_en else w
                it[f"n_tokens_{k}"] = len(tk.encode(form, add_special_tokens=False))
                it[f"n_tokens_{k}_nospace"] = len(tk.encode(w, add_special_tokens=False))
            g.write(json.dumps(it, ensure_ascii=False) + "\n")
            n += 1
    os.replace(out_tmp, args.out)
    print(f"annotated {n} items -> {args.out}")


if __name__ == "__main__":
    main()
