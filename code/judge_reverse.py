"""Judge the reversal-control stage: is each model-produced definition of the
target word correct? (word -> description direction; MPC5 reducibility defense).

For every result dir given, reads reverse.jsonl and writes reverse_judged.jsonl
adding {"def_correct": bool}. Run with the 72B judge on 2 GPUs.

  python -u judge_reverse.py --items ../data/items_full_en_final.jsonl --lang en \
      --model Qwen/Qwen2.5-72B-Instruct --tp 2 \
      --dirs ../results/full_en_qwen7b,../results/full_en_llama8b
"""
from __future__ import annotations

import argparse
import json
import os

JUDGE_EN = (
    'The word is "{word}". Its dictionary definition is: "{gloss}"\n\n'
    'Someone described this word as: "{definition}"\n\n'
    "Is that description a correct account of what the word means (same sense)? "
    "Minor omissions are fine; it must not describe a different meaning or a "
    "different word. Answer YES or NO only."
)
JUDGE_ZH = (
    '词语是"{word}"。它的词典释义是：“{gloss}”\n\n'
    '有人把这个词解释为：“{definition}”\n\n'
    "这个解释是否正确地说明了该词的意思（同一义项）？可以有小的遗漏，"
    "但不能解释成别的意思或别的词。只回答「是」或「否」。"
)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--lang", choices=["en", "zh"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out-name", default="reverse_judged.jsonl")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    items = load_jsonl(args.items)
    tmpl = JUDGE_ZH if args.lang == "zh" else JUDGE_EN
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=4096,
              gpu_memory_utilization=0.90)
    tok = AutoTokenizer.from_pretrained(args.model)

    def chat(u):
        return tok.apply_chat_template([{"role": "user", "content": u}],
                                       tokenize=False, add_generation_prompt=True)

    for d in args.dirs.split(","):
        d = d.strip()
        rev = load_jsonl(os.path.join(d, "reverse.jsonl"))
        out_path = os.path.join(d, args.out_name)
        done = set(load_jsonl(out_path))
        todo = [(iid, r) for iid, r in rev.items()
                if iid in items and iid not in done]
        print(f"{d}: {len(todo)} to judge ({len(done)} resumed)")
        if not todo:
            continue
        ps = [chat(tmpl.format(word=items[iid]["word"],
                               gloss=items[iid].get("gloss_original",
                                                    items[iid]["gloss"]),
                               definition=r["definition"][:400]))
              for iid, r in todo]
        outs = llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=4))
        with open(out_path, "a", encoding="utf-8") as g:
            for (iid, r), o in zip(todo, outs):
                v = o.outputs[0].text.strip().upper()
                ok = v.startswith("YES") or v.startswith("是")
                g.write(json.dumps({"item_id": iid, "def_correct": ok,
                                    "definition": r["definition"]},
                                   ensure_ascii=False) + "\n")
        print(f"{d}: judged")
    print("REVERSE_JUDGE_DONE")


if __name__ == "__main__":
    main()
