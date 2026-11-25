"""Judge-model screening of candidate items (run with a strong model, e.g. Qwen2.5-72B TP=2).

For every candidate item this produces:
  1. acceptable answers: words that fit the gloss as well as the target
     (union with WordNet synonyms; used by production scoring)
  2. uniqueness verdict: whether the gloss pins down the target well enough
  3. paraphrased gloss: same meaning, different surface form, never containing
     the target (contamination defense — surface form != any indexed dictionary)

Output: items file enriched with answers / unique / gloss_paraphrased.
Items failing uniqueness or paraphrase checks are kept but flagged (filter later).

  python -u judge_screen.py --pool ../data/items_en_pool.jsonl --lang en \
      --model Qwen/Qwen2.5-72B-Instruct --tp 2 --out ../data/items_en_judged.jsonl \
      [--max-items 300]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import normalize_en, normalize_zh

JUDGE_SYNONYMS_EN = (
    'Here is a dictionary definition and the word it defines.\n\n'
    'Definition: "{gloss}"\nWord: "{word}"\n\n'
    "List other English words that are TRUE SYNONYMS of this word in this sense — "
    "words for which a dictionary could use this same definition. Do NOT list merely "
    "related words, examples, category members, or broader/narrower terms. "
    "Reply with a comma-separated list of words only, or NONE if there is no true synonym."
)
VERIFY_SYN_EN = (
    'Definition: "{gloss}"\n\n'
    'Could a dictionary use this definition as a definition of the word "{syn}"? '
    "Answer YES or NO only."
)
VERIFY_SYN_ZH = (
    '释义：“{gloss}”\n\n'
    '一部词典能否用这条释义作为词语"{syn}"的释义？只回答「能」或「否」。'
)
JUDGE_UNIQUE_EN = (
    'Here is a dictionary definition and the word it defines.\n\n'
    'Definition: "{gloss}"\nWord: "{word}"\n\n'
    "Question: does this definition identify this word clearly enough that a person "
    "with a large vocabulary could name it from the definition alone? "
    "Answer YES or NO only."
)
PARAPHRASE_EN = (
    'Rewrite this dictionary definition in your own words, keeping the meaning '
    'precise. Requirements: do NOT use the word "{word}" or any form of it; '
    'do not make it easier or harder; one sentence; similar length.\n\n'
    'Definition: "{gloss}"\n\nRewritten definition only:'
)
JUDGE_SYNONYMS_ZH = (
    '下面是一个词典释义和它定义的词。\n\n释义：“{gloss}”\n词：“{word}”\n\n'
    "列出这个词在此义项下的真同义词——即词典可以用同一条释义来解释的其他中文词语。"
    "不要列出仅仅相关的词、示例、上位词或下位词。"
    "只回答一个用逗号分隔的词语列表；如果没有真同义词，回答「无」。"
)
JUDGE_UNIQUE_ZH = (
    '下面是一个词典释义和它定义的词。\n\n释义：“{gloss}”\n词：“{word}”\n\n'
    "问题：这个释义是否足够清楚地界定了这个词，使得词汇量大的人仅凭释义就能"
    "说出它？只回答「是」或「否」。"
)
PARAPHRASE_ZH = (
    '用你自己的话改写这个词典释义，保持意思精确。要求：不得使用"{word}"这个词'
    "或它的任何一个字；不要让它变得更容易或更难猜；一句话；长度相近。\n\n"
    '释义：“{gloss}”\n\n只输出改写后的释义：'
)


def parse_list(text, lang):
    t = text.strip()
    if not t or t.upper().startswith("NONE") or t.startswith("无"):
        return []
    t = re.sub(r"^[^:：]*[:：]", "", t, count=1) if ("：" in t[:15] or ":" in t[:15]) else t
    parts = re.split(r"[,，、;；\n]+", t)
    norm = normalize_zh if lang == "zh" else normalize_en
    out = []
    for p in parts[:15]:
        w = norm(p)
        if w and len(w) <= 30 and " " not in w:
            out.append(w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--lang", choices=["en", "zh"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-items", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    with open(args.pool, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    if args.max_items:
        items = items[: args.max_items]

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["word"])
                except Exception:
                    pass
    items = [it for it in items if it["word"] not in done]
    print(f"{len(items)} items to judge ({len(done)} resumed)")
    if not items:
        return

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=4096,
              gpu_memory_utilization=float(os.environ.get("VLLM_GPU_UTIL",
                                                          "0.90")))
    tok = AutoTokenizer.from_pretrained(args.model)

    def chat(u):
        return tok.apply_chat_template([{"role": "user", "content": u}],
                                       tokenize=False, add_generation_prompt=True)

    syn_t = JUDGE_SYNONYMS_ZH if args.lang == "zh" else JUDGE_SYNONYMS_EN
    unq_t = JUDGE_UNIQUE_ZH if args.lang == "zh" else JUDGE_UNIQUE_EN
    par_t = PARAPHRASE_ZH if args.lang == "zh" else PARAPHRASE_EN
    norm = normalize_zh if args.lang == "zh" else normalize_en

    fout = open(args.out, "a", encoding="utf-8")
    CHUNK = 200
    for c0 in range(0, len(items), CHUNK):
        chunk = items[c0:c0 + CHUNK]
        ps_syn = [chat(syn_t.format(gloss=it["gloss"], word=it["word"])) for it in chunk]
        ps_unq = [chat(unq_t.format(gloss=it["gloss"], word=it["word"])) for it in chunk]
        ps_par = [chat(par_t.format(gloss=it["gloss"], word=it["word"])) for it in chunk]
        o_syn = llm.generate(ps_syn, SamplingParams(temperature=0.0, max_tokens=80))
        o_unq = llm.generate(ps_unq, SamplingParams(temperature=0.0, max_tokens=6))
        o_par = llm.generate(ps_par, SamplingParams(temperature=0.0, max_tokens=120))
        # verification pass: keep a proposed synonym only if the judge affirms the
        # gloss could serve as ITS definition (kills related-but-not-synonym words)
        ver_t = VERIFY_SYN_ZH if args.lang == "zh" else VERIFY_SYN_EN
        proposals = []
        for it, s in zip(chunk, o_syn):
            syns = parse_list(s.outputs[0].text, args.lang)
            tgt = norm(it["word"])
            proposals.append([w for w in syns if w != tgt])
        pair_ps, pair_idx = [], []
        for i, (it, syns) in enumerate(zip(chunk, proposals)):
            for w in syns:
                pair_ps.append(chat(ver_t.format(gloss=it["gloss"], syn=w)))
                pair_idx.append((i, w))
        verified = [[] for _ in chunk]
        if pair_ps:
            o_ver = llm.generate(pair_ps, SamplingParams(temperature=0.0, max_tokens=4))
            for (i, w), o in zip(pair_idx, o_ver):
                v = o.outputs[0].text.strip().upper()
                if v.startswith("YES") or v.startswith("能"):
                    verified[i].append(w)

        for it, ver_syns, u, p in zip(chunk, verified, o_unq, o_par):
            syns = ver_syns
            tgt = norm(it["word"])
            uniq_raw = u.outputs[0].text.strip().upper()
            unique = uniq_raw.startswith("YES") or uniq_raw.startswith("是")
            para = p.outputs[0].text.strip().strip('"“”')
            para_ok = bool(para) and tgt not in norm(para)
            if args.lang == "zh":
                para_ok = para_ok and not any(ch in para for ch in it["word"])
            answers = sorted(set([it["word"]] + it.get("wn_synonyms", []) + syns))
            rec = dict(it)
            rec.update({"answers": answers, "judge_synonyms": syns,
                        "unique": unique, "gloss_paraphrased": para if para_ok else None,
                        "paraphrase_ok": para_ok})
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"{min(c0 + CHUNK, len(items))}/{len(items)} judged", flush=True)
    fout.close()
    print("JUDGE SCREEN COMPLETE")


if __name__ == "__main__":
    main()
