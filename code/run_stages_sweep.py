"""Battery runner (vLLM offline).

Runs the staged battery over an item file. Every stage is fresh-context per item.
Results are written incrementally (chunked) as JSONL, one line per item per stage,
and runs are resumable: existing (stage, item_id) pairs are skipped.

Stages:
  production  greedy x1 + n=N_PROD samples @ T_PROD; scored against item's answers
  fok         verbalized 0-100 (greedy) + yes/no first-token logprob   [all items]
  partial     channel probes, greedy + n=N_PART samples @ 0.7          [all items]
  recognition 4AFC, 4 latin-square rotations, first-token logprob over A-D
  selfreport  answer-or-"TOT"-or-"DK" (greedy)
  perform     instructed TOT performance paragraph (greedy)
  cuerequest  agentive cue request + self-repair: turn 1 offers answer-or-
              request-one-hint on failed items (same pool as `cue`); turn 2
              delivers the requested hint and retries in the same dialogue
  circum      controlled circumlocution (fresh context; failed items + produced
              control sample), SPEAKER model, written into --out-dir/circum.jsonl
  listener    4AFC over the SAME options as recognition, single seeded rotation,
              scored by first-token logprob; run with the LISTENER model, reading
              --circum-dir/circum.jsonl and writing into --out-dir/listener.jsonl
              (point --out-dir at the SPEAKER's dir to keep listener.jsonl there)

Usage (on GPU node):
  python -u run_stages.py --items ../data/items_full_en_final.jsonl --lang en \
      --model Qwen/Qwen2.5-7B-Instruct --tp 1 --out-dir ../rebuild/full_en_qwen7b \
      --stages production,fok,partial,recognition,selfreport,perform
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts as P
from scoring import score_production, extract_candidate

N_PROD = 20
T_PROD = 1.0
N_PART = 5
CHUNK = 250
MAX_TOKENS_WORD = 16

EN_PARTIAL = {
    "first_letter": P.PARTIAL_FIRST_LETTER_EN,
    "n_letters": P.PARTIAL_N_LETTERS_EN,
    "n_syllables": P.PARTIAL_N_SYLLABLES_EN,
    "sounds_like": P.PARTIAL_SOUNDS_LIKE_EN,
}
ZH_PARTIAL = {
    "first_char": P.PARTIAL_FIRST_CHAR_ZH,
    "n_chars": P.PARTIAL_N_CHARS_ZH,
    "pinyin_initial": P.PARTIAL_PINYIN_INITIAL_ZH,
    "tone1": P.PARTIAL_TONE_ZH,
}
# Latin square: position of the correct answer across the 4 rotations, with the
# three distractors cycling through the remaining slots.
ROTATIONS = [
    (0, 1, 2, 3),
    (1, 2, 3, 0),
    (2, 3, 0, 1),
    (3, 0, 1, 2),
]
LETTERS = ["A", "B", "C", "D"]


def load_items(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def done_ids(path):
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.add(json.loads(line)["item_id"])
            except Exception:
                continue  # tolerate a half-written trailing line
    return out


class Runner:
    def __init__(self, model, tp, max_model_len=8192, gpu_mem_util=0.90):
        from vllm import LLM
        from transformers import AutoTokenizer
        self.llm = LLM(model=model, tensor_parallel_size=tp,
                       max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_mem_util,
                       trust_remote_code=True)
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    def chat_prompt(self, user, system=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)

    def generate(self, prompts_, **kw):
        from vllm import SamplingParams
        if getattr(self, "strip_think", False):
            kw["max_tokens"] = max(kw.get("max_tokens", 16),
                                   getattr(self, "think_budget", 768))
        sp = SamplingParams(**kw)
        outs = self.llm.generate(prompts_, sp)
        if getattr(self, "strip_think", False):
            outs = [_strip_think_out(o) for o in outs]
        # vLLM preserves input order
        return outs


class _O:
    __slots__ = ("text", "logprobs")

    def __init__(self, text, logprobs=None):
        self.text = text
        self.logprobs = logprobs


class _Out:
    __slots__ = ("outputs",)

    def __init__(self, outputs):
        self.outputs = outputs


def _strip_think_out(o):
    """Reasoning models emit <think>...</think> before the answer; keep only
    the post-think text for scoring (logprobs dropped — they describe the
    think prefix, not the answer)."""
    new = []
    for oo in o.outputs:
        t = oo.text
        if "</think>" in t:
            t = t.split("</think>", 1)[1]
        new.append(_O(t.strip(), None))
    return _Out(new)


def first_token_letter_logprobs(out):
    """Extract normalized P(A..D) from the first generated token's logprob table."""
    try:
        lps = out.outputs[0].logprobs[0]
    except (IndexError, TypeError):
        return None
    best = {}
    for _tid, lp in lps.items():
        t = lp.decoded_token.strip().strip(".:)").upper()
        if t in LETTERS:
            best[t] = max(best.get(t, -1e9), lp.logprob)
    if not best:
        return None
    mx = max(best.values())
    z = sum(math.exp(v - mx) for v in best.values())
    return {k: math.exp(v - mx) / z for k, v in best.items()}


def parse_int(text, lo=0, hi=100):
    m = re.search(r"-?\d+", text)
    if not m:
        return None
    v = int(m.group())
    return v if lo <= v <= hi else None


def run_stage(stage, items, runner, lang, out_path, args):
    done = done_ids(out_path)
    todo = [it for it in items if it["item_id"] not in done]
    if not todo:
        print(f"[{stage}] nothing to do ({len(done)} already done)")
        return
    print(f"[{stage}] {len(todo)} items to run ({len(done)} resumed)")
    f = open(out_path, "a", encoding="utf-8")

    def emit(rec):
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for c0 in range(0, len(todo), CHUNK):
        chunk = todo[c0:c0 + CHUNK]

        if stage == "production":
            sys_p = P.PROD_SYSTEM_ZH if lang == "zh" else P.PROD_SYSTEM_EN
            usr_t = P.PROD_USER_ZH if lang == "zh" else P.PROD_USER_EN
            ps = [runner.chat_prompt(usr_t.format(gloss=it["gloss"]), sys_p)
                  for it in chunk]
            greedy = runner.generate(ps, temperature=0.0, max_tokens=MAX_TOKENS_WORD)
            sampled = runner.generate(ps, temperature=args.temperature, top_p=args.top_p,
                                      n=N_PROD, max_tokens=MAX_TOKENS_WORD, seed=args.seed)
            for it, g, s in zip(chunk, greedy, sampled):
                g_text = g.outputs[0].text
                s_texts = [o.text for o in s.outputs]
                answers = it["answers"]
                g_ok = score_production(g_text, answers, lang)
                s_ok = [score_production(t, answers, lang) for t in s_texts]
                emit({"item_id": it["item_id"], "greedy": g_text.strip(),
                      "greedy_correct": g_ok,
                      "samples": [t.strip() for t in s_texts],
                      "samples_correct": s_ok, "k_prod": sum(s_ok), "n_prod": N_PROD,
                      "temperature": args.temperature, "top_p": args.top_p,
                      "candidates": [extract_candidate(t, lang) for t in s_texts]})

        elif stage == "fok":
            usr_v = P.FOK_USER_ZH if lang == "zh" else P.FOK_USER_EN
            usr_yn = P.FOK_YESNO_ZH if lang == "zh" else P.FOK_YESNO_EN
            ps_v = [runner.chat_prompt(usr_v.format(gloss=it["gloss"])) for it in chunk]
            ps_yn = [runner.chat_prompt(usr_yn.format(gloss=it["gloss"])) for it in chunk]
            out_v = runner.generate(ps_v, temperature=0.0, max_tokens=8)
            out_yn = runner.generate(ps_yn, temperature=0.0, max_tokens=4, logprobs=20)
            for it, ov, oy in zip(chunk, out_v, out_yn):
                fok_num = parse_int(ov.outputs[0].text)
                p_yes = None
                try:
                    lps = oy.outputs[0].logprobs[0]
                    yes_lp, no_lp = None, None
                    for _tid, lp in lps.items():
                        t = lp.decoded_token.strip().lower()
                        if t in ("yes", "能"):
                            yes_lp = max(yes_lp, lp.logprob) if yes_lp is not None else lp.logprob
                        if t in ("no", "不", "不能"):
                            no_lp = max(no_lp, lp.logprob) if no_lp is not None else lp.logprob
                    if yes_lp is not None and no_lp is not None:
                        p_yes = 1.0 / (1.0 + math.exp(no_lp - yes_lp))
                    elif yes_lp is not None:
                        p_yes = 1.0
                    elif no_lp is not None:
                        p_yes = 0.0
                except (IndexError, TypeError):
                    pass
                emit({"item_id": it["item_id"], "fok_verbal": fok_num,
                      "fok_verbal_raw": ov.outputs[0].text.strip(),
                      "fok_p_yes": p_yes, "fok_yn_raw": oy.outputs[0].text.strip()})

        elif stage == "ptrue":
            # P(True) baseline: token-level P(Yes) that the model's own greedy
            # production answer is correct (Kadavath-style self-evaluation of
            # the produced answer, distinct from prospective FOK).
            prod_path = os.path.join(args.out_dir, "production.jsonl")
            prods = {}
            with open(prod_path, encoding="utf-8") as pf:
                for line in pf:
                    if line.strip():
                        r = json.loads(line)
                        prods[r["item_id"]] = r
            usr_t = P.PTRUE_ZH if lang == "zh" else P.PTRUE_EN
            metas, ps = [], []
            for it in chunk:
                pr = prods.get(it["item_id"])
                if not pr or not pr.get("greedy", "").strip():
                    continue
                ans = pr["greedy"].strip()
                ps.append(runner.chat_prompt(usr_t.format(gloss=it["gloss"],
                                                          answer=ans)))
                metas.append((it["item_id"], ans))
            if ps:
                outs = runner.generate(ps, temperature=0.0, max_tokens=4,
                                       logprobs=20)
                for (iid, ans), o in zip(metas, outs):
                    p_yes = None
                    try:
                        lps = o.outputs[0].logprobs[0]
                        yes_lp, no_lp = None, None
                        for _tid, lp in lps.items():
                            t = lp.decoded_token.strip().lower()
                            if t in ("yes", "是"):
                                yes_lp = max(yes_lp, lp.logprob) if yes_lp is not None else lp.logprob
                            if t in ("no", "否", "不"):
                                no_lp = max(no_lp, lp.logprob) if no_lp is not None else lp.logprob
                        if yes_lp is not None and no_lp is not None:
                            p_yes = 1.0 / (1.0 + math.exp(no_lp - yes_lp))
                        elif yes_lp is not None:
                            p_yes = 1.0
                        elif no_lp is not None:
                            p_yes = 0.0
                    except (IndexError, TypeError):
                        pass
                    emit({"item_id": iid, "answer": ans, "p_true": p_yes,
                          "raw": o.outputs[0].text.strip()})

        elif stage == "fokstrong":
            # Stronger verbalized-confidence elicitation (full-range integer,
            # anti-round-number instruction).
            usr_t = P.FOKSTRONG_ZH if lang == "zh" else P.FOKSTRONG_EN
            ps = [runner.chat_prompt(usr_t.format(gloss=it["gloss"]))
                  for it in chunk]
            outs = runner.generate(ps, temperature=0.0, max_tokens=8)
            for it, o in zip(chunk, outs):
                emit({"item_id": it["item_id"],
                      "fokstrong_verbal": parse_int(o.outputs[0].text),
                      "raw": o.outputs[0].text.strip()})

        elif stage == "partial":
            channels = ZH_PARTIAL if lang == "zh" else EN_PARTIAL
            recs = {it["item_id"]: {"item_id": it["item_id"]} for it in chunk}
            for ch_name, tmpl in channels.items():
                ps = [runner.chat_prompt(tmpl.format(gloss=it["gloss"])) for it in chunk]
                greedy = runner.generate(ps, temperature=0.0, max_tokens=8)
                sampled = runner.generate(ps, temperature=0.7, n=N_PART,
                                          max_tokens=8, seed=args.seed)
                for it, g, s in zip(chunk, greedy, sampled):
                    recs[it["item_id"]][ch_name] = {
                        "greedy": g.outputs[0].text.strip(),
                        "samples": [o.text.strip() for o in s.outputs]}
            for r in recs.values():
                emit(r)

        elif stage == "recognition":
            usr_t = P.RECOG_USER_ZH if lang == "zh" else P.RECOG_USER_EN
            metas, ps = [], []
            for it in chunk:
                opts = [it["target_option"]] + it["distractors"]  # [target, d1, d2, d3]
                for rot_i, rot in enumerate(ROTATIONS):
                    arranged = [None] * 4
                    for src_idx, dst_pos in enumerate(rot):
                        arranged[dst_pos] = opts[src_idx]
                    ps.append(runner.chat_prompt(usr_t.format(
                        gloss=it["gloss"], opt_a=arranged[0], opt_b=arranged[1],
                        opt_c=arranged[2], opt_d=arranged[3])))
                    metas.append((it["item_id"], rot_i, rot[0]))
            outs = runner.generate(ps, temperature=0.0, max_tokens=4, logprobs=20)
            per_item = {}
            for (iid, rot_i, tgt_pos), o in zip(metas, outs):
                probs = first_token_letter_logprobs(o)
                gen = o.outputs[0].text.strip().upper()[:1]
                if probs:
                    choice = max(probs, key=probs.get)
                    p_target = probs.get(LETTERS[tgt_pos], 0.0)
                else:
                    choice, p_target = (gen if gen in LETTERS else None), None
                per_item.setdefault(iid, []).append({
                    "rotation": rot_i, "target_pos": LETTERS[tgt_pos],
                    "choice": choice, "gen": gen, "p_target": p_target,
                    "correct": choice == LETTERS[tgt_pos]})
            for it in chunk:
                rots = per_item.get(it["item_id"], [])
                emit({"item_id": it["item_id"], "rotations": rots,
                      "k_rec": sum(r["correct"] for r in rots), "n_rec": len(rots),
                      "mean_p_target": (sum(r["p_target"] for r in rots
                                            if r["p_target"] is not None)
                                        / max(1, sum(1 for r in rots
                                                     if r["p_target"] is not None)))})

        elif stage == "selfreport":
            usr_t = P.SELFREPORT_ZH if lang == "zh" else P.SELFREPORT_EN
            ps = [runner.chat_prompt(usr_t.format(gloss=it["gloss"])) for it in chunk]
            outs = runner.generate(ps, temperature=0.0, max_tokens=MAX_TOKENS_WORD)
            for it, o in zip(chunk, outs):
                txt = o.outputs[0].text.strip()
                up = txt.upper().strip("「」\"'. ")
                label = ("TOT" if "TOT" in up[:8] else
                         "DK" if up[:8].startswith("DK") else "ANSWER")
                emit({"item_id": it["item_id"], "raw": txt, "label": label,
                      "answer_correct": (score_production(txt, it["answers"], lang)
                                         if label == "ANSWER" else None)})

        elif stage in ("selfreport_ep", "selfreport_logit",
                       "selfreport_logit_ep"):
            # (a) selfreport_ep re-elicits the self-report
            # inside the failed production episode (referent pinned, like
            # partial_episode); (b) *_logit variants force a TOT/DK choice
            # and read token-level P("TOT") vs P("DK") off the logits.
            def totdk_logprobs(o):
                tot_lp = dk_lp = None
                try:
                    lps = o.outputs[0].logprobs[0]
                    for _tid, lp in lps.items():
                        t = lp.decoded_token.strip().upper()
                        if t in ("TOT", "T", "TO"):
                            tot_lp = max(tot_lp, lp.logprob) \
                                if tot_lp is not None else lp.logprob
                        if t in ("DK", "D"):
                            dk_lp = max(dk_lp, lp.logprob) \
                                if dk_lp is not None else lp.logprob
                except Exception:
                    pass
                if tot_lp is not None and dk_lp is not None:
                    p_tot = 1.0 / (1.0 + math.exp(dk_lp - tot_lp))
                elif tot_lp is not None:
                    p_tot = 1.0
                elif dk_lp is not None:
                    p_tot = 0.0
                else:
                    p_tot = None
                return p_tot, tot_lp, dk_lp

            if stage == "selfreport_logit":
                usr_t = P.SELFREPORT_FC_ZH if lang == "zh" else P.SELFREPORT_FC_EN
                ps = [runner.chat_prompt(usr_t.format(gloss=it["gloss"]))
                      for it in chunk]
                outs = runner.generate(ps, temperature=0.0, max_tokens=4,
                                       logprobs=20)
                for it, o in zip(chunk, outs):
                    p_tot, tot_lp, dk_lp = totdk_logprobs(o)
                    emit({"item_id": it["item_id"],
                          "gen": o.outputs[0].text.strip(),
                          "p_tot": p_tot, "tot_lp": tot_lp, "dk_lp": dk_lp})
            else:
                prod_path = os.path.join(args.out_dir, "production.jsonl")
                prod = {}
                with open(prod_path, encoding="utf-8") as pf:
                    for line in pf:
                        if line.strip():
                            r = json.loads(line)
                            prod[r["item_id"]] = r
                sys_p = P.PROD_SYSTEM_ZH if lang == "zh" else P.PROD_SYSTEM_EN
                usr_t = P.PROD_USER_ZH if lang == "zh" else P.PROD_USER_EN
                prefix_t = P.EPISODE_PREFIX_ZH if lang == "zh" \
                    else P.EPISODE_PREFIX_EN
                if stage == "selfreport_ep":
                    q = P.SELFREPORT_EP_Q_ZH if lang == "zh" \
                        else P.SELFREPORT_EP_Q_EN
                else:
                    q = P.SELFREPORT_FC_EP_Q_ZH if lang == "zh" \
                        else P.SELFREPORT_FC_EP_Q_EN
                usable = []
                for it in chunk:
                    pr = prod.get(it["item_id"])
                    if pr and not pr["greedy_correct"]:
                        usable.append((it, pr))
                ps = []
                for it, pr in usable:
                    follow = prefix_t.format(
                        greedy=pr["greedy"].strip()[:60]) + q
                    msgs = [{"role": "system", "content": sys_p},
                            {"role": "user",
                             "content": usr_t.format(gloss=it["gloss"])},
                            {"role": "assistant",
                             "content": pr["greedy"].strip()},
                            {"role": "user", "content": follow}]
                    ps.append(runner.tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True))
                if stage == "selfreport_ep":
                    outs = runner.generate(ps, temperature=0.0,
                                           max_tokens=MAX_TOKENS_WORD)
                    for (it, pr), o in zip(usable, outs):
                        txt = o.outputs[0].text.strip()
                        up = txt.upper().strip("「」\"'. ")
                        label = ("TOT" if "TOT" in up[:8] else
                                 "DK" if up[:8].startswith("DK") else "ANSWER")
                        emit({"item_id": it["item_id"], "raw": txt,
                              "label": label,
                              "answer_correct": (
                                  score_production(txt, it["answers"], lang)
                                  if label == "ANSWER" else None)})
                else:
                    outs = runner.generate(ps, temperature=0.0, max_tokens=4,
                                           logprobs=20)
                    for (it, pr), o in zip(usable, outs):
                        p_tot, tot_lp, dk_lp = totdk_logprobs(o)
                        emit({"item_id": it["item_id"],
                              "gen": o.outputs[0].text.strip(),
                              "p_tot": p_tot, "tot_lp": tot_lp,
                              "dk_lp": dk_lp})

        elif stage == "partial_episode":
            # requires production.jsonl in out_dir (runs after production stage)
            prod_path = os.path.join(args.out_dir, "production.jsonl")
            prod = {}
            with open(prod_path, encoding="utf-8") as pf:
                for line in pf:
                    if line.strip():
                        r = json.loads(line)
                        prod[r["item_id"]] = r
            sys_p = P.PROD_SYSTEM_ZH if lang == "zh" else P.PROD_SYSTEM_EN
            usr_t = P.PROD_USER_ZH if lang == "zh" else P.PROD_USER_EN
            qs = P.EPISODE_Q_ZH if lang == "zh" else P.EPISODE_Q_EN
            ceil_qs = P.EPISODE_CEILING_ZH if lang == "zh" else P.EPISODE_CEILING_EN
            prefix_t = P.EPISODE_PREFIX_ZH if lang == "zh" else P.EPISODE_PREFIX_EN
            recs = {}
            usable = [it for it in chunk if it["item_id"] in prod]
            for ch_name in qs:
                metas, ps = [], []
                for it in usable:
                    pr = prod[it["item_id"]]
                    greedy_ans = pr["greedy"].strip()[:60]
                    if pr["greedy_correct"]:
                        follow = ceil_qs[ch_name]
                        grp = "ceiling_retrieval"
                    else:
                        follow = prefix_t.format(greedy=greedy_ans) + qs[ch_name]
                        grp = "failed_greedy"
                    msgs = [{"role": "system", "content": sys_p},
                            {"role": "user", "content": usr_t.format(gloss=it["gloss"])},
                            {"role": "assistant", "content": pr["greedy"].strip()},
                            {"role": "user", "content": follow}]
                    ps.append(runner.tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True))
                    metas.append((it["item_id"], grp))
                greedy = runner.generate(ps, temperature=0.0, max_tokens=8)
                sampled = runner.generate(ps, temperature=0.7, n=N_PART,
                                          max_tokens=8, seed=args.seed)
                for (iid, grp), g, s in zip(metas, greedy, sampled):
                    recs.setdefault(iid, {"item_id": iid, "group": grp})
                    recs[iid][ch_name] = {
                        "greedy": g.outputs[0].text.strip(),
                        "samples": [o.text.strip() for o in s.outputs]}
            for r in recs.values():
                emit(r)

        elif stage == "feature_ceiling":
            qs = P.FEATURE_CEILING_ZH if lang == "zh" else P.FEATURE_CEILING_EN
            recs = {it["item_id"]: {"item_id": it["item_id"]} for it in chunk}
            for ch_name, tmpl in qs.items():
                ps = [runner.chat_prompt(tmpl.format(word=it["word"])) for it in chunk]
                greedy = runner.generate(ps, temperature=0.0, max_tokens=8)
                for it, g in zip(chunk, greedy):
                    recs[it["item_id"]][ch_name] = {"greedy": g.outputs[0].text.strip()}
            for r in recs.values():
                emit(r)

        elif stage == "rescue":
            # ZH-only: on production-failed items with an English equivalent, retry
            # production with the translation as a cue (Gollan translation priming).
            prod_path = os.path.join(args.out_dir, "production.jsonl")
            prod = {}
            with open(prod_path, encoding="utf-8") as pf:
                for line in pf:
                    if line.strip():
                        r = json.loads(line)
                        prod[r["item_id"]] = r
            usable = []
            for it in chunk:
                pr = prod.get(it["item_id"])
                en = (it.get("en_equiv") or "").split(";")[0].strip()
                if pr and not pr["greedy_correct"] and pr["k_prod"] <= 2 and en:
                    usable.append((it, en))
            if usable:
                ps = [runner.chat_prompt(
                    P.RESCUE_ZH_WITH_EN.format(gloss=it["gloss"], en_word=en),
                    P.PROD_SYSTEM_ZH) for it, en in usable]
                greedy = runner.generate(ps, temperature=0.0, max_tokens=MAX_TOKENS_WORD)
                sampled = runner.generate(ps, temperature=T_PROD, n=N_PROD,
                                          max_tokens=MAX_TOKENS_WORD, seed=args.seed)
                for (it, en), g, s in zip(usable, greedy, sampled):
                    s_ok = [score_production(t.text, it["answers"], lang)
                            for t in s.outputs]
                    emit({"item_id": it["item_id"], "en_cue": en,
                          "greedy": g.outputs[0].text.strip(),
                          "greedy_correct": score_production(
                              g.outputs[0].text, it["answers"], lang),
                          "k_rescue": sum(s_ok), "n_rescue": N_PROD})

        elif stage == "cue":
            # First-unit cue on failed items, with a wrong-cue placebo condition.
            # Human prediction: correct phonological cues preferentially resolve
            # TOT (vs DK); the wrong cue controls for generic hint effects.
            prod_path = os.path.join(args.out_dir, "production.jsonl")
            prod = {}
            with open(prod_path, encoding="utf-8") as pf:
                for line in pf:
                    if line.strip():
                        r = json.loads(line)
                        prod[r["item_id"]] = r
            import random as _random
            rng = _random.Random(args.seed)
            usr_t = P.CUE_USER_ZH if lang == "zh" else P.CUE_USER_EN
            sys_p = P.PROD_SYSTEM_ZH if lang == "zh" else P.PROD_SYSTEM_EN
            alphabet = ("的一是不了人在有我他这中大来上国和"
                        if lang == "zh" else "abcdefghijklmnopqrstuvwxyz")
            usable = []
            for it in chunk:
                pr = prod.get(it["item_id"])
                if pr and not pr["greedy_correct"] and pr["k_prod"] <= 2:
                    true_cue = it["word"][0]
                    wrong = rng.choice([c for c in alphabet if c != true_cue])
                    usable.append((it, true_cue, wrong))
            for cond, idx in (("correct", 1), ("wrong", 2)):
                ps = [runner.chat_prompt(
                    usr_t.format(gloss=u[0]["gloss"], cue=u[idx]), sys_p)
                    for u in usable]
                greedy = runner.generate(ps, temperature=0.0,
                                         max_tokens=MAX_TOKENS_WORD)
                sampled = runner.generate(ps, temperature=T_PROD, n=N_PROD,
                                          max_tokens=MAX_TOKENS_WORD,
                                          seed=args.seed)
                for u, g, s in zip(usable, greedy, sampled):
                    it = u[0]
                    s_ok = [score_production(t.text, it["answers"], lang)
                            for t in s.outputs]
                    emit({"item_id": it["item_id"], "condition": cond,
                          "cue": u[idx],
                          "greedy_correct": score_production(
                              g.outputs[0].text, it["answers"], lang),
                          "k_cue": sum(s_ok), "n_cue": N_PROD})

        elif stage == "retest":
            # The fully matched neutral control for the cue arm.
            #
            # CUE_USER is PROD_USER with one line inserted, and the cue stage
            # already passes PROD_SYSTEM, MAX_TOKENS_WORD, N_PROD and T_PROD.
            # So replaying production on the cue stage's own item pool differs
            # from the cue arm in the first unit and nothing else: same
            # template, same system prompt, same 20 samples at T=1, same single
            # turn with no history. The CoT retry cannot do this job -- it
            # changes the instruction, the output format and the sample budget
            # at the same time as it removes the cue.
            #
            # SEED. The pool is selected on k_prod<=2 out of 20, so replaying
            # production's seed would redraw the same misses and the arm would
            # read ~0 by construction -- a silent no-op that would look like a
            # huge cue effect. The offset makes this an independent draw, which
            # is what turns it into a measurement of regression to the mean.
            #
            # The greedy decode is deliberately still recorded, but it is
            # degenerate here: temperature 0 on the production prompt
            # reproduces production's own greedy answer, which is wrong for
            # every item in the pool. Only the sampled arm is informative, so
            # the contrast to draw is k_retest>0 against k_cue>0.
            prod_path = os.path.join(args.out_dir, "production.jsonl")
            prod = {}
            with open(prod_path, encoding="utf-8") as pf:
                for line in pf:
                    if line.strip():
                        r = json.loads(line)
                        prod[r["item_id"]] = r
            usr_t = P.PROD_USER_ZH if lang == "zh" else P.PROD_USER_EN
            sys_p = P.PROD_SYSTEM_ZH if lang == "zh" else P.PROD_SYSTEM_EN
            usable = [it for it in chunk
                      if (prod.get(it["item_id"])
                          and not prod[it["item_id"]]["greedy_correct"]
                          and prod[it["item_id"]]["k_prod"] <= 2)]
            ps = [runner.chat_prompt(usr_t.format(gloss=it["gloss"]), sys_p)
                  for it in usable]
            greedy = runner.generate(ps, temperature=0.0,
                                     max_tokens=MAX_TOKENS_WORD)
            sampled = runner.generate(ps, temperature=T_PROD, n=N_PROD,
                                      max_tokens=MAX_TOKENS_WORD,
                                      seed=args.seed + 977)
            for it, g, s in zip(usable, greedy, sampled):
                s_ok = [score_production(t.text, it["answers"], lang)
                        for t in s.outputs]
                emit({"item_id": it["item_id"],
                      "greedy_correct": score_production(
                          g.outputs[0].text, it["answers"], lang),
                      "k_retest": sum(s_ok), "n_retest": N_PROD,
                      "k_prod": prod[it["item_id"]]["k_prod"]})

        elif stage == "cuerequest":
            # Agentive cue request + self-repair, on the same failed-item pool as
            # `cue`. Turn 1: answer or request exactly one hint. Turn 2 (same
            # dialogue): deliver the requested hint, retry. Human prediction: TOT
            # preferentially requests a hint instead of guessing wrong (vs DK).
            prod_path = os.path.join(args.out_dir, "production.jsonl")
            prod = {}
            with open(prod_path, encoding="utf-8") as pf:
                for line in pf:
                    if line.strip():
                        r = json.loads(line)
                        prod[r["item_id"]] = r
            if lang == "zh":
                usr_t = P.CUEREQUEST_USER_ZH
            else:
                usr_t = {
                    "default": P.CUEREQUEST_USER_EN,
                    "encourage": P.CUEREQUEST_USER_EN_ENCOURAGE,
                    "neutral2": P.CUEREQUEST_USER_EN_NEUTRAL2,
                }[args.cuereq_variant]
            hint_letter_t = (P.CUEREQUEST_HINT_LETTER_ZH if lang == "zh"
                             else P.CUEREQUEST_HINT_LETTER_EN)
            hint_context_t = (P.CUEREQUEST_HINT_CONTEXT_ZH if lang == "zh"
                              else P.CUEREQUEST_HINT_CONTEXT_EN)
            valid_tokens = ["LETTER", "CONTEXT"] + (["TRANSLATION"] if lang == "zh" else [])
            hint_re = re.compile(r"HINT-(" + "|".join(valid_tokens) + r")")
            usable = []
            for it in chunk:
                pr = prod.get(it["item_id"])
                if pr and not pr["greedy_correct"] and pr["k_prod"] <= 2:
                    usable.append(it)
            if usable:
                ps1 = [runner.chat_prompt(usr_t.format(gloss=it["gloss"]))
                       for it in usable]
                turn1 = runner.generate(ps1, temperature=0.0, max_tokens=24)

                metas = []
                turn2_todo = []  # (item, turn1_raw, meta) needing a turn-2 call
                for it, o in zip(usable, turn1):
                    raw = o.outputs[0].text.strip()
                    m = hint_re.search(raw.upper())
                    if m:
                        meta = {"item_id": it["item_id"], "mode": "requested",
                                 "cue_type": m.group(1).lower(), "fallback": False,
                                 "turn1_correct": None,
                                 "turn2_greedy_correct": None, "k_turn2": None,
                                 "n_turn2": 0}
                        turn2_todo.append((it, raw, meta))
                    else:
                        metas.append({"item_id": it["item_id"], "mode": "answered",
                                      "cue_type": None, "fallback": False,
                                      "turn1_correct": score_production(
                                          raw, it["answers"], lang),
                                      "turn2_greedy_correct": None, "k_turn2": None,
                                      "n_turn2": 0})

                if turn2_todo:
                    ps2 = []
                    for it, raw, meta in turn2_todo:
                        if meta["cue_type"] == "letter":
                            hint = hint_letter_t.format(cue=it["word"][0])
                        elif meta["cue_type"] == "context":
                            hint = hint_context_t.format(gloss=it["gloss"])
                        else:  # translation (zh only)
                            en = (it.get("en_equiv") or "").split(";")[0].strip()
                            if en:
                                hint = P.CUEREQUEST_HINT_TRANSLATION_ZH.format(en=en)
                            else:
                                meta["fallback"] = True
                                hint = hint_letter_t.format(cue=it["word"][0])
                        msgs = [{"role": "user",
                                 "content": usr_t.format(gloss=it["gloss"])},
                                {"role": "assistant", "content": raw},
                                {"role": "user", "content": hint}]
                        ps2.append(runner.tok.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True))
                    greedy2 = runner.generate(ps2, temperature=0.0,
                                              max_tokens=MAX_TOKENS_WORD)
                    sampled2 = runner.generate(ps2, temperature=T_PROD, n=10,
                                               max_tokens=MAX_TOKENS_WORD,
                                               seed=args.seed)
                    for (it, raw, meta), g, s in zip(turn2_todo, greedy2, sampled2):
                        s_ok = [score_production(o.text, it["answers"], lang)
                                for o in s.outputs]
                        meta["turn2_greedy_correct"] = score_production(
                            g.outputs[0].text, it["answers"], lang)
                        meta["k_turn2"] = sum(s_ok)
                        meta["n_turn2"] = 10
                        metas.append(meta)

                for meta in metas:
                    emit(meta)

        elif stage in ("cot", "cot20"):
            # CoT rescue: strict/weak-fail items plus a produced control sample.
            # cot20 is the same arm at the production sampling budget,
            # so that cue, placebo, retest and this one are all read
            # off greedy plus twenty draws and no budget footnote is
            # needed anywhere. It writes cot20.jsonl; cot.jsonl stays.
            n_cot = N_PROD if stage == "cot20" else 5
            prod_path = os.path.join(args.out_dir, "production.jsonl")
            prod = {}
            with open(prod_path, encoding="utf-8") as pf:
                for line in pf:
                    if line.strip():
                        r = json.loads(line)
                        prod[r["item_id"]] = r
            import random as _random
            rng2 = _random.Random(args.seed + 1)
            fail_items, prod_items = [], []
            for it in chunk:
                pr = prod.get(it["item_id"])
                if not pr:
                    continue
                if not pr["greedy_correct"] and pr["k_prod"] <= 2:
                    fail_items.append(it)
                elif pr["k_prod"] >= 6:
                    prod_items.append(it)
            rng2.shuffle(prod_items)
            usable = [(it, "failed") for it in fail_items]
            if stage == "cot":
                # the five-draw arm keeps its ceiling group; cot20
                # does not, because every analysis that reads it
                # requires a strict production failure and a control
                # item (k_prod>=6) can never pass that filter.
                usable += [(it, "produced_control") for it in
                           prod_items[: max(1, len(fail_items) // 3)]]
            usr_t = P.COT_USER_ZH if lang == "zh" else P.COT_USER_EN
            ps = [runner.chat_prompt(usr_t.format(gloss=it["gloss"]))
                  for it, _ in usable]
            greedy = runner.generate(ps, temperature=0.0, max_tokens=400)
            sampled = runner.generate(ps, temperature=T_PROD, n=n_cot,
                                      max_tokens=400, seed=args.seed)
            ans_re = re.compile(r"(?:ANSWER|答案)[:：]\s*(.+)", re.IGNORECASE)

            def cot_answer(text):
                m = ans_re.findall(text)
                return m[-1].strip() if m else text.strip().splitlines()[-1]
            for (it, grp), g, s in zip(usable, greedy, sampled):
                g_ans = cot_answer(g.outputs[0].text)
                s_ok = [score_production(cot_answer(o.text), it["answers"], lang)
                        for o in s.outputs]
                emit({"item_id": it["item_id"], "group": grp,
                      "greedy_correct": score_production(g_ans, it["answers"], lang),
                      "greedy_answer": g_ans[:60],
                      "k_cot": sum(s_ok), "n_cot": n_cot,
                      "trace": g.outputs[0].text[:600]})

        elif stage == "circum":
            # Controlled circumlocution: describe the concept without reusing the
            # gloss wording, on failed-production items (like cue/cot select) plus
            # a produced-control sample (like cot's ceiling group). SPEAKER model,
            # fresh context, written to out_dir/circum.jsonl for a separate
            # listener model to consume via --circum-dir.
            prod_path = os.path.join(args.out_dir, "production.jsonl")
            prod = {}
            with open(prod_path, encoding="utf-8") as pf:
                for line in pf:
                    if line.strip():
                        r = json.loads(line)
                        prod[r["item_id"]] = r
            import random as _random
            rng3 = _random.Random(args.seed + 5)
            fail_items, prod_items = [], []
            for it in chunk:
                pr = prod.get(it["item_id"])
                if not pr:
                    continue
                if not pr["greedy_correct"] and pr["k_prod"] <= 2:
                    fail_items.append(it)
                elif pr["k_prod"] >= 6:
                    prod_items.append(it)
            rng3.shuffle(prod_items)
            usable = ([(it, "failed") for it in fail_items]
                      + [(it, "produced_control") for it in prod_items[
                          : max(1, len(fail_items) // 3)]])
            usr_t = P.CIRCUM_USER_ZH if lang == "zh" else P.CIRCUM_USER_EN
            ps = [runner.chat_prompt(usr_t.format(gloss=it["gloss"]))
                  for it, _ in usable]
            outs = runner.generate(ps, temperature=0.0, max_tokens=200)
            for (it, grp), o in zip(usable, outs):
                emit({"item_id": it["item_id"], "group": grp,
                      "circum": o.outputs[0].text.strip()})

        elif stage == "listener_name":
            # Downstream name-transmission condition: the listener receives the
            # SPEAKER's greedy production answer (right or wrong) instead of a
            # description, same 4 options, single seeded rotation (same seeds as
            # `listener` so conditions are position-matched). Measures the
            # downstream cost of confabulated names vs descriptive fallback.
            if not args.circum_dir:
                raise SystemExit("--circum-dir (speaker dir) required")
            prod_path = os.path.join(args.circum_dir, "production.jsonl")
            sprod = {}
            with open(prod_path, encoding="utf-8") as pf:
                for line in pf:
                    if line.strip():
                        r = json.loads(line)
                        sprod[r["item_id"]] = r
            import random as _random
            usr_t = (P.LISTENER_NAME_ZH if lang == "zh" else P.LISTENER_NAME_EN)
            metas, ps = [], []
            for it in chunk:
                pr = sprod.get(it["item_id"])
                if not pr or not it.get("distractors"):
                    continue
                name = pr["greedy"].strip()[:40]
                if not name:
                    continue
                opts = [it["target_option"]] + it["distractors"]
                rng = _random.Random(f"{args.seed}-{it['item_id']}")
                tgt_pos = rng.randrange(4)
                slots = [None] * 4
                slots[tgt_pos] = opts[0]
                rest = [p for p in range(4) if p != tgt_pos]
                for pos, d in zip(rest, opts[1:]):
                    slots[pos] = d
                ps.append(runner.chat_prompt(usr_t.format(
                    name=name, opt_a=slots[0], opt_b=slots[1],
                    opt_c=slots[2], opt_d=slots[3])))
                metas.append((it["item_id"], tgt_pos,
                              bool(pr["greedy_correct"])))
            if ps:
                outs = runner.generate(ps, temperature=0.0, max_tokens=4,
                                       logprobs=20)
                for (iid, tgt_pos, g_ok), o in zip(metas, outs):
                    probs = first_token_letter_logprobs(o)
                    choice = (max(probs, key=probs.get) if probs else None)
                    emit({"item_id": iid, "speaker_greedy_correct": g_ok,
                          "target_pos": LETTERS[tgt_pos], "choice": choice,
                          "correct": choice == LETTERS[tgt_pos]})

        elif stage == "listener":
            # 4AFC over a circumlocution produced by a DIFFERENT (speaker) model.
            # Reads circum.jsonl from --circum-dir; same 4 options as recognition
            # (target_option + distractors) but a single seeded-random rotation per
            # item (the listener never sees the gloss). Scored like recognition,
            # by first-token logprob argmax over A-D.
            if not args.circum_dir:
                raise SystemExit("--circum-dir is required for the listener stage")
            circum_path = os.path.join(args.circum_dir, "circum.jsonl")
            circum = {}
            with open(circum_path, encoding="utf-8") as cf:
                for line in cf:
                    if line.strip():
                        r = json.loads(line)
                        circum[r["item_id"]] = r
            import random as _random
            usr_t = P.LISTENER_USER_ZH if lang == "zh" else P.LISTENER_USER_EN
            metas, ps = [], []
            for it in chunk:
                c = circum.get(it["item_id"])
                if not c or not it.get("distractors"):
                    continue
                opts = [it["target_option"]] + it["distractors"]  # [target, d1, d2, d3]
                rng = _random.Random(f"{args.seed}-{it['item_id']}")
                tgt_pos = rng.randrange(4)
                slots = [None] * 4
                slots[tgt_pos] = opts[0]
                rest = [p for p in range(4) if p != tgt_pos]
                for pos, d in zip(rest, opts[1:]):
                    slots[pos] = d
                ps.append(runner.chat_prompt(usr_t.format(
                    circum=c["circum"], opt_a=slots[0], opt_b=slots[1],
                    opt_c=slots[2], opt_d=slots[3])))
                metas.append((it["item_id"], tgt_pos, c.get("group")))
            if ps:
                outs = runner.generate(ps, temperature=0.0, max_tokens=4, logprobs=20)
                for (iid, tgt_pos, grp), o in zip(metas, outs):
                    probs = first_token_letter_logprobs(o)
                    gen = o.outputs[0].text.strip().upper()[:1]
                    if probs:
                        choice = max(probs, key=probs.get)
                        p_target = probs.get(LETTERS[tgt_pos], 0.0)
                    else:
                        choice, p_target = (gen if gen in LETTERS else None), None
                    emit({"item_id": iid, "group": grp,
                          "listener_model": args.model,
                          "choice": choice, "target_pos": LETTERS[tgt_pos],
                          "p_target": p_target, "correct": choice == LETTERS[tgt_pos]})

        elif stage == "listener_open":
            # Open-ended listener: names the target from the description with NO
            # candidate set (breaks the 4AFC geometry the state labels share).
            # Scored against the acceptable-answer set like production.
            if not args.circum_dir:
                raise SystemExit("--circum-dir is required for listener_open")
            circum_path = os.path.join(args.circum_dir, "circum.jsonl")
            circum = {}
            with open(circum_path, encoding="utf-8") as cf:
                for line in cf:
                    if line.strip():
                        r = json.loads(line)
                        circum[r["item_id"]] = r
            usr_t = P.LISTENER_OPEN_ZH if lang == "zh" else P.LISTENER_OPEN_EN
            metas, ps = [], []
            for it in chunk:
                c = circum.get(it["item_id"])
                if not c:
                    continue
                ps.append(runner.chat_prompt(usr_t.format(circum=c["circum"])))
                metas.append((it, c.get("group")))
            if ps:
                outs = runner.generate(ps, temperature=0.0, max_tokens=16)
                for (it, grp), o in zip(metas, outs):
                    txt = o.outputs[0].text.strip()
                    emit({"item_id": it["item_id"], "group": grp,
                          "listener_model": args.model, "answer": txt,
                          "correct": score_production(txt, it["answers"], lang)})

        elif stage == "reverse":
            usr_t = P.REVERSE_ZH if lang == "zh" else P.REVERSE_EN
            ps = [runner.chat_prompt(usr_t.format(word=it["word"])) for it in chunk]
            outs = runner.generate(ps, temperature=0.0, max_tokens=80)
            for it, o in zip(chunk, outs):
                emit({"item_id": it["item_id"],
                      "definition": o.outputs[0].text.strip()})

        elif stage == "perform":
            usr_t = P.PERFORM_TOT_ZH if lang == "zh" else P.PERFORM_TOT_EN
            ps = [runner.chat_prompt(usr_t.format(gloss=it["gloss"])) for it in chunk]
            outs = runner.generate(ps, temperature=0.0, max_tokens=200)
            for it, o in zip(chunk, outs):
                emit({"item_id": it["item_id"], "performance": o.outputs[0].text.strip()})

        f.flush()
        print(f"[{stage}] {min(c0 + CHUNK, len(todo))}/{len(todo)} done", flush=True)
    f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang", choices=["en", "zh"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--circum-dir", default=None,
                    help="dir containing circum.jsonl (the SPEAKER's out-dir); "
                         "required for --stages listener")
    ap.add_argument("--stages", default="production,fok,partial,partial_episode,feature_ceiling,recognition,selfreport,perform,reverse,rescue")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90,
                    help="vLLM gpu_memory_utilization (raise on H100 PCIe "
                         "where usable VRAM is slightly below A100-80G)")
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="sampling temperature for the production stage (paper: 1.0)")
    ap.add_argument("--top-p", type=float, default=1.0,
                    help="nucleus top-p for the production stage (paper: 1.0)")
    ap.add_argument("--cuereq-variant", choices=["default", "encourage", "neutral2"],
                    default="default",
                    help="swap the turn-1 user template for the cuerequest stage "
                         "(EN only; turn-2 hint templates unchanged). Non-default "
                         "variants write to cuerequest_{variant}.jsonl instead of "
                         "cuerequest.jsonl")
    ap.add_argument("--strip-think", action="store_true",
                    help="reasoning models: budget tokens for <think> and "
                         "score only post-think text")
    ap.add_argument("--think-budget", type=int, default=768,
                    help="token budget for <think> plus answer (--strip-think "
                         "only).")
    args = ap.parse_args()

    items = load_items(args.items)
    if args.max_items:
        items = items[: args.max_items]
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    need_rec = "recognition" in stages or "listener" in stages
    if need_rec and any("distractors" not in it for it in items):
        missing = sum(1 for it in items if "distractors" not in it)
        print(f"WARNING: {missing} items lack distractors; recognition/listener "
              f"will skip them")
        if missing == len(items):
            stages = [s for s in stages if s not in ("recognition", "listener")]
    if "listener" in stages and not args.circum_dir:
        raise SystemExit("--circum-dir is required for --stages listener")

    os.makedirs(args.out_dir, exist_ok=True)
    runner = Runner(args.model, args.tp, args.max_model_len,
                    args.gpu_mem_util)
    runner.strip_think = args.strip_think
    runner.think_budget = args.think_budget
    for stage in stages:
        run_items = items
        if stage in ("recognition", "listener"):
            run_items = [it for it in items if it.get("distractors")]
        if stage == "cuerequest" and args.cuereq_variant != "default":
            out_path = os.path.join(args.out_dir, f"cuerequest_{args.cuereq_variant}.jsonl")
        else:
            out_path = os.path.join(args.out_dir, f"{stage}.jsonl")
        run_stage(stage, run_items, runner, args.lang, out_path, args)
    print("ALL STAGES COMPLETE")


if __name__ == "__main__":
    main()
