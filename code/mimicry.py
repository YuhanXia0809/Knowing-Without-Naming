"""Surface mimicry quantification: when instructed to PERFORM a
tip-of-the-tongue state, models fluently claim partial knowledge — is any of it
real? Extract first-letter/first-character claims from perform.jsonl free text,
score against targets, and compare with the same items' in-episode probe accuracy.

  python -u mimicry.py
"""
from __future__ import annotations

import os
import re
from itertools import product

from analyze_pilot import classify, load_jsonl, score_partial

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODELS = ["qwen7b", "llama8b", "qwen72b", "llama70b"]

# Article-aware patterns come first: without them the article in
# `starts with a "N"` is captured as the claimed letter "a".
EN_PATTERNS = [
    # article + quoted letter: `starts with a "N"`, `first letter is an "e"`
    re.compile(r"first letter\s+(?:is\s+|might be\s+|could be\s+|would be\s+)?"
               r"(?:a|an)\s+[\"'“”‘’]([A-Za-z])[\"'“”‘’]"),
    re.compile(r"(?:starts?|begins?|starting|beginning)\s+with\s+"
               r"(?:the letter\s+)?(?:a|an)\s+[\"'“”‘’]([A-Za-z])[\"'“”‘’]"),
    # article + bare capital: `starts with a T`
    re.compile(r"(?:starts?|begins?|starting|beginning)\s+with\s+"
               r"(?:the letter\s+)?(?:a|an)\s+([A-Z])\b"),
    # no article, optionally quoted — with a guard so a captured `a`/`an` is
    # not immediately followed by a quoted letter (the article case)
    re.compile(r"first letter\s+(?:is\s+|might be\s+|could be\s+|would be\s+)?"
               r"[\"'“”‘’]?([A-Za-z])\b(?!\s*[\"'“”‘’][A-Za-z])"),
    re.compile(r"(?:starts?|begins?|starting|beginning)\s+with\s+(?:the letter\s+)?"
               r"[\"'“”‘’]?([A-Za-z])\b(?!\s*[\"'“”‘’][A-Za-z])"),
    re.compile(r"letter\s+[\"'“”‘’]([A-Za-z])[\"'“”‘’]"),
]
ZH_PATTERNS = [
    re.compile(r"第一个字(?:是|应该是|可能是|好像是)?[\s“”\"'「『]*([一-鿿])"),
    re.compile(r"[以是]?[“”\"'「『]([一-鿿])[”\"'」』]?\s*(?:字)?开头"),
    re.compile(r"开头(?:的字)?(?:是|应该是|可能是)[\s“”\"'「『]*([一-鿿])"),
]


def extract_claim(text, lang):
    pats = ZH_PATTERNS if lang == "zh" else EN_PATTERNS
    for p in pats:
        m = p.search(text)
        if m:
            g = m.group(1)
            return g.lower() if lang == "en" else g
    return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    items_by_lang = {
        lang: load_jsonl(os.path.join(ROOT, "data", f"items_full_{lang}_final.jsonl"))
        for lang in ("en", "zh")}
    print(f"{'dataset':16s} {'state':6s} {'n':>4s} {'claim%':>7s} "
          f"{'claim_acc':>9s} {'probe_acc':>9s}")
    for lang, model in product(("en", "zh"), MODELS):
        base = os.path.join(ROOT, "results", f"full_{lang}_{model}")
        prod = load_jsonl(os.path.join(base, "production.jsonl"))
        rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
        perf = load_jsonl(os.path.join(base, "perform.jsonl"))
        epis = load_jsonl(os.path.join(base, "partial_episode.jsonl"))
        items = items_by_lang[lang]
        ch = "first_letter" if lang == "en" else "first_char"
        stats = {}
        for iid, it in items.items():
            st = classify(prod.get(iid), rec.get(iid))
            if st not in ("known", "TOT", "DK"):
                continue
            pf = perf.get(iid)
            if not pf:
                continue
            claim = extract_claim(pf.get("performance", ""), lang)
            claim_ok = None
            if claim is not None:
                tgt_first = it["word"][0].lower() if lang == "en" else it["word"][0]
                claim_ok = claim == tgt_first
            ep = epis.get(iid)
            probe_ok = None
            if ep and (ep.get("group") == "failed_greedy" or st == "known"):
                sc = score_partial(ch, ep.get(ch, {}).get("greedy"), it, lang,
                                   "target")
                probe_ok = None if sc is None else (
                    sc >= 0.5 if isinstance(sc, float) else sc)
            d = stats.setdefault(st, {"n": 0, "claimed": 0, "claim_ok": [],
                                      "probe_ok": []})
            d["n"] += 1
            if claim is not None:
                d["claimed"] += 1
                d["claim_ok"].append(1.0 if claim_ok else 0.0)
            if probe_ok is not None:
                d["probe_ok"].append(1.0 if probe_ok else 0.0)
        for st in ("known", "TOT", "DK"):
            d = stats.get(st)
            if not d:
                continue
            print(f"{lang}/{model:12s} {st:6s} {d['n']:>4d} "
                  f"{d['claimed'] / d['n']:>6.1%} "
                  f"{mean(d['claim_ok']):>9.3f} {mean(d['probe_ok']):>9.3f}")


if __name__ == "__main__":
    main()
