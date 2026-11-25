"""Cross-family answer-key validation.

Given the Llama-3.1-70B rebuild of acceptable-answer sets on the stratified
299-item sample (data/items_en_rejudge_llama.jsonl), for each confirmatory
model:
  1. re-score production (greedy + 20 samples) against the Llama answer set;
  2. strict-failure label flip rate vs the original Qwen-72B answer set;
  3. the headline number: of items originally scored strict
     failures, what fraction have their MODAL response accepted by the
     Llama-built answer set (the spurious-TOT candidate rate)?

  python rejudge_flip.py [--lang zh]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from analyze_pilot import classify, load_jsonl
from scoring import extract_candidate, score_production

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODELS = ["qwen7b", "llama8b", "qwen72b", "llama70b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en", choices=["en", "zh"])
    args = ap.parse_args()
    lang = args.lang
    orig = load_jsonl(os.path.join(ROOT, "data",
                                   f"items_full_{lang}_final.jsonl"))
    rej_path = os.path.join(ROOT, "data",
                            f"items_{lang}_rejudge_llama.jsonl")
    rej = load_jsonl(rej_path)
    shared = [iid for iid in rej if iid in orig]
    print(f"rejudged sample: {len(shared)} items")

    # answer-set comparison
    n_super = n_sub = n_diff = n_same = 0
    for iid in shared:
        a0 = set(x.lower() for x in orig[iid]["answers"])
        a1 = set(x.lower() for x in rej[iid].get("answers", []))
        if a0 == a1:
            n_same += 1
        elif a1 >= a0:
            n_super += 1
        elif a1 <= a0:
            n_sub += 1
        else:
            n_diff += 1
    print(f"answer sets: identical={n_same} llama-superset={n_super} "
          f"llama-subset={n_sub} crossing={n_diff}")

    for tag in MODELS:
        base = os.path.join(ROOT, "results", f"full_{lang}_{tag}")
        prod = load_jsonl(os.path.join(base, "production.jsonl"))
        rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
        n = flips = 0
        sf_orig = 0
        modal_ok = 0
        for iid in shared:
            p = prod.get(iid)
            if not p:
                continue
            n += 1
            a1 = rej[iid].get("answers", [])
            g_ok1 = score_production(p["greedy"], a1, lang)
            s_ok1 = [score_production(t, a1, lang)
                     for t in p.get("samples", [])]
            sf1 = (not g_ok1) and sum(s_ok1) == 0
            sf0 = (not p["greedy_correct"]) and p["k_prod"] == 0
            if sf0 != sf1:
                flips += 1
            if sf0:
                sf_orig += 1
                cands = [c for c in (extract_candidate(t, lang)
                                     for t in p.get("samples", [])) if c]
                if cands:
                    modal = Counter(cands).most_common(1)[0][0]
                    if score_production(modal, a1, lang):
                        modal_ok += 1
        print(f"{tag}: n={n} strict-fail(orig)={sf_orig} "
              f"label-flips={flips} ({flips / max(1, n):.1%}) "
              f"modal-accepted-under-llama-key={modal_ok}"
              f" ({modal_ok / max(1, sf_orig):.1%} of orig strict-fails)")


if __name__ == "__main__":
    main()
