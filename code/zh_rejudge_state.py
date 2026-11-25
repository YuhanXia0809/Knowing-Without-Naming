"""Zh state-conditional label-flip rates under the
Llama-rebuilt answer key (300-item stratified sample) -- mirrors the
block (b) of mwe_rescore.py for Chinese, plus recomputed sample ratios.

  python zh_rejudge_state.py
"""
from __future__ import annotations

import os

from analyze_pilot import classify, load_jsonl
from scoring import score_production

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODELS = ["qwen7b", "llama8b", "qwen72b", "llama70b"]


def main():
    items = load_jsonl(os.path.join(ROOT, "data",
                                    "items_full_zh_final.jsonl"))
    rej = load_jsonl(os.path.join(ROOT, "data",
                                  "items_zh_rejudge_llama.jsonl"))
    shared = [i for i in rej if i in items]
    print(f"shared: {len(shared)}")
    for tag in MODELS:
        base = os.path.join(ROOT, "results", f"full_zh_{tag}")
        prod = load_jsonl(os.path.join(base, "production.jsonl"))
        rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
        t_flip = d_flip = t_n = d_n = 0
        tot1 = dk1 = 0
        for iid in shared:
            p = prod.get(iid)
            if not p:
                continue
            st0 = classify(p, rec.get(iid))
            a1 = rej[iid].get("answers", [])
            g1 = score_production(p["greedy"], a1, "zh")
            k1 = sum(score_production(t, a1, "zh")
                     for t in p.get("samples", []))
            sf1 = (not g1) and k1 == 0
            if st0 == "TOT":
                t_n += 1
                if not sf1:
                    t_flip += 1
                else:
                    tot1 += 1
            elif st0 == "DK":
                d_n += 1
                if not sf1:
                    d_flip += 1
                else:
                    dk1 += 1
        r0 = t_n / max(1, d_n)
        r1 = tot1 / max(1, dk1)
        print(f"{tag}: TOT flips {t_flip}/{t_n} ({t_flip/max(1,t_n):.1%}), "
              f"DK flips {d_flip}/{d_n} ({d_flip/max(1,d_n):.1%}); "
              f"sample ratio {r0:.1f}:1 -> {r1:.1f}:1 under Llama key")


if __name__ == "__main__":
    main()
