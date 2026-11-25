"""Two checks: (a) MWE/hyphen/whitespace-normalized rescoring of
English strict failures against expanded answer sets (vetted answers +
WordNet synonyms already stored per item in wn_synonyms), and (b)
state-conditional label-flip rates under the Llama-rebuilt key on the
299-item sample (which TOT vs DK items flip), with recomputed sample
ratios under the alternative key.

  python mwe_rescore.py
"""
from __future__ import annotations

import os
import re
from collections import Counter

from analyze_pilot import classify, load_jsonl
from scoring import extract_candidate, score_production

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODELS = ["qwen7b", "llama8b", "qwen72b", "llama70b"]


def norm(s):
    return re.sub(r"[\s\-]+", "", (s or "").lower())


def main():
    items = load_jsonl(os.path.join(ROOT, "data", "items_full_en_final.jsonl"))
    # expanded normalized answer sets: vetted answers + wn_synonyms
    exp = {}
    for iid, it in items.items():
        ans = set(norm(a) for a in it.get("answers", []))
        for s in it.get("wn_synonyms", []) or []:
            ans.add(norm(s))
        exp[iid] = ans

    print("== (a) MWE/hyphen-normalized rescoring of strict failures ==")
    for tag in MODELS:
        base = os.path.join(ROOT, "results", f"full_en_{tag}")
        prod = load_jsonl(os.path.join(base, "production.jsonl"))
        rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
        flips = 0
        tot_flips = dk_flips = 0
        n_sf = 0
        for iid, p in prod.items():
            sf = (not p["greedy_correct"]) and p["k_prod"] == 0
            if not sf:
                continue
            n_sf += 1
            st = classify(p, rec.get(iid))
            texts = [p.get("greedy", "")] + list(p.get("samples", []))
            hit = any(norm(extract_candidate(t, "en") or t) in exp[iid]
                      for t in texts if t)
            if hit:
                flips += 1
                if st == "TOT":
                    tot_flips += 1
                elif st == "DK":
                    dk_flips += 1
        print(f"{tag}: strict-fails={n_sf} normalized-hits={flips} "
              f"({flips/max(1,n_sf):.1%}) of which TOT={tot_flips} "
              f"DK={dk_flips}")

    print("== (b) state-conditional flips under Llama key (299 sample) ==")
    rej = load_jsonl(os.path.join(ROOT, "data", "items_en_rejudge_llama.jsonl"))
    shared = [i for i in rej if i in items]
    for tag in MODELS:
        base = os.path.join(ROOT, "results", f"full_en_{tag}")
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
            g1 = score_production(p["greedy"], a1, "en")
            k1 = sum(score_production(t, a1, "en")
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
