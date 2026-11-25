"""English enrichment sets (ext_en = ultrarare extension, dk2_en = opaque-word set):
taxonomy, recognition ceiling, FOK AUC, reversal control.

Conventions come straight from analyze_pilot (classify, load_jsonl, auc) so the
numbers are directly comparable to the confirmatory battery.

  python -u enrichment_taxonomy.py --family dk2_en --items ../data/items_dk2_en_final.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from analyze_pilot import auc, classify, load_jsonl

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODELS = ["qwen7b", "llama8b", "qwen72b", "llama70b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="dk2_en")
    ap.add_argument("--items", required=True)
    args = ap.parse_args()

    items = load_jsonl(args.items)
    n_distr = sum(1 for it in items.values() if it.get("distractors"))
    n_uniq = sum(1 for it in items.values() if it.get("unique"))
    print(f"### family={args.family}  items={len(items)}  "
          f"with-distractors(recognition-eligible)={n_distr}  unique={n_uniq}")

    tot_all = dk_all = 0
    for m in MODELS:
        base = os.path.join(ROOT, "results", f"{args.family}_{m}")
        prod = load_jsonl(os.path.join(base, "production.jsonl"))
        rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
        fok = load_jsonl(os.path.join(base, "fok.jsonl"))
        if not prod:
            print(f"  {m}: MISSING {base}")
            continue
        states = {i: classify(prod.get(i), rec.get(i)) for i in items}
        c = Counter(states.values())
        n_scored = sum(1 for i in items if i in prod)
        strict = sum(1 for i in items if i in prod
                     and prod[i]["k_prod"] == 0 and not prod[i]["greedy_correct"])
        nT, nD = c.get("TOT", 0), c.get("DK", 0)
        tot_all += nT
        dk_all += nD
        ratio = (nT / nD) if nD else float("inf")
        print(f"\n== {m}: scored={n_scored} strict_fail={strict} "
              f"({strict / n_scored:.1%})  TOT={nT} DK={nD}  T:D={ratio:.1f}")
        print(f"   states: {dict(c.most_common())}")

        # instrument check: recognition ceiling on KNOWN items (paper gate 85%)
        kn = [1.0 if rec[i]["k_rec"] >= 3 else 0.0 for i in items
              if states[i] == "known" and i in rec and rec[i].get("n_rec") == 4]
        ceil = sum(kn) / len(kn) if kn else float("nan")
        print(f"   recognition ceiling on KNOWN: {ceil:.1%} (n={len(kn)}) "
              f"{'PASS' if ceil > 0.85 else 'FAIL'}")

        # FOK AUC, TOT vs DK
        fail_ids = [i for i in items if states[i] in ("TOT", "DK")]
        for meas in ("fok_verbal", "fok_p_yes"):
            a = auc([(fok.get(i, {}).get(meas), 1 if states[i] == "TOT" else 0)
                     for i in fail_ids])
            print(f"   FOK {meas}: AUC={'NA' if a is None else round(a, 3)} "
                  f"(n={len(fail_ids)})")

        # reversal control (availability): def_correct by state
        rev = load_jsonl(os.path.join(base, "reverse_judged.jsonl"))
        if rev:
            for g in ("known", "TOT", "DK"):
                v = [1.0 if rev[i]["def_correct"] else 0.0 for i in items
                     if states[i] == g and i in rev
                     and rev[i].get("def_correct") is not None]
                if v:
                    print(f"   reverse def_correct {g}: {sum(v) / len(v):.1%} "
                          f"(n={len(v)})")

    print(f"\n### POOLED {args.family}: TOT={tot_all} DK={dk_all} "
          f"T:D={tot_all / dk_all if dk_all else float('inf'):.1f}")


if __name__ == "__main__":
    main()
