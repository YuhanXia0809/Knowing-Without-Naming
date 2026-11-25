"""R- scarcity across the English enrichment sets.

For the main English bank and the two enrichment sets (the 500-item
ultrarare extension and the 500-item opaque-word set), over the four models
run on all three (Qwen2.5-7B / -72B, Llama-3.1-8B / -70B): strict
failures, R+ (TOT) and R- (DK) counts, and the share of strict failures
with a full recognition record that score 4/4 -- the series the paper
prints as 84.4 -> 92.2 -> 93.5%, and the R- count 113 -> 154 that the
1.67x larger English pool buys.

Conventions are analyze_pilot.classify's: strict failure = greedy wrong
and 0/20; TOT = strict failure with k_rec >= 3/4, DK = k_rec <= 1/4.

  python enrichment_ceiling.py
"""
from __future__ import annotations

import os

from analyze_pilot import classify, load_jsonl

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODELS = ["qwen7b", "llama8b", "qwen72b", "llama70b"]
SETS = [("full_en", "items_full_en_final.jsonl", "main English bank"),
        ("ext_en", "items_ext_en_final.jsonl", "ultrarare extension"),
        ("dk2_en", "items_dk2_en_final.jsonl", "opaque-word set")]


KNOWN = []


def main():
    cum_items = cum_dk = 0
    print("%-22s %6s %7s %6s %5s %7s %13s" % ("set", "items", "strict",
                                              "TOT", "DK", "TOT:DK",
                                              "4/4 of strict"))
    for fam, fn, label in SETS:
        items = load_jsonl(os.path.join(ROOT, "data", fn))
        strict = full = k4 = tot = dk = 0
        for m in MODELS:
            base = os.path.join(ROOT, "results", "%s_%s" % (fam, m))
            prod = load_jsonl(os.path.join(base, "production.jsonl"))
            rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
            for iid in items:
                p = prod.get(iid)
                if p is None or p["k_prod"] != 0 or p["greedy_correct"]:
                    continue
                strict += 1
                r = rec.get(iid)
                if r is not None and r.get("n_rec", 0) >= 4:
                    full += 1
                    k4 += r["k_rec"] == 4
                st = classify(p, r)
                tot += st == "TOT"
                dk += st == "DK"
        cum_items += len(items)
        cum_dk += dk
        ceil = []
        for m in MODELS:
            base = os.path.join(ROOT, "results", "%s_%s" % (fam, m))
            prod = load_jsonl(os.path.join(base, "production.jsonl"))
            rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
            kn = [1.0 if rec[i]["k_rec"] >= 3 else 0.0 for i in items
                  if classify(prod.get(i), rec.get(i)) == "known"
                  and i in rec and rec[i].get("n_rec") == 4]
            ceil.append("%s %.1f%% (n=%d)" % (m, 100.0 * sum(kn) / len(kn),
                                             len(kn)))
        KNOWN.append((label, ceil))
        print("%-22s %6d %7d %6d %5d %7.1f %6d/%d = %.1f%%"
              % (label, len(items), strict, tot, dk, tot / dk, k4, full,
                 100.0 * k4 / full))
        print("%-22s cumulative English items %d (x%.2f of the main bank), "
              "cumulative DK %d" % ("", cum_items, cum_items / 1503.0,
                                    cum_dk))
    print("\nknown-item recognition ceiling (items with k_prod >= 6/20 and a "
          "full recognition record, share with k_rec >= 3/4)")
    for label, ceil in KNOWN:
        print("%-22s %s" % (label, "; ".join(ceil)))


if __name__ == "__main__":
    main()
