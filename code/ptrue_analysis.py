"""Detection-baseline additions: P(True) and
strong-elicitation verbalized FOK, for the detector comparison of
Appendix E (Detection Details).

Per confirmatory dataset (4 models x 2 languages), from full_{lang}_{tag}/:
  ptrue.jsonl     -> p_yes on the model's own greedy answer (Kadavath-style
                     self-evaluation) : AUC for (a) greedy failure,
                     (b) TOT-vs-DK among strict failures.
  fokstrong.jsonl -> full-range verbalized probability (0-100, anti-round
                     instruction): same two AUCs; also % distinct values
                     (does the full-range instruction actually spread the
                     distribution vs the near-binary default elicitation?).
Bootstrap CIs (1000 item resamples) on the state AUCs.

  python ptrue_analysis.py
"""
from __future__ import annotations

import os
import random

from analyze_pilot import classify, load_jsonl

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONF = [("en", "qwen7b"), ("en", "llama8b"), ("en", "qwen72b"),
        ("en", "llama70b"), ("zh", "qwen7b"), ("zh", "llama8b"),
        ("zh", "qwen72b"), ("zh", "llama70b")]


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def boot(pos, neg, n_boot=1000, seed=0):
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        bp = [pos[rng.randrange(len(pos))] for _ in pos]
        bn = [neg[rng.randrange(len(neg))] for _ in neg]
        vals.append(auc(bp, bn))
    vals.sort()
    return vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]


def main():
    for lang, tag in CONF:
        base = os.path.join(ROOT, "results", f"full_{lang}_{tag}")
        items = load_jsonl(os.path.join(ROOT, "data",
                                        f"items_full_{lang}_final.jsonl"))
        prod = load_jsonl(os.path.join(base, "production.jsonl"))
        rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
        for stage, field, parse in (
                ("ptrue", "p_true", float),
                ("fokstrong", "fokstrong_verbal", lambda v: float(v))):
            data = load_jsonl(os.path.join(base, f"{stage}.jsonl"))
            if not data:
                print(f"{lang}_{tag} {stage}: missing")
                continue
            fail_pos, fail_neg = [], []   # signal for failure prediction
            st_pos, st_neg = [], []       # among failures: TOT vs DK
            vals = []
            for iid in items:
                p = prod.get(iid)
                r = data.get(iid)
                if p is None or r is None or r.get(field) is None:
                    continue
                try:
                    v = parse(r[field])
                except (TypeError, ValueError):
                    continue
                vals.append(v)
                failed = not p["greedy_correct"]
                # higher signal should mean "will succeed"
                (fail_neg if failed else fail_pos).append(v)
                st = classify(p, rec.get(iid))
                if st == "TOT":
                    st_pos.append(v)
                elif st == "DK":
                    st_neg.append(v)
            a_fail = auc(fail_pos, fail_neg)
            a_st = auc(st_pos, st_neg)
            lo, hi = boot(st_pos, st_neg) if st_pos and st_neg else (float("nan"),) * 2
            n_distinct = len(set(vals))
            print(f"{lang}_{tag} {stage}: fail-AUC={a_fail:.3f} "
                  f"state-AUC={a_st:.3f} [{lo:.3f},{hi:.3f}] "
                  f"nTOT={len(st_pos)} nDK={len(st_neg)} "
                  f"distinct-values={n_distinct}")


if __name__ == "__main__":
    main()
