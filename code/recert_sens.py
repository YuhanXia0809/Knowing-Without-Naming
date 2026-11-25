"""Taxonomy sensitivity to the paraphrase re-certification.

Recompute the pooled TOT:DK ratio over the six primary models, before and
after dropping the items whose uniqueness certificate is lost under the
prompt (paraphrased) gloss -- i.e. unique_orig true and unique false.

Conventions follow analyze_pilot.classify / analysis_common.PRIMARY6, exactly as
every other ratio in the paper.

  python -u recert_sens.py
"""
from __future__ import annotations

import json
import os

from analysis_common import PRIMARY6, PRETTY, ROOT, classify, load_jsonl

DATA = os.path.join(ROOT, "data")


def lost_ids(lang):
    path = os.path.join(DATA, f"items_recert_{lang}_judged.jsonl")
    out, seen = set(), {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                seen[r["item_id"]] = r
    for iid, r in seen.items():
        if r.get("unique_orig") and not r.get("unique"):
            out.add(iid)
    return out


def counts(lang, drop=frozenset()):
    items = list(load_jsonl(os.path.join(
        ROOT, "data", f"items_full_{lang}_final.jsonl")))
    per = {}
    for tag in PRIMARY6:
        base = os.path.join(ROOT, "results", f"full_{lang}_{tag}")
        prod = load_jsonl(os.path.join(base, "production.jsonl"))
        rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
        t = d = 0
        for iid in items:
            if iid in drop:
                continue
            st = classify(prod.get(iid), rec.get(iid))
            if st == "TOT":
                t += 1
            elif st == "DK":
                d += 1
        per[tag] = (t, d)
    return per


def show(lang, per, label):
    T = sum(v[0] for v in per.values())
    D = sum(v[1] for v in per.values())
    print(f"  {label:<22} TOT {T:>5}  DK {D:>4}  ratio {T / D:>6.2f}:1  "
          f"P(TOT|cls) {T / (T + D):.3f}")
    for tag in PRIMARY6:
        t, d = per[tag]
        print(f"      {PRETTY[tag]:<16} {t:>5} / {d:<4} = "
              f"{(t / d if d else float('nan')):.2f}")
    return T, D


def main():
    for lang in ("en", "zh"):
        drop = lost_ids(lang)
        print(f"\n[{lang}] YES->NO items dropped: {len(drop)}")
        before = counts(lang)
        after = counts(lang, drop)
        T0, D0 = show(lang, before, "before (full bank)")
        T1, D1 = show(lang, after, "after (excl. YES->NO)")
        print(f"  DELTA ratio {T0 / D0:.2f} -> {T1 / D1:.2f} "
              f"({(T1 / D1) - (T0 / D0):+.2f}); "
              f"P(TOT|cls) {T0 / (T0 + D0):.3f} -> {T1 / (T1 + D1):.3f}")


if __name__ == "__main__":
    main()
