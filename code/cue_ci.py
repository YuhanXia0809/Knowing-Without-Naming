"""Item-cluster bootstrap intervals for the pooled cue-arm rates in the
cue table's caption, following the inference policy that pooled
quantities on the shared bank are item-clustered.

  python -u cue_ci.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_common import PRIMARY6, ROOT, classify, load_jsonl   # noqa: E402


def main():
    rng = np.random.default_rng(26)
    for lang in ("en", "zh"):
        rows = {"TOT": {}, "DK": {}}   # state -> item -> [outcomes]
        for tag in PRIMARY6:
            base = os.path.join(ROOT, "results", f"full_{lang}_{tag}")
            prod = load_jsonl(os.path.join(base, "production.jsonl"))
            rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
            with open(os.path.join(base, "cue.jsonl"), encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r.get("condition") != "correct":
                        continue
                    st = classify(prod.get(r["item_id"]), rec.get(r["item_id"]))
                    if st not in ("TOT", "DK"):
                        continue
                    y = 1 if (r["greedy_correct"] or r["k_cue"] > 0) else 0
                    rows[st].setdefault(r["item_id"], []).append(y)
        for st in ("TOT", "DK"):
            items = sorted(rows[st])
            k = sum(sum(v) for v in rows[st].values())
            n = sum(len(v) for v in rows[st].values())
            boots = []
            for _ in range(2000):
                take = rng.integers(0, len(items), len(items))
                kk = nn = 0
                for i in take:
                    v = rows[st][items[i]]
                    kk += sum(v)
                    nn += len(v)
                boots.append(kk / nn)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"{lang} {st}: {k}/{n} = {k/n:.3f}  cluster-bootstrap "
                  f"[{lo:.3f}, {hi:.3f}]  ({len(items)} items)")


if __name__ == "__main__":
    main()
