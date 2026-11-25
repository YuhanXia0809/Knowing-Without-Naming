"""Population-matched decoy-cue arm.

As first run, the decoy-cue rate is computed over ALL usable decoy items,
while the real-target rate it is set against is computed only over cleanly
classified failures (TOT + DK).  The mismatch runs in the authors' favour:
on an item the model CAN retrieve, a strong stored competitor for that
gloss is available to suppress production of the decoy, so including such
items depresses the decoy rate and widens the margin.

This script recomputes both sides of the contrast on ONE population: the
decoy items whose source bank item is a cleanly classified failure for
that same model.  It also reports the full-population rate so the size and
direction of the mismatch are visible.

  python -u decoycue_match.py --out ../results/decoycue_match.txt
"""
from __future__ import annotations

import argparse
import os
import sys

from analysis_common import CORE, PRETTY, ROOT, classify, load_jsonl

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def unlocked(r):
    return bool(r["greedy_correct"]) or r["k_cue"] > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "decoycue_match.txt"))
    args = ap.parse_args()
    say("=" * 100)
    say("DECOY-CUE ARM ON THE MATCHED POPULATION")
    say("=" * 100)
    say("'all' = every usable decoy item.  'matched' = decoy items whose "
        "source bank item is a cleanly classified failure (TOT or DK) for "
        "that model -- the population the real-target rate is computed on.  "
        "'real' = the same model's real-target cue unlock on exactly those "
        "source items.  All rates are any-of-20 (greedy OR >=1/20).")
    say(f"\n{'cell':22s} {'n_all':>6s} {'decoy_all':>10s} {'n_match':>8s} "
        f"{'decoy_match':>12s} {'real_match':>11s} {'margin':>7s}")

    for lang in ("en", "zh"):
        dec = {it["item_id"]: it for it in
               load_jsonl(os.path.join(ROOT, "data",
                                       f"items_decoy_{lang}.jsonl")).values()}
        for tag in CORE:
            cdir = os.path.join(ROOT, "results", f"decoycue_{lang}_{tag}")
            base = os.path.join(ROOT, "results", f"full_{lang}_{tag}")
            cp = os.path.join(cdir, "cue.jsonl")
            if not os.path.exists(cp):
                continue
            prod = load_jsonl(os.path.join(base, "production.jsonl"))
            rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
            realcue = load_jsonl(os.path.join(base, "cue.jsonl"))
            # full-bank cue records are keyed by item only; keep the
            # correct-cue condition
            realc = {}
            with open(os.path.join(base, "cue.jsonl"), encoding="utf-8") as f:
                import json as _j
                for line in f:
                    if not line.strip():
                        continue
                    r = _j.loads(line)
                    if r.get("condition") == "correct":
                        realc[r["item_id"]] = r
            del realcue

            n_all = k_all = 0
            n_m = k_m = 0
            n_r = k_r = 0
            with open(cp, encoding="utf-8") as f:
                import json as _j
                for line in f:
                    if not line.strip():
                        continue
                    r = _j.loads(line)
                    if r.get("condition") != "correct":
                        continue
                    n_all += 1
                    k_all += 1 if unlocked(r) else 0
                    it = dec.get(r["item_id"])
                    if not it:
                        continue
                    src = it["true_item_id"]
                    st = classify(prod.get(src), rec.get(src))
                    if st not in ("TOT", "DK"):
                        continue
                    n_m += 1
                    k_m += 1 if unlocked(r) else 0
                    rr = realc.get(src)
                    if rr is not None:
                        n_r += 1
                        k_r += 1 if unlocked(rr) else 0
            da = k_all / n_all if n_all else float("nan")
            dm = k_m / n_m if n_m else float("nan")
            rm = k_r / n_r if n_r else float("nan")
            say(f"{lang}/{PRETTY[tag]:16s} {n_all:6d} {da:10.3f} {n_m:8d} "
                f"{dm:12.3f} {rm:11.3f} {(rm / dm if dm else float('nan')):6.2f}x")

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(("\n".join(OUT) + "\n").rstrip("\n") + "\n")
    print("wrote", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
