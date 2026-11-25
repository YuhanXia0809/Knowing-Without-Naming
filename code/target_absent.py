"""One released summary for every target-absent control.

The decoy 4AFC (both variants), the decoy-cue arm, the pseudoword-cue
floor and the non-lexical battery were reported only in running prose.
This script emits every number the paper prints for them, from the raw
per-cell records, into a single released file.

  python -u target_absent.py --out ../results/target_absent.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from analysis_common import CORE, PRETTY, ROOT, classify, load_jsonl

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def rd(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "target_absent.txt"))
    args = ap.parse_args()
    say("=" * 96)
    say("TARGET-ABSENT CONTROLS: EVERY PRINTED NUMBER, FROM THE RAW "
        "RECORDS")
    say("=" * 96)

    # ---------------------------------------------- decoy 4AFC, both variants
    for tag_dir, label in (("decoy", "rung (ii): borrowed fillers"),
                           ("decoyb", "rung (ii-b): option-matched")):
        say(f"\n### DECOY-TARGET 4AFC -- {label}"
            f"   (consistent selection = k_rec >= 3/4; chance ~ 5%)")
        say(f"{'cell':22s} {'n':>5s} {'k>=3':>6s} {'rate':>7s} {'mean p':>7s}")
        for lang in ("en", "zh"):
            for m in CORE:
                rows = rd(os.path.join(ROOT, "results",
                                       f"{tag_dir}_{lang}_{m}",
                                       "recognition.jsonl"))
                if not rows:
                    continue
                n = len(rows)
                k = sum(1 for r in rows if r["k_rec"] >= 3)
                mp = sum(r["mean_p_target"] for r in rows) / n
                say(f"{lang}/{PRETTY[m]:16s} {n:5d} {k:6d} {k / n:7.3f} "
                    f"{mp:7.3f}")

    # ---------------------------------------------------- cue arms on decoys
    for tag_dir, label in (("decoycue", "DECOY-CUE (real word, not the "
                                        "stored answer for that gloss)"),
                           ("pseudocue", "PSEUDOWORD-CUE (novel form, no "
                                         "stored entry at all)")):
        say(f"\n### {label}   (any-of-20 = greedy OR >=1/20)")
        say(f"{'cell':22s} {'n':>5s} {'greedy':>7s} {'any20':>7s} "
            f"{'plac.g':>7s} {'plac.20':>8s}")
        for lang in ("en", "zh"):
            for m in CORE:
                rows = rd(os.path.join(ROOT, "results",
                                       f"{tag_dir}_{lang}_{m}", "cue.jsonl"))
                if not rows:
                    continue
                c = [r for r in rows if r["condition"] == "correct"]
                p = [r for r in rows if r["condition"] != "correct"]
                u = lambda R: sum(1 for r in R                     # noqa: E731
                                  if r["greedy_correct"] or r["k_cue"] > 0)
                g = lambda R: sum(1 for r in R if r["greedy_correct"])
                say(f"{lang}/{PRETTY[m]:16s} {len(c):5d} "
                    f"{g(c) / len(c):7.3f} {u(c) / len(c):7.3f} "
                    f"{(g(p) / len(p) if p else float('nan')):7.3f} "
                    f"{(u(p) / len(p) if p else float('nan')):8.3f}")

    # ------------------------------------------------- non-lexical battery
    say("\n### NON-LEXICAL CONTROL BATTERY (300 English facts; identical "
        "pipeline and classifier)")
    say("DK here is the complement of TOT within strict failures (the 2/4 "
        "recognition band included).")
    say(f"{'cell':22s} {'n':>5s} {'strictfail':>11s} {'TOT':>5s} {'DK':>5s} "
        f"{'T:D':>7s}")
    for m in CORE:
        base = os.path.join(ROOT, "results", f"nonlex_en_{m}")
        prod = load_jsonl(os.path.join(base, "production.jsonl"))
        rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
        if not prod:
            continue
        sf = tot = 0
        for iid, p in prod.items():
            if p["k_prod"] == 0 and not p["greedy_correct"]:
                sf += 1
                r = rec.get(iid)
                if r and r["k_rec"] >= 3:
                    tot += 1
        dk = sf - tot
        say(f"en/{PRETTY[m]:16s} {len(prod):5d} {sf:11d} {tot:5d} {dk:5d} "
            f"{(tot / dk if dk else float('inf')):7.2f}")

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(("\n".join(OUT) + "\n").rstrip("\n") + "\n")
    print("wrote", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
