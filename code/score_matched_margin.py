"""Scoring-matched decoy-vs-target margin check.

One reading makes the decoy-target margin a scoring artifact: the real-target arm is credited against the vetted
acceptable-answer SET (mean 2.8 En / 1.9 Zh) while the decoy arm is
credited on a single string. Restricting BOTH sides of the matched
comparison to source items whose answer set contains exactly one word
makes the two arms scoring-symmetric by construction (target credit =
exact string, same as decoy credit). If the margin holds on that subset,
the mismatch explains none of it; if it collapses, the reported margin
is inflated and must be corrected.

  python -u score_matched_margin.py --out ../results/score_matched_margin.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from analysis_common import ROOT, classify, load_jsonl

CORE = ["qwen7b", "qwen72b", "llama8b", "llama70b"]
PRETTY = {"qwen7b": "Qwen2.5-7B", "qwen72b": "Qwen2.5-72B",
          "llama8b": "Llama-3.1-8B", "llama70b": "Llama-3.1-70B"}
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
                                                  "score_matched_margin.txt"))
    args = ap.parse_args()

    say("=" * 96)
    say("SCORING-MATCHED DECOY-VS-TARGET MARGIN (|answer set| = 1 subset)")
    say("=" * 96)
    say("Same matched population as decoycue_match (decoy items whose")
    say("source item is a cleanly classified failure), additionally split by")
    say("whether the source item's vetted answer set is a single word (=")
    say("target credit is exact-string, scoring-symmetric with the decoy arm).")

    for lang in ("en", "zh"):
        items = {it["item_id"]: it for it in
                 load_jsonl(os.path.join(
                     ROOT, "data", f"items_full_{lang}_final.jsonl")).values()}
        n1 = sum(1 for it in items.values()
                 if len(it.get("answers") or [it["word"]]) == 1)
        say(f"\n--- {lang.upper()}: {n1}/{len(items)} bank items have "
            f"|answers| = 1")
        dec = {it["item_id"]: it for it in
               load_jsonl(os.path.join(ROOT, "data",
                                       f"items_decoy_{lang}.jsonl")).values()}
        say(f"{'cell':22s} {'nA1':>5s} {'decoy_A1':>9s} {'real_A1':>8s} "
            f"{'marg_A1':>8s} | {'nA2+':>5s} {'decoy_A2+':>9s} "
            f"{'real_A2+':>8s} {'marg_A2+':>8s}")
        for tag in CORE:
            cdir = os.path.join(ROOT, "results", f"decoycue_{lang}_{tag}")
            base = os.path.join(ROOT, "results", f"full_{lang}_{tag}")
            cp = os.path.join(cdir, "cue.jsonl")
            if not os.path.exists(cp):
                continue
            prod = load_jsonl(os.path.join(base, "production.jsonl"))
            rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
            realc = {}
            with open(os.path.join(base, "cue.jsonl"), encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("condition") == "correct":
                            realc[r["item_id"]] = r
            acc = {1: [0, 0, 0, 0], 2: [0, 0, 0, 0]}  # nd,kd,nr,kr
            with open(cp, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r.get("condition") != "correct":
                        continue
                    it = dec.get(r["item_id"])
                    if not it:
                        continue
                    src = it["true_item_id"]
                    st = classify(prod.get(src), rec.get(src))
                    if st not in ("TOT", "DK"):
                        continue
                    srcit = items.get(src)
                    if srcit is None:
                        continue
                    nans = len(srcit.get("answers") or [srcit["word"]])
                    b = 1 if nans == 1 else 2
                    acc[b][0] += 1
                    acc[b][1] += 1 if unlocked(r) else 0
                    rr = realc.get(src)
                    if rr is not None:
                        acc[b][2] += 1
                        acc[b][3] += 1 if unlocked(rr) else 0
            row = []
            for b in (1, 2):
                nd, kd, nr, kr = acc[b]
                d = kd / nd if nd else float("nan")
                t = kr / nr if nr else float("nan")
                m = t / d if (nd and kd) else float("nan")
                row.append((nd, d, t, m))
            say(f"{lang}/{PRETTY[tag]:16s} {row[0][0]:5d} {row[0][1]:9.3f} "
                f"{row[0][2]:8.3f} {row[0][3]:7.2f}x | {row[1][0]:5d} "
                f"{row[1][1]:9.3f} {row[1][2]:8.3f} {row[1][3]:7.2f}x")

        # pooled over the four cells
        for b, lab in ((1, "|A|=1"), (2, "|A|>=2")):
            nd = kd = nr = kr = 0
            for tag in CORE:
                cdir = os.path.join(ROOT, "results", f"decoycue_{lang}_{tag}")
                base = os.path.join(ROOT, "results", f"full_{lang}_{tag}")
                cp = os.path.join(cdir, "cue.jsonl")
                if not os.path.exists(cp):
                    continue
                prod = load_jsonl(os.path.join(base, "production.jsonl"))
                rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
                realc = {}
                with open(os.path.join(base, "cue.jsonl"),
                          encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            r = json.loads(line)
                            if r.get("condition") == "correct":
                                realc[r["item_id"]] = r
                with open(cp, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        if r.get("condition") != "correct":
                            continue
                        it = dec.get(r["item_id"])
                        if not it:
                            continue
                        src = it["true_item_id"]
                        st = classify(prod.get(src), rec.get(src))
                        if st not in ("TOT", "DK"):
                            continue
                        srcit = items.get(src)
                        if srcit is None:
                            continue
                        nans = len(srcit.get("answers") or [srcit["word"]])
                        if (1 if nans == 1 else 2) != b:
                            continue
                        nd += 1
                        kd += 1 if unlocked(r) else 0
                        rr = realc.get(src)
                        if rr is not None:
                            nr += 1
                            kr += 1 if unlocked(rr) else 0
            say(f"{lang} pooled {lab:7s}: decoy {kd}/{nd} = "
            f"{(kd / nd if nd else float('nan')):.3f}   target {kr}/{nr} = "
            f"{(kr / nr if nr else float('nan')):.3f}   margin "
            f"{((kr / nr) / (kd / nd) if nd and kd and nr else float('nan')):.2f}x")

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(("\n".join(OUT) + "\n").rstrip("\n") + "\n")
    print("wrote", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
