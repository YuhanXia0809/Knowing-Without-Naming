"""Inference for the fit-matched decoy-vs-target contrast.

The fit>=.8 decoy-vs-target comparison is otherwise a point estimate
with no interval or test. This computes a source-item
cluster bootstrap (2,000 resamples) for the rate difference and ratio,
per language, on exactly the fit_gradient fit>=.8 population.

  python -u fit_ci.py --out ../results/fit_ci.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_common import CORE, ROOT, classify, load_jsonl   # noqa: E402

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "fit_ci.txt"))
    args = ap.parse_args()

    say("=" * 96)
    say(f"SOURCE-ITEM CLUSTER BOOTSTRAP FOR THE FIT>={args.thr} "
        f"DECOY-VS-TARGET CONTRAST ({args.reps} resamples)")
    say("=" * 96)

    rng = np.random.default_rng(args.seed)
    for lang in ("en", "zh"):
        dec = {it["item_id"]: it for it in
               load_jsonl(os.path.join(ROOT, "data",
                                       f"items_decoy_{lang}.jsonl")).values()}
        dec_rows = []   # (src, y)
        tgt_rows = []   # (src, y)
        for tag in CORE:
            base = os.path.join(ROOT, "results", f"full_{lang}_{tag}")
            prod = load_jsonl(os.path.join(base, "production.jsonl"))
            rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
            fit = {}
            bdir = os.path.join(ROOT, "results", f"decoyb_{lang}_{tag}")
            for r in load_jsonl(os.path.join(bdir,
                                             "recognition.jsonl")).values():
                rots = r.get("rotations") or []
                ps = [x.get("p_target") for x in rots
                      if x.get("p_target") is not None]
                mv = r.get("mean_p_target")
                if mv is None and ps:
                    mv = sum(ps) / len(ps)
                fit[r["item_id"].replace("decoyb_", "decoy_", 1)] = mv
            realc = {}
            with open(os.path.join(base, "cue.jsonl"), encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("condition") == "correct":
                            realc[r["item_id"]] = r
            cp = os.path.join(ROOT, "results",
                              f"decoycue_{lang}_{tag}", "cue.jsonl")
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
                    if classify(prod.get(src), rec.get(src)) \
                            not in ("TOT", "DK"):
                        continue
                    fv = fit.get(r["item_id"])
                    if fv is None or fv < args.thr:
                        continue
                    dec_rows.append((src, 1 if (r["greedy_correct"] or
                                                r["k_cue"] > 0) else 0))
                    rr = realc.get(src)
                    if rr is not None:
                        tgt_rows.append((src, 1 if (rr["greedy_correct"] or
                                                    rr["k_cue"] > 0) else 0))

        srcs = sorted({s for s, _ in dec_rows} | {s for s, _ in tgt_rows})
        by_src = {s: ([], []) for s in srcs}
        for s, y in dec_rows:
            by_src[s][0].append(y)
        for s, y in tgt_rows:
            by_src[s][1].append(y)

        dr = np.mean([y for _, y in dec_rows])
        tr = np.mean([y for _, y in tgt_rows])
        say(f"\n--- {lang.upper()}  fit>={args.thr}: decoy {dr:.3f} "
            f"({len(dec_rows)} rows) vs target {tr:.3f} ({len(tgt_rows)} "
            f"rows), {len(srcs)} source items")

        diffs, ratios = [], []
        idx = np.arange(len(srcs))
        for _ in range(args.reps):
            take = rng.integers(0, len(srcs), len(srcs))
            dks = dns = tks = tns = 0
            for k in take:
                dy, ty = by_src[srcs[k]]
                dks += sum(dy)
                dns += len(dy)
                tks += sum(ty)
                tns += len(ty)
            if dns and tns and dks:
                d = tks / tns - dks / dns
                diffs.append(d)
                ratios.append((tks / tns) / (dks / dns))
        lo_d, hi_d = np.percentile(diffs, [2.5, 97.5])
        lo_r, hi_r = np.percentile(ratios, [2.5, 97.5])
        pd_ = float(np.mean(np.array(diffs) <= 0))
        say(f"  target-minus-decoy difference {tr - dr:+.3f}  "
        f"95% CI [{lo_d:+.3f}, {hi_d:+.3f}]  P(diff<=0) = {pd_:.4f}")
        say(f"  ratio {tr / dr:.2f}x  95% CI [{lo_r:.2f}, {hi_r:.2f}]")

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(("\n".join(OUT) + "\n").rstrip("\n") + "\n")
    print("wrote", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
