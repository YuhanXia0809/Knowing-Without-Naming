"""Three hardening analyses.

  A. Item-cluster bootstrap CIs for the joint ratios on the literal
     denominator. Replicates the joint-criterion classification exactly,
     then bootstraps items.

  B. Wilson 95% intervals for every rate in the composition series
     (pseudoword-cue -> decoy-cue -> stored target, per cell and pooled).

  C. OpenCC t2s headword-collision count for the Chinese bank, replicating
     load_moedict()'s first-wins map.

  python -u hardening.py --reps 2000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_common import ROOT, classify, load_jsonl          # noqa: E402
from paths import items_path                          # noqa: E402

CONF = [("qwen7b", "Qwen2.5-7B"), ("qwen72b", "Qwen2.5-72B"),
        ("llama8b", "Llama-3.1-8B"), ("llama70b", "Llama-3.1-70B")]

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def wilson(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - hw, c + hw


# ---------------------------------------------------------------- part A ---
def part_a(reps, seed):
    say("=" * 92)
    say("A. LITERAL-DENOMINATOR (DK_jnt) JOINT RATIOS: item-cluster "
        "bootstrap 95% CIs")
    say("=" * 92)
    say(f"reps={reps} seed={seed}; DK_jnt = strict failure failing either "
        "control (joint-criterion semantics, replicated per item).")
    rng = np.random.default_rng(seed)
    for lang in ("en", "zh"):
        items = list(load_jsonl(items_path(lang)))
        idx = {iid: k for k, iid in enumerate(items)}
        n = len(items)
        cells = {}
        for tag, label in CONF:
            base = os.path.join(ROOT, "results", f"full_{lang}_{tag}")
            adv = os.path.join(ROOT, "results", f"adv_{lang}_{tag}")
            prod = load_jsonl(os.path.join(base, "production.jsonl"))
            arec = load_jsonl(os.path.join(adv, "recognition.jsonl"))
            rev = load_jsonl(os.path.join(base, "reverse_judged.jsonl"))
            tj = np.zeros(n)
            dj = np.zeros(n)
            for iid in items:
                s_adv = classify(prod.get(iid), arec.get(iid))
                if s_adv not in ("TOT", "DK", "ambiguous_rec"):
                    continue
                dc = rev.get(iid, {}).get("def_correct")
                if dc is None:
                    continue
                if s_adv == "TOT" and dc:
                    tj[idx[iid]] = 1
                else:
                    dj[idx[iid]] = 1
            cells[label] = (tj, dj)

        boots = {lab: np.empty(reps) for lab in cells}
        boots['POOLED'] = np.empty(reps)
        for b in range(reps):
            w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            T = D = 0.0
            for lab, (tj, dj) in cells.items():
                t, d = tj @ w, dj @ w
                boots[lab][b] = t / d if d else np.nan
                T += t
                D += d
            boots['POOLED'][b] = T / D if D else np.nan

        say(f"\n--- {lang.upper()}")
        for lab in list(cells) + ['POOLED']:
            if lab == 'POOLED':
                t = sum(v[0].sum() for v in cells.values())
                d = sum(v[1].sum() for v in cells.values())
            else:
                t, d = cells[lab][0].sum(), cells[lab][1].sum()
            x = boots[lab]
            x = x[np.isfinite(x)]
            lo, hi = np.percentile(x, [2.5, 97.5])
            excl = 'excludes parity' if (hi < 1 or lo > 1) else 'covers parity'
            say(f"  {lab:14s} TOTjnt={int(t):4d} DKjnt={int(d):4d} "
                f"r={t / d:5.2f}  CI[{lo:5.2f},{hi:5.2f}]  ({excl})")


# ---------------------------------------------------------------- part B ---
def unlocked(r):
    return bool(r["greedy_correct"]) or r["k_cue"] > 0


def cue_counts(path):
    k = n = 0
    if not os.path.exists(path):
        return 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("condition") != "correct":
                continue
            n += 1
            k += 1 if unlocked(r) else 0
    return k, n


def part_b():
    say("")
    say("=" * 92)
    say("B. COMPOSITION SERIES: Wilson 95% intervals for every rate "
        "(any-of-20, correct-cue arm)")
    say("=" * 92)
    say(f"{'cell':22s} {'pseudo k/n':>12s} {'Wilson':>15s} | "
        f"{'decoy k/n':>10s} {'Wilson':>15s} | {'target k/n':>10s} "
        f"{'Wilson':>15s}   (decoy+target = matched population)")
    for lang in ("en", "zh"):
        dec = {it["item_id"]: it for it in
               load_jsonl(os.path.join(ROOT, "data",
                                       f"items_decoy_{lang}.jsonl")).values()}
        agg = dict(pk=0, pn=0, dk=0, dn=0, tk=0, tn=0)
        for tag, label in CONF:
            pk, pn = cue_counts(os.path.join(
                ROOT, "results", f"pseudocue_{lang}_{tag}", "cue.jsonl"))
            base = os.path.join(ROOT, "results", f"full_{lang}_{tag}")
            prod = load_jsonl(os.path.join(base, "production.jsonl"))
            rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
            realc = {}
            with open(os.path.join(base, "cue.jsonl"), encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("condition") == "correct":
                            realc[r["item_id"]] = r
            dk = dn = tk = tn = 0
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
                    if classify(prod.get(src), rec.get(src)) not in ("TOT", "DK"):
                        continue
                    dn += 1
                    dk += 1 if unlocked(r) else 0
                    rr = realc.get(src)
                    if rr is not None:
                        tn += 1
                        tk += 1 if unlocked(rr) else 0
            for key, val in (('pk', pk), ('pn', pn), ('dk', dk), ('dn', dn),
                             ('tk', tk), ('tn', tn)):
                agg[key] += val
            f1 = wilson(pk, pn)
            f2 = wilson(dk, dn)
            f3 = wilson(tk, tn)
            say(f"{lang}/{label:16s} {pk:4d}/{pn:<6d} [{f1[0]:.4f},{f1[1]:.4f}] | "
                f"{dk:3d}/{dn:<5d} [{f2[0]:.3f},{f2[1]:.3f}] | "
                f"{tk:3d}/{tn:<5d} [{f3[0]:.3f},{f3[1]:.3f}]")
        f1 = wilson(agg['pk'], agg['pn'])
        f2 = wilson(agg['dk'], agg['dn'])
        f3 = wilson(agg['tk'], agg['tn'])
        say(f"{lang}/POOLED{'':10s} {agg['pk']:4d}/{agg['pn']:<6d} "
            f"[{f1[0]:.4f},{f1[1]:.4f}] | "
            f"{agg['dk']:3d}/{agg['dn']:<5d} [{f2[0]:.3f},{f2[1]:.3f}] | "
            f"{agg['tk']:3d}/{agg['tn']:<5d} [{f3[0]:.3f},{f3[1]:.3f}]")


# ---------------------------------------------------------------- part C ---
def part_c():
    say("")
    say("=" * 92)
    say("C. OPENCC t2s HEADWORD-COLLISION AUDIT")
    say("=" * 92)
    import re

    try:
        import opencc
    except ImportError:
        say("  skipped: part C needs opencc-python-reimplemented and the "
            "upstream dict-revised.json (see README, optional dependencies)")
        return False
    han = re.compile(r"^[一-鿿]+$")
    t2s = opencc.OpenCC("t2s")
    path = os.path.join(os.environ.get("KWN_RESOURCES", os.path.join(
        ROOT, "resources")), "moedict", "dict-revised.json")
    if not os.path.exists(path):
        say("  skipped: upstream dict-revised.json not found at %s "
            "(set KWN_RESOURCES; see README)" % path)
        return False
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    gloss_map = {}
    dropped = 0
    collided_keys = set()
    considered = 0
    for entry in data:
        title_t = entry.get("title", "")
        if not (2 <= len(title_t) <= 4) or not han.match(title_t):
            continue
        all_defs = [d.get("def", "").strip()
                    for h in entry.get("heteronyms", [])
                    for d in h.get("definitions", [])
                    if d.get("def", "").strip()]
        if not all_defs:
            continue
        word_s = t2s.convert(title_t)
        if not (2 <= len(word_s) <= 4) or not han.match(word_s):
            continue
        considered += 1
        if word_s in gloss_map:
            dropped += 1
            collided_keys.add(word_s)
        else:
            gloss_map[word_s] = title_t
    say(f"entries passing the builder's filters : {considered}")
    say(f"kept (first occurrence per simplified key): {len(gloss_map)}")
    say(f"dropped by first-wins collision        : {dropped} "
        f"({dropped / considered:.2%}) over {len(collided_keys)} keys")

    items = load_jsonl(os.path.join(ROOT, "data", "items_full_zh_final.jsonl"))
    hit = [it["word"] for it in items.values() if it["word"] in collided_keys]
    say(f"of the 903 bank items, words whose key had >=1 dropped competitor: "
        f"{len(hit)} ({len(hit) / len(items):.2%})")
    if hit:
        say("  affected items: " + " ".join(sorted(hit)[:40]) +
            (" ..." if len(hit) > 40 else ""))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "hardening.txt"))
    args = ap.parse_args()
    part_a(args.reps, args.seed)
    part_b()
    if not part_c():
        # without part C the file would lose numbers the paper cites
        print("not written: %s (part C skipped; the stored file keeps "
              "all three parts)" % args.out)
        return
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(("\n".join(OUT) + "\n").rstrip("\n") + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
