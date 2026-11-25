"""Does pure renormalization explain cue-unlocked production?

The constrained-search / renormalization account says: the target already
carries small sub-threshold probability mass p pre-cue (0/21 bounds it at
~.139); the first-unit cue merely filters the model's candidate
distribution to cue-consistent forms and renormalizes, amplifying p by
roughly 1/w, where w is the pre-cue mass on cue-consistent candidates.

Three deliverables, all from existing records (production.jsonl sample
strings + cue.jsonl flags):

  (1) DESCRIPTIVE. Per failed item, w-hat = share of the 20 pre-cue
      samples whose first unit matches the correct cue. Gives the
      amplification the account has to work with (~1/w).
  (2) IDENTIFICATION. The pooled per-draw post-cue target share implied
      by observed unlock (q = 1-(1-rate)^(1/21)) and the pre-cue mass p
      the account needs (q = p/(p+(1-p)w)). Reported per language.
  (3) THE SIGNED TEST. Renormalization predicts unlock DECREASES in
      w-hat (more consistent competitors -> less amplification), holding
      item covariates fixed. Storage-style access predicts ~no
      dependence. Logistic GLM: unlocked ~ z(w-hat) + z(Zipf) +
      z(length) + z(tokens) + dataset FE, item-clustered SEs, per
      language, on the correct-cue arm of the matched population
      (strict TOT/DK failures). Sign and magnitude decide.

  python -u renorm_test.py --out ../results/renorm_test.txt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from analysis_common import PRIMARY6, ROOT, classify, load_jsonl

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def first_unit(word, lang):
    w = (word or "").strip().strip('"“”。，.,;:!?()').lower()
    if not w:
        return None
    if lang == "en":
        for ch in w:
            if ch.isalpha():
                return ch
        return None
    return w[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "renorm_test.txt"))
    args = ap.parse_args()

    say("=" * 96)
    say("RENORMALIZATION BASELINE FOR CUE-UNLOCKED PRODUCTION")
    say("=" * 96)

    for lang in ("en", "zh"):
        rows = []
        for tag in PRIMARY6:
            base = os.path.join(ROOT, "results", f"full_{lang}_{tag}")
            prod = load_jsonl(os.path.join(base, "production.jsonl"))
            rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
            items = load_jsonl(os.path.join(
                ROOT, "data", f"items_full_{lang}_final.jsonl"))
            import json
            with open(os.path.join(base, "cue.jsonl"), encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r.get("condition") != "correct":
                        continue
                    iid = r["item_id"]
                    st = classify(prod.get(iid), rec.get(iid))
                    if st not in ("TOT", "DK"):
                        continue
                    pr = prod.get(iid)
                    it = items.get(iid)
                    if pr is None or it is None:
                        continue
                    samples = pr.get("samples") or []
                    if not samples:
                        continue
                    cue = (r.get("cue") or "").strip().lower()
                    if not cue:
                        continue
                    fus = [first_unit(x, lang) for x in samples]
                    fus = [x for x in fus if x]
                    if not fus:
                        continue
                    w_hat = sum(1 for x in fus if x == cue) / len(fus)
                    rows.append(dict(
                        item_id=iid, dataset=f"{lang}_{tag}", state=st,
                        unlocked=1.0 if (r["greedy_correct"] or
                                         r["k_cue"] > 0) else 0.0,
                        w_hat=w_hat, zipf=it.get("zipf") or 0.0,
                        length=it.get("n_letters") or len(it["word"]),
                        tokens=it.get("tok_qwen") or it.get("n_tokens") or 0,
                    ))
        df = pd.DataFrame(rows)
        n = len(df)
        say(f"\n--- {lang.upper()}  n={n} correct-cue rows "
            f"({df['item_id'].nunique()} items, {df['dataset'].nunique()} "
            f"datasets)")

        # (1) descriptive
        say(f"  (1) w-hat: mean {df.w_hat.mean():.3f}  median "
            f"{df.w_hat.median():.3f}  zero-share "
            f"{(df.w_hat == 0).mean():.3f}  P90 "
            f"{df.w_hat.quantile(.9):.3f}")
        say(f"      unlock rate: {df.unlocked.mean():.3f}  "
            f"(TOT {df[df.state=='TOT'].unlocked.mean():.3f} / DK "
            f"{df[df.state=='DK'].unlocked.mean():.3f})")

        # (2) identification
        rate = df.unlocked.mean()
        q = 1 - (1 - rate) ** (1 / 21)
        wbar = df.w_hat.mean()
        # p solving q = p / (p + (1-p) wbar)
        p_need = q * wbar / (1 - q + q * wbar)
        say(f"  (2) implied per-draw post-cue target share q = {q:.4f}; "
            f"with mean w-hat {wbar:.3f} the account needs pre-cue mass "
            f"p = {p_need:.4f} (rule-of-three ceiling .139)")
        # ceiling prediction
        qc = 0.139 / (0.139 + (1 - 0.139) * np.maximum(df.w_hat, 1e-9))
        pred_ceil = float(np.mean(1 - (1 - qc) ** 21))
        say(f"      at the p=.139 ceiling the account predicts mean unlock "
            f"{pred_ceil:.3f} vs observed {rate:.3f}")

        # (3) signed test
        d = df.copy()
        zcols = []
        for c in ("w_hat", "zipf", "length", "tokens"):
            sd = d[c].std()
            if sd > 0:
                d["z_" + c] = (d[c] - d[c].mean()) / sd
                zcols.append("z_" + c)
        X = pd.get_dummies(d["dataset"], drop_first=True).astype(float)
        X = pd.concat([d[zcols], X], axis=1)
        X = sm.add_constant(X)
        fit = sm.GLM(d["unlocked"], X,
                     family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": d["item_id"]})
        b = fit.params["z_w_hat"]
        se = fit.bse["z_w_hat"]
        pv = fit.pvalues["z_w_hat"]
        say(f"  (3) unlocked ~ z(w-hat)+covs+dataset FE, item-clustered:")
        say(f"      b(z_w-hat) = {b:+.3f} (SE {se:.3f})  95% CI "
            f"[{b - 1.96 * se:+.3f}, {b + 1.96 * se:+.3f}]  p = {pv:.4g}")
        say(f"      renormalization predicts strongly NEGATIVE; "
            f"storage-style access predicts ~0.")
        # raw stratified check, no model
        for lo, hi, lab in ((0, 0, "w=0"), (0.0001, 0.10, "0<w<=.10"),
                            (0.1001, 1.0, "w>.10")):
            m = (df.w_hat >= lo) & (df.w_hat <= hi) if lo else \
                (df.w_hat == 0)
            sub = df[m]
            if len(sub):
                say(f"      stratum {lab:11s}: n={len(sub):5d}  unlock "
                    f"{sub.unlocked.mean():.3f}")

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(("\n".join(OUT) + "\n").rstrip("\n") + "\n")
    print("wrote", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
