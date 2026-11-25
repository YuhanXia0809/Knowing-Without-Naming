"""Gloss-fit gradient among decoys, the analysis half of the
gloss-fit-matched control.

One reading of the composition series: the target-over-decoy unlock margin reflects
better SEMANTIC FIT under a first-unit constraint, not access to a stored
target. Existing data lets us measure how strongly fit drives unlock among
the decoys themselves: each decoy item carries a continuous fit score (its
mean selection probability in the target-absent B' 4AFC, decoyb recognition)
and an unlock outcome (decoy-cue arm, any-of-20).

If unlock is ~flat in fit across the decoys' range, the fit gradient is too
weak for extrapolated fit to explain the stored-target rate; if steep, fit
could explain part of it. Either way it is reported.

  python -u fit_gradient.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_common import CORE, PRETTY, ROOT, classify, load_jsonl   # noqa: E402

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def main():
    import pandas as pd
    import statsmodels.formula.api as smf

    say("=" * 96)
    say("GLOSS-FIT GRADIENT AMONG DECOYS (decoy-cue unlock ~ B' fit)")
    say("=" * 96)
    say("fit = mean_p_target of the decoy in the option-matched target-absent "
        "4AFC (decoyb recognition);")
    say("unlock = decoy produced under its first-unit cue, any-of-20, correct "
        "arm; matched population (source item a clean failure).")

    for lang in ("en", "zh"):
        dec = {it["item_id"]: it for it in
               load_jsonl(os.path.join(ROOT, "data",
                                       f"items_decoy_{lang}.jsonl")).values()}
        rows = []
        tgt_rate = {}
        for tag in CORE:
            base = os.path.join(ROOT, "results", f"full_{lang}_{tag}")
            prod = load_jsonl(os.path.join(base, "production.jsonl"))
            rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
            fit = {}
            bdir = os.path.join(ROOT, "results", f"decoyb_{lang}_{tag}")
            for r in load_jsonl(os.path.join(bdir, "recognition.jsonl")).values():
                rots = r.get("rotations") or []
                ps = [x.get("p_target") for x in rots if x.get("p_target") is not None]
                mv = r.get("mean_p_target")
                if mv is None and ps:
                    mv = sum(ps) / len(ps)
                # decoyb_<lang>_<...> -> decoy_<lang>_<...> (the cue arm's ids)
                fit[r["item_id"].replace("decoyb_", "decoy_", 1)] = mv
            realc = {}
            with open(os.path.join(base, "cue.jsonl"), encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("condition") == "correct":
                            realc[r["item_id"]] = r
            tk = tn = 0
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
                    fv = fit.get(r["item_id"])
                    if fv is None:
                        continue
                    unlocked = 1 if (r["greedy_correct"] or r["k_cue"] > 0) else 0
                    rows.append(dict(item=r["item_id"], model=tag,
                                     fit=float(fv), y=unlocked))
                    rr = realc.get(src)
                    if rr is not None:
                        tn += 1
                        tk += 1 if (rr["greedy_correct"] or rr["k_cue"] > 0) else 0
            tgt_rate[tag] = (tk, tn)

        df = pd.DataFrame(rows)
        say(f"\n--- {lang.upper()}  rows={len(df)}  items={df['item'].nunique()}  "
            f"mean fit={df.fit.mean():.3f}  fit range "
            f"[{df.fit.min():.3f},{df.fit.max():.3f}]  "
            f"unlock={df.y.mean():.3f}")

        df["zfit"] = (df.fit - df.fit.mean()) / df.fit.std()
        m = smf.logit("y ~ zfit + C(model)", data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["item"]}, disp=0)
        b, se, p = (m.params["zfit"], m.bse["zfit"], m.pvalues["zfit"])
        say(f"  logit unlock ~ z(fit) + model FE, item-clustered: "
            f"b={b:+.3f} (SE {se:.3f}), p={p:.4f}")

        q = df.fit.quantile([1 / 3, 2 / 3]).values
        lo = df[df.fit <= q[0]].y.mean()
        hi = df[df.fit > q[1]].y.mean()
        say(f"  unlock by fit tercile: bottom {lo:.3f} vs top {hi:.3f} "
            f"(diff {hi - lo:+.3f})")

        tk = sum(v[0] for v in tgt_rate.values())
        tn = sum(v[1] for v in tgt_rate.values())
        say(f"  stored-target unlock on the same source items: "
            f"{tk / tn:.3f} ({tk}/{tn})")
        # per-fit-unit slope on the probability scale near the decoy mean,
        # and the fit gap the target would need to close the margin
        import math
        pbar = df.y.mean()
        slope_prob = b / df.fit.std() * pbar * (1 - pbar)
        gap = (tk / tn - pbar) / slope_prob if slope_prob > 0 else float('inf')
        say(f"  local slope ~{slope_prob:+.2f} unlock per fit unit; the "
            f"target margin would require a fit gap of ~{gap:.2f} "
            f"(observed decoy-fit spread {df.fit.max() - df.fit.min():.2f})")
        # the sharpest cut: decoys whose B' fit is essentially answer-level
        for thr in (0.8, 0.6):
            hi = df[df.fit >= thr]
            items_hi = set(hi['item'])
            srcs = {dec[i]['true_item_id'] for i in items_hi}
            tk2 = tn2 = 0
            for tag in CORE:
                base = os.path.join(ROOT, 'results', f'full_{lang}_{tag}')
                realc = {}
                with open(os.path.join(base, 'cue.jsonl'), encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            r = json.loads(line)
                            if r.get('condition') == 'correct':
                                realc[r['item_id']] = r
                for src in srcs:
                    rr = realc.get(src)
                    if rr is not None:
                        tn2 += 1
                        tk2 += 1 if (rr['greedy_correct'] or rr['k_cue'] > 0) else 0
            if len(hi):
                say(f"  fit>={thr}: decoy unlock {hi.y.mean():.3f} "
                    f"(n={len(hi)} rows, {len(items_hi)} items) vs stored "
                    f"target on those source items {tk2/tn2:.3f} "
                    f"({tk2}/{tn2})")

    out = os.path.join(ROOT, "results", "fit_gradient.txt")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(("\n".join(OUT) + "\n").rstrip("\n") + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
