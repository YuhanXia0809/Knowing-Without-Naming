"""Signal-detection analysis of the self-report vs behavioral-state
dissociation.

For every results dir full_{lang}_{model} with selfreport.jsonl:

  A. Failure-conditional discrimination (the Gollan step-2 analogue):
     among strict-failure items classified TOT or DK, does self-reported
     TOT distinguish behavioral TOT from DK?
       2x2: [behav TOT, behav DK] x [self TOT, self not-TOT]
       -> hit rate, FA rate, d' (log-linear 0.5 correction), odds ratio
          with Fisher exact p and Woolf CI.

  B. All-item discrimination: among all classified items (known/TOT/DK),
     does self-reported TOT pick out behavioral TOT vs everything else?
       Same statistics, plus phi.

Also dumps the raw 2x2 counts (e.g. how many of a model's self-reported
TOTs are behavioral TOT).

  python selfreport_sdt.py --out ../results/selfreport_sdt.json
"""
from __future__ import annotations

import argparse
import json
import math
import os

from scipy import stats as sps

from analyze_pilot import classify, load_jsonl

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

MODELS = ["qwen7b", "qwen14b", "qwen32b", "qwen72b", "llama8b", "llama70b",
          "gemma12b", "gemma27b", "glm9b", "mistral7b", "olmo13b", "olmo32b",
          "phi4", "r1d14b"]


def _z(p):
    return sps.norm.ppf(p)


def sdt(n_hit, n_miss, n_fa, n_cr):
    """d' with log-linear correction; OR with Woolf CI + Fisher exact p."""
    hr = (n_hit + 0.5) / (n_hit + n_miss + 1.0)
    fa = (n_fa + 0.5) / (n_fa + n_cr + 1.0)
    dprime = _z(hr) - _z(fa)
    a, b, c, d = n_hit + 0.5, n_miss + 0.5, n_fa + 0.5, n_cr + 0.5
    or_ = (a * d) / (b * c)
    se_log = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ci = (math.exp(math.log(or_) - 1.96 * se_log),
          math.exp(math.log(or_) + 1.96 * se_log))
    _, p = sps.fisher_exact([[n_hit, n_miss], [n_fa, n_cr]])
    return {"hit_rate": round(hr, 4), "fa_rate": round(fa, 4),
            "dprime": round(dprime, 3), "odds_ratio": round(or_, 3),
            "or_ci": [round(ci[0], 3), round(ci[1], 3)],
            "fisher_p": round(p, 5),
            "counts": {"hit": n_hit, "miss": n_miss, "fa": n_fa, "cr": n_cr}}


def analyze_dir(base, items):
    prod = load_jsonl(os.path.join(base, "production.jsonl"))
    rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
    selfr = load_jsonl(os.path.join(base, "selfreport.jsonl"))
    st, sr = {}, {}
    for iid in items:
        s = classify(prod.get(iid), rec.get(iid))
        if s:
            st[iid] = s
        if iid in selfr:
            sr[iid] = selfr[iid].get("label") == "TOT"
    ids = [i for i in st if i in sr]

    # A: failure-conditional (TOT vs DK)
    fail = [i for i in ids if st[i] in ("TOT", "DK")]
    a_hit = sum(1 for i in fail if st[i] == "TOT" and sr[i])
    a_miss = sum(1 for i in fail if st[i] == "TOT" and not sr[i])
    a_fa = sum(1 for i in fail if st[i] == "DK" and sr[i])
    a_cr = sum(1 for i in fail if st[i] == "DK" and not sr[i])

    # B: all classified items (TOT vs known+DK)
    b_hit = sum(1 for i in ids if st[i] == "TOT" and sr[i])
    b_miss = sum(1 for i in ids if st[i] == "TOT" and not sr[i])
    b_fa = sum(1 for i in ids if st[i] != "TOT" and sr[i])
    b_cr = sum(1 for i in ids if st[i] != "TOT" and not sr[i])

    n_self_tot = sum(1 for i in ids if sr[i])
    return {
        "n_classified": len(ids),
        "n_self_tot": n_self_tot,
        "self_tot_that_are_behav_tot": b_hit,
        "tot_vs_dk": sdt(a_hit, a_miss, a_fa, a_cr),
        "tot_vs_rest": sdt(b_hit, b_miss, b_fa, b_cr),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    items_by_lang = {
        lang: load_jsonl(os.path.join(ROOT, "data", f"items_full_{lang}_final.jsonl"))
        for lang in ("en", "zh")}
    report = {}
    for lang in ("en", "zh"):
        for m in MODELS:
            base = os.path.join(ROOT, "results", f"full_{lang}_{m}")
            if not os.path.isdir(base) or \
               not os.path.exists(os.path.join(base, "selfreport.jsonl")):
                continue
            key = f"full_{lang}_{m}"
            report[key] = analyze_dir(base, items_by_lang[lang])
            r = report[key]
            print(f"{key}: selfTOT={r['n_self_tot']} "
                  f"(of which behavTOT={r['self_tot_that_are_behav_tot']}); "
                  f"TOTvsDK d'={r['tot_vs_dk']['dprime']} "
                  f"OR={r['tot_vs_dk']['odds_ratio']} "
                  f"CI={r['tot_vs_dk']['or_ci']} p={r['tot_vs_dk']['fisher_p']}; "
                  f"TOTvsRest d'={r['tot_vs_rest']['dprime']} "
                  f"OR={r['tot_vs_rest']['odds_ratio']} p={r['tot_vs_rest']['fisher_p']}")
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1)
    print("written", args.out)


if __name__ == "__main__":
    main()
