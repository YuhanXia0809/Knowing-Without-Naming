"""Plain-text versions of the two sweep tables, computed from the records.

  python paper_tables.py taxonomy      # -> stdout (results/taxonomy_full.txt)
  python paper_tables.py selfreport    # -> stdout (results/selfreport_full.txt)

Row logic is that of the two tables in the paper: analyze_pilot.classify for
the states, the recognition ceiling on
known items, selfreport.jsonl labels for the self-reported TOT set,
results/selfreport_cells.csv (from selfreport_meta3.py) for the odds ratio,
and mimicry.extract_claim on perform.jsonl for the performed-TOT first-unit
claims.
"""
from __future__ import annotations

import csv
import io
import os
import sys
from itertools import product

from analyze_pilot import classify, load_jsonl
from mimicry import extract_claim

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PRIMARY = [("qwen7b", "Qwen2.5-7B"), ("qwen14b", "Qwen2.5-14B"),
           ("qwen32b", "Qwen2.5-32B"), ("qwen72b", "Qwen2.5-72B"),
           ("llama8b", "Llama-3.1-8B"), ("llama70b", "Llama-3.1-70B")]
EXTENDED = [("gemma12b", "Gemma-3-12B"), ("gemma27b", "Gemma-3-27B"),
            ("glm9b", "GLM-4-9B"), ("mistral7b", "Mistral-7B"),
            ("olmo13b", "OLMo-2-13B"), ("olmo32b", "OLMo-2-32B"),
            ("phi4", "Phi-4"), ("r1d14b", "R1-Distill-14B")]
MODELS = PRIMARY + EXTENDED

TAX_HEAD = [
    "# Taxonomy across fourteen models (six primary + eight replication), 28 datasets",
    "# State taxonomy per dataset. Fail% = strict production-failure rate (greedy"
    " wrong and 0/20 samples); Known = items produced in >= 6/20 samples; TOT ="
    " strict failures with recognition >= 3/4 (R+ in the paper); DK = strict"
    " failures with recognition <= 1/4 (R-); Ceil% = recognition accuracy on known"
    " items (instrument check, gated at 85%). Counts are items; strict failures"
    " with 2/4 recognition are in neither TOT nor DK. Computed by"
    " code/paper_tables.py.",
    "",
    "Lang\tModel\tFail%\tKnown\tTOT\tDK\tCeil%"]
SR_HEAD = [
    "# Self-report vs behavioral state, all 28 datasets",
    "# Behav. TOT = strict failures with recognition >= 3/4; Self-rep. TOT ="
    " items the model self-reports as TOT; OR(TOT/DK) = odds ratio for"
    " self-reported TOT discriminating behavioral TOT from DK among strict"
    " failures (log-linear 0.5 correction; * = Fisher exact p < .05);"
    " n/a^dagger = no self-reports inside that set, so no odds ratio is defined;"
    " Jaccard = overlap of the two TOT sets (descriptive, base-rate-bound);"
    " Claim%/Acc = rate and accuracy of first-unit claims in the performed-TOT"
    " probe on failed items. Per-dataset 2x2 cells: results/selfreport_cells.csv"
    " (code/selfreport_meta3.py). Computed by code/paper_tables.py.",
    "",
    "Lang\tModel\tBehav. TOT\tSelf-rep. TOT\tOR(TOT/DK)\tJaccard\tClaim% (TOT)"
    "\tAcc (TOT)\tClaim% (DK)\tAcc (DK)"]


def items(lang):
    return load_jsonl(os.path.join(ROOT, "data", f"items_full_{lang}_final.jsonl"))


def taxonomy():
    out = list(TAX_HEAD)
    for lang, (mdir, label) in product(("en", "zh"), MODELS):
        base = os.path.join(ROOT, "results", f"full_{lang}_{mdir}")
        if not os.path.isdir(base):
            continue
        prod = load_jsonl(os.path.join(base, "production.jsonl"))
        rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
        its = items(lang)
        st = {}
        for iid in its:
            s = classify(prod.get(iid), rec.get(iid))
            st[s] = st.get(s, 0) + 1
        fail = sum(1 for i in its if i in prod and prod[i]["k_prod"] == 0
                   and not prod[i]["greedy_correct"])
        known = [1.0 if rec[i]["k_rec"] >= 3 else 0.0 for i in its
                 if i in rec and rec[i].get("n_rec") == 4
                 and classify(prod.get(i), rec.get(i)) == "known"]
        ceil = 100 * sum(known) / max(1, len(known))
        out.append("\t".join(str(x) for x in (
            lang.upper(), label, f"{100 * fail / len(its):.1f}",
            st.get("known", 0), st.get("TOT", 0), st.get("DK", 0),
            f"{ceil:.1f}")))
    return out


def selfreport():
    with io.open(os.path.join(ROOT, "results", "selfreport_cells.csv"),
                 encoding="utf-8") as f:
        cells = {r["dataset"]: r for r in csv.DictReader(
            [ln for ln in f if not ln.startswith("#")])}
    out = list(SR_HEAD)
    for lang, (mdir, label) in product(("en", "zh"), MODELS):
        base = os.path.join(ROOT, "results", f"full_{lang}_{mdir}")
        if not os.path.isdir(base):
            continue
        prod = load_jsonl(os.path.join(base, "production.jsonl"))
        rec = load_jsonl(os.path.join(base, "recognition.jsonl"))
        selfr = load_jsonl(os.path.join(base, "selfreport.jsonl"))
        perf = load_jsonl(os.path.join(base, "perform.jsonl"))
        its = items(lang)
        bh, sr = set(), set()
        claim = {"TOT": [0, 0, []], "DK": [0, 0, []]}
        for iid, it in its.items():
            st = classify(prod.get(iid), rec.get(iid))
            if st == "TOT":
                bh.add(iid)
            if selfr.get(iid, {}).get("label") == "TOT":
                sr.add(iid)
            if st in ("TOT", "DK") and iid in perf:
                d = claim[st]
                d[0] += 1
                c = extract_claim(perf[iid].get("performance", ""), lang)
                if c is not None:
                    d[1] += 1
                    tgt = it["word"][0].lower() if lang == "en" else it["word"][0]
                    d[2].append(1.0 if c == tgt else 0.0)
        union = len(bh | sr)
        jac = len(bh & sr) / union if union else float("nan")

        def acc(st):
            v = claim[st][2]
            return f"{sum(v) / len(v):.2f}" if v else "---"

        def rate(st):
            d = claim[st]
            return f"{d[1] / d[0]:.2f}" if d[0] else "---"
        c = cells.get(f"full_{lang}_{mdir}")
        if c is None:
            or_str = "---"
        elif c["status"] == "zero_report_na":
            or_str = "n/a^dagger"
        else:
            or_str = (f"{float(c['or_haldane']):.2f}"
                      + ("*" if float(c["fisher_p"]) < 0.05 else ""))
        out.append("\t".join(str(x) for x in (
            lang.upper(), label, len(bh), len(sr), or_str, f"{jac:.3f}",
            rate("TOT"), acc("TOT"), rate("DK"), acc("DK"))))
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "taxonomy"
    print("\n".join(taxonomy() if which == "taxonomy" else selfreport()))
