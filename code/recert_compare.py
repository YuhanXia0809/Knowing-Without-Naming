"""Compare re-issued certificates (on the paraphrased gloss actually used in
production prompts) against the original certificates (issued on gloss_original).

Reads ../data/items_recert_{lang}_judged.jsonl, which carries both the new
verdict (`unique`, `answers`) and the original one (`unique_orig`,
`answers_orig`) on every record. Prints a compact per-language table.

Answer-set relation is computed on the judge-visible answer set. Note both sets
always contain the target word plus wn_synonyms (judge_screen unions them in),
so "crossing" means the two judge passes proposed genuinely incompatible
synonym sets, not merely different targets.

  python -u recert_compare.py
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def relation(new, old):
    n, o = set(new), set(old)
    if n == o:
        return "identical"
    if n > o:
        return "superset"
    if n < o:
        return "subset"
    return "crossing"


def load(lang):
    path = os.path.join(DATA, f"items_recert_{lang}_judged.jsonl")
    if not os.path.exists(path):
        return None
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r["item_id"]] = r  # dedupe on resume-appended files
    return list(recs.values())


def summarize(lang, recs, only_paraphrased):
    if only_paraphrased:
        recs = [r for r in recs if r.get("gloss_is_paraphrase")]
    n = len(recs)
    if not n:
        return None
    agree = sum(1 for r in recs if bool(r.get("unique")) == bool(r.get("unique_orig")))
    yes_no = sum(1 for r in recs if r.get("unique_orig") and not r.get("unique"))
    no_yes = sum(1 for r in recs if not r.get("unique_orig") and r.get("unique"))
    rels = {"identical": 0, "superset": 0, "subset": 0, "crossing": 0}
    for r in recs:
        rels[relation(r.get("answers", []), r.get("answers_orig", []))] += 1
    # items that were certified unique originally but lose it under recert are
    # the ones that actually threaten the production bank
    lost = [r["item_id"] for r in recs if r.get("unique_orig") and not r.get("unique")]
    return dict(lang=lang, n=n, agree=agree, yes_no=yes_no, no_yes=no_yes,
                rels=rels, lost=lost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-paraphrased", action="store_true",
                    help="restrict to items whose prompt gloss differs from the "
                         "certified original")
    ap.add_argument("--list-lost", type=int, default=0,
                    help="print N item_ids that lost their uniqueness certificate")
    args = ap.parse_args()

    rows = []
    for lang in ("en", "zh"):
        recs = load(lang)
        if recs is None:
            print(f"[{lang}] items_recert_{lang}_judged.jsonl not found - skipped")
            continue
        s = summarize(lang, recs, args.only_paraphrased)
        if s:
            rows.append(s)
    if not rows:
        return

    scope = "paraphrased items only" if args.only_paraphrased else "full bank"
    print(f"\nPARAPHRASE RE-CERTIFICATION ({scope})")
    print("-" * 78)
    hdr = (f"{'lang':<5}{'n':>6}{'uniq agree':>12}{'YES->NO':>9}{'NO->YES':>9}"
           f"{'ans=':>7}{'ans+':>7}{'ans-':>7}{'ansX':>7}")
    print(hdr)
    print("-" * 78)
    for s in rows:
        r = s["rels"]
        print(f"{s['lang']:<5}{s['n']:>6}"
              f"{s['agree'] / s['n'] * 100:>11.1f}%"
              f"{s['yes_no']:>9}{s['no_yes']:>9}"
              f"{r['identical']:>7}{r['superset']:>7}{r['subset']:>7}"
              f"{r['crossing']:>7}")
    print("-" * 78)
    tot = sum(s["n"] for s in rows)
    tag = sum(s["agree"] for s in rows)
    ident = sum(s["rels"]["identical"] for s in rows)
    print(f"{'both':<5}{tot:>6}{tag / tot * 100:>11.1f}%"
          f"{sum(s['yes_no'] for s in rows):>9}"
          f"{sum(s['no_yes'] for s in rows):>9}"
          f"{ident:>7}"
          f"{sum(s['rels']['superset'] for s in rows):>7}"
          f"{sum(s['rels']['subset'] for s in rows):>7}"
          f"{sum(s['rels']['crossing'] for s in rows):>7}")
    print("\nans= identical / ans+ recert superset / ans- recert subset / "
          "ansX crossing")
    print("YES->NO = lost uniqueness certificate under the prompt gloss "
          "(the failure mode that matters)")

    if args.list_lost:
        for s in rows:
            if s["lost"]:
                print(f"\n[{s['lang']}] lost uniqueness ({len(s['lost'])}), "
                      f"first {args.list_lost}:")
                print("  " + ", ".join(s["lost"][:args.list_lost]))


if __name__ == "__main__":
    main()
