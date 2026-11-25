"""Build re-certification pools: certify the gloss text PRODUCTION ACTUALLY USES.

Why this is needed:
  judge_screen.py issues certificates (acceptable `answers` + `unique` verdict)
  from the gloss it is given, and *in the same pass* emits `gloss_paraphrased`.
  merge_judged.py --use-paraphrase then does `it["gloss"] = j["gloss_paraphrased"]`
  and stashes the pre-paraphrase text in `gloss_original`. run_stages.py formats
  every production/cue prompt with `it["gloss"]`.
  => the certificates were issued on `gloss_original`, but the models are prompted
     with `gloss`. This script re-poses the judge questions on `gloss` itself.

Output pool keeps the original certificate under *_orig keys, which survive
judge_screen (it does `rec = dict(it)` and only overwrites answers/unique/
gloss_paraphrased/judge_synonyms/paraphrase_ok), so the judged file is
self-contained for recert_compare.py.

  python -u recert_prep.py --out-dir ../rebuild            # both languages
  python -u recert_prep.py --out-dir ../rebuild --lang en

Needs the un-withheld item files (gloss text); the shipped Chinese files
withhold it, so the Chinese pass needs a local rebuild first.
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

# fields judge_screen.py reads, plus provenance we want to keep for comparison
CARRY = ["item_id", "word", "pos", "wn_synonyms", "freq_stratum", "source_pool",
         "target_option", "distractors", "en_equiv", "is_loanword"]


def build(lang, out_dir):
    src = os.path.join(DATA, f"items_full_{lang}_final.jsonl")
    if not os.path.exists(src):
        raise SystemExit("%s not found" % src)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"items_recert_{lang}.jsonl")
    if os.path.abspath(dst) == os.path.abspath(
            os.path.join(DATA, f"items_recert_{lang}.jsonl")):
        raise SystemExit("refusing to overwrite the shipped %s" % dst)
    with open(src, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]

    n_same, n_missing = 0, 0
    out = []
    for it in items:
        gloss = it.get("gloss")
        orig = it.get("gloss_original")
        if not gloss:
            n_missing += 1
            continue
        if orig is not None and gloss.strip() == orig.strip():
            # item was never paraphrased (paraphrase_ok False at merge time):
            # certificate and prompt already agree, but keep it so the recert
            # covers the identical bank and the compare table is complete.
            n_same += 1
        rec = {k: it[k] for k in CARRY if k in it}
        # THE text to certify: exactly what run_stages.py puts in the prompt
        rec["gloss"] = gloss
        rec["gloss_original"] = orig
        rec["gloss_is_paraphrase"] = (orig is not None and
                                      gloss.strip() != orig.strip())
        # original certificate, under names judge_screen will not clobber
        rec["answers"] = it.get("answers", [])
        rec["answers_orig"] = it.get("answers", [])
        rec["unique_orig"] = it.get("unique")
        out.append(rec)

    with open(dst, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{lang}] {len(out)} items -> {dst}"
          f"  (paraphrased: {len(out) - n_same}, unchanged gloss: {n_same},"
          f" skipped no-gloss: {n_missing})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "zh"], default=None)
    ap.add_argument("--out-dir", required=True,
                    help="where to write items_recert_<lang>.jsonl (never data/)")
    args = ap.parse_args()
    for lang in ([args.lang] if args.lang else ["en", "zh"]):
        build(lang, args.out_dir)


if __name__ == "__main__":
    main()
