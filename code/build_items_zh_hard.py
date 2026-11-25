#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a HARDER Chinese TOT (tip-of-the-tongue) item pool: moedict entries that
are absent from SUBTLEX-CH, or present there at very low frequency, so an LLM
is expected to fail on a meaningful fraction of them (DK / "don't know"
enrichment for the reverse-dictionary experiment).

Reuses, rather than reimplements, the existing pipeline pieces:
  - moedict parsing / Traditional->Simplified conversion / SUBTLEX-CH loading /
    CC-CEDICT loading / pypinyin helpers / strict gloss-containment check, all
    from build_items_zh.py (this script imports them; build_items_zh.py itself
    is not modified -- its top-level pipeline only runs under
    `if __name__ == "__main__"`, so importing it is side-effect free).
  - gloss cleaning (Latin-script / 音译-metalanguage segment removal), from
    clean_zh_pool.py's clean_gloss().
  - the proper-name exclusion rule (gloss-marker/biographical-shape patterns +
    transliterated-name detection), from filter_zh_propernames.py's
    is_proper_name(), so the proper-name screens of the main and the hard
    pool stay identical by construction.

Candidate universe: moedict simplified 2-4 char words ABSENT from SUBTLEX-CH,
or present there at freq_per_million < 0.05. Frequency field for these hard
items comes from wordfreq.zipf_frequency(word, "zh") instead of SUBTLEX-CH
(SUBTLEX-CH is used only as the candidacy screen, not as the frequency value,
since it does not cover most of this pool by construction). Kept range:
zipf in [0.0, 3.0], split into three strata:
    offcorpus  : zipf == 0.0   (absent from wordfreq's Chinese corpus entirely;
                                 included as its own stratum rather than
                                 dropped, since many legitimate rare/literary
                                 words are simply missing from wordfreq's
                                 Chinese wordlist, not because they're bad
                                 items)
    ultrarare  : 0.0 < zipf < 1.8
    rare       : 1.8 <= zipf <= 3.0

Additional filters (beyond the base build_items_zh.py pipeline):
  - definition >= 8 chars after clean_gloss() cleaning
  - strict containment: neither the target word nor any of its individual
    characters may appear in the gloss (gloss_ok_strict, reused)
  - proper-name exclusion (is_proper_name, reused from the main pool)
  - cross-reference / abbreviation entries excluded: gloss contains
    姓。|参见「|见「|之简称|的简称 (checked on the pre-clean simplified gloss,
    since these markers are plain Chinese punctuation/text unaffected by the
    Latin-script cleaning step, and checking pre-clean avoids any chance the
    sentence-splitting in clean_gloss fragments the marker away)
  - 儿化 (erhua) words excluded: word ends in 儿
  - single-morpheme bound-form fragments excluded: moedict tags these
    entries' primary definition with type == "缀" (affix/bound morpheme,
    e.g. prefixes/suffixes that are not free-standing words), which is the
    one structural signal available in the source data for "this is a
    fragment, not a word" (parsed directly here since build_items_zh.py's
    load_moedict() does not retain the per-definition type field)

Target size: 1500-4000 items. If the filtered candidate count exceeds 4000,
items with a 10-40 char cleaned gloss (clean, single-idea definitional style)
are preferred over longer/shorter ones when truncating to 4000.

Output (same schema as items_zh_pool.jsonl, plus zipf/freq_stratum/
subtlex_status fields documenting what makes each item "hard"):
    $KWN_REBUILD_DIR/items_zh_hard_pool.jsonl   (default: rebuild/)
    $KWN_REBUILD_DIR/items_zh_hard_pool_stats.txt

"""

import json
import os
import random
import re
import statistics
import sys
from collections import Counter

import opencc
import wordfreq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from build_items_zh import (  # noqa: E402
    CEDICT_TXT,
    MOEDICT_JSON,
    HAN_RE,
    char_pinyin_tone3,
    first_char_initial,
    gloss_ok_strict,
    load_aoa,
    load_cedict,
    load_subtlex_freq,
    tone_from_tone3,
)
from clean_zh_pool import clean_gloss  # noqa: E402
from filter_zh_propernames import is_proper_name  # noqa: E402

# inputs and outputs are settable; outputs never default to data/
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.environ.get("KWN_RESOURCES", os.path.join(_ROOT, "resources"))
DATA = os.environ.get("KWN_REBUILD_DIR", os.path.join(_ROOT, "rebuild"))

OUT_POOL = f"{DATA}/items_zh_hard_pool.jsonl"
OUT_STATS = f"{DATA}/items_zh_hard_pool_stats.txt"

SUBTLEX_ABSENT_OR_RARE_MAX_WPM = 0.05
MIN_GLOSS_LEN = 8
ZIPF_MIN, ZIPF_MAX = 0.0, 3.0
TARGET_MAX = 4000
TARGET_MIN = 1500
PREFERRED_GLOSS_LEN_MIN, PREFERRED_GLOSS_LEN_MAX = 10, 40

CROSSREF_ABBREV_RE = re.compile(r"姓。|参见「|见「|之简称|的简称")


# ---------------------------------------------------------------------------
# moedict parsing, extended to also retain the primary definition's "type"
# field (needed to detect bound-morpheme/affix entries, type == "缀"), which
# build_items_zh.py's load_moedict() discards. Same source file, same
# title-length / HAN_RE / first-occurrence / Traditional->Simplified logic as
# build_items_zh.load_moedict().
# ---------------------------------------------------------------------------
def load_moedict_with_type():
    with open(MOEDICT_JSON, encoding="utf-8") as f:
        data = json.load(f)
    t2s = opencc.OpenCC("t2s")
    gloss_map = {}
    for entry in data:
        title_t = entry.get("title", "")
        if not (2 <= len(title_t) <= 4) or not HAN_RE.match(title_t):
            continue
        heteronyms = entry.get("heteronyms", [])
        all_defs = [
            d
            for h in heteronyms
            for d in h.get("definitions", [])
            if d.get("def", "").strip()
        ]
        if not all_defs:
            continue
        n_defs = len(all_defs)
        primary_def = all_defs[0]
        primary_def_t = primary_def.get("def", "").strip()
        primary_type = primary_def.get("type", "")
        word_s = t2s.convert(title_t)
        if not (2 <= len(word_s) <= 4) or not HAN_RE.match(word_s):
            continue
        gloss_s = t2s.convert(primary_def_t)
        if word_s not in gloss_map:  # keep first occurrence
            gloss_map[word_s] = {
                "gloss": gloss_s,
                "n_defs": n_defs,
                "primary_type": primary_type,
            }
    return gloss_map


def zipf_stratum(zipf):
    if zipf == 0.0:
        return "offcorpus"
    if zipf < 1.8:
        return "ultrarare"
    return "rare"


def main():
    os.makedirs(DATA, exist_ok=True)
    print("Loading SUBTLEX-CH frequencies (candidacy screen only)...")
    freq_map = load_subtlex_freq()
    print(f"  {len(freq_map)} SUBTLEX-CH 2-4 char words with frequency")

    print("Loading moedict definitions (Traditional -> Simplified via OpenCC)...")
    gloss_map = load_moedict_with_type()
    print(f"  {len(gloss_map)} simplified word entries with a moedict gloss")

    print("Loading CC-CEDICT...")
    en_map = load_cedict()
    print(f"  {len(en_map)} simplified word entries with an English gloss")

    print("Loading Xu et al. 2021 Chinese AoA norms...")
    aoa_map = load_aoa()
    print(f"  {len(aoa_map)} words with AoA norms")

    counts = Counter()
    candidates = []

    for word, g in gloss_map.items():
        counts["n_moedict"] += 1

        # SUBTLEX-CH candidacy screen: absent, or present but very low freq.
        sub = freq_map.get(word)
        if sub is not None and sub[0] >= SUBTLEX_ABSENT_OR_RARE_MAX_WPM:
            continue
        counts["after_subtlex_screen"] += 1

        # wordfreq zipf range
        zipf = wordfreq.zipf_frequency(word, "zh")
        if not (ZIPF_MIN <= zipf <= ZIPF_MAX):
            continue
        counts["after_zipf_range"] += 1

        gloss_raw = g["gloss"]

        # cross-reference / abbreviation entries (checked pre-clean)
        if CROSSREF_ABBREV_RE.search(gloss_raw):
            continue
        counts["after_crossref_abbrev"] += 1

        # single-morpheme bound-form fragments (affix entries)
        if g["primary_type"] == "缀":
            continue
        counts["after_affix_type"] += 1

        # 儿化 (erhua) words
        if word.endswith("儿"):
            continue
        counts["after_erhua"] += 1

        # gloss cleaning (Latin-script / 音译-metalanguage removal), reused
        cleaned = clean_gloss(gloss_raw)
        if len(cleaned) < MIN_GLOSS_LEN:
            continue
        counts["after_gloss_len"] += 1

        # strict containment: word / any of its chars must not appear in gloss
        if not gloss_ok_strict(cleaned, word):
            continue
        counts["after_containment"] += 1

        en_equiv = en_map.get(word)

        # proper-name exclusion, reused from the main pool (identical rule)
        if is_proper_name({"word": word, "gloss": cleaned, "en_equiv": en_equiv}):
            continue
        counts["after_proper_name"] += 1

        try:
            py = char_pinyin_tone3(word)
        except Exception:
            continue
        if len(py) != len(word):
            continue
        initial = first_char_initial(word)
        tone1 = tone_from_tone3(py[0])

        fpm = wordfreq.word_frequency(word, "zh") * 1_000_000
        aoa = aoa_map.get(word)

        subtlex_status = "absent" if sub is None else f"present_low_freq={sub[0]}"

        candidates.append({
            "word": word,
            "gloss": cleaned,
            "gloss_raw": gloss_raw,
            "n_chars": len(word),
            "pinyin": py,
            "initial": initial,
            "tone1": tone1,
            "zipf": zipf,
            "freq_per_million": fpm,
            "freq_stratum": zipf_stratum(zipf),
            "subtlex_status": subtlex_status,
            "aoa": aoa,
            "n_defs": g["n_defs"],
            "en_equiv": en_equiv,
            "gloss_source": "moedict dict-revised.json (Taiwan MoE Revised Mandarin Chinese Dictionary, Traditional->Simplified via OpenCC t2s)",
        })

    print("\nFilter pipeline counts:")
    for k in (
        "n_moedict", "after_subtlex_screen", "after_zipf_range",
        "after_crossref_abbrev", "after_affix_type", "after_erhua",
        "after_gloss_len", "after_containment", "after_proper_name",
    ):
        print(f"  {k:24s} = {counts[k]}")
    print(f"  final candidate count    = {len(candidates)}")

    truncated = False
    if len(candidates) > TARGET_MAX:
        truncated = True

        def sort_key(it):
            in_range = PREFERRED_GLOSS_LEN_MIN <= len(it["gloss"]) <= PREFERRED_GLOSS_LEN_MAX
            return (0 if in_range else 1, it["word"])

        candidates.sort(key=sort_key)
        candidates = candidates[:TARGET_MAX]
        print(f"\nOver {TARGET_MAX} candidates -- truncated to {TARGET_MAX}, "
              f"preferring 10-40 char cleaned glosses.")

    candidates.sort(key=lambda it: (it["freq_stratum"], it["word"]))

    print(f"\nFinal hard pool size: {len(candidates)}")
    if len(candidates) < TARGET_MIN:
        print(f"WARNING: final pool size {len(candidates)} is below the "
              f"{TARGET_MIN} target minimum.")

    with open(OUT_POOL, "w", encoding="utf-8") as f:
        for it in candidates:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Wrote {len(candidates)} items to {OUT_POOL}")

    write_stats(candidates, counts, truncated)


def write_stats(items, counts, truncated):
    strata = Counter(it["freq_stratum"] for it in items)
    n_len = Counter(it["n_chars"] for it in items)
    n_aoa = sum(1 for it in items if it["aoa"] is not None)
    n_en = sum(1 for it in items if it["en_equiv"])
    n_absent = sum(1 for it in items if it["subtlex_status"] == "absent")

    lines = []
    lines.append("Chinese HARD candidate item pool -- build stats")
    lines.append("=" * 60)
    lines.append("")
    lines.append("SOURCE: moedict dict-revised.json (Traditional -> Simplified via OpenCC),")
    lines.append("same source as items_zh_pool.jsonl. Candidacy screen: absent from")
    lines.append("SUBTLEX-CH, or present there at freq_per_million < "
                  f"{SUBTLEX_ABSENT_OR_RARE_MAX_WPM}.")
    lines.append("Frequency VALUE (zipf / freq_per_million fields) comes from the")
    lines.append("wordfreq package, not SUBTLEX-CH -- SUBTLEX-CH is only the candidacy")
    lines.append("screen and does not cover most of this pool by construction.")
    lines.append("")
    lines.append("FILTER PIPELINE COUNTS")
    lines.append("-" * 60)
    for k in (
        "n_moedict", "after_subtlex_screen", "after_zipf_range",
        "after_crossref_abbrev", "after_affix_type", "after_erhua",
        "after_gloss_len", "after_containment", "after_proper_name",
    ):
        lines.append(f"  {k:24s} = {counts[k]}")
    lines.append("")
    lines.append(f"TRUNCATION: {'applied (>' + str(TARGET_MAX) + ' candidates, kept ' + str(PREFERRED_GLOSS_LEN_MIN) + '-' + str(PREFERRED_GLOSS_LEN_MAX) + ' char glosses preferentially)' if truncated else 'not needed'}")
    lines.append("")
    lines.append("FINAL POOL")
    lines.append("-" * 60)
    lines.append(f"Total items: {len(items)}")
    lines.append(f"  by n_chars: {dict(sorted(n_len.items()))}")
    lines.append("  by freq_stratum:")
    for k in ("offcorpus", "ultrarare", "rare"):
        lines.append(f"    {k:10s}: {strata.get(k, 0)}")
    lines.append(f"  of which absent from SUBTLEX-CH entirely: {n_absent}")
    if items:
        lines.append(f"AoA coverage: {n_aoa}/{len(items)} ({100.0 * n_aoa / len(items):.1f}%)")
        lines.append(f"CC-CEDICT en_equiv coverage: {n_en}/{len(items)} ({100.0 * n_en / len(items):.1f}%)")
        gloss_lens = [len(it["gloss"]) for it in items]
        lines.append(f"gloss length (chars): min={min(gloss_lens)} max={max(gloss_lens)} "
                      f"mean={statistics.mean(gloss_lens):.1f} median={statistics.median(gloss_lens):.1f}")
    lines.append("")
    lines.append("10 EXAMPLE ITEMS")
    lines.append("-" * 60)
    random.seed(42)
    examples = random.sample(items, min(10, len(items)))
    for it in examples:
        lines.append(json.dumps(it, ensure_ascii=False))
    lines.append("")

    with open(OUT_STATS, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote stats to {OUT_STATS}")


if __name__ == "__main__":
    main()
