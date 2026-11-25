#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the Chinese candidate item pool for the reverse-dictionary / tip-of-the-tongue
experiment.

Pipeline:
  1. Load SUBTLEX-CH word frequencies (2-4 character words).
  2. Load moedict dict-revised.json (Ministry of Education, Taiwan, Revised
     Mandarin Chinese Dictionary, Traditional Chinese; CC BY-ND 3.0 TW) and
     convert to Simplified Chinese with OpenCC (t2s). This is the source of
     every Chinese definition.
  3. Join with CC-CEDICT for an English-equivalent field.
  4. Join with Xu et al. (2021) Chinese AoA norms.
  5. Compute per-character pinyin (with tone) via pypinyin.
  6. Filter: 2-4 characters, has SUBTLEX-CH frequency, frequency in a documented
     low-to-mid Zipf band, gloss >=6 chars, gloss must not contain the target word or
     any of its individual characters, not a function word.
  7. Write items_zh_pool.jsonl + items_zh_pool_stats.txt.

Also builds a curated transliterated-loanword (音译词) candidate list,
loanwords_zh.jsonl, from a hardcoded seed list cross-checked against SUBTLEX-CH.

"""

import json
import math
import re
import statistics
from collections import Counter

import openpyxl
import opencc
from pypinyin import pinyin, Style

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
import os as _os
# inputs and outputs are settable; outputs never default to data/
_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")
RES = _os.environ.get("KWN_RESOURCES", _os.path.join(_ROOT, "resources"))
DATA = _os.environ.get("KWN_REBUILD_DIR", _os.path.join(_ROOT, "rebuild"))

SUBTLEX_WF_XLSX = f"{RES}/subtlexch_wf/SUBTLEX-CH-WF.xlsx"
MOEDICT_JSON = f"{RES}/moedict/dict-revised.json"
CEDICT_TXT = f"{RES}/cedict_ts.u8"
AOA_XLSX = f"{RES}/chinese_aoa_xu2021.xlsx"

OUT_POOL = f"{DATA}/items_zh_pool.jsonl"
OUT_STATS = f"{DATA}/items_zh_pool_stats.txt"
OUT_LOANWORDS = f"{DATA}/loanwords_zh.jsonl"

# ---------------------------------------------------------------------------
# Frequency cutoff (documented, chosen after inspecting the distribution below).
# log10(freq_per_million) in [LOG_FREQ_MIN, LOG_FREQ_MAX].
#
# The joined (SUBTLEX x moedict, post gloss/stopword/length filters) candidate
# distribution has median=-0.268, p10=-1.222, p90=0.947 (n=22871). An earlier
# suggested default band (-0.5 to 1.5, ~Zipf 2.5-4.5) applied on top of the
# STRICT gloss-containment rule only yields 2326 items (< the 3000 target).
# Widening to (-1.0, 2.0) (~Zipf 2.0-5.0) -- still comfortably low-to-mid
# frequency, not touching the very-high-frequency function-word-dominated tail
# above Zipf 5 -- yields 3125 items while keeping the STRICT containment rule,
# so that band is used instead. See items_zh_pool_stats.txt for the full
# per-band comparison that motivated this choice.
# ---------------------------------------------------------------------------
LOG_FREQ_MIN = -1.0
LOG_FREQ_MAX = 2.0

MIN_GLOSS_LEN = 6
TARGET_POOL = 3000
RELAX_THRESHOLD = 1500

HAN_RE = re.compile(r"^[一-鿿]+$")

# ---------------------------------------------------------------------------
# Function-word stoplist (closed-class grammatical words only: conjunctions,
# prepositions/coverbs, pronouns, demonstrative/interrogative pro-forms,
# common modal auxiliaries, structural/aspect particles). Content words
# (verbs/nouns/adjectives), even very common ones, are NOT included here.
# ---------------------------------------------------------------------------
FUNCTION_WORDS = set("""
因为 所以 但是 如果 虽然 并且 而且 可是 不过 只是 除了 关于 对于 由于 以及
还是 或者 然而 因此 于是 既然 无论 不管 尽管 即使 以便 以免 为了 通过 根据
按照 随着 至于 除非 以致 以至 甚至 况且 何况 反而 反正 幸亏 亏得 索性 干脆
就算 即便 哪怕 假如 假使 倘若 要是 若是 不仅 不但 不只 不光
这个 那个 这些 那些 这里 那里 这样 那样 什么 怎么 为什么 哪里 哪个 哪些
谁的 我们 你们 他们 她们 它们 自己 大家 别人 有些 有的 一些 每个 各个
任何 某个 某些 其他 另外 另一 之前 之后 之中 之内 之外 之间 以上 以下
以内 以外 以前 以后 期间 首先 其次 然后 接着 最终 总之 总的 一直 一向
从来 曾经 已经 依然 仍然 依旧 再次 又是 也是 就是 都是 只有 只要
可以 可能 也许 或许 一定 必须 应该 应当 能够 愿意
""".split())

# ---------------------------------------------------------------------------
# Step 1: SUBTLEX-CH frequencies
# ---------------------------------------------------------------------------
def load_subtlex_freq():
    wb = openpyxl.load_workbook(SUBTLEX_WF_XLSX, read_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    next(rows)  # "Total word count: ..."
    next(rows)  # "Context number: ..."
    header = next(rows)
    assert header[0] == "Word", f"unexpected header: {header}"
    freq_map = {}
    for r in rows:
        word = r[0]
        if not word:
            continue
        if not (2 <= len(word) <= 4):
            continue
        if not HAN_RE.match(word):
            continue
        try:
            wpm = float(r[2])
            logw = float(r[3])
        except (TypeError, ValueError):
            continue
        if word not in freq_map:  # keep first (highest-freq) occurrence
            freq_map[word] = (wpm, logw)
    wb.close()
    return freq_map


# ---------------------------------------------------------------------------
# Step 2: moedict definitions -> Simplified Chinese
# ---------------------------------------------------------------------------
def load_moedict():
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
            d.get("def", "").strip()
            for h in heteronyms
            for d in h.get("definitions", [])
            if d.get("def", "").strip()
        ]
        if not all_defs:
            continue
        n_defs = len(all_defs)
        primary_def_t = all_defs[0]
        word_s = t2s.convert(title_t)
        if not (2 <= len(word_s) <= 4) or not HAN_RE.match(word_s):
            continue
        gloss_s = t2s.convert(primary_def_t)
        if word_s not in gloss_map:  # keep first occurrence
            gloss_map[word_s] = {"gloss": gloss_s, "n_defs": n_defs}
    return gloss_map


# ---------------------------------------------------------------------------
# Step 3: CC-CEDICT English equivalents
# ---------------------------------------------------------------------------
CEDICT_LINE_RE = re.compile(r"^(\S+)\s+(\S+)\s+\[([^\]]+)\]\s+/(.*)/\s*$")
PAREN_RE = re.compile(r"[\(（][^\)）]*[\)）]")
CL_RE = re.compile(r"\bCL:[^/]*")


def clean_en_def(d):
    d = CL_RE.sub("", d)
    d = PAREN_RE.sub("", d)
    d = d.strip(" ,;")
    return d.strip()


def load_cedict():
    en_map = {}
    with open(CEDICT_TXT, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            m = CEDICT_LINE_RE.match(line.strip())
            if not m:
                continue
            simp = m.group(2)
            defs = [d for d in m.group(4).split("/") if d.strip()]
            if not defs:
                continue
            if simp in en_map:
                continue  # keep first entry encountered
            cleaned = clean_en_def(defs[0])
            if cleaned:
                en_map[simp] = cleaned
    return en_map


# ---------------------------------------------------------------------------
# Step 4: Xu et al. 2021 Chinese AoA norms
# ---------------------------------------------------------------------------
def load_aoa():
    wb = openpyxl.load_workbook(AOA_XLSX, read_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    idx_word = header.index("Word")
    idx_aoa = header.index("AoA Mean")
    aoa_map = {}
    for r in rows:
        word = r[idx_word]
        aoa = r[idx_aoa]
        if word and aoa is not None and word not in aoa_map:
            aoa_map[word] = float(aoa)
    wb.close()
    return aoa_map


# ---------------------------------------------------------------------------
# Gloss containment checks
# ---------------------------------------------------------------------------
def gloss_ok_strict(gloss, word):
    """Exclude if full word appears, or any individual character appears."""
    if word in gloss:
        return False
    for ch in word:
        if ch in gloss:
            return False
    return True


def gloss_ok_relaxed(gloss, word):
    """Exclude only if the full word appears."""
    return word not in gloss


# ---------------------------------------------------------------------------
# Pinyin
# ---------------------------------------------------------------------------
def char_pinyin_tone3(word):
    """Per-character pinyin with tone number, e.g. 沙发 -> ['sha1', 'fa1']."""
    result = pinyin(word, style=Style.TONE3, neutral_tone_with_five=True, heteronym=False)
    return [r[0] for r in result]


def first_char_initial(word):
    result = pinyin(word[0], style=Style.INITIALS, strict=False, heteronym=False)
    init = result[0][0] if result else ""
    return init


def tone_from_tone3(syll):
    m = re.search(r"(\d)$", syll)
    return int(m.group(1)) if m else 5


# ---------------------------------------------------------------------------
# Main pool build
# ---------------------------------------------------------------------------
def build_pool():
    print("Loading SUBTLEX-CH frequencies...")
    freq_map = load_subtlex_freq()
    print(f"  {len(freq_map)} candidate 2-4 char words with SUBTLEX-CH frequency")

    print("Loading moedict definitions (Traditional -> Simplified via OpenCC)...")
    gloss_map = load_moedict()
    print(f"  {len(gloss_map)} simplified word entries with a moedict gloss")

    print("Loading CC-CEDICT...")
    en_map = load_cedict()
    print(f"  {len(en_map)} simplified word entries with an English gloss")

    print("Loading Xu et al. 2021 Chinese AoA norms...")
    aoa_map = load_aoa()
    print(f"  {len(aoa_map)} words with AoA norms")

    # Join SUBTLEX x moedict, apply non-frequency filters first
    joined = []
    n_no_gloss = 0
    n_stopword = 0
    n_short_gloss = 0
    n_fail_strict = 0
    for word, (wpm, logw) in freq_map.items():
        if word in FUNCTION_WORDS:
            n_stopword += 1
            continue
        g = gloss_map.get(word)
        if g is None:
            n_no_gloss += 1
            continue
        gloss = g["gloss"]
        if len(gloss) < MIN_GLOSS_LEN:
            n_short_gloss += 1
            continue
        strict_ok = gloss_ok_strict(gloss, word)
        relaxed_ok = gloss_ok_relaxed(gloss, word)
        if not relaxed_ok:
            continue  # fails even the relaxed rule -> never usable
        if not strict_ok:
            n_fail_strict += 1
        joined.append({
            "word": word,
            "gloss": gloss,
            "n_defs": g["n_defs"],
            "freq_per_million": wpm,
            "log_freq": logw,
            "strict_ok": strict_ok,
        })

    print(f"  joined candidates (pre freq-cutoff): {len(joined)}")
    print(f"    excluded: no gloss={n_no_gloss}, stopword={n_stopword}, "
          f"short gloss={n_short_gloss}, fail-strict-containment={n_fail_strict}")

    # Frequency distribution (log10 freq per million) over joined candidates
    log10freqs = [math.log10(c["freq_per_million"]) for c in joined if c["freq_per_million"] > 0]
    log10freqs.sort()
    def pct(p):
        if not log10freqs:
            return float("nan")
        idx = min(len(log10freqs) - 1, int(p * len(log10freqs)))
        return log10freqs[idx]

    dist_summary = (
        f"n={len(log10freqs)}  "
        f"min={log10freqs[0]:.3f}  p10={pct(0.10):.3f}  p25={pct(0.25):.3f}  "
        f"median={pct(0.50):.3f}  p75={pct(0.75):.3f}  p90={pct(0.90):.3f}  "
        f"max={log10freqs[-1]:.3f}  mean={statistics.mean(log10freqs):.3f}"
    )
    print("log10(freq_per_million) distribution over joined candidates:")
    print("  " + dist_summary)

    # Document a few alternative frequency bands considered before settling on
    # LOG_FREQ_MIN/MAX (for the stats file).
    band_candidates = [(-0.5, 1.5), (-0.7, 1.5), (-0.8, 1.8), (-1.0, 1.5),
                        (-1.0, 2.0), (-1.2, 1.5), (-0.9, 1.6)]
    band_table = []
    for lo, hi in band_candidates:
        n_s = sum(1 for c in joined if lo <= math.log10(c["freq_per_million"]) <= hi and c["strict_ok"])
        n_r = sum(1 for c in joined if lo <= math.log10(c["freq_per_million"]) <= hi)
        band_table.append((lo, hi, n_s, n_r))

    # Apply frequency cutoff + strict containment
    def in_band(c):
        lf = math.log10(c["freq_per_million"]) if c["freq_per_million"] > 0 else -99
        return LOG_FREQ_MIN <= lf <= LOG_FREQ_MAX

    pool_strict = [c for c in joined if in_band(c) and c["strict_ok"]]
    pool_relaxed = [c for c in joined if in_band(c)]  # relaxed containment = all in-band joined

    print(f"pool size with STRICT containment + freq band [{LOG_FREQ_MIN}, {LOG_FREQ_MAX}]: "
          f"{len(pool_strict)}")
    print(f"pool size with RELAXED containment (full-word only) + freq band: {len(pool_relaxed)}")

    used_relaxed = False
    if len(pool_strict) >= RELAX_THRESHOLD:
        final_candidates = pool_strict
        containment_rule = "strict (word AND any character excluded from gloss)"
    else:
        used_relaxed = True
        final_candidates = pool_relaxed
        containment_rule = "relaxed (only full-word match excluded from gloss) -- strict rule yielded < 1500 items"

    print(f"Using containment rule: {containment_rule}")
    print(f"Final candidate count before item assembly: {len(final_candidates)}")

    # Assemble full items
    items = []
    n_aoa_hit = 0
    for c in final_candidates:
        word = c["word"]
        try:
            py = char_pinyin_tone3(word)
        except Exception:
            continue
        if len(py) != len(word):
            continue
        initial = first_char_initial(word)
        tone1 = tone_from_tone3(py[0])
        aoa = aoa_map.get(word)
        if aoa is not None:
            n_aoa_hit += 1
        en_equiv = en_map.get(word)
        items.append({
            "word": word,
            "gloss": c["gloss"],
            "n_chars": len(word),
            "pinyin": py,
            "initial": initial,
            "tone1": tone1,
            "freq_per_million": c["freq_per_million"],
            "log_freq": c["log_freq"],
            "aoa": aoa,
            "n_defs": c["n_defs"],
            "en_equiv": en_equiv,
            "gloss_source": "moedict dict-revised.json (Taiwan MoE Revised Mandarin Chinese Dictionary, Traditional->Simplified via OpenCC t2s)",
        })

    # Sort by frequency descending for readability
    items.sort(key=lambda x: -x["freq_per_million"])

    with open(OUT_POOL, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"Wrote {len(items)} items to {OUT_POOL}")

    write_stats(
        items, freq_map, gloss_map, joined, log10freqs, dist_summary,
        n_no_gloss, n_stopword, n_short_gloss, n_fail_strict,
        len(pool_strict), len(pool_relaxed), used_relaxed, n_aoa_hit, band_table,
    )
    return items


def write_stats(items, freq_map, gloss_map, joined, log10freqs, dist_summary,
                 n_no_gloss, n_stopword, n_short_gloss, n_fail_strict,
                 n_strict_pool, n_relaxed_pool, used_relaxed, n_aoa_hit, band_table):
    n_len = Counter(it["n_chars"] for it in items)
    aoa_cov = 100.0 * n_aoa_hit / len(items) if items else 0.0
    en_cov = 100.0 * sum(1 for it in items if it["en_equiv"]) / len(items) if items else 0.0

    lines = []
    lines.append("Chinese candidate item pool -- build stats")
    lines.append("=" * 60)
    lines.append("")
    lines.append("SOURCE OF CHINESE DEFINITIONS")
    lines.append("-" * 60)
    lines.append("moedict dict-revised.json: Ministry of Education, Taiwan, Revised")
    lines.append("Mandarin Chinese Dictionary (Traditional Chinese), licensed CC BY-ND")
    lines.append("3.0 TW; converted Traditional->Simplified with")
    lines.append("opencc-python-reimplemented (t2s profile), applied to both")
    lines.append("headwords and definition text.")
    lines.append("")
    lines.append("SOURCE USED: moedict dict-revised.json (traditional->simplified via OpenCC)")
    lines.append("")
    lines.append("PIPELINE / FILTER COUNTS")
    lines.append("-" * 60)
    lines.append(f"SUBTLEX-CH 2-4 char words with frequency: {len(freq_map)}")
    lines.append(f"moedict simplified word entries with a gloss: {len(gloss_map)}")
    lines.append(f"Joined (SUBTLEX x moedict), pre freq-cutoff, post gloss/stopword filters: {len(joined)}")
    lines.append(f"  excluded -- no moedict gloss: {n_no_gloss}")
    lines.append(f"  excluded -- function word (stoplist): {n_stopword}")
    lines.append(f"  excluded -- gloss shorter than {MIN_GLOSS_LEN} chars: {n_short_gloss}")
    lines.append(f"  of joined candidates, fail STRICT containment (but pass relaxed): {n_fail_strict}")
    lines.append("")
    lines.append("FREQUENCY DISTRIBUTION (log10 freq-per-million, over joined candidates)")
    lines.append("-" * 60)
    lines.append("  " + dist_summary)
    lines.append("")
    lines.append("  Frequency bands considered (strict = full word+chars excluded from gloss,")
    lines.append("  relaxed = only full-word match excluded), evaluated over joined candidates:")
    lines.append(f"  {'band':>16}  {'strict':>8}  {'relaxed':>8}")
    for lo, hi, n_s, n_r in band_table:
        marker = "  <-- chosen" if (lo, hi) == (LOG_FREQ_MIN, LOG_FREQ_MAX) else ""
        lines.append(f"  [{lo:>5.1f}, {hi:>4.1f}]  {n_s:>8}  {n_r:>8}{marker}")
    lines.append(f"  chosen band: log10(freq_per_million) in [{LOG_FREQ_MIN}, {LOG_FREQ_MAX}]"
                  f"  (~Zipf {LOG_FREQ_MIN+3:.1f}-{LOG_FREQ_MAX+3:.1f}); widened from the earlier")
    lines.append(f"  suggested default (-0.5, 1.5) because that band, combined with the STRICT")
    lines.append(f"  containment rule, only produced 2326 items (< 3000 target); see table above.")
    lines.append("")
    lines.append("CONTAINMENT RULE")
    lines.append("-" * 60)
    lines.append(f"  pool size with STRICT rule (word + all its chars excluded from gloss) "
                 f"in freq band: {n_strict_pool}")
    lines.append(f"  pool size with RELAXED rule (only full-word match excluded) in freq band: "
                 f"{n_relaxed_pool}")
    if used_relaxed:
        lines.append(f"  STRICT rule yielded < {RELAX_THRESHOLD} items -> RELAXED rule used for final pool.")
    else:
        lines.append(f"  STRICT rule yielded >= {RELAX_THRESHOLD} items -> STRICT rule used for final pool.")
    lines.append("")
    lines.append("FINAL POOL")
    lines.append("-" * 60)
    lines.append(f"Total items: {len(items)}")
    lines.append(f"  by n_chars: {dict(sorted(n_len.items()))}")
    lines.append(f"AoA coverage: {n_aoa_hit}/{len(items)} = {aoa_cov:.1f}%")
    lines.append(f"CC-CEDICT en_equiv coverage: {en_cov:.1f}%")
    if items:
        freqs = [it["freq_per_million"] for it in items]
        lines.append(f"freq_per_million: min={min(freqs):.3f} max={max(freqs):.3f} "
                      f"mean={statistics.mean(freqs):.3f} median={statistics.median(freqs):.3f}")
    lines.append("")
    lines.append("10 EXAMPLE ITEMS")
    lines.append("-" * 60)
    import random
    random.seed(42)
    examples = random.sample(items, min(10, len(items)))
    for it in examples:
        lines.append(json.dumps(it, ensure_ascii=False, indent=None))
    lines.append("")

    with open(OUT_STATS, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote stats to {OUT_STATS}")


# ---------------------------------------------------------------------------
# Loanwords (音译词) candidate list
# ---------------------------------------------------------------------------
LOANWORD_SEED = {
    "沙发": "sofa", "咖啡": "coffee", "巧克力": "chocolate", "引擎": "engine",
    "幽默": "humor", "逻辑": "logic", "沙拉": "salad", "坦克": "tank",
    "吉他": "guitar", "摩托": "motor", "芭蕾": "ballet", "香槟": "champagne",
    "尼龙": "nylon", "维他命": "vitamin", "麦克风": "microphone", "巴士": "bus",
    "的士": "taxi", "披萨": "pizza", "汉堡": "hamburger", "三明治": "sandwich",
    "苏打": "soda", "咖喱": "curry", "柠檬": "lemon", "芒果": "mango",
    "木乃伊": "mummy", "雷达": "radar", "声纳": "sonar", "基因": "gene",
    "克隆": "clone", "荷尔蒙": "hormone", "卡路里": "calorie", "淋巴": "lymph",
    "尼古丁": "nicotine", "阿司匹林": "aspirin", "吗啡": "morphine", "扑克": "poker",
    "高尔夫": "golf", "保龄球": "bowling", "马拉松": "marathon", "瑜伽": "yoga",
    "桑拿": "sauna", "华尔兹": "waltz", "探戈": "tango", "爵士": "jazz",
    "朋克": "punk", "博客": "blog", "黑客": "hacker", "拷贝": "copy",
    "培根": "bacon", "芝士": "cheese", "吐司": "toast", "布丁": "pudding",
    "慕斯": "mousse", "拿铁": "latte", "摩卡": "mocha", "威士忌": "whisky",
    "伏特加": "vodka", "白兰地": "brandy", "香波": "shampoo", "凡士林": "vaseline",
}


def build_loanwords():
    print("Building loanwords_zh.jsonl...")
    freq_map = load_subtlex_freq_all()  # unrestricted length, for loanword lookup
    items = []
    missing = []
    for word, en_src in LOANWORD_SEED.items():
        f = freq_map.get(word)
        if f is None:
            missing.append(word)
            continue
        wpm, logw = f
        py_syllables = pinyin(word, style=Style.TONE, heteronym=False)
        py_str = " ".join(s[0] for s in py_syllables)
        items.append({
            "word": word,
            "pinyin": py_str,
            "freq_per_million": wpm,
            "en_source_word": en_src,
        })
    items.sort(key=lambda x: -x["freq_per_million"])
    with open(OUT_LOANWORDS, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"  {len(items)}/{len(LOANWORD_SEED)} seed loanwords found in SUBTLEX-CH")
    if missing:
        print(f"  NOT found in SUBTLEX-CH (excluded): {missing}")
    print(f"Wrote {OUT_LOANWORDS}")
    return items, missing


def load_subtlex_freq_all():
    """Same as load_subtlex_freq but without the 2-4 char restriction (loanwords
    can occasionally be 2 chars up to a handful of chars, e.g. 阿司匹林 is 4)."""
    wb = openpyxl.load_workbook(SUBTLEX_WF_XLSX, read_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    next(rows)
    next(rows)
    header = next(rows)
    assert header[0] == "Word"
    freq_map = {}
    for r in rows:
        word = r[0]
        if not word:
            continue
        try:
            wpm = float(r[2])
            logw = float(r[3])
        except (TypeError, ValueError):
            continue
        if word not in freq_map:
            freq_map[word] = (wpm, logw)
    wb.close()
    return freq_map


if __name__ == "__main__":
    _os.makedirs(DATA, exist_ok=True)
    build_pool()
    build_loanwords()
    print("Done.")
