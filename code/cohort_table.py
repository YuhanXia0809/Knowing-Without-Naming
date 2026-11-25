# -*- coding: utf-8 -*-
"""Write the per-item cohort counts that cohort_control.py reads.

For every target word of items_full_{en,zh}_final.jsonl and every decoy word
of items_decoy_{en,zh}.jsonl, the cohort variants of the cohort-size control
(number of lexicon entries sharing the word's first letter / first
character, and its sensitivity variants). The lexicon loaders and cohort
classes below are the ones the analysis was run with; this script needs
the lexicons themselves, which are NOT redistributed with the release:

  <resources>/clearpond_english/englishCPdatabase2.txt   CLEARPOND English
  <resources>/dict/index.{noun,verb,adj,adv}             WordNet 3.1
  <resources>/cedict_ts.u8                               CC-CEDICT
  <resources>/subtlexch_wf/SUBTLEX-CH-WF                 SUBTLEX-CH (WF)

  python cohort_table.py --out-dir <folder>

<resources> is --resources if given, else $KWN_RESOURCES, else
resources/ at the repository root (the convention of the other
rebuild scripts).
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import math
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = None          # set from --resources
COLS = ["coh_n", "log_coh", "log_coh_freqw", "log_coh_len",
        "log_coh_hifreq", "log_coh_wn"]


# ------------------------------------------------------------------ lexicons
def load_clearpond():
    """CLEARPOND English lexicon: word -> SUBTLEX-US frequency per million.

    27,751 orthographic types, every one attested at >= 0.33 per million,
    i.e. it is already a 'words a speaker/model plausibly produces' list
    rather than an unabridged dictionary.
    """
    path = os.path.join(RES, "clearpond_english", "englishCPdatabase2.txt")
    out = {}
    with io.open(path, encoding="latin-1") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5 or not p[0]:
                continue
            w = p[0].strip().lower()
            if not (w.isalpha() and w.isascii()):
                continue
            try:
                fr = float(p[4])
            except Exception:
                continue
            out[w] = max(out.get(w, 0.0), fr)
    return out


def load_wordnet():
    """WordNet 3.1 single-word lemmas (an unabridged-dictionary alternative)."""
    out = set()
    for pos in ("noun", "verb", "adj", "adv"):
        path = os.path.join(RES, "dict", "index.%s" % pos)
        if not os.path.exists(path):
            continue
        with io.open(path, encoding="latin-1") as f:
            for line in f:
                if line.startswith(" "):
                    continue
                lem = line.split(" ", 1)[0].strip().lower()
                if lem and lem.isalpha() and lem.isascii():
                    out.add(lem)
    return out


def load_cedict():
    """CC-CEDICT simplified headwords (the Chinese 'dictionary' cohort)."""
    path = os.path.join(RES, "cedict_ts.u8")
    out = set()
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split(" ", 2)
            if len(parts) >= 2 and parts[1]:
                out.add(parts[1])
    return out


def load_subtlexch():
    """SUBTLEX-CH-WF: word -> per-million frequency (GBK encoded)."""
    path = os.path.join(RES, "subtlexch_wf", "SUBTLEX-CH-WF")
    out = {}
    hdr = False
    with io.open(path, encoding="gbk", errors="ignore") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if not hdr:
                if p and p[0].strip() == "Word":
                    hdr = True
                continue
            if len(p) < 3 or not p[0]:
                continue
            try:
                out[p[0]] = float(p[2])
            except Exception:
                pass
    return out


# ----------------------------------------------------------- cohort builders
class CohortEn(object):
    """Cohort size of an English word = lexicon entries sharing its first
    letter.  `primary` is the raw type count in CLEARPOND; the rest are
    the sensitivity variants."""

    def __init__(self):
        self.lex = load_clearpond()
        self.wn = load_wordnet()
        by = collections.defaultdict(list)
        for w, fr in self.lex.items():
            by[w[0]].append((w, fr))
        self.by = by
        self.count = {c: len(v) for c, v in by.items()}
        self.fmass = {c: sum(fr for _, fr in v) for c, v in by.items()}
        self.hifreq = {c: sum(1 for _, fr in v if fr >= 1.0)
                       for c, v in by.items()}
        self.wn_count = collections.Counter(w[0] for w in self.wn)
        # length-matched needs the word, so it is computed on demand
        self.by_len = {}
        for c, v in by.items():
            d = collections.Counter()
            for w, _ in v:
                d[len(w)] += 1
            self.by_len[c] = d

    def unit(self, word):
        return word[0].lower()

    def variants(self, word):
        c = self.unit(word)
        n = self.count.get(c, 0)
        fm = self.fmass.get(c, 0.0)
        hf = self.hifreq.get(c, 0)
        wnn = self.wn_count.get(c, 0)
        d = self.by_len.get(c, {})
        L = len(word)
        lm = sum(v for k, v in d.items() if abs(k - L) <= 1)
        return {
            "coh_n": n,
            "log_coh": math.log10(n + 1.0),                 # PRIMARY
            "log_coh_freqw": math.log10(fm + 1.0),          # frequency-weighted
            "log_coh_len": math.log10(lm + 1.0),            # +/-1 letter matched
            "log_coh_hifreq": math.log10(hf + 1.0),         # Zipf >= 3 subset
            "log_coh_wn": math.log10(wnn + 1.0),            # WordNet 3.1
        }


class CohortZh(object):
    """Cohort size of a Chinese word = dictionary entries sharing its first
    character.  `primary` is the CC-CEDICT simplified-headword count (best
    coverage of the bank's first characters); SUBTLEX-CH is the
    frequency-attested alternative."""

    def __init__(self):
        self.ced = load_cedict()
        self.sub = load_subtlexch()
        self.ced_count = collections.Counter(w[0] for w in self.ced)
        self.sub_count = collections.Counter(w[0] for w in self.sub)
        self.sub_mass = collections.defaultdict(float)
        for w, fr in self.sub.items():
            self.sub_mass[w[0]] += fr
        self.ced_len = collections.defaultdict(collections.Counter)
        for w in self.ced:
            self.ced_len[w[0]][len(w)] += 1

    def unit(self, word):
        return word[0]

    def variants(self, word):
        c = self.unit(word)
        n = self.ced_count.get(c, 0)
        sn = self.sub_count.get(c, 0)
        sm_ = self.sub_mass.get(c, 0.0)
        L = len(word)
        lm = self.ced_len.get(c, {}).get(L, 0)
        return {
            "coh_n": n,
            "log_coh": math.log10(n + 1.0),                 # PRIMARY (CEDICT)
            "log_coh_freqw": math.log10(sm_ + 1.0),         # SUBTLEX freq mass
            "log_coh_len": math.log10(lm + 1.0),            # same n_chars
            "log_coh_hifreq": math.log10(sn + 1.0),         # SUBTLEX type count
            "log_coh_wn": math.log10(sn + 1.0),             # (alias; see note)
        }



def words(lang):
    out = {}
    for fn, kind in (("items_full_%s_final.jsonl" % lang, "target"),
                     ("items_decoy_%s.jsonl" % lang, "decoy")):
        with io.open(os.path.join(ROOT, "data", fn), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r["item_id"]] = (kind, r["word"])
    return out


def main():
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--resources", default=os.environ.get(
        "KWN_RESOURCES", os.path.join(ROOT, "resources")))
    ap.add_argument("--out-dir", required=True,
                    help="output folder (data/ holds the shipped tables)")
    args = ap.parse_args()
    RES = args.resources
    en, zh = CohortEn(), CohortZh()
    meta = {"en": {"clearpond_types": len(en.lex),
                   "clearpond_first_letters": len(en.count),
                   "wordnet_lemmas": len(en.wn)},
            "zh": {"cedict_headwords": len(zh.ced),
                   "cedict_first_chars": len(zh.ced_count),
                   "subtlexch_types": len(zh.sub)}}
    for lang, coh in (("en", en), ("zh", zh)):
        p = os.path.join(args.out_dir, "cohort_%s.tsv" % lang)
        with io.open(p, "w", encoding="utf-8", newline="") as f:
            for k, v in meta[lang].items():
                f.write("# resource: %s=%d\n" % (k, v))
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["item_id", "kind", "word", "unit"] + COLS)
            for iid, (kind, word) in sorted(words(lang).items()):
                v = coh.variants(word)
                w.writerow([iid, kind, word, coh.unit(word), v["coh_n"]]
                           + [repr(float(v[c])) for c in COLS[1:]])
        print("wrote", p)


if __name__ == "__main__":
    main()
