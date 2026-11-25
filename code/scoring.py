"""Answer scoring for definition->word production and recognition stages.

Design-stable core: normalization + acceptable-answer matching. Every item carries
an `answers` list (target + vetted synonyms/variants); scoring is membership after
normalization. English gets light morphological loosening; Chinese is exact-match
on normalized strings (full-width -> half-width, whitespace stripped).
"""
from __future__ import annotations

import re
import unicodedata

_EN_ARTICLES = re.compile(r"^(the|a|an|to)\s+", re.IGNORECASE)
_EN_NONWORD = re.compile(r"[^a-z' -]")
_WS = re.compile(r"\s+")


def normalize_en(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.strip().lower()
    s = _EN_ARTICLES.sub("", s)
    s = _EN_NONWORD.sub("", s)
    s = _WS.sub(" ", s).strip().strip("'-")
    return s


def normalize_zh(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)  # full-width -> half-width
    s = re.sub(r"[\s，。、！？；：" "" r"''《》（）()\[\]【】.,!?;:'" '"-]', "", s)
    return s.strip()


def en_variants(word: str) -> set[str]:
    """Small, conservative set of morphological variants treated as the same lexeme."""
    w = normalize_en(word)
    out = {w}
    if len(w) > 3:
        if w.endswith("s"):
            out.add(w[:-1])
        else:
            out.add(w + "s")
        if w.endswith("es"):
            out.add(w[:-2])
        if w.endswith("y"):
            out.add(w[:-1] + "ies")
    return out


def extract_candidate(response: str, lang: str) -> str:
    """Pull the answer word out of a (possibly chatty) model response.

    Production prompts instruct 'respond with the single word only', but models
    sometimes wrap it ('The word is "wrench".'). Take the last quoted span if any,
    else the trailing word of the first line after stripping boilerplate.
    """
    text = response.strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    quoted = re.findall(r"[\"“”'‘’]([^\"“”'‘’]{1,40})[\"“”'‘’]", text)
    cand = quoted[-1] if quoted else first_line
    cand = re.sub(r"^(the word is|answer:|the answer is|word:)\s*", "", cand,
                  flags=re.IGNORECASE)
    cand = cand.rstrip(".。!！?？")
    return normalize_zh(cand) if lang == "zh" else normalize_en(cand)


def score_production(response: str, answers: list[str], lang: str) -> bool:
    cand = extract_candidate(response, lang)
    if not cand:
        return False
    if lang == "zh":
        return cand in {normalize_zh(a) for a in answers}
    accept: set[str] = set()
    for a in answers:
        accept |= en_variants(a)
    return cand in accept or normalize_en(response) in accept
