"""Decoy-target 4AFC control.

The recognition stage does not otherwise test whether the model can
reject a plausible-but-wrong option when the TRUE target is absent.

For each source item, build a 4AFC item whose true target is ABSENT from the
options. The "correct" option (stored as `target_option`, per the schema
run_stages.py's recognition stage consumes) is a DECOY: the item's own
strongest gloss-compatible non-answer -- i.e. the semantic associate a model
could pick via gloss-overlap inference alone, not lexical retrieval of the
real word. The other 3 options are distractors pooled from OTHER items (never
the source item's own distractors, never the true target/answers), so the
option set contains zero mentions of the true target. Scoring the recognition
stage on this file measures the decoy-selection rate: a model that reliably
"recognizes" the decoy as if it were the answer is pattern-matching gloss
overlap, not retrieving a stored word.

Decoy selection rule (in priority order):
  1. --associate-from items_adv_{lang}.jsonl (make_adversarial_items.py output):
     that file's `distractors` are pool candidates ranked by IDF-weighted
     content-token gloss-overlap score, descending (see make_adversarial_items
     .pick_distractors). distractors[0] is therefore the single most
     gloss-confusable non-answer for that item -- exactly the "decoy" this
     control wants. Matched by item_id.
  2. Fallback (no --associate-from, or the item_id is missing from it): among
     the item's own 3 stored `distractors`, pick the one with the highest
     character/word overlap with the item's gloss -- for zh, shared CJK
     characters with the gloss's content characters; for en, shared character
     bigrams between the distractor word and the gloss's content tokens. This
     approximates the same "gloss-compatible associate" signal directly from
     what's already stored on the item, without needing the adversarial file.
     ASSUMPTION TO VERIFY: distractor provenance (d1/orthographic, d2/semantic,
     d3/random) is not labeled per-slot in items_full_*_final.jsonl, so this is
     a heuristic re-ranking, not a lookup of the "true" d2 slot. Passing
     --associate-from is the more principled option whenever items_adv_{lang}
     .jsonl is available.

Filler distractors: 3 words drawn from the DISTRACTOR POOL of all OTHER items
in the same pool file (never the current item's own distractors/target/
answers, never each other). If `wordfreq` is importable, fillers are sampled
from candidates in the same (or nearest available) Zipf-frequency band as the
decoy, widening the band search outward if the exact band is too small;
otherwise sampling is uniform random over the candidate pool. All sampling is
seeded (--seed) for determinism.

Schema of the emitted item matches what run_stages.py --stages recognition
consumes (target_option + distractors + gloss), plus extra bookkeeping fields
(true_target, true_item_id, decoy_provenance) that recognition/production
ignore but are useful for auditing:

  {"item_id": "decoy_en_en_foo", "gloss": "...", "word": "<decoy>",
   "target_option": "<decoy>", "distractors": ["f1", "f2", "f3"],
   "answers": ["<decoy>"], "true_target": "foo", "true_item_id": "en_foo",
   "decoy_provenance": "adv_top1" | "char_overlap_fallback", "lang": "en"}

  python gen_decoy_items.py --pool ../data/items_full_en_final.jsonl \\
      --lang en --associate-from ../data/items_adv_en.jsonl \\
      --out ../rebuild/items_decoy_en.jsonl --n 400 --seed 47
"""
from __future__ import annotations

import argparse
import json
import random
import re

try:
    from wordfreq import zipf_frequency
    HAVE_WORDFREQ = True
except ImportError:
    HAVE_WORDFREQ = False

EN_STOP = set(
    "a an the of or and to in for on with that which is are was were be being "
    "been by as at from it its this these those someone something who whose "
    "used use having has have not no other any some more most very".split()
)


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def content_tokens(gloss, lang):
    if lang == "zh":
        return set(gloss) - set("的了在是有和与或及等一个把被对从")
    toks = re.findall(r"[a-z]+", gloss.lower())
    return {t for t in toks if t not in EN_STOP and len(t) > 2}


def char_overlap_score(distractor, gloss, lang):
    """Fallback decoy-ranking heuristic: character/word overlap between a
    candidate distractor word and the item's gloss (see module docstring,
    fallback rule)."""
    if lang == "zh":
        return len(set(str(distractor)) & content_tokens(gloss, "zh"))
    d = str(distractor).lower()
    d_bigrams = {d[i:i + 2] for i in range(len(d) - 1)}
    g_joined = "".join(sorted(content_tokens(gloss, "en")))
    g_bigrams = {g_joined[i:i + 2] for i in range(len(g_joined) - 1)}
    return len(d_bigrams & g_bigrams)


def pick_decoy(item, adv_by_id, lang):
    """Returns (decoy_word, provenance)."""
    if adv_by_id is not None:
        a = adv_by_id.get(item["item_id"])
        if a and a.get("distractors"):
            return str(a["distractors"][0]), "adv_top1"
    dists = item.get("distractors", [])
    if not dists:
        return None, None
    gloss = item.get("gloss", "")
    scored = sorted(
        ((char_overlap_score(d, gloss, lang), i, d) for i, d in enumerate(dists)),
        key=lambda x: (-x[0], x[1]),
    )
    return str(scored[0][2]), "char_overlap_fallback"


def zipf_band(word, lang):
    if not HAVE_WORDFREQ:
        return None
    try:
        return round(zipf_frequency(str(word).lower(), lang))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True,
                    help="items_full_{lang}_final.jsonl (or similar)")
    ap.add_argument("--lang", choices=["en", "zh"], required=True)
    ap.add_argument("--associate-from", default=None,
                    help="items_adv_{lang}.jsonl (make_adversarial_items.py "
                         "output); distractors[0] there is the top gloss-"
                         "overlap-ranked candidate, used as the decoy")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--keep-own-distractors", action="store_true",
                    help="criterial-composition variant: options = decoy + "
                         "the item's own 3 original distractors (exact "
                         "option structure of the standard 4AFC, target "
                         "swapped for decoy) instead of borrowed fillers")
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    items = load_jsonl(args.pool)
    items = [it for it in items
              if len(it.get("distractors", []) or []) == 3 and it.get("gloss")]
    print(f"pool items with gloss + 3 distractors: {len(items)}")

    adv_by_id = None
    if args.associate_from:
        adv_by_id = {it["item_id"]: it for it in load_jsonl(args.associate_from)}
        print(f"associate-from: {len(adv_by_id)} adversarial items loaded")

    # Filler pool: every distractor word appearing anywhere in the source
    # pool (deduped). Per-item exclusion below keeps fillers from ever being
    # the current item's own distractors/target/answers.
    all_distractor_words = sorted({
        str(d) for it in items for d in (it.get("distractors") or [])
    })
    band_pool = {}
    if HAVE_WORDFREQ:
        for w in all_distractor_words:
            band_pool.setdefault(zipf_band(w, args.lang), []).append(w)

    sample = list(items)
    rng.shuffle(sample)
    sample = sample[:args.n]

    out, n_skip = [], 0
    for it in sample:
        decoy, provenance = pick_decoy(it, adv_by_id, args.lang)
        if not decoy:
            n_skip += 1
            continue

        if args.keep_own_distractors:
            own = [str(d) for d in (it.get("distractors") or [])]
            if len(own) != 3 or decoy.lower() in {d.lower() for d in own}:
                n_skip += 1
                continue
            out.append({
                "item_id": f"decoyb_{args.lang}_{it['item_id']}",
                "gloss": it["gloss"],
                "word": decoy,
                "target_option": decoy,
                "distractors": own,
                "answers": [decoy],
                "true_target": it.get("word"),
                "true_item_id": it["item_id"],
                "decoy_provenance": provenance + "+own_distractors",
                "lang": args.lang,
            })
            continue

        exclude = {decoy.lower(), str(it.get("word", "")).lower(),
                   str(it.get("target_option", "")).lower()}
        for a in it.get("answers", [it.get("word")]) or []:
            exclude.add(str(a).lower())
        for d in it.get("distractors", []) or []:
            exclude.add(str(d).lower())

        if HAVE_WORDFREQ:
            band = zipf_band(decoy, args.lang)
            candidates, radius = [], 0
            while len(candidates) < 20 and radius <= 6:
                for bb in ({band} if radius == 0 else (band - radius, band + radius)):
                    candidates += [w for w in band_pool.get(bb, [])
                                   if w.lower() not in exclude]
                candidates = list(dict.fromkeys(candidates))
                radius += 1
        else:
            candidates = [w for w in all_distractor_words
                          if w.lower() not in exclude]

        rng.shuffle(candidates)
        fillers, seen = [], set(exclude)
        for w in candidates:
            wl = w.lower()
            if wl in seen:
                continue
            fillers.append(w)
            seen.add(wl)
            if len(fillers) == 3:
                break
        if len(fillers) < 3:
            n_skip += 1
            continue

        out.append({
            "item_id": f"decoy_{args.lang}_{it['item_id']}",
            "gloss": it["gloss"],
            "word": decoy,
            "target_option": decoy,
            "distractors": fillers,
            "answers": [decoy],
            "true_target": it.get("word"),
            "true_item_id": it["item_id"],
            "decoy_provenance": provenance,
            "lang": args.lang,
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} decoy items ({n_skip} skipped) -> {args.out}")
    for r in out[:5]:
        print(" ", r["item_id"], "| decoy:", r["word"],
              "| true:", r["true_target"], "| provenance:", r["decoy_provenance"])


if __name__ == "__main__":
    main()
