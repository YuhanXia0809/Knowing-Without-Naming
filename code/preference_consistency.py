"""Consistent-preference decomposition of the 4AFC, from the released per-rotation records.

Rotation r places option src_idx at position (src_idx + r) mod 4 (cyclic Latin square,
run_stages.py ROTATIONS), so the word chosen at position c is w = (c - r) mod 4:
0 = target slot, 1..3 = the three distractors. A model that consistently prefers one
WORD picks the same w across rotations; one that prefers a POSITION picks the same letter.
The decoyb_* settings replace the target by the strongest gloss-compatible non-answer, so
"consistent on slot 0" there measures preference for the best available option with no
target present: a preference-preserving null.

  python preference_consistency.py
"""
import json, os, collections
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
R = os.path.join(ROOT, 'results')
L = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
PRIMARY = {'EN': ['full_en_qwen7b', 'full_en_qwen14b', 'full_en_qwen32b', 'full_en_qwen72b', 'full_en_llama8b', 'full_en_llama70b'],
           'ZH': ['full_zh_qwen7b', 'full_zh_qwen14b', 'full_zh_qwen32b', 'full_zh_qwen72b', 'full_zh_llama8b', 'full_zh_llama70b']}
DECOY = {'EN': ['decoyb_en_qwen7b', 'decoyb_en_llama8b', 'decoyb_en_qwen72b', 'decoyb_en_llama70b'],
         'ZH': ['decoyb_zh_qwen7b', 'decoyb_zh_llama8b', 'decoyb_zh_qwen72b', 'decoyb_zh_llama70b']}
def load(p): return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

def words_and_positions(rec):
    ws, ps = [], []
    for r in rec['rotations']:
        c = r.get('choice'); t = r.get('target_pos')
        if c not in L or t not in L: continue
        ws.append((L[c] - L[t]) % 4); ps.append(L[c])
    return ws, ps

def modal(xs):
    if not xs: return None, 0
    v, n = collections.Counter(xs).most_common(1)[0]
    return v, n

def analyse(settings, label, strict_only):
    tot = collections.Counter()
    for s in settings:
        d = os.path.join(R, s)
        rec = {r['item_id']: r for r in load(os.path.join(d, 'recognition.jsonl'))}
        prod_path = os.path.join(d, 'production.jsonl')
        strict = None
        if strict_only and os.path.exists(prod_path):
            strict = {p['item_id'] for p in load(prod_path) if (not p['greedy_correct']) and p['k_prod'] == 0}
        for iid, rr in rec.items():
            if strict is not None and iid not in strict: continue
            ws, ps = words_and_positions(rr)
            if len(ws) < 4: continue
            tot['items'] += 1
            wv, wn = modal(ws); pv, pn = modal(ps)
            if wn >= 3:
                tot['word_consistent'] += 1
                tot['consistent_on_%d' % wv] += 1
            if pn >= 3:
                tot['position_consistent'] += 1
            if rr['k_rec'] >= 3: tot['pass'] += 1
    n = tot['items']; wc = tot['word_consistent']
    print('%-34s items %5d | pass>=3/4 %5.1f%% | same WORD >=3/4: %5.1f%%  of which slot0 %5.1f%%  d1 %4.1f%%  d2 %4.1f%%  d3 %4.1f%% | same POSITION >=3/4: %4.1f%%'
          % (label, n, 100.0 * tot['pass'] / n, 100.0 * wc / n,
             100.0 * tot['consistent_on_0'] / wc, 100.0 * tot['consistent_on_1'] / wc,
             100.0 * tot['consistent_on_2'] / wc, 100.0 * tot['consistent_on_3'] / wc,
             100.0 * tot['position_consistent'] / n))

for lang in ('EN', 'ZH'):
    print('===', lang, '===')
    analyse(PRIMARY[lang], 'target present, strict failures', True)
    analyse(PRIMARY[lang], 'target present, all items', False)
    analyse(DECOY[lang], 'TARGET ABSENT (decoyb), all items', False)
    for s in PRIMARY[lang]:
        analyse([s], '  ' + s + ' strict', True)
    print()
