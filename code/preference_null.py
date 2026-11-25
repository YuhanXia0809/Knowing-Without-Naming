"""Exact preference-preserving null: keep each item's four choices
fixed, permute which option identity is the target. Null pass probability per item =
(# identities chosen >=3 times) / 4. Compare to the observed target pass rate and
compute a floor-corrected TOT:DK (subtract expected preference passes from TOT,
add them to DK).

  python preference_null.py
"""
import json, os, collections
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
R = os.path.join(ROOT, 'results')
L = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
SETS = {'EN': ['full_en_qwen7b', 'full_en_qwen14b', 'full_en_qwen32b', 'full_en_qwen72b', 'full_en_llama8b', 'full_en_llama70b'],
        'ZH': ['full_zh_qwen7b', 'full_zh_qwen14b', 'full_zh_qwen32b', 'full_zh_qwen72b', 'full_zh_llama8b', 'full_zh_llama70b']}
def load(p): return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
for lang, sets in SETS.items():
    print('=== %s ===' % lang)
    pooled = collections.Counter(); pooled_null = 0.0
    for s in sets:
        d = os.path.join(R, s)
        strict = {p['item_id'] for p in load(os.path.join(d, 'production.jsonl')) if (not p['greedy_correct']) and p['k_prod'] == 0}
        n = tot = dk = 0; null = 0.0
        for r in load(os.path.join(d, 'recognition.jsonl')):
            if r['item_id'] not in strict: continue
            ws = [(L[x['choice']] - L[x['target_pos']]) % 4 for x in r['rotations'] if x.get('choice') in L and x.get('target_pos') in L]
            if len(ws) < 4: continue
            n += 1
            cnt = collections.Counter(ws)
            null += sum(1 for v in cnt.values() if v >= 3) / 4.0
            if r['k_rec'] >= 3: tot += 1
            elif r['k_rec'] <= 1: dk += 1
        exp_pass = null / n
        f = exp_pass
        true_dk = dk / (1 - f); corr_tot = tot - true_dk * f; corr_dk = true_dk
        print('  %-18s strict %4d | observed pass %.3f | null pass %.3f | TOT:DK %5.1f -> floor-corrected %4.1f'
              % (s, n, tot / n, exp_pass, tot / dk if dk else float('inf'), corr_tot / corr_dk if corr_dk else float('inf')))
        pooled['n'] += n; pooled['tot'] += tot; pooled['dk'] += dk; pooled_null += null
    n = pooled['n']; tot = pooled['tot']; dk = pooled['dk']
    f = pooled_null / n; true_dk = dk / (1 - f)
    print('  POOLED             strict %4d | observed pass %.3f | null pass %.3f | TOT:DK %5.1f -> floor-corrected %4.1f'
          % (n, tot / n, f, tot / dk, (tot - true_dk * f) / true_dk))
