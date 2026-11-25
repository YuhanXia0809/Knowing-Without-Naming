"""Sample-budget sensitivity from the stored 20 samples (no new runs).

Strict failure at budget k = greedy wrong and the first k samples all wrong.
Recognition labels are fixed, so TOT:DK at each budget follows directly.

  python budget_curve.py
"""
import json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
R = os.path.join(ROOT, 'results')
SETS = {'EN': ['full_en_qwen7b', 'full_en_qwen14b', 'full_en_qwen32b', 'full_en_qwen72b', 'full_en_llama8b', 'full_en_llama70b'],
        'ZH': ['full_zh_qwen7b', 'full_zh_qwen14b', 'full_zh_qwen32b', 'full_zh_qwen72b', 'full_zh_llama8b', 'full_zh_llama70b']}
def load(p): return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
for lang, sets in SETS.items():
    print('=== %s ===' % lang)
    print('%-8s %8s %8s %8s %8s %10s' % ('budget', 'items', 'fail%', 'TOT', 'DK', 'TOT:DK'))
    for k in (0, 1, 5, 10, 15, 20):
        items = fails = tot = dk = 0
        for s in sets:
            d = os.path.join(R, s)
            rec = {r['item_id']: r['k_rec'] for r in load(os.path.join(d, 'recognition.jsonl'))}
            for p in load(os.path.join(d, 'production.jsonl')):
                items += 1
                sc = p['samples_correct'][:k]
                if p['greedy_correct'] or any(sc):
                    continue
                fails += 1
                kr = rec.get(p['item_id'])
                if kr is None: continue
                if kr >= 3: tot += 1
                elif kr <= 1: dk += 1
        print('%-8s %8d %7.1f%% %8d %8d %10.1f' % ('g+%d' % k, items, 100.0 * fails / items, tot, dk, tot / dk if dk else float('inf')))
    print()
