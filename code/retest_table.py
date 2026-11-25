"""Per dataset and state, the cue-table numbers with the matched no-cue
retest arm: one block per state (TOT then DK), each block n / arm / arm /
arm. Rates count an item as unlocked when the greedy answer is correct or
at least one sample hits.

  python -u retest_table.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

from analysis_common import PRIMARY6, ROOT, classify, load_jsonl

PRETTY = {'qwen7b': 'Qwen-7B', 'qwen14b': 'Qwen-14B', 'qwen32b': 'Qwen-32B',
          'qwen72b': 'Qwen-72B', 'llama8b': 'Llama-8B', 'llama70b': 'Llama-70B'}
ORDER = ['qwen7b', 'qwen14b', 'qwen32b', 'qwen72b', 'llama8b', 'llama70b']


def unlocked(r, k):
    return 1.0 if (r and (r['greedy_correct'] or r.get(k, 0) > 0)) else 0.0


def main():
    cell = defaultdict(lambda: defaultdict(list))   # (lang,tag)[state] -> rows
    for lang in ('en', 'zh'):
        for tag in PRIMARY6:
            base = os.path.join(ROOT, 'results', f'full_{lang}_{tag}')
            prod = load_jsonl(os.path.join(base, 'production.jsonl'))
            rec = load_jsonl(os.path.join(base, 'recognition.jsonl'))
            ret = load_jsonl(os.path.join(base, 'retest.jsonl'))
            cues = defaultdict(dict)
            with open(os.path.join(base, 'cue.jsonl'), encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        cues[r['item_id']][r['condition']] = r
            for iid, conds in cues.items():
                p, cc, rr = prod.get(iid), conds.get('correct'), ret.get(iid)
                ww = conds.get('wrong')
                if not p or cc is None or rr is None:
                    continue
                st = classify(p, rec.get(iid))
                if st not in ('TOT', 'DK'):
                    continue
                cell[(lang, tag)][st].append(
                    (unlocked(rr, 'k_retest'), unlocked(cc, 'k_cue'),
                     unlocked(ww, 'k_cue')))

    def pct(v):
        return 100.0 * sum(v) / len(v) if v else float('nan')

    print('% Lang & Model & n & retest & cue & plac. & n & retest & cue & plac.')
    pool = defaultdict(lambda: defaultdict(list))
    for lang in ('en', 'zh'):
        for tag in ORDER:
            out = []
            for st in ('TOT', 'DK'):
                rows = cell[(lang, tag)][st]
                pool[lang][st] += rows
                out.append('%d & %.1f & %.1f & %.1f' % (
                    len(rows), pct([r[0] for r in rows]),
                    pct([r[1] for r in rows]), pct([r[2] for r in rows])))
            print('%s & %s & %s & %s \\\\' % (
                lang.upper(), PRETTY[tag], out[0], out[1]))
    for lang in ('en', 'zh'):
        out = []
        for st in ('TOT', 'DK'):
            rows = pool[lang][st]
            out.append('%d & %.1f & \\textbf{%.1f} & %.1f' % (
                len(rows), pct([r[0] for r in rows]),
                pct([r[1] for r in rows]), pct([r[2] for r in rows])))
        print('%s & pooled & %s & %s \\\\' % (lang.upper(), out[0], out[1]))

    print('\n% pooled over TOT+DK together (the numbers the text quotes):')
    for lang in ('en', 'zh'):
        rows = pool[lang]['TOT'] + pool[lang]['DK']
        print('%%   %s n=%d  retest %.1f  cue %.1f  plac %.1f' % (
            lang, len(rows), pct([r[0] for r in rows]),
            pct([r[1] for r in rows]), pct([r[2] for r in rows])))


if __name__ == '__main__':
    main()
