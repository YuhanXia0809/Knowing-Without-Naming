"""What chain-of-thought buys, at the chain-of-thought budget.

The headline contrast is cue@20 against retest@20 and needs nothing from this
file. Pricing chain-of-thought against the retest at CoT greedy+5 versus
retest greedy+20 would reintroduce the budget asymmetry the retest was built
to remove, only pointing the other way.

No rerun is needed. The retest arm stores k_retest out of 20, so it can be
downsampled to 5 draws exactly, with no independence assumption:

    P(no hit in 5 | k hits in 20) = C(20-k, 5) / C(20, 5)

Both arms then stand at greedy plus five. The retest greedy is degenerate
(temperature 0 replays production's own wrong answer), so its six attempts are
really five, which if anything favours the CoT arm.

  python -u cot_matched.py --out ../results/cot_matched.txt
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from math import comb

import numpy as np

from analysis_common import PRIMARY6, ROOT, classify, load_jsonl

OUT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    OUT.append(s)


def any_of_m(k, n, m):
    """Exact P(>=1 hit) when m of the n stored draws are kept, k of which hit."""
    if k <= 0:
        return 0.0
    if n - k < m:
        return 1.0
    return 1.0 - comb(n - k, m) / comb(n, m)


def build():
    rows = []
    for lang in ('en', 'zh'):
        for tag in PRIMARY6:
            base = os.path.join(ROOT, 'results', f'full_{lang}_{tag}')
            prod = load_jsonl(os.path.join(base, 'production.jsonl'))
            rec = load_jsonl(os.path.join(base, 'recognition.jsonl'))
            cot = load_jsonl(os.path.join(base, 'cot.jsonl'))
            ret = load_jsonl(os.path.join(base, 'retest.jsonl'))
            cues = defaultdict(dict)
            with open(os.path.join(base, 'cue.jsonl'), encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        cues[r['item_id']][r['condition']] = r
            for iid, conds in cues.items():
                p, ct, rr = prod.get(iid), cot.get(iid), ret.get(iid)
                if not p or ct is None or rr is None:
                    continue
                if classify(p, rec.get(iid)) not in ('TOT', 'DK'):
                    continue
                nt = ct.get('n_cot', 5) or 5
                nr = rr.get('n_retest', 20) or 20
                rows.append({
                    'lang': lang, 'ds': f'{lang}/{tag}', 'item': f'{lang}_{iid}',
                    'cot': 1.0 if (ct['greedy_correct']
                                   or ct.get('k_cot', 0) > 0) else 0.0,
                    'ret20': 1.0 if rr.get('k_retest', 0) > 0 else 0.0,
                    'ret5': any_of_m(rr.get('k_retest', 0), nr, min(nt, nr))})
    return rows


def pooled(rows, a, b, lang, boot=4000, seed=0):
    R = [r for r in rows if r['lang'] == lang]
    d = np.array([r[a] - r[b] for r in R])
    uniq, inv = np.unique(np.array([r['item'] for r in R]),
                          return_inverse=True)
    by = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    bs = np.empty(boot)
    for i in range(boot):
        pick = rng.integers(0, len(by), len(by))
        bs[i] = d[np.concatenate([by[j] for j in pick])].mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return (np.mean([r[a] for r in R]), np.mean([r[b] for r in R]),
            d.mean(), lo, hi, len(R))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'results',
                                                  'cot_matched.txt'))
    args = ap.parse_args()
    rows = build()
    say('=' * 88)
    say('CHAIN-OF-THOUGHT AT ITS OWN BUDGET, AGAINST A DOWNSAMPLED '
        'RETEST')
    say('=' * 88)
    say(f'rows = {len(rows)}')
    for lang in ('en', 'zh'):
        say(f'\n--- {lang.upper()} ' + '-' * 74)
        for label, a, b in (
                ('unmatched budgets             CoT@6 vs retest@20',
                 'cot', 'ret20'),
                ('BUDGET-MATCHED               CoT@6 vs retest@6 ',
                 'cot', 'ret5')):
            ra, rb, d, lo, hi, n = pooled(rows, a, b, lang)
            say(f'  {label}  {ra:.4f} vs {rb:.4f} | '
                f'{d:+.4f} [{lo:+.4f},{hi:+.4f}]  n={n}')
    with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(('\n'.join(OUT) + '\n').rstrip("\n") + "\n")
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
