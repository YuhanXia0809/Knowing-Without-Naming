"""Which scoring convention does each printed cue number use?

Recomputes, from the released records, the per-setting cue and wrong-unit rates of the appendix
cue table under two conventions:
  any-of-20 : the target appears in at least one of the 20 post-cue samples
  any-of-21 : ... or in the post-cue greedy decode
and compares both with the values printed in the paper's appendix cue table.
Also recomputes the pooled-by-label cue rates with the item-cluster bootstrap of code/cue_ci.py.

  python cue_conventions.py
"""
import io, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
MODELS = ['qwen7b', 'qwen14b', 'qwen32b', 'qwen72b', 'llama8b', 'llama70b']
out = []


def say(s=''):
    out.append(s)
    sys.stdout.write(s + '\n')


def jl(n):
    with io.open(os.path.join(ROOT, n), encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def truthy(v):
    return v is True or (isinstance(v, str) and v.strip().lower() == 'true')

# the appendix cue table as printed in the paper: per setting, R+ n / retest / cue / wrong unit,
# then the same for R-
printed = {
    ('en', 'qwen7b'): [395, 12.2, 55.2, 4.3, 30, 0.0, 40.0, 3.3],
    ('en', 'qwen14b'): [331, 8.5, 64.4, 3.3, 26, 3.8, 42.3, 0.0],
    ('en', 'qwen32b'): [321, 12.1, 63.9, 6.9, 20, 0.0, 50.0, 0.0],
    ('en', 'qwen72b'): [253, 10.7, 64.4, 4.7, 18, 5.6, 55.6, 0.0],
    ('en', 'llama8b'): [292, 14.0, 65.1, 6.5, 41, 9.8, 56.1, 2.4],
    ('en', 'llama70b'): [187, 15.5, 68.4, 5.3, 24, 0.0, 50.0, 0.0],
    ('zh', 'qwen7b'): [513, 4.7, 20.3, 0.6, 95, 3.2, 10.5, 1.1],
    ('zh', 'qwen14b'): [469, 3.2, 39.7, 1.9, 80, 2.5, 36.2, 2.5],
    ('zh', 'qwen32b'): [469, 3.6, 39.9, 1.9, 63, 1.6, 28.6, 4.8],
    ('zh', 'qwen72b'): [423, 4.3, 44.2, 0.9, 75, 2.7, 29.3, 1.3],
    ('zh', 'llama8b'): [429, 6.1, 35.2, 1.2, 137, 2.9, 16.1, 0.7],
    ('zh', 'llama70b'): [386, 8.3, 42.7, 1.0, 101, 1.0, 34.7, 0.0],
}

def main():
    rows_by = {}
    for lang in ('en', 'zh'):
        say('=== %s ===' % lang.upper())
        say('%-10s %-3s %5s | cue: printed  any20  any21 | wrong unit: printed  any20  any21' % ('model', 'lab', 'n'))
        for tag in MODELS:
            d = 'results/full_%s_%s' % (lang, tag)
            prod = {r['item_id']: r for r in jl(d + '/production.jsonl')}
            rec = {r['item_id']: r for r in jl(d + '/recognition.jsonl')}
            cue = {}
            for r in jl(d + '/cue.jsonl'):
                cue.setdefault(r['condition'], {})[r['item_id']] = r
            for lab, test in (('R+', lambda k: k >= 3), ('R-', lambda k: k <= 1)):
                ids = [i for i, p in prod.items() if not truthy(p['greedy_correct']) and int(p['k_prod']) == 0
                       and i in rec and test(int(rec[i]['k_rec'])) and i in cue.get('correct', {})]
                n = len(ids)
                c20 = sum(int(cue['correct'][i]['k_cue']) > 0 for i in ids)
                c21 = sum(int(cue['correct'][i]['k_cue']) > 0 or truthy(cue['correct'][i]['greedy_correct']) for i in ids)
                w = [i for i in ids if i in cue.get('wrong', {})]
                w20 = sum(int(cue['wrong'][i]['k_cue']) > 0 for i in w)
                w21 = sum(int(cue['wrong'][i]['k_cue']) > 0 or truthy(cue['wrong'][i]['greedy_correct']) for i in w)
                pv = printed[(lang, tag)]
                off = 0 if lab == 'R+' else 4
                say('%-10s %-3s %5d |      %5.1f  %5.1f  %5.1f |             %5.1f  %5.1f  %5.1f   %s' % (
                    tag, lab, n, pv[off + 2], 100.0 * c20 / n, 100.0 * c21 / n, pv[off + 3], 100.0 * w20 / max(len(w), 1), 100.0 * w21 / max(len(w), 1),
                    '' if abs(pv[off] - n) < 0.5 else '<-- n differs from printed %d' % pv[off]))
                rows_by.setdefault((lang, lab), {})
                for i in ids:
                    rows_by[(lang, lab)].setdefault(i, []).append((int(cue['correct'][i]['k_cue']) > 0, int(cue['correct'][i]['k_cue']) > 0 or truthy(cue['correct'][i]['greedy_correct'])))
        say()

    say('=== pooled by label, item-cluster bootstrap as in code/cue_ci.py (numpy default_rng(26), 2000 resamples per cell) ===')
    rng = np.random.default_rng(26)
    for lang in ('en', 'zh'):
        for lab in ('R+', 'R-'):
            d = rows_by[(lang, lab)]
            items = sorted(d)
            res = {}
            # the published script draws for TOT then DK, en then zh, any-of-21 only; keep that draw order for the any-of-21 column
            takes = [rng.integers(0, len(items), len(items)) for _ in range(2000)]
            for conv, idx in (('any21', 1), ('any20', 0)):
                k = sum(sum(v[idx] for v in d[i]) for i in items); n = sum(len(d[i]) for i in items)
                boots = []
                for take in takes:
                    kk = nn = 0
                    for t in take:
                        v = d[items[t]]
                        kk += sum(x[idx] for x in v); nn += len(v)
                    boots.append(kk / nn)
                lo, hi = np.percentile(boots, [2.5, 97.5])
                res[conv] = (k, n, 100.0 * k / n, 100 * lo, 100 * hi)
            say('%s %s  any-of-21: %d/%d = %.2f%% [%.1f, %.1f]   any-of-20: %d/%d = %.2f%% [%.1f, %.1f]   (%d items)' % (
                lang, lab, res['any21'][0], res['any21'][1], res['any21'][2], res['any21'][3], res['any21'][4],
                res['any20'][0], res['any20'][1], res['any20'][2], res['any20'][3], res['any20'][4], len(items)))
    say()
    say('Printed in the paper\'s appendix: En 62.8% [58.9, 66.4] / 49.1% [35.6, 63.1]; Zh 36.4% [33.5, 39.4] / 24.7% [19.5, 30.6]')
    with io.open(os.path.join(ROOT, 'results', 'cue_conventions.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(('\n'.join(out) + '\n').rstrip("\n") + "\n")


if __name__ == "__main__":
    main()
