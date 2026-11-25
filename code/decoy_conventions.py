"""Which scoring do the printed decoy-cue rates use? (Appendix D: "Decoy production, in the
greedy decode or any of the 20 samples, runs at ...")

Recomputes decoy production after the decoy's first unit, over all usable decoy items, from the released
records, as (a) any of the 20 samples and (b) greedy decode or any of the 20 samples, and prints both
next to the values printed in the paper.

  python decoy_conventions.py
"""
import io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
PRINTED = {('en', 'qwen7b'): 15.1, ('en', 'llama8b'): 13.2, ('en', 'qwen72b'): 16.9, ('en', 'llama70b'): 18.1,
           ('zh', 'qwen7b'): 10.4, ('zh', 'llama8b'): 14.6, ('zh', 'qwen72b'): 29.2, ('zh', 'llama70b'): 27.7}
out = []


def say(s=''):
    out.append(s)
    sys.stdout.write(s + '\n')


def jl(n):
    with io.open(os.path.join(ROOT, n), encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def truthy(v):
    return v is True or (isinstance(v, str) and v.strip().lower() == 'true')

def main():
    say('setting          n | printed | any of 20 samples | greedy or any of 20 | greedy alone | wrong unit (greedy or any of 20)')
    for (lang, tag), pv in PRINTED.items():
        rows = jl('results/decoycue_%s_%s/cue.jsonl' % (lang, tag))
        c = [r for r in rows if r.get('condition') == 'correct']
        w = [r for r in rows if r.get('condition') == 'wrong']
        n = len(c)
        a20 = sum(int(r['k_cue']) > 0 for r in c)
        a21 = sum(int(r['k_cue']) > 0 or truthy(r['greedy_correct']) for r in c)
        g = sum(truthy(r['greedy_correct']) for r in c)
        w21 = sum(int(r['k_cue']) > 0 or truthy(r['greedy_correct']) for r in w)
        say('%s %-9s %4d |  %5.1f  |       %5.1f       |        %5.1f        |    %5.1f     | %5.1f  (n=%d)' % (
            lang, tag, n, pv, 100.0 * a20 / n, 100.0 * a21 / n, 100.0 * g / n, 100.0 * w21 / max(len(w), 1), len(w)))
    with io.open(os.path.join(ROOT, 'results', 'decoy_conventions.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(('\n'.join(out) + '\n').rstrip("\n") + "\n")


if __name__ == "__main__":
    main()
