"""Post-process the decoding-sensitivity reruns (production rerun at T=0.7 and at
top-p 0.9 for Qwen2.5-7B and Llama-3.1-8B, both languages; results/sweep_*).

For each sweep setting: strict-failure rate; TOT:DK using the published recognition
records (item-level, decoding-independent); overlap with the published T=1.0 failure
set; and the correct-cue increment on the INTERSECTION failure set using the
published cue and retest records (retest_contrast convention: any of 20 samples).
Usage: python decoding_sweep.py [<results_dir> [<sweep_results_dir>]]   (both default to results/)
Self-test: pass the released dir twice (sweep == published -> overlap 1.0).
"""
import json, os, sys, collections
def load(p): return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
SWP = sys.argv[2] if len(sys.argv) > 2 else REL
SETTINGS = [(l, m) for l in ('en', 'zh') for m in ('qwen7b', 'llama8b')]
CONDS = [('published_t10', 'full_%s_%s'), ('t07_p10', 'sweep_t07_%s_%s'), ('t10_p09', 'sweep_p09_%s_%s')]
def strict_set(d):
    return {p['item_id'] for p in load(os.path.join(d, 'production.jsonl')) if (not p['greedy_correct']) and p['k_prod'] == 0}
for lang, model in SETTINGS:
    base = os.path.join(REL, 'full_%s_%s' % (lang, model))
    rec = {r['item_id']: r['k_rec'] for r in load(os.path.join(base, 'recognition.jsonl'))}
    n_items = len(load(os.path.join(base, 'production.jsonl')))
    cue = {r['item_id']: r for r in load(os.path.join(base, 'cue.jsonl')) if r['condition'] == 'correct'}
    retest_path = os.path.join(base, 'retest.jsonl')
    retest = {r['item_id']: r for r in load(retest_path)} if os.path.exists(retest_path) else {}
    pub = strict_set(base)
    print('=== %s / %s  (%d items; published strict failures %d = %.1f%%) ===' % (lang, model, n_items, len(pub), 100.0 * len(pub) / n_items))
    for label, pattern in CONDS:
        d = os.path.join(SWP if label != 'published_t10' else REL, pattern % (lang, model))
        if not os.path.exists(os.path.join(d, 'production.jsonl')):
            print('  %-14s (not available yet)' % label); continue
        fails = strict_set(d)
        tot = sum(1 for i in fails if rec.get(i, -1) >= 3); dk = sum(1 for i in fails if 0 <= rec.get(i, -1) <= 1)
        inter = fails & pub
        jac = len(inter) / len(fails | pub) if (fails | pub) else float('nan')
        # cue increment on the intersection, any-of-20 in both arms (published records)
        rows = [i for i in inter if i in cue and (0 <= rec.get(i, -1) <= 1 or rec.get(i, -1) >= 3)]
        cue_rate = sum(1 for i in rows if cue[i]['k_cue'] >= 1) / len(rows) if rows else float('nan')
        if retest:
            rr = [i for i in rows if i in retest]
            re_rate = sum(1 for i in rr if retest[i].get('k_retest', 0) >= 1) / len(rr) if rr else float('nan')
        else:
            re_rate = float('nan')
        print('  %-14s fail %5.1f%%  TOT %4d  DK %3d  TOT:DK %5.1f | overlap with T=1.0 set: Jaccard %.3f, %d of %d in common | cue@20 on intersection %.3f (n=%d), retest@20 %s'
              % (label, 100.0 * len(fails) / n_items, tot, dk, tot / dk if dk else float('inf'), jac, len(inter), len(fails), cue_rate, len(rows), ('%.3f' % re_rate) if re_rate == re_rate else 'n/a (retest records not in release)'))
