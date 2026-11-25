# Knowing Without Naming

Data, code and per-item records for

> Yuhan Xia, Jinming Zhang, Yucheng Lin, Yunfei Long, and Massimo Poesio.
> **Knowing Without Naming: Availability Without Accessibility in Language
> Models, and the Tip-of-the-Tongue Talk That Misses It.** In *Findings of
> the Association for Computational Linguistics: EACL 2027*.

Repository:
https://github.com/YuhanXia0809/Knowing-Without-Naming

## What this is

A definition-to-word battery for language models in English and Chinese,
with the per-item records behind the paper. Each item gives a definition
and asks the model to name the word (production: greedy plus 20 samples),
then to pick it among four options (recognition: a 4x4 Latin square), and
probes what the model can say or do about a word it cannot produce
(feeling of knowing, in-episode self-report, partial-form probes, the
first-unit cue, reversal, decoy and pseudoword controls). A strict
production failure (greedy wrong and 0/20) with clean recognition
(at least 3 of 4 rotations) is a recognition-positive failure (R+, `TOT`
in the files); with failed recognition (at most 1 of 4) it is a
recognition-negative failure (R-, `DK` in the files). The labels in the
files and scripts are the ones the code has always used; the paper calls
them R+ and R-.

The release contains the **main dataset** of 2,406 items (1,503 English =
501 normed + 1,002 beyond-corpus; 903 Chinese = 402 + 501), a 500-item
English ultrarare extension, a 500-item morphologically opaque rare-word
set, 799 pseudoword controls, a 300-item English non-lexical set, and the
option-set overlays for the adversarial-distractor, decoy-target
(borrowed-filler and option-matched), decoy-cue and pseudoword-cue
conditions, with the raw per-setting records and the analysis scripts.

## Layout

- `data/` item banks, option-set overlays and per-item norm values.
  English items ship in full. For Chinese, **the traditional moedict
  originals ship verbatim** in `data/moedict_source_zh.jsonl`, one entry
  per item with attribution, since CC BY-ND permits redistribution but not
  adaptation; the file covers every Chinese item whose definition comes
  from moedict. **What is
  withheld is the adapted text**: the OpenCC Traditional-to-Simplified
  conversion and the answer-avoiding paraphrase. Each withheld string's
  sha256 is in `data/withheld_sha256_zh.json` (per item file, keyed by
  `item_id` then field). To rebuild it locally (`data/UPSTREAM.json`
  pins the upstream file by sha256 and the package versions):
  1. `code/build_items_zh.py` and `code/build_items_zh_hard.py` read the
     pinned `dict-revised.json` from `$KWN_RESOURCES/moedict/`, convert
     headwords and definitions with OpenCC `t2s`
     (opencc-python-reimplemented 0.1.7), clean the definitions
     (`clean_zh_pool.py`, `filter_zh_propernames.py`) and write candidate
     pools to `$KWN_REBUILD_DIR` (default `rebuild/`, never `data/`); the
     converted definition of a shipped item is the pool entry of its word
     (`gloss_raw` as converted, `gloss` after cleaning). The shipped Chinese
     item files carry neither field, so copy both into a local copy of the
     item file, matching on `word`, before the next step;
  2. `code/judge_screen.py` (Qwen2.5-72B-Instruct, temperature 0, no
     sampling seed, vLLM 0.15.0) writes the answer-avoiding paraphrase;
  3. `code/merge_judged.py --use-paraphrase` puts the paraphrase in
     `gloss` and keeps the cleaned converted text as `gloss_original`;
  4. compare the sha256 of every rebuilt string with
     `data/withheld_sha256_zh.json`. The
     digests are the check: greedy decoding on other hardware or software
     can differ, so a mismatch shows where, not that the procedure failed.

  Model outputs sometimes restate the definition they were shown. So that
  the adapted text does not leave through them either, every Chinese
  results record is masked. Comparison ignores whitespace, punctuation and
  symbols (only letters, digits and Han characters count). Masked are
  (A) every span of 8 or more such characters that also occurs in an
  adapted definition of the record's item (for overlay conditions, of the
  item the definition came from), (B) every occurrence of a complete
  adapted definition of the item of 4 or more characters, and (C) every
  occurrence of a complete adapted definition of any item of 6 or more
  characters; in addition, (L) every range that the same three rules mask
  when whitespace alone is dropped and punctuation counts (spans of 8,
  complete own definitions of 5, complete definitions of any item of 8
  characters). Three quotations written partly in Traditional characters,
  which these Simplified comparisons miss, are masked by explicit entries
  (M): `results/full_zh_r1d14b/production.jsonl` line 548 (`zh_大庭`, two
  spans) and `results/full_zh_qwen7b/production.jsonl` line 349
  (`zh_身心`). Each masked range, from its first to its last matched
  character, becomes
  `[withheld sha256:<first 16 hex of the sha256 of the removed text>]`; the
  item's target word and accepted answers are never masked. A definition,
  or part of one, that consists only of the item's accepted answers
  therefore stays visible; the answer key itself is released. Ranges
  shorter than these thresholds are masked only as parts of longer runs,
  and a placeholder's digest identifies the removed text only to someone
  who already has that text. Keys, numbers and flags in the records are
  untouched and every number printed in the paper is unchanged; a rate
  that a script scores by parsing free text can shift where a placeholder
  removed the parsed text. Three stored values differ in this way from
  the unmasked records: `mimicry.py`, zh/qwen7b TOT claim rate 87.1%
  (unmasked 87.5%); `results/selfreport_full.txt`, ZH Qwen2.5-7B Claim%
  (TOT) 0.87 (0.88); `analyze_pilot.py` on `results/full_zh_llama70b`,
  fresh-context `pinyin_initial`, TOT 0.094 (0.095).
  The stored outputs are computed on the released, masked records, so a
  re-run reproduces them. In all, 10,494 ranges in 77 results
  files are masked.
  - `models.jsonl` (and `MODELS.md`) the model identifiers and revisions.
  - Frequency values: `zipf` and `freq_per_million` come from SUBTLEX for
    the normed pools and from wordfreq for the English beyond-corpus and
    Chinese hard pools (and the sets drawn from them).
  - `items_full_{en,zh}*.jsonl` the main dataset at each construction
    stage (pool sample, judged, final); `*_final.jsonl` is what the
    models were run on.
  - `items_ext_en*.jsonl` the 500-item English ultrarare extension
    (`items_ext_en.jsonl.pool` is its candidate pool) and
    `items_advext_en.jsonl` its adversarial-distractor overlay.
  - `items_dk2_en*.jsonl` the 500-item morphologically opaque rare-word
    set (`items_dk2_en.jsonl.pool` its candidate pool).
  - `items_{en,zh}_rejudge_llama.jsonl` the answer keys rebuilt by
    Llama-3.1-70B on the stratified samples (299 English / 300 Chinese
    items); the original keys of the same items are in
    `items_full_{en,zh}_final.jsonl`.
  - `items_recert_*` the re-certification of the paraphrased definitions
    (`items_recert_en_x400*` the stratified 400-item English sample
    re-certified by Llama-3.1-70B);
    `items_pseudo*`, `items_decoy*`, `items_adv*`, `items_nonlex_en`
    the control and overlay sets.
  - `cohort_{en,zh}.tsv` per-item cohort counts (number of lexicon entries
    sharing the cued word's first letter / first character) used by the
    cohort-size control. Only these per-item values ship; the lexicons
    they are counted from (CLEARPOND, WordNet, CC-CEDICT, SUBTLEX-CH) do
    not.
- `results/` per-item stage records for every dataset and setting in the
  paper (`full_{en,zh}_<model>/`, `ext_en_*`, `dk2_en_*`, `adv*`,
  `decoy*` (including the decoy-target recognition records
  `decoy_{en,zh}_*`), `pseudo*`, `nonlex_en_*`, and the decoding reruns
  `sweep_{t07,p09}_{en,zh}_{qwen7b,llama8b}/`), plus the stored summaries
  of the analyses (`.txt`, a few `.json`, one `.csv`) and `worked_example.json`, the one episode the paper
  prints with its full stage records (item *taps*, Llama-3.1-70B,
  English).
- `code/` build, run, judging, scoring and analysis scripts (see below).

## Record fields

Item files (`data/items_*.jsonl`, one item per line): `item_id`; `word` the
target; `gloss` the definition shown to the model and `gloss_original` the
definition before paraphrase (both withheld for Chinese, see above);
`answers` the accepted answers; `target_option` and `distractors` the
recognition options; `source_pool` and `freq_stratum` the pool and frequency
stratum the item was drawn from; `unique` the judge's verdict that the
definition identifies the word; `zipf` (log10 occurrences per billion
words), `freq_per_million`, `aoa` (years), `concreteness` (1-5 rating),
`n_letters` / `n_chars`, `n_phonemes`, `n_syllables`, `ortho_N`,
`pnd_clearpond`, `ond_clearpond` (neighbour counts), `n_tokens_*` (tokens
under the Qwen2.5 / Llama-3.1 tokenizers); Chinese also `pinyin`, `initial`,
`tone1` (tone of the first syllable, 5 = neutral) and `en_equiv` (CC-CEDICT
gloss). Which fields a file has depends on its construction stage and set;
15 English and 10 Chinese items of the main set have no `distractors` and
so no recognition record.

Per-item records (`results/<set>_<model>/<stage>.jsonl`, one record per `item_id`; `cue`
has one per item and condition):
- `production`: `greedy` and `samples` the model's responses, `greedy_correct`
  and `samples_correct` their scores, `k_prod` of `n_prod` samples
  acceptable, `candidates` the word extracted from each sample.
- `recognition`: `rotations` one entry per Latin-square rotation
  (`target_pos`, `choice`, `gen`, `p_target`, `correct`); `k_rec` of `n_rec`
  rotations correct; `mean_p_target` the target letter's mean first-token
  probability, normalized over A-D. `p_target` is null where no option
  letter was found among the first generated token's probabilities (every
  R1-Distill record, one rotation elsewhere); `mean_p_target` averages the
  non-null values and is 0.0 when there are none.
- `cue`: `condition` (`correct` or `wrong` first-unit cue) and `cue`;
  `greedy_correct`; `k_cue` of `n_cue` post-cue samples acceptable.
- `retest`: the cue pool re-sampled without a cue: `k_retest` of
  `n_retest` samples acceptable; `k_prod` from production.
- `cot`: `group` (`failed` or `produced_control`); `greedy_answer`,
  `greedy_correct` and `trace` (cut to 600 characters before masking) of
  the greedy chain of thought; `k_cot` of `n_cot` samples acceptable.
- `selfreport` (and in-episode `selfreport_ep`): `raw` the reply, `label`
  `TOT`, `DK` or `ANSWER`, `answer_correct` the score of an `ANSWER`.

## How to reproduce

The analysis scripts in `code/` need Python 3 with `numpy`, `scipy`,
`pandas` and `statsmodels` (tested with Python 3.12) and read only the released
records; no GPU is needed. They resolve paths relative to the
repository root (`code/paths.py`, `code/analysis_common.py`), so they can
be started from anywhere:

    python code/fit_gradient.py       # prints, or writes results/<name>.txt

Scripts that write a summary into `results/` (for example
`target_absent.py`, `ratio_ci.py`, `fit_gradient.py`, `regression.py`) write
it exactly as stored, so a re-run reproduces the
stored file byte for byte, with one exception: `hardening.py` writes its
file only when part C can run (it needs OpenCC and the upstream dictionary,
see the optional dependencies below) and otherwise leaves the stored file
as it is. Floating-point values in `results/selfreport_meta3.json` can
differ in their last digits on another machine. By design, an
analysis script regenerates its stored output file in place when run.
Regression fits with (quasi-)separation (fitted probabilities at 0 or 1)
store `null` standard errors and p values with a `separation` note, and
`key_relabel.py` prints the cell counts instead of a coefficient for a fit
with an empty cell. Scripts that only print (for example
`budget_curve.py`, `preference_null.py`, `enrichment_ceiling.py`,
`key_relabel.py`, `paper_tables.py`) print the stored file's content.
No script writes into `data/` by default:
the builders write to `$KWN_REBUILD_DIR` (default `rebuild/`), and
`recert_prep.py`, `cohort_table.py` and `selfreport_meta3.py` need an
explicit output folder. Some usage lines:

    python code/regression.py --items data/items_full_en_final.jsonl \
        --results-root results --lang en --out /tmp/regression_en.json \
        --models full_en_qwen7b,full_en_llama8b,full_en_qwen72b,full_en_llama70b
    python code/analyze_pilot.py --items data/items_full_en_final.jsonl \
        --results results/full_en_qwen7b --lang en
    python code/enrichment_taxonomy.py --family dk2_en \
        --items data/items_dk2_en_final.jsonl

Optional dependencies, needed only for the steps named: vLLM 0.15.0 and
the model weights (`run_stages.py`, `run_stages_sweep.py`, `judge_*.py`);
`opencc-python-reimplemented` 0.1.7, `pypinyin` 0.55.0, `wordfreq` 3.1.1,
`openpyxl` and the upstream files (`build_items_zh*.py`, and part C of
`hardening.py`: without them it prints parts A and B and a skip notice and
leaves `results/hardening.txt` as stored; set `KWN_RESOURCES`); `wordfreq`
also for `gen_pseudo_items.py` and `make_dk2_items.py` (and
`gen_decoy_items.py` uses it when installed); the lexicons named above
(`cohort_table.py`).

### Which script prints which number

- Taxonomy and band accounting: `scoring.py`, `analyze_pilot.py`; the
  fourteen-model taxonomy and self-report tables: `paper_tables.py`
  (`results/taxonomy_full.txt`, `results/selfreport_full.txt`; the
  self-report odds ratios come from `results/selfreport_cells.csv`, written
  by `selfreport_meta3.py` from the records, with `mimicry.py` for the
  performed-TOT claims; `selfreport_sdt.py`, `results/selfreport_sdt.json`,
  the signal-detection analysis of the self-report vs behavioural-state
  dissociation); per-model norm regressions: `regression.py`
  (`results/regression_full_{en,zh}.json`).
- Target-absent and decoy-cue controls (including the decoy-target
  rung and the controls-table column on `results/decoy_*`) and the
  non-lexical control: `target_absent.py`,
  `decoycue_match.py`; certificate perturbation: `label_bootstrap.py`;
  ratio and joint-criterion intervals: `ratio_ci.py`,
  `hardening.py`; the ratio by normed / beyond-corpus pool:
  `pool_stratified_ratio.py`.
- Gloss-fit contrast: `fit_gradient.py`, `fit_ci.py`; scoring-matched decoy
  margins: `score_matched_margin.py`; cue-arm intervals: `cue_ci.py`.
- The matched no-cue retest and the cue-minus-retest increment:
  `retest_contrast.py`, `retest_table.py`; chain-of-thought at its own
  budget: `cot_budget.py`, `cot_matched.py`; the
  renormalization test: `renorm_test.py`.
- Per-condition cue pools and the state contrast models:
  `cue_availability.py`, `heterogeneity.py`, `cue_state_model.py`,
  `cue_meta.py` (with `aoa_tost.py`, `results/aoa_tost.json`).
- Cross-family answer-key rebuild: `rejudge_flip.py`, `mwe_rescore.py`
  (English), `zh_rejudge_state.py` (Chinese); the fully relabelled sample
  ratios and the item-level contrasts under the rebuilt key:
  `key_relabel.py` (`results/key_relabel.txt`); paraphrase
  re-certification: `recert_prep.py`, `recert_compare.py`,
  `recert_sens.py`; P(True) and strong-elicitation FOK: `ptrue_analysis.py`.
- R- scarcity: `enrichment_taxonomy.py` (taxonomy, known-item ceilings,
  FOK and reversal per enrichment set: 643 R+ / 23 R- and 635 R+ / 18 R-),
  `enrichment_ceiling.py` (share of strict failures scoring 4/4,
  84.4 -> 92.2 -> 93.5%, the R- count 113 -> 154, and the known-item
  ceilings per model),
  `pooled_contrast.py` (the in-episode first-unit contrast with the
  enrichment sets pooled).
- Decoding sensitivity (Table: production rerun under tighter decoding):
  `decoding_sweep.py` on `results/sweep_*` (`results/decoding_sweep.txt`);
  the sample-budget curve: `budget_curve.py`.
- The 4AFC chance level that respects item-level preference:
  `preference_consistency.py`, `preference_null.py`.
- Which scoring convention each printed cue and decoy rate uses:
  `cue_conventions.py`, `decoy_conventions.py`.
- Cohort-size control and the same-first-letter contrast:
  `cohort_control.py` on `data/cohort_{en,zh}.tsv` (built by
  `cohort_table.py`).
- Prompts for every task, in both languages: `prompts.py`.

Numbers in the paper that come from human annotation are not recomputable
from this release; the annotation sheets are retained by the authors.

## Licences

Everything we author is under **CC BY 4.0**, and the code additionally under
the **MIT** licence (`LICENSE-MIT`). Upstream sources keep their own licences
and terms; `LICENSE.md` lists each of them with what it is used for. In
short: the WordNet definitions ship in full; the Chinese source definitions
(CC BY-ND 3.0 TW) ship verbatim, and our two adaptations of them are
withheld and rebuilt locally; norms and frequencies ship only as per-item
values; the per-item records contain text generated by the models listed in
`MODELS.md`.

## Citation

Please cite the paper:

```bibtex
@inproceedings{xia-etal-2027-knowing,
  title     = {Knowing Without Naming: Availability Without Accessibility in
               Language Models, and the Tip-of-the-Tongue Talk That Misses It},
  author    = {Xia, Yuhan and Zhang, Jinming and Lin, Yucheng and
               Long, Yunfei and Poesio, Massimo},
  booktitle = {Findings of the Association for Computational Linguistics:
               EACL 2027},
  year      = {2027}
}
```

## Contact

Questions and problems: please open an issue at https://github.com/YuhanXia0809/Knowing-Without-Naming/issues .
