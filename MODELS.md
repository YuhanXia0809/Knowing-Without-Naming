# Models

Every evaluated model and both judges, with the full Hugging Face identifier and the snapshot commit the runs served. Each identifier is the `--model` argument of the inference job configuration and each revision the snapshot recorded by the inference engine; one revision per model occurs across all runs. All inference used vLLM 0.15.0. Machine-readable twin: `data/models.jsonl`.

| tag | model | identifier | revision (snapshot commit) | role | context window seen |
|---|---|---|---|---|---|
| qwen7b | Qwen2.5-7B | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | evaluated (primary) | 8192 |
| qwen14b | Qwen2.5-14B | `Qwen/Qwen2.5-14B-Instruct` | `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` | evaluated (primary) | 8192 |
| qwen32b | Qwen2.5-32B | `Qwen/Qwen2.5-32B-Instruct` | `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd` | evaluated (primary) | 8192 |
| qwen72b | Qwen2.5-72B | `Qwen/Qwen2.5-72B-Instruct` | `495f39366efef23836d0cfae4fbe635880d2be31` | evaluated (primary); primary judge (answer sets, uniqueness, paraphrase, reversal) | 4096 / 8192 |
| llama8b | Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | evaluated (primary) | 8192 |
| llama70b | Llama-3.1-70B | `meta-llama/Llama-3.1-70B-Instruct` | `1605565b47bb9346c5515c34102e054115b4f98b` | evaluated (primary); cross-family judge (answer-key rebuild, re-certification, reversal re-judging) | 4096 / 8192 |
| gemma12b | Gemma-3-12B | `google/gemma-3-12b-it` | `96b6f1eccf38110c56df3a15bffe176da04bfd80` | evaluated (sweep) | 4096 |
| gemma27b | Gemma-3-27B | `google/gemma-3-27b-it` | `005ad3404e59d6023443cb575daa05336842228a` | evaluated (sweep) | 4096 |
| glm9b | GLM-4-9B | `zai-org/glm-4-9b-chat` | `bd8234fe5e0c09c48637a92abb0c797cb5fa0e73` | evaluated (sweep) | 4096 |
| mistral7b | Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | evaluated (sweep) | 4096 |
| olmo13b | OLMo-2-13B | `allenai/OLMo-2-1124-13B-Instruct` | `3a5c85baefbb1896a54d56fe2e76c0395627ddf4` | evaluated (sweep) | 4096 |
| olmo32b | OLMo-2-32B | `allenai/OLMo-2-0325-32B-Instruct` | `b96024342a77a69aa0dda815c3454a671f477463` | evaluated (sweep) | 4096 |
| phi4 | Phi-4 | `microsoft/phi-4` | `2db69c1c3e91a05d2c64a3185acfbaf36f744e25` | evaluated (sweep) | 4096 |
| r1d14b | R1-Distill-14B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | `1df8507178afcc1bef68cd8c393f61a886323761` | evaluated (sweep) | 4096 |

**Generation caps (max new tokens).** Evaluated models (code/run_stages.py: production/cue/retest/self-report 16; cue-request first turn 24; FOK verbal, partial, feature ceiling 8; recognition and yes/no probes 4; reversal 80; circumlocution/perform 200; chain-of-thought 400). Judges (code/judge_screen.py: synonyms 80, uniqueness 6, paraphrase 120, verification 4; code/judge_reverse.py: 4; context 4096). Reasoning models run with `--strip-think` get the think budget (768 tokens) as cap. The context window column is vLLM's `max_seq_len` as recorded in the logs; it varied between runs of the same model.

- `olmo13b`: identifier and revision are those the inference engine recorded for the run.
- `r1d14b`: run with --strip-think (think budget 768 tokens, code/run_stages.py). Where the reasoning did not close within the budget, the unterminated text was scored as the answer (greedy production output: 96 of the 1,503 English and 80 of the 903 Chinese items).
