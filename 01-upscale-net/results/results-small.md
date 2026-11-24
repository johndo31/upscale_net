| task | source | grown, step 0 | source + same training (control) | grown + healed | pythia-1b | SmolLM2-1.7B | Qwen2.5-1.5B | TinyLlama_v1.1 | OLMo-2-0425-1B |
|---|---|---|---|---|---|---|---|---|---|
| hellaswag (%) | 52.1 | 52.0 | 51.4 | 51.1 | 47.2 | 71.3 | 67.8 | 55.6 | 68.4 |
| arc_easy (%) | 58.4 | 58.2 | 61.0 | 61.4 | 49.2 | 73.5 | 71.5 | 42.8 | 73.4 |
| arc_challenge (%) | 32.5 | 32.1 | 32.7 | 31.1 | 26.9 | 47.4 | 45.1 | 30.3 | 41.9 |
| piqa (%) | 70.3 | 70.0 | 69.9 | 70.4 | 69.7 | 77.4 | 76.1 | 70.2 | 76.0 |
| winogrande (%) | 56.7 | 56.6 | 56.7 | 56.0 | 53.4 | 65.4 | 63.3 | 48.8 | 64.2 |
| gsm8k (%) | – | – | – | – | – | – | – | – | – |
| wikitext (ppl ↓) | 18.45 | 18.46 | 20.37 | 20.27 | 99.72 | 11.06 | 12.59 | 583.64 | 11.55 |

heal-small: 15258 steps, 1000M tokens, adamw lr 3e-05, ~30810 tok/s. Held-out loss 2.618 -> 2.578

control-small: 15258 steps, 1000M tokens, adamw lr 3e-05, ~48772 tok/s. Held-out loss 2.618 -> 2.584
