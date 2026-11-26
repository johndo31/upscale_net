# 06 — Pedro Domingos's Modernized RNN vs. MicroGPT on Common Crawl

Empirical test of Pedro Domingos's claim (Nov 23, 2026):
> *"Transformers have no reason to exist at this point. An RNN with softmax layer, residual stream, layer norm and positional encoding does the same."*
> *"Sure, apply it to Common Crawl or such and compare perplexity with MicroGPT."*

## Design (strictly parameter-matched, token-matched, and wall-clock-tracked)

* **Dataset:** Streamed Common Crawl (`HuggingFaceFW/fineweb-edu`, `sample-10BT`), GPT-2 tokenizer, `seq_len = 512`, identical batch stream across all 3 models, fixed 131K-token held-out validation split.
* **Three parameter-matched models** (~10.6M non-embedding params / ~30M total, 6 layers, `d_model = 384`, identical sinusoidal positional encoding, Pre-LayerNorm, residual stream, GELU MLP, tied softmax LM head, AdamW + cosine LR):
  1. **`MicroGPT`** — standard causal multi-head self-attention (6 heads)
  2. **`Domingos-LSTM`** — cuDNN LSTM + Pre-LN + residual stream + pos enc + GELU MLP + softmax LM head
  3. **`Domingos-RNN`** — plain Elman RNN + softmax gating layer + Pre-LN + residual stream + pos enc + GELU MLP + softmax LM head
* **Metrics:**
  1. Held-out Common Crawl perplexity vs. tokens **and** vs. GPU wall-clock seconds.
  2. Per-position loss across context (`pos 1..32` → `256..512`).
  3. Zero-shot in-context associative recall (copying a novel code from `32 / 128 / 320` tokens earlier in context).
* **250M-token run** adds a fourth model, **`MicroGPT-RoPE`** (same as `MicroGPT`, rotary position encoding instead of the sinusoidal one, no extra parameters).

## Results

Validation perplexity at the end of each run (lower is better):

| Run | `microgpt` | `microgpt_rope` | `domingos_lstm` | `domingos_rnn` |
|---|---|---|---|---|
| 65.5M tokens (batch 32, 4,000 steps) | 170.92 | — | **125.38** | 158.71 |
| 250M tokens (batch 64, 7,630 steps) | 66.84 | **58.16** | 79.43 | 96.96 |

* **The Transformers win at 250M tokens; the recurrent models lead early only against the sinusoidal GPT.** At 65.5M tokens both recurrent models are ahead of `microgpt` (that run had no rotary model). At 250M tokens both GPT variants are ahead of both recurrent models. Within the 250M run, `microgpt_rope` is ahead of the LSTM at every checkpoint, though only just at the first (176.8 vs 180.8 at 31M tokens); the LSTM is ahead of sinusoidal `microgpt` at 31M and 63M tokens and behind from 94M tokens on. So the early lead of the recurrent models may come from the weaker position encoding of the baseline.
* **The claim "an RNN with these additions does the same" is not supported at 250M tokens** in this small setup.
* **In-context recall is at chance** (about 17–38%, chance 25%) for all four models in both runs, so the recall test supports no conclusion.
* **Caveats.** The two runs differ in batch size and step count, so they are two separate runs, not one curve. One seed each. Models have about 30M parameters (10.6M non-embedding).

Result files: `runs/domingos_results.json` (65.5M), `runs/domingos_250m_results.json` (250M).
