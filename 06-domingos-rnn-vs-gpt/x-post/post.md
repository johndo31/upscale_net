# X Post — Experiment 06 (final version, saved as a quote-post draft on X, 2026-11-26)

> Quote post of https://x.com/pmddomingos/status/2125008371845836829
> Attach: `01-ppl-crossover.png` and `02-pos-loss.png`

I ran it. The Transformers win, and the gap grows with data. The recurrent models lead early only against a GPT with the older sinusoidal position encoding.

Setup as you suggested: perplexity on Common Crawl text (FineWeb-Edu), a small GPT against an LSTM and an RNN, each with softmax layer, residual stream, layer norm and positional encoding. All have about 30M parameters and 6 layers, and see the same data in the same order.

Validation perplexity (lower is better)

First run, after 65M tokens:
LSTM 125
RNN 159
GPT, sinusoidal positions 171

Second run, after 250M tokens:
GPT, rotary positions 58
GPT, sinusoidal positions 67
LSTM 79
RNN 97

In the second run the sinusoidal GPT passes the LSTM between 63M and 94M tokens. The rotary GPT is ahead of the LSTM at every checkpoint, though only just at 31M tokens: 177 vs 181 (chart 1). The first run had no rotary GPT.

Why: the GPTs keep improving with more context, across all 512 tokens. The LSTM stops improving after about 64 tokens, and the RNN gets slightly worse (chart 2).

So "does the same" holds only early in training, and only against the sinusoidal GPT.

Limits: small models, one seed, one learning rate for all, and the two runs are separate. The softmax layer in the RNN is my reading of your description. A bigger test may look different.

Code and logs: github.com/johndo31/upscale_net (folder 06-domingos-rnn-vs-gpt)

---

# Earlier drafts

---

## Option A — Single Post

> Attach: `01-ppl-crossover.png` and `02-pos-loss.png`

Pedro Domingos argued that a modernized RNN (softmax layer, residual stream, LayerNorm, positional encoding) matches a Transformer, and suggested comparing perplexity against MicroGPT on Common Crawl.

I ran the parameter-matched comparison on FineWeb-Edu (~30M total params / 10.6M non-embedding, 6 layers, `seq_len = 512`, identical data stream).

At 65.5M tokens, both recurrent models actually led sinusoidal MicroGPT:
- Domingos-LSTM: 125.4 PPL
- Domingos-RNN: 158.7 PPL
- MicroGPT (sinusoidal): 170.9 PPL

Extended to 250M tokens (and adding RoPE with zero extra params), the ranking flipped completely:
- MicroGPT-RoPE: 58.2 PPL
- MicroGPT (sinusoidal): 66.8 PPL
- Domingos-LSTM: 79.4 PPL
- Domingos-RNN: 97.0 PPL

The per-position loss curve shows why: both GPTs improve monotonically across all 512 tokens (`4.58 → 4.01` nats for RoPE), while LSTM flattens and the RNN U-curves upward past position 64. The Transformers were also ~22–35% faster in wall-clock time.

Caveats: small scale (~30M params), 1 seed, separate 65.5M vs. 250M runs (batch 32 vs. 64), and zero-shot in-context recall stayed near chance (~25%) for all four models.

Code & full JSON logs: github.com/johndo31/upscale_net (`06-domingos-rnn-vs-gpt/`)

---

## Option B — 3-Post Thread

### Post 1/3
> Attach: `01-ppl-crossover.png`

Pedro Domingos claimed a modernized RNN (softmax layer, residual stream, LayerNorm, positional encoding) does the same as a Transformer, and suggested testing perplexity on Common Crawl against MicroGPT.

I ran the parameter-matched test on FineWeb-Edu (~30M params / 10.6M non-embedding, `seq_len = 512`).

In a short 65.5M-token run, the recurrent models led sinusoidal MicroGPT (LSTM 125.4, RNN 158.7 vs. MicroGPT 170.9).

In a 250M-token run, the Transformers overtook both (crossing LSTM between 63M and 94M tokens):
- MicroGPT-RoPE: 58.2 PPL
- MicroGPT (sinusoidal): 66.8 PPL
- Domingos-LSTM: 79.4 PPL
- Domingos-RNN: 97.0 PPL

### Post 2/3
> Attach: `02-pos-loss.png`

Per-position validation loss at 250M tokens explains the gap.

In the first 32 tokens, Domingos-LSTM (4.63 nats) is actually ahead of sinusoidal MicroGPT (4.68) and close to MicroGPT-RoPE (4.58).

Past position 64, the curves diverge:
- Both GPTs drop monotonically all the way to positions 256–512 (`4.01` for RoPE, `4.15` for sinusoidal).
- LSTM flattens (`4.34 → 4.36`), and the modernized Elman RNN curves back up (`4.50 → 4.58`).

At `seq_len = 512`, both GPT variants were also faster on GPU (248–268k tok/s vs. 198–204k tok/s).

### Post 3/3

Setup & caveats:
- 6 layers, `d_model = 384`, ~30M total / 10.6M non-embedding params, identical Pre-LN + residual + GELU MLP + tied LM head, identical FineWeb-Edu batch stream, fixed 131K-token val split.
- The 65.5M (batch 32) and 250M (batch 64) runs are separate runs, not one continuous curve.
- One seed per run, small model scale.
- Zero-shot in-context associative recall was at chance (~17–38%, chance 25%) for all four models, so that test supports no conclusion.

Full code and result logs: github.com/johndo31/upscale_net under `06-domingos-rnn-vs-gpt/`
