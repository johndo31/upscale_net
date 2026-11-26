# Weight-Space & Self-Training Experiments

Six controlled experiments testing how far you can push models by **writing, compiling, grafting, or self-distilling weights directly** rather than relying only on standard pretraining.

> **Interactive Report:** Open [index.html](index.html) in any browser for the full synthesis, interactive SVG charts, and live 8-weight circuit simulator.

| # | Folder | Question | Status | Headline Result |
|---|---|---|---|---|
| 1 | [01-upscale-net/](01-upscale-net/) | Can you double a trained LLM (`Qwen2.5-0.5B` → `1.05B`) via exact function-preserving growth and heal it cheaply? | Done (`small`) · Polling G4 (`big` 8.22B) | `2.578` vs `2.584` loss (`-0.006` nats); tied on 5 benchmarks ([post.md](01-upscale-net/post.md)) |
| 2 | [02-weight-writer/](02-weight-writer/) | Can `Qwen2.5-0.5B` teach itself to write tiny neural network weights from nothing but an exact checker? | Done | `0% → 42.0%` train (`0%` held-out); random search beat it (`85.4%`) ([post.md](02-weight-writer/post.md)) |
| 3 | [03-transformer-writer/](03-transformer-writer/) | Can a verifier-bootstrapped LLM compile variable-length algorithms into tiny Transformer weights — beating SGD on `L=64` length generalization? | Done | `18/32` exact at `L=64` vs SGD `10–12/32` & random `1/32`; invented 8-weight `all_0` veto head; `0` multi-turn repairs at 3B & 7B |
| 4 | [04-neural-graft/](04-neural-graft/) | Can you graft a frozen compiled Transformer organ into a real LLM (`Qwen2.5-1.5B`) so it generalizes to `L=64` where LoRA and SGD grafts collapse? | Done | Written circuit `100%` on every seed & test up to 16× training length (64-digit carry chains, hard brackets at `L=128`) vs tuned LoRA `52.3%`/`53.6%` & trained circuit `48.3%`/`50.1%`; whole sums in normal notation exact at 64 digits; damage at `=` removed by a 1,537-number gate; injected at layer 9 of 28, Qwen uses the result on unseen fact questions (`94–100%`) about as well as told in words (`88–98%`); 14/17 model-written circuits graft at `100%`. Limits: the circuit computes, Qwen does not learn the skill; 5 held-out questions; normal notation needs position input & equal-length operands ([post.md](04-neural-graft/post.md)) |
| 5 | [05-context-to-weights/](05-context-to-weights/) | Can a frozen LLM internalize 16 unseen documents from its context window into its weights with zero labels — and do Gram-Schmidt orthogonal LoRA slots stop catastrophic forgetting? | Done | Orthogonal rank-2 merged slots retained `4.3×` more facts (`26.6%` vs `6.2%`) w/ no router; downstream interference still pushed NLL `2.81 → 3.13` |
| 6 | [06-domingos-rnn-vs-gpt/](06-domingos-rnn-vs-gpt/) | Parameter-matched modernized RNN & LSTM (residual, LayerNorm, pos enc, softmax) vs. MicroGPT on Common Crawl perplexity & in-context recall | Done | At 65.5M tokens (batch 32, 4k steps) `domingos_lstm` `125.4` & `domingos_rnn` `158.7` PPL lead `microgpt` `170.9`; at 250M tokens (batch 64, 7.6k steps) `microgpt` `66.8` & `microgpt_rope` `58.2` lead LSTM `79.4` & RNN `97.0`. `microgpt_rope` leads the LSTM at every checkpoint of the 250M run (176.8 vs 180.8 at 31M tokens); the recurrent models lead early only against sinusoidal `microgpt`, which passes the LSTM between 63M and 94M tokens; two separate runs, one seed each, ~30M params; recall at chance in both |

Licence: MIT.

Author: Niklas — [drniklas.com](https://drniklas.com) · podcast: [venturegrade.drniklas.com](https://venturegrade.drniklas.com)
