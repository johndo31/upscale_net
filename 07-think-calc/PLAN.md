# Experiment Roadmap — `07-think-calc` & Next Steps

**Working title:** *Grafting: Adding an Inline Neural Co-Processor Inside a Model's Thinking*
**Subtitle / Lead:** *I gave a frozen thinking model a 100%-exact inline calculator and it still failed because it stared at `=53623887026008` and muttered "(Wait, is that right? Let me check.)"*

---

## Planned Sequence

### 1. `07a` — `r1-1.5b-diag` Diagnostic Run *(in progress on Colab L4)*
- **Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (frozen) + compiled 2-layer `+`/`-` `SparseOrgan` (`3,561` nnz) + `W_out` (`18,432` params, 12th trailing-space channel) + `GateProbe` (`1,537` params).
- **Goal:** Pinpoint where answers are lost on 9–14 digit word problems via per-problem `traces.jsonl` and 8 failure tags (`truncated_think`, `hit_token_limit`, `hit_limit_before_boxed`, `no_final_answer`, `wrote_big_binary_eq`, `column_by_column_only`, `wrote_multi_term_sum`, `missing_input_numbers`), plus fair chat-template Eval 1, gated Eval 1, and problem-level held-out real-`<think>` gate F1 / Python `=` NLL damage.

### 2. `07b` — Co-Processor Awareness Test on `R1-1.5B` *(Colab L4, ~1–2 units)*
- **Hypothesis:** A frozen thinking model was RL-trained not to trust itself on big sums, so it reflexively breaks 12-digit numbers into 20 lines of 1-digit column additions instead of writing `a + b =` directly.
- **Test:** Run the exact same frozen `R1-1.5B` graft with a 1-line system prompt (given equally to Base and Grafted) instructing the model to write full binary equations directly as `a + b =` / `a - b =` without digit-by-digit column breakdowns (and with sufficient `max_new_tokens`).
- **Metrics:** Final-answer accuracy delta (Base vs. Grafted), `<think>` token reduction, re-check/loop count, and `graft_recheck_rate`.

### 3. `07c` — Scale to Stronger Thinking Models *(Colab A100)*
- **Models:** `Qwen/Qwen3-4B` first, then `XiaomiMiMo/MiMo-7B-RL`.
- **Goal:** Test the winning setup against a **non-zero** 4B/7B baseline where the host model copies operands and plans multi-step word problems cleanly. Measure whether the inline neural co-processor both beats the baseline on 12–16 digit word problems and shortens `<think>` verification loops (e.g., MiMo overthinking loops).
- **Engineering note:** Batch Base-arm generation in Eval 2/3 if needed to keep A100 runtime/cost tight.

### 4. `08` — Option 2: Full Inline Math Kernel Inside `<think>` *(follow-up)*
- **Idea:** Keep the exact same frozen host + tiny hidden-state `GateProbe`, but swap the pure-weight `+`/`-` Transformer circuit for a tiny safe deterministic math/CAS evaluator (`+ - * / ^ sqrt`, decimals, fractions, multi-term expressions) running inline in `<0.1 ms` per `=` with zero Python sandbox and zero tool-calling RL.
- **Benchmark:** Hard multi-step GSM8K / MATH / big-number word problems — direct 3-way comparison of **Base** vs. **Pure-Weight `+`/`-` Organ (`07`)** vs. **Inline Math Kernel (`08`)** on accuracy, `<think>` token savings, latency, and portability.

---

## Key Prior Art to Cite in Write-Up
- **Sah (Nov 2026, arXiv:2611.26450):** *A CPU Inside an LLM* (zero-gradient width-extended 6-digit ALU compilation into Granite 3B / Qwen 4B).
- **Gilerson (NeurIPS 2026):** *Don't Learn What You Can Compute* (frozen SmolLM2-360M + frozen RNS math block via gated injection + `lm_head` LoRA).
- **Josiah Wilson:** *Typed Neural Firmware* (frozen Qwen2.5-0.5B + learned router + deterministic ripple-carry cell).
- **Vergara-Browne & Soto (ACL Findings 2025, arXiv:2505.10719):** *Tracr-Injection* (distilling Tracr/RASP programs into LLM weights).
- **Dietz & Klakow (Jan 2025, arXiv:2501.00684):** *IGC* (gated non-differentiable GPU calculator fine-tuned into Llama 3.1 8B).
- **Foundations:** RASP (Weiss et al., 2021), Tracr (Lindner et al., 2023), CALM (Bansal et al., ICLR 2024), Abacus Embeddings (McLeish et al., NeurIPS 2024).

---

## Article Notes & Framing (for X Post / Write-Up)
- **Core distinction vs. prior art:** Prior 2025–2026 work (`Sah 2026`, `Gilerson 2026`, `Wilson`) tests one-shot `prompt -> answer` arithmetic. Embedding a gated co-processor **inside `<think>`** tests whether we can keep multi-step reasoning straight without tool-call syntax, sandboxes, or weight retraining.
- **Why it matters for local & open-weight models:** Local runtimes (Ollama, MLX, llama.cpp) don't ship Dockerized Python sandboxes; 4B–14B models actually trip on arithmetic; at 20–50 tok/s every wasted `<think>` token costs real seconds; an ~8 KB linear gate can upgrade any frozen open-weight release in ~2 min with zero re-RL.
- **Why it's even more efficient for frontier labs (CPU ALU vs. OS syscall):** Pausing continuous-batching GPU clusters for a 50–300 ms Python container round-trip hurts throughput and KV-cache residency, and burning 400 `<think>` tokens on manual carries wastes 400 giant forward passes. CPUs don't skip having an on-chip FPU/ALU just because the OS has syscalls — an inline fast-path co-processor for instant exact math *plus* Python for heavy code is strictly cheaper and faster.
- **Pure-weight organ (`07`) vs. deterministic math kernel (`08`):** Be honest about the trade-off — Option 2 (same hidden-state gate + `<0.1 ms` safe math/CAS kernel) wins for real-world shipping (all of math, zero repetition-logit tug-of-war), while pure-weight compiled grafting (`04`/`07`) wins for 100%-in-weights portability and **mid-layer latent injection** (which a token-level kernel cannot do).
- **The messy-`<think>` lesson (`07a` → `07a-v2`):** A frozen RL model doesn't know it has a co-processor — it inserts (sometimes mis-grouped) commas, uses vertical newlines, and grinds 1-digit columns. Adapting the graft to natural `<think>` formatting (`07a-v2`) vs. prompt/habit awareness (`07b`) vs. stronger 4B/7B hosts (`07c`) tells the complete, honest story.

