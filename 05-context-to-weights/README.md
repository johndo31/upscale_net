# 05 — Zero-Label Context-to-Weights Internalization

Can a frozen LLM (`Qwen2.5-1.5B-Instruct`) read 16 completely novel documents/rules in its context window and **bake them permanently into its weights with ZERO labeled training data** (matching its own with-context distribution via KL self-distillation on unlabeled questions) — and can **Gram-Schmidt orthogonalized LoRA slots** prevent catastrophic forgetting as cards 1 → 16 accumulate into a single merged weight matrix?

## Arms (sequential absorption of 16 unseen knowledge cards, evaluated WITH ZERO CONTEXT)

* **`with_context` (ceiling)** — Card kept in the system prompt.
* **`no_context` (floor)** — Base model with card deleted (~0%).
* **`sft_shared_lora`** — Sequential hard-label SFT on a shared rank-16 LoRA (using the model's own with-context rollouts).
* **`kl_shared_lora`** — Sequential full-logit KL self-distillation (`KL(P_base(·|card,q) || P_student(·|q))`) on a shared rank-16 LoRA.
* **`kl_orthogonal_lora`** — Sequential KL self-distillation into **Gram-Schmidt orthogonalized rank-2 LoRA slots** (`A_k ⊥ span(A_1..A_{k-1})`), all merged into one weight matrix with no external router.
