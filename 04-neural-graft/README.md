# 04 — Neural Organ Grafting

Can you transplant a **pre-compiled, sparse 2-layer Transformer circuit** (from `03-transformer-writer`) directly into a **frozen real LLM (`Qwen2.5-1.5B-Instruct`)** via a tiny zero-initialized residual socket — giving the LLM 100% OOD length generalization (`L=4..8` → `L=64`) where standard LoRA collapses?

Write-up: [post.md](post.md). Results below are from the multi-seed runs `robust_graft_exp.py` (`runs/robust_graft_results.json`) and `natural_graft_exp.py` (`runs/natural_graft_results.json`); they replace the first single-seed claims.

## Results (Qwen frozen; all arms trained only on lengths 4–8; up to 5 seeds; tuned LoRA and trained-circuit baselines)

* **Exact at up to 16× the training length.** The written circuit scores `100%` on every seed and every test of three real-token tasks, including 64-digit carry chains and the hard bracket set at length 128. Tuned LoRA: `52.3%` and `53.6%` there; trained circuit of the same shape: `48.3%` and `50.1%` (near chance). Ascending digits at 64: LoRA `84.5%` (seeds `54.9–100%`), trained circuit `62.9%`.
* **Whole sums in normal notation** (`Add: 4821+3957=`, generated digit by digit): written circuit exact in `100%` of cases at 16, 32 and 64 digits and on 64-digit carry chains, 3 seeds. Trained circuit and LoRA: `0%` at 32 and 64 digits.
* **Effect on ordinary text.** The earlier "no damage" result held only for text without `=`. A larger test (prose, Python code, maths) found damage at `=` positions (brackets circuit in code: loss `1.84 → 13.35`). A learned gate of 1,537 numbers on Qwen's last hidden state removes it (all changes `0.00` except `+0.02` for ascending digits in code), with task accuracy still `100%`.
* **Qwen uses the result.** Injected at layer 9 of 28, the bracket circuit's result is used by Qwen's frozen upper layers on unseen fact questions: `94.2–99.9%` by bracket length, versus `88.1–97.9%` when Qwen is told the result in words and `61.1–100%` for LoRA.
* **Model-written circuits.** Of 17 circuits written by the Exp 03 writer model, 14 reach `100%` at length 64 once grafted.
* **Limits.** The circuit does the computing; Qwen does not learn the skill. The combined test has five held-out fact questions. Normal-notation addition needs a fixed position input and equal-length operands.

## Four Arms of the first run (per task, trained ONLY on `L = 4..8`, evaluated on `L = 4..8, 16, 32, 64` + English NLL)

1. **`base`** — Untrained `Qwen/Qwen2.5-1.5B-Instruct` (zero-shot).
2. **`lora`** — Standard LoRA (`r=16`, all attn+MLP projections) trained on `L=4..8`.
3. **`sgd_graft`** — Frozen Qwen + tiny linear socket (`W_in`, `W_out`) + **trainable random `TinyTransformer` organ** (exact same architecture & extra params as `compiled_graft`, trained end-to-end on `L=4..8`).
4. **`compiled_graft`** — Frozen Qwen + tiny linear socket (`W_in`, `W_out`, zero-init `W_out`) + **frozen exact sparse `TinyTransformer` organ** (only the 30K socket params trained on `L=4..8`).
