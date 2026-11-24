# 03 — Transformer Writer

Verifier-bootstrapped compilation of algorithmic sequence tasks into tiny standard PyTorch Transformer weights.

## Hypothesis

In [02-weight-writer](../02-weight-writer/post.md), `Qwen2.5-0.5B` learned to write 2–3 input Boolean MLP weights from an exact checker, but:
1. Held-out generalization stayed at ~0% (lookup-table memorization).
2. Random weight search beat every self-training loop (85.4% vs 42.0%).
3. Gains did not compound across tasks.

Here we test the next step:
* **Target:** Tiny standard PyTorch 1–2 layer Transformers (`d_model = 8–16`, 1–2 heads, standard softmax attention + ReLU MLP), parameterized as sparse weight specifications.
* **Tasks:** Variable-length sequence algorithms (copy, shift, max/min, majority, induction, sort, reverse, Dyck-1, parity, histogram, 2-hop lookup) verified on short lengths (`L ≤ 8`) and evaluated on out-of-distribution lengths (`L = 16..64`) and held-out task variants.
* **Controls:**
  1. Random weight search at matched verifier budget.
  2. Ordinary SGD/Adam training of the exact same tiny Transformer on `L ≤ 8`, tested on `L = 16..64`.
  3. Base model / unverified loop.
