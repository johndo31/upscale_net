# Process log: growing Qwen2.5-0.5B to 1B, and an 8B run left on hold

Chronological record of what was done, what was measured, and what was
decided. The blog post is written from this file.

## Goal

Take an MIT-licensed ~1B open-weight model from Hugging Face and turn it into an
~8B model whose extra capacity can be trained ("healed") instead of training an
8B model from scratch.

## Day 1 (2026-11-03)

### Starting point

The idea came from reading "Sparks of Weight-Space Programming" (Takehi, Teoh,
Zhang, Khattab, MIT CSAIL). That post is about an LLM reading and editing
weights directly. It does not cover growing a network, so the method here is
the classic one instead: function-preserving growth (Net2Net, Chen et al. 2015;
depth-up-scaling as in SOLAR; zero-initialised output projections as in LiGO /
bert2BERT).

### Model choice

Hugging Face API, filter `license:mit`, `pipeline_tag:text-generation`, sorted
by downloads. Candidates near 1B with MIT weights:

| model | params | arch | notes |
|---|---|---|---|
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.78B | Qwen2 | most downloaded MIT model near 1B; reasoning distill |
| microsoft/rho-math-1b-v0.1 | 1.1B | Llama | maths-specialised |
| microsoft/phi-1_5 | 1.3B | Phi | non-standard block (parallel attn+MLP) |

Picked DeepSeek-R1-Distill-Qwen-1.5B: plain Qwen2 architecture (RMSNorm,
SwiGLU, GQA, RoPE, head_dim 128), the same family as Llama 3 8B, so the target
can be exactly the Llama-3-8B shape.

For proving the method, Qwen/Qwen2.5-0.5B (same architecture class, Apache
licence, 494M params) is used because it is small enough for exact float32
comparisons on a 16 GB Mac.

### Machine

Apple M1, 16 GB unified memory, ~50 GB free disk. Python 3.13 venv with
torch 2.15, transformers 5.22, lm-eval 0.4.13.

### Growth recipe (`grow.py`)

Width d -> D: the residual stream is zero-padded. Writers into the residual
(embedding, o_proj, down_proj) get zero rows/columns for the new dims. Readers
(q/k/v, gate/up, lm_head) get random columns so the writers receive gradient.
RMSNorm over a zero-padded vector changes the mean of squares by d/D, which is
cancelled exactly by `w' = w * sqrt(d/D)` and `eps' = eps * d/D`.

Heads: old query heads are placed into new GQA groups whose KV head is a copy
of their original KV head. New heads are random with zeroed o_proj columns.

MLP: new gate/up rows random, matching down_proj columns zero.

Depth: new blocks are copies of a neighbour with o_proj and down_proj zeroed,
so each is the identity on the residual stream. They are spread evenly.

Everything streams tensor by tensor into safetensors shards; peak RAM is about
one layer.

### Proof on Qwen2.5-0.5B

```
python grow.py --src Qwen/Qwen2.5-0.5B --dst <out> \
  --hidden 1280 --intermediate 6912 --layers 28 --heads 20 --kv-heads 4 --dtype fp32
```

0.494B -> 1.048B params (2.12x), 18 s on CPU.

`verify.py` in float32, four prompts, 34 token positions:

| metric | value |
|---|---|
| max abs logit difference | 1.07e-4 |
| mean abs logit difference | ~7e-6 |
| argmax agreement | 34/34 |
| greedy 16-token continuations identical | 4/4 |

That is float32 rounding noise. The grown model is the same function.

### 8B growth

```
python grow.py --src deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --dst models/deepseek-r1-qwen-grown-8b \
  --hidden 4096 --intermediate 14336 --layers 32 --heads 32 --kv-heads 8 --dtype bf16
```

1.777B -> 8.224B params, 16.4 GB in bf16. Shape is Llama-3-8B's. Head map:
12 old query heads live in 4 of the 8 KV groups (two copies each of the two
original KV heads); 20 query heads and 4 KV heads are new.

### Healing plan

The grown model behaves exactly like its source until trained. Training needs
weights + gradients + optimizer state, so:

* 1B test model: fits on the Mac with fp32 weights, bf16 autocast, Adafactor,
  gradient checkpointing (`heal.py`). Used for the demonstration.
* 8B model: does not fit in 16 GB. Needs a rented GPU (one or more 80 GB
  cards). `heal.py --device cuda` is the same script.

### Benchmarks

`bench.sh` runs hellaswag, arc_easy, piqa, winogrande (300 examples each) and
wikitext perplexity through lm-eval on MPS. Measured at: source, grown step 0
(must equal source), grown after healing.

_(results filled in below as they arrive)_

### Decisions after the proof (same day)

* Skip all heavy runs on the Mac; rent a GPU instead (see `runpod.md`). The
  Mac is only used to validate scripts with tiny smoke tests.
* Local 8B verification was stopped: the 8B does not fit in 16 GB RAM and
  disk offload was too slow. It is verified on the GPU box as step 2 of
  `run_all.sh` instead.
* `bench.sh` crashed on wikitext (27 GiB buffer): the models advertise 32k to
  131k context and lm-eval scored wikitext in windows that long. Fixed by
  capping `max_length=4096` and running wikitext at batch 1.
* Bug found by the smoke test: `heal.py` shifted labels by one *and* HF's loss
  shifts internally, so the model was trained to predict token t+2. Held-out
  loss read 9.5 instead of 3.3. Fixed; loss is now 3.34 on fineweb-edu for
  Qwen2.5-0.5B, which is the expected range.
* Risk noted: DeepSeek-R1-Distill is a reasoning model and web-text healing
  could erase its think-then-answer habit while the general benchmarks look
  fine. Mitigations wired in: (1) GSM8K added to `bench.sh` for the big run
  (`BENCH_REASONING=1`), (2) `heal.py --mix-dataset open-r1/OpenR1-Math-220k
  --mix-ratio 0.2` blends DeepSeek-R1 reasoning traces into the healing data,
  rendered through the chat template.
* Control added: the source model is trained on the same tokens with the
  same settings (`CONTROL=1`, default). The claim "growth helped" only holds
  if grown+healed beats source+same training.
* Prior work found (cite in the post): Net2Net 2015; Masked Structural Growth
  2023; HyperCloning (Apple 2024); Stacking Your Transformers 2024; SOLAR,
  Yi-9B, LLaMA Pro as depth-grown production models. The weight-space
  programming post is inspiration, not method.

### Self-healing without an external corpus

Question raised: can the healing be done by the small model itself, with no
external training text? Pure self-imitation cannot help: the grown model is
an exact copy of the source, so the expected gradient of imitating the
source's own samples is zero. What does work is rejection-sampled
self-distillation (STaR-style): the source model solves GSM8K and MATH
training problems several times, only solutions with the right final answer
are kept (the correctness filter is the learning signal), and the grown model
trains on those. Implemented in `self_distill.py`; `HEAL_DATA=self` in
`run_all.sh` runs that variant with the same control. It mostly improves
maths reasoning, so GSM8K is the headline metric for it and the general
benchmarks may not move. Both variants are run on the GPU box and compared.

### Script validation on the Mac (smoke tests, all passing)

* `heal.py` fineweb: held-out loss 3.34 for Qwen2.5-0.5B (was 9.5 before the
  label-shift fix). Saves bf16 safetensors; only the latest intermediate
  checkpoint is kept and it is removed once the final model is saved.
* `heal.py --mix-dataset open-r1/OpenR1-Math-220k`: streams in 6 s, traces
  contain `<think>`, rendered through the chat template. Works.
* `self_distill.py` on 4 GSM8K problems x 2 samples with the 0.5B base: 1 of
  8 kept (the base model is weak at maths; the DeepSeek distill will keep far
  more). `heal.py --dataset runs/.../data.jsonl` trains on it and loops
  epochs over the small file. Works.
* `bench.sh`: wikitext fixed (max_length 4096, batch 1); the five
  multiple-choice tasks run at batch 16. GSM8K added behind `BENCH_REASONING=1`.
* `results.py` produces the comparison table (source / grown step 0 /
  control / grown+healed) plus healing-run summaries.

Everything is packed by `make_bundle.sh` into `bundle.sh` for pasting into a
RunPod terminal. Next step: run `run_all.sh` there (see `runpod.md`).

### GPU session 1 (2026-11-03, RunPod)

Secure Cloud showed "out of capacity" for every card for ~20 minutes; after
switching the deploy filter to Community Cloud and reloading, an RTX PRO 6000
Blackwell Workstation (96 GB) was available at $1.70/h and was deployed as pod
`upscale-test`. The account has no SSH key registered, so a dedicated
passphrase-free key (`~/.ssh/runpod_ed25519`) was added to the pod through
the web console; `pod.sh` wraps ssh/scp with it. Pip is PEP-668 locked on the
template, so deps live in `~/venv` (system site packages, keeps torch 2.8).

Sanity run: `LIMIT=50 ./run_all.sh small 10000000`.

Finding: with a bf16 grown checkpoint the verify step shows max |Δlogit| 1.7,
mean 0.08, argmax agreement 34/34, 3/4 greedy continuations identical (the
4th diverges after the EOS token). Cause: the norm weights are multiplied by
sqrt(d/D) and then rounded to bf16, a ~0.4% perturbation per norm. The Mac
proof in fp32 gave 1e-4. Fix: the small experiment now stores the grown model
in fp32 (4 GB) and verification always runs in fp32.

### GPU session 2: B300 -> H100 (2026-11-03 afternoon)

* Secure Cloud B300 (`upscale-1b`, $9/h) measured 52k tok/s on the 1B model,
  only ~3x the RTX PRO 6000, so ~$0.05 per 1M tokens, about 1.5x an H100's
  cost per token. Switched to a Secure H100 SXM (`upscale-1b-h100`, $4/h,
  47 GB volume at /workspace). Baseline benchmarks from the B300 were synced
  to the Mac and copied over so `run_all.sh` skips them.
* H100 verify in fp32: max |Δlogit| 5.6e-5, argmax 34/34.
* First H100 healing attempt OOM'd (80 GB): batch 16 x 2048 with the 152k
  vocab needs ~20 GB for logits alone. Changed small mode to batch 8 x
  grad-accum 4 (same 65,536 tokens/step). Budget: 1B tokens, ~15.3k steps.

### Learning-rate correction (H100 run, 2026-11-03 evening)

With peak lr 1e-4 and a 65,536-token batch the held-out loss went 2.62 (step
0) -> 2.82 (100) -> 2.90 (300) -> 2.89 (400) -> 2.87 (500): the pretrained
weights were being knocked off their minimum and the budget would have gone
into repair. The original pretraining used batches ~60x larger, so 1e-4 here
is a much bigger step. Restarted with peak lr 3e-5 (same warmup 100, cosine
to 10%). The 1e-4 log is kept as `runs/heal-small-lr1e-4.jsonl` for the post.
Cost of the false start: ~35 min of H100, ~$2.30.

### Reference models (added during the H100 run)

To give the numbers scale, `bench_refs.sh` scores open ~1B base models on
the same suite: Qwen2.5-1.5B (same family, 18T tokens), SmolLM2-1.7B (11T),
OLMo-2-0425-1B (4T), TinyLlama 1.1B (3T), Pythia-1B (300B, the fairest
comparison on training budget). Llama-3.2-1B is gated and left out. Run on
the H100 concurrently with healing (GPU had ~30 GB headroom). Expectation,
stated before seeing results: grown+healed lands near its 0.5B source and
below the real 1B models; Pythia-1B is the one it might match.

lr 3e-5 run, held-out loss: 2.62 (0) -> 2.65 (100) -> 2.68 (300) -> 2.68
(500) -> 2.68 (600). Bump of +0.05 instead of +0.25; waiting on step 1500 to
confirm the downward trend.

### Future work (collected during the run, for the post and the next session)

1. **LLM-written initialization (the paper's method applied to growth).** A first version from weight and activation statistics ran on day 2; raw-weight reading is still open. A frontier
   model reads the small model (weights plus the activity statistics from
   `analyze.py`), decides which heads/neurons to copy, split, perturb or replace,
   and writes the rule or the weights. Needs the paper's harness and long context;
   each variant costs one 100M-token healing run (~$4) to compare.
2. **Recursive self-improvement ladder.** (1) rule-written init, no model in the
   loop [done on day 2: zero and split]; (2) self-signal init: the model's own activations choose what to
   split, a human/frontier model writes the rule [done on day 2: guided]; (3) a frontier model
   writes the rule itself; (4) the grown model does (3) on itself. (3) needs the
   paper's harness; (4) needs a model that can reason about weights, which neither
   the 0.5B nor the 1B can. A 0.5B-instruct "pick neurons from a table" demo would
   be a predictable failure; decided not to spend the $4.
3. **Scale the healing budget.** 5B tokens (~$280) to see the new capacity clearly
   in use; 20B (~$1,100) for a compute-optimal 1B-class model.
4. **The 8B.** DeepSeek-R1-Distill-Qwen-1.5B -> 8.2B (Llama-3-8B shape) with the
   reasoning-trace mix and GSM8K; 100M tokens (~$30) shows whether reasoning survives
   healing; 500M+ to beat the 1.5B control. On hold.
5. **Self-distillation healing** (`self_distill.py`, `HEAL_DATA=self`): verified
   solutions written by the source model as the only training data. GSM8K is the
   metric. Cheap (~$4 generation + short healing).
6. **Exactness vs. noise.** The 2024/2026 papers disagree on whether exact function
   preservation matters; compare zero-init vs. small-noise-on-new-parts vs. split
   under matched 100M-token schedules (same harness as the init ablation).
7. **Width growth of the residual stream.** New residual dims are zero at init and
   only come alive through the random reader columns; a split-style scheme for
   residual dims needs an integer d/D ratio (or a norm rewrite) to stay exact.
8. **Publish models.** Push healed/control models to the Hugging Face Hub as private
   repos, flip to public with the post.

### Reference models and the self-read of the 0.5B (H100 run, hour 2)

Reference benchmarks (full task sets, bf16, max_length 4096):

| task | Qwen2.5-0.5B (source) | grown step 0 | Pythia-1B | TinyLlama 1.1B | OLMo-2-1B | Qwen2.5-1.5B | SmolLM2-1.7B |
|---|---|---|---|---|---|---|---|
| hellaswag | 52.1 | 52.0 | 47.2 | 55.6 | 68.4 | 67.8 | 71.3 |
| arc_easy | 58.4 | 58.2 | 49.2 | 42.8 | 73.4 | 71.5 | 73.5 |
| arc_challenge | 32.5 | 32.1 | 26.9 | 30.3 | 41.9 | 45.1 | 47.4 |
| piqa | 70.3 | 70.0 | 69.7 | 70.2 | 76.0 | 76.1 | 77.4 |
| winogrande | 56.7 | 56.6 | 53.4 | 48.8 | 64.2 | 63.3 | 65.4 |
| wikitext ppl | 18.45 | 18.46 | 99.7 | 583.6 | 11.55 | 12.59 | 11.06 |

Grown step 0 equals the source within bf16 benchmark noise, as it must. The
wikitext perplexities of Pythia and TinyLlama look off (likely tokenizer /
detokenisation interaction in the harness); report accuracies for references
and treat their ppl with caution.

`analyze.py` on 200k fineweb-edu tokens: no dead neurons in any layer; load
is highly skewed (per-layer max |activation| 10-100x the mean, 30.2 vs 0.37 in
layer 23); heads come in two GQA groups of 7, and in layers 1, 17, 20, 21 one
group carries 3-4x the output norm of the other. Guided init rule written from
this: new neurons copy the highest-activity neurons, new heads copy the
highest-norm heads of their group, outputs halved (exact: fp32 max |Δlogit|
7.8e-4, argmax 34/34). Queued as the third ablation variant (zero / split /
guided, 100M tokens each, identical schedule).

## Day 2 (2026-11-04) and the morning after

### The host died

At 19:40 UTC on day 1 the H100 host in AP-IN-1 failed (Docker daemon gone, SSH
refused, RunPod console: "critical error on this machine"). The run was at step
7,852 of 15,258, held-out loss 2.623 and falling. Everything lived on the pod's
volume disk, which is tied to that host, so the run was lost. The pod kept showing
as running at $4/h until I stopped it at 06:01 UTC. A refund ticket for the 4.7 h
of lost training plus the dead hours went to RunPod support.

Two mistakes of mine made this worse than it had to be: `heal.py` saved only bf16
model snapshots, no optimizer state and no data position, so nothing was resumable;
and a volume disk instead of a network volume meant that even the snapshots died
with the host.

### Fixes before restarting

* `heal.py` now writes a full training state every 1,000 steps (fp32 weights, AdamW
  moments, scheduler step, RNG, dataset position) and has `--resume` and
  `--stop-after`. Tested on the Mac: a run stopped after step 3 and resumed gives
  the same losses as an uninterrupted run to every printed digit, including a
  mixed-dataset stream.
* The new pod uses a 120 GB network volume (EU-FR-1) mounted at /workspace, so a
  dead host costs a redeploy, not the data.
* Pilot-first protocol (`pilot.sh`): each init variant trains on the *full*
  15,258-step schedule but stops after 2,000 steps (131M tokens) with its state
  saved; the variants are compared at matched steps; the winner is resumed, so the
  pilot is not thrown away. Decision rule fixed in advance (`pick.py`): lowest
  held-out loss at step 2,000; if the best is within 0.01 of zero-init, zero wins.
  2,000 rather than 1,526 steps because the lost run showed the first ~1,500
  steps are a disruption plateau (loss rises from 2.65 to 2.68 and only turns down
  around step 2,000).

### A fourth variant: a model-written plan

Added mid-run at the user's request ("a model chooses, you for example"). The
LLM assistant running the experiment read the 0.5B beyond the activity
statistics (`analyze2.py`: per-neuron weight norms, near-duplicate neurons,
firing rates, per-head norms, within-group head similarity, per-layer attention
and MLP contribution to the residual stream) and wrote `plan_llm.py`, which
emits an explicit per-layer plan executed by `grow.py --init plan`. The choices
below are the model's, the code only guarantees exactness:

* identity blocks after layers 0, 12, 21, 23 instead of evenly spaced (layer 0's
  MLP changes the residual by 12x its norm; layers 20-23 carry the skewed
  specialist neurons);
* new neurons: 50% (skewed layers) / 25% (others) are *reader copies* of the
  busiest non-duplicate neurons with a zero writer, the rest fresh;
* new heads: reader copies of the strongest head of their kv group with a zero
  o_proj, unless the group is redundant (mean q cosine > 0.3) or has a dead head.

Function preservation verified for all four variants (fp32 max |Δlogit|: zero
5.6e-5, split 3.9e-4, guided 1.9e-4, plan 5.6e-5; greedy agreement 34/34).

### Finding: exact splits make gradient twins

A Net2Net split copies a neuron's reader exactly and halves the writer of both
copies. Both copies then see the same input and have the same writer, so they get
identical gradients and never separate. Measured on the split pilot's step-2,000
checkpoint: mean cosine between twin neurons' gate, up and down weights = 1.0000 in layers 0, 10, 20, 23 (random pairs: 0.007). The checkpoint is bf16, so differences too small for bf16 would not show; to that precision the twins have not separated.
The split model is therefore the zero model with ~40% of its new parameters
(the 2,048 neurons and 6 heads per layer) locked; the residual widening and the new
layers are still free. Net2Net added explicit noise for this reason; I left it out
to keep growth exact. The reader-copy construction in the plan is exact and *not*
symmetric (zero writer vs. intact writer), so it does not have this problem.

### Pilots (131M tokens each, identical schedule and data)

| init | held-out @ 2,000 | @ 1,500 | slope | pilot bench mean |
|---|---|---|---|---|
| zero | 2.666 | 2.679 | -0.013 | 51.9 |
| plan (llm) | 2.669 | 2.684 | -0.015 | 52.6 |
| split | 2.676 | 2.690 | -0.014 | 51.9 |
| guided | 2.679 | 2.687 | -0.008 | 52.2 |

Winner by the pre-registered rule: zero. The plan took the early hit the model had
predicted (doubled first updates on the busiest units: 2.671 vs 2.652 at step 100),
recovered by step 500 and tracked zero within 0.002-0.005 afterwards, clearly ahead
of split and guided. Benchmarks at 131M tokens are at source level for all four and
differ within task noise. The losing pilots' states stay on the volume.

### The long run

Zero resumed from its step-2,000 state at 13:00 UTC and reproduced the lost run's
curve to the third decimal (2.629 at step 7,000 both times). Final held-out loss
2.578 (start 2.618). The control (Qwen2.5-0.5B itself, same tokens, same
hyperparameters) ran at 48.8k tok/s vs 30.8k for the 1B and finished at 2.584.

| step | grown 1B | control 0.5B |
|---|---|---|
| 100 | 2.652 | 2.622 |
| 2,000 | 2.666 | 2.646 |
| 5,000 | 2.648 | 2.636 |
| 10,000 | 2.604 | 2.604 |
| 15,258 | 2.578 | 2.584 |

The control leads for 10,000 steps (no noisy new units to absorb), the grown model
crosses it around step 10,000 and pulls away by 0.006 over the last third.

### Results (full task sets, bf16, max_length 4096)

| task | source | grown step 0 | control (0.5B + 1B tokens) | grown + healed (1B + 1B tokens) |
|---|---|---|---|---|
| hellaswag | 52.1 | 52.0 | 51.4 | 51.1 |
| arc_easy | 58.4 | 58.2 | 61.0 | 61.4 |
| arc_challenge | 32.5 | 32.1 | 32.7 | 31.1 |
| piqa | 70.3 | 70.0 | 69.9 | 70.4 |
| winogrande | 56.7 | 56.6 | 56.7 | 56.0 |
| wikitext ppl | 18.45 | 18.46 | 20.37 | 20.27 |

Reading: at equal tokens the grown model beats the control by 0.006 in held-out
loss and by nothing on benchmarks (all differences within one standard error).
Both healing runs lose two points of wikitext perplexity: a fineweb-edu vs.
Wikipedia data effect, identical for both, not a growth effect. At equal
*compute* the picture is worse for growth: the control trains 1.6x faster per
token, so for the same GPU hours it would have seen ~1.6B tokens with its curve
still falling. The best current models in this size class (OLMo-2-1B, Qwen2.5-1.5B, SmolLM2-1.7B) stay 10-20 points ahead, and were trained on trillions of tokens. Pythia-1B scores below the source. That compute-matched result is an expectation; the run was not done. All numbers are from one seed.

The pipeline finished at 02:45 UTC on 2026-11-05 and the pod stopped itself at
02:55 UTC via the RunPod API. Models (`heal-small`, `control-small`) were copied
to the Mac with the pod started CPU-only.

### Costs

Second attempt, pilot to results: 19 h of H100 at $4/h, about $77 (balance $84.42 to $7.09): pilots $21, long run $32, control $23. First attempt lost to the host failure: about $22 of pod time. The balance stayed at $64.42 through the night, so the dead hours do not appear to have been billed even though the console showed $4/h.
Test pods and the B300 detour on day 1: roughly $10.

### What I would change next time

1. Full-state checkpoints and a network volume from the first minute.
2. Compute-matched control alongside the token-matched one.
3. Pilot with noise-broken splits as a fifth variant, to separate "exactness"
   from "twins".
4. The negated-gate reader copy (fires where the donor is silent) as a sixth.
5. A larger healing budget (5B+ tokens) before expecting benchmark movement.
