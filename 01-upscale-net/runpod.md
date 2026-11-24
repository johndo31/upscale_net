# Running the pipeline on RunPod

## What to rent

| experiment | GPU | volume | time for 1B healing tokens (incl. control) | approx. cost |
|---|---|---|---|---|
| `small` (0.5B -> 1.05B) | 1x A100 80 GB (or H100, or RTX 4090 at 2x the time) | 50 GB | ~16 h on A100, ~10 h on H100 | $25 to $35 |
| `big` (1.5B -> 8.2B) | 1x H200 141 GB preferred, 1x H100 80 GB works | 150 GB | ~45 h on H100 (500M tokens: ~23 h) | $140 to $200 (500M: ~$80) |

The control (the un-grown source trained on the same tokens) adds about a
third to a half of the healing time. It is what makes the result
interpretable; skip it with `CONTROL=0` only for a dry run.

Template: RunPod's official PyTorch 2.x CUDA image (any recent one). Pick a
pod with a **network volume** or a container disk of the size above so
checkpoints survive. Leave Jupyter/SSH on. The throughput and cost numbers
are estimates; the first 100 steps print a measured tok/s.

## 1. Get the code onto the pod

On your Mac:

```
./make_bundle.sh
pbcopy < bundle.sh
```

Open the pod's web terminal (or ssh in), paste, press enter. That recreates
`~/upscale_net` with all scripts. Alternatively `scp -r` the folder, or push
it to a git remote and clone.

## 2. Set up

```
cd ~/upscale_net
pip install -r requirements.txt
export HF_HOME=/workspace/hf            # keep downloads on the volume
mkdir -p /workspace/hf && ln -sfn /workspace/models models && ln -sfn /workspace/runs runs
mkdir -p /workspace/models /workspace/runs
nvidia-smi
```

## 3. Run

Small experiment, 1B healing tokens, everything in one go:

```
nohup ./run_all.sh small 1000000000 > /workspace/run-small.log 2>&1 &
tail -f /workspace/run-small.log
```

Big experiment, 500M tokens:

```
nohup ./run_all.sh big 500000000 > /workspace/run-big.log 2>&1 &
```

`run_all.sh` does: grow -> verify (logit comparison) -> full benchmarks of
source and grown-step-0 -> heal -> benchmarks of healed -> control (source
model trained the same way) + its benchmarks -> results table in
`runs/results-<mode>.md`. The big run also scores GSM8K (reasoning) and mixes
20% DeepSeek-R1 reasoning traces into the healing data. It is restartable: if the pod dies, run the same
command again and it skips the finished steps. Healing checkpoints are saved
every 1000 steps in `runs/heal-<mode>/checkpoint-N/`; only the latest is kept, and it is removed once the final model is saved (not auto-resumed; pass
`--model runs/heal-<mode>/checkpoint-N` to `heal.py` by hand if needed).

Self-healing variant (no external text corpus: the source model writes
verified solutions to GSM8K/MATH training problems, the grown model trains on
them; see `self_distill.py`):

```
pip install vllm                      # optional, 10x faster generation
HEAL_DATA=self SD_ENGINE=vllm nohup ./run_all.sh small 50000000 > /workspace/run-small-self.log 2>&1 &
```

Generation: ~15k problems x 4 samples; ~30 min with vLLM on an A100, several
hours with the default transformers engine (`SD_ENGINE=hf`). Expect 30-60%
of samples to be kept for the DeepSeek distill, far fewer for Qwen2.5-0.5B
base. The healing budget is small (tens of millions of tokens, several
epochs over the kept solutions). Results land in `runs/results-<mode>-self.md`
and can be compared with the fineweb run side by side.

Quick sanity run first (~15 min, tiny token budget, 50 examples per task):

```
LIMIT=50 ./run_all.sh small 20000000
```

Delete `runs/` and `models/qwen05-grown-1b` afterwards so the real run starts clean.

## 4. What to watch

* `runs/verify-<mode>.log`: max abs logit difference should be ~1e-1 or less
  in bf16 with 100% argmax agreement. That is the function-preservation check.
* `runs/heal-<mode>/log.jsonl`: `loss` per step, `heldout_loss` every 100
  steps. The step-0 held-out loss equals the source model's. It should go down.
  If it spikes above the start value and stays there, lower `LR` in
  `run_all.sh` (halve it) and restart from the last checkpoint.
* `tok_s` in the same log gives real throughput; steps x tokens/step / tok_s
  is the remaining time.

## 5. Bring results home

```
tar cz runs/results-*.md runs/verify-*.log runs/heal-*/log.jsonl runs/bench/*/*.json runs/bench/*/**/results_*.json > results.tgz
```

Download `results.tgz` from the pod's file browser. That is all the blog post
needs; the healed model weights are optional (`runs/heal-<mode>/model/`,
2 GB for small, 16 GB for big).

## Memory notes

* `small` uses AdamW with fp32 master weights, bf16 compute: ~17 GB for the
  1.05B model plus activations. Fits 24 GB cards with `BATCH=8 ACC=4`.
* `big` uses Adafactor with fp32 weights: ~66 GB before activations. On an
  H100 80 GB set `BATCH=2 ACC=16` if you see out-of-memory; on an H200 the
  defaults are fine. Switching `big` to `OPT=adamw` needs ~130 GB (2x H100
  with FSDP, not supported by this script).
