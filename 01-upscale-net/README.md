# upscale_net

Grow a trained open-weight LLM into a bigger one without changing what it
computes, then train ("heal") it so the new capacity gets used.

* `grow.py`    function-preserving width + depth growth for Qwen2/Llama models (streams to safetensors)
* `verify.py`  proves the grown model gives the same logits as its source
* `heal.py`    continued pre-training on streamed fineweb-edu text (or any HF dataset / local jsonl, with optional data mix)
* `self_distill.py`  the source model solves GSM8K/MATH train problems; verified-correct solutions become healing data (no external corpus)
* `bench.sh`   lm-eval suite (hellaswag, arc, piqa, winogrande, wikitext ppl)
* `results.py` turns benchmark outputs into a markdown table
* `analyze.py`, `analyze2.py` read the small model: activity, weight and residual-stream statistics
* `plan_llm.py` the LLM-written growth plan, with its reasoning, executed by `grow.py --init plan`
* `pilot.sh`   the pipeline behind the published numbers: four init pilots, `pick.py` picks one, resume it to 1B tokens, then the control, unattended on one GPU
* `run_all.sh` the earlier single-init pipeline: `./run_all.sh small|big [tokens]` (`HEAL_DATA=self` for self-distillation, `CONTROL=0` to skip the control)
* `runpod.md`  what to rent and how to run it there
* `PROCESS.md` chronological log of what was done and measured

Experiments:

| mode | source | grown | shape |
|---|---|---|---|
| small | Qwen/Qwen2.5-0.5B (0.49B) | 1.05B | hidden 1280, 28 layers, 20/4 heads |
| big | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (1.78B, MIT) | 8.22B | Llama-3-8B shape: hidden 4096, 32 layers, 32/8 heads |

Local setup (Mac): `python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt`

## Result

Qwen2.5-0.5B grown to 1.05B with identical outputs, then trained on 1B fineweb-edu tokens against a control (the 0.5B on the same tokens), one seed each:

| | source 0.5B | control | grown 1B |
|---|---|---|---|
| held-out loss | 2.618 | 2.584 | 2.578 |
| mean of five benchmarks | 54.0 | 54.3 | 54.0 |

Growth is exact, but at this budget the extra capacity bought a rounding error. Logs, loss curves and benchmark JSON are in `results/`. The 8B mode is implemented, but it was grown and not healed.

The write-up is `post.md`, the full log is `PROCESS.md`. Licence: MIT.

Author: Niklas — [drniklas.com](https://drniklas.com) · podcast: [venturegrade.drniklas.com](https://venturegrade.drniklas.com)
