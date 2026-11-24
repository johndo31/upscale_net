#!/usr/bin/env python
"""Self-distillation data from the *small* model, no external text corpus.

The source model solves training problems with checkable answers (GSM8K and
MATH train splits), several samples per problem. Only solutions whose final
answer matches the reference are kept (rejection sampling / STaR). The output
is a JSONL of chat `messages` that heal.py can train the grown model on.

The learning signal comes from the correctness filter: the grown model starts
as an exact copy of the small one, so imitating *unfiltered* samples would
have zero expected gradient.

    python self_distill.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --out runs/self-big/data.jsonl
    python self_distill.py --model ... --engine vllm      # much faster if vllm is installed
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

from datasets import load_dataset

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
MATH_CONFIGS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]


def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(s):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
        k += 1
    return None


def normalize(a):
    if a is None:
        return None
    a = a.strip().rstrip(".")
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    for t in ["\\left", "\\right", "\\!", "\\,", "\\;", "\\ ", "$", "\\$", "^\\circ", "^{\\circ}", "\\%", "%", " "]:
        a = a.replace(t, "")
    a = a.replace("dfrac", "frac").replace("tfrac", "frac")
    a = re.sub(r"\\boxed\{(.*)\}", r"\1", a)
    if a.startswith("{") and a.endswith("}"):
        a = a[1:-1]
    a = a.replace(",", "") if re.fullmatch(r"-?[\d,]+(\.\d+)?", a) else a
    return a


def same_answer(pred, gold):
    p, g = normalize(pred), normalize(gold)
    if p is None or g is None:
        return False
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return False


def load_problems(names, limit, seed):
    probs = []
    if "gsm8k" in names:
        for ex in load_dataset("openai/gsm8k", "main", split="train"):
            probs.append({"source": "gsm8k", "problem": ex["question"], "gold": ex["answer"].split("####")[-1].strip()})
    if "math" in names:
        for cfg in MATH_CONFIGS:
            for ex in load_dataset("EleutherAI/hendrycks_math", cfg, split="train"):
                probs.append({"source": f"math/{cfg}", "problem": ex["problem"], "gold": last_boxed(ex["solution"])})
    probs = [p for p in probs if p["gold"]]
    random.Random(seed).shuffle(probs)
    return probs[:limit] if limit else probs


def gen_hf(model_name, prompts, n, max_new, temperature, batch, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device).eval()
    out = [[] for _ in prompts]
    order = [(i, _) for i in range(len(prompts)) for _ in range(n)]
    for b in range(0, len(order), batch):
        idx = [i for i, _ in order[b:b + batch]]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temperature,
                                 top_p=0.95, pad_token_id=tok.pad_token_id)
        for i, g in zip(idx, gen[:, enc.input_ids.shape[1]:]):
            out[i].append(tok.decode(g, skip_special_tokens=True))
        yield b + len(idx), out


def gen_vllm(model_name, prompts, n, max_new, temperature, batch, device):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=max_new + 1024)
    sp = SamplingParams(n=n, temperature=temperature, top_p=0.95, max_tokens=max_new)
    res = llm.generate(prompts, sp)
    yield len(prompts) * n, [[o.text for o in r.outputs] for r in res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the small source model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--problems", default="gsm8k,math")
    ap.add_argument("--limit", type=int, default=0, help="max problems (0 = all, ~15k)")
    ap.add_argument("--samples", type=int, default=4, help="samples per problem")
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--engine", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    probs = load_problems(args.problems.split(","), args.limit, args.seed)
    prompts = [tok.apply_chat_template([{"role": "user", "content": p["problem"] + SUFFIX}],
                                       tokenize=False, add_generation_prompt=True) for p in probs]
    print(f"{len(probs)} problems x {args.samples} samples, engine={args.engine}", flush=True)

    gen = gen_vllm if args.engine == "vllm" else gen_hf
    t0 = time.time()
    outs = None
    for done, outs in gen(args.model, prompts, args.samples, args.max_new_tokens, args.temperature, args.batch, args.device):
        print(f"  {done}/{len(probs) * args.samples} generations, {time.time() - t0:.0f}s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    kept = total = solved = 0
    with open(args.out, "w") as f:
        for p, samples in zip(probs, outs):
            got_one = False
            for s in samples:
                total += 1
                if "</think>" in s or "<think>" not in s:   # complete reasoning (or a non-thinking model)
                    if same_answer(last_boxed(s), p["gold"]):
                        kept += 1; got_one = True
                        f.write(json.dumps({"messages": [{"role": "user", "content": p["problem"] + SUFFIX},
                                                         {"role": "assistant", "content": s}],
                                            "source": p["source"]}) + "\n")
            solved += got_one
    stats = dict(problems=len(probs), solved_at_least_once=solved, samples=total, kept=kept,
                 keep_rate=round(kept / max(1, total), 3), seconds=round(time.time() - t0), out=args.out)
    print(json.dumps(stats))
    json.dump(stats, open(Path(args.out).with_suffix(".stats.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
