#!/usr/bin/env python
"""Check that a grown model computes the same function as its source.

Compares next-token logits on a few prompts and greedy continuations.
With --dtype fp32 the max abs logit difference should be ~1e-4 or smaller;
with bf16 expect ~1e-1 differences from rounding but identical argmax
almost everywhere.
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "In 1905, Albert Einstein published",
    "Q: What is 17 * 23?\nA:",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--dtype", choices=DTYPES, default="fp32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gen-tokens", type=int, default=16)
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--offload-gb", type=float, default=0,
                    help="if >0, load the grown model with at most this many GiB in RAM and the rest offloaded to disk")
    args = ap.parse_args()
    dt = DTYPES[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.src)
    t0 = time.time()
    a = AutoModelForCausalLM.from_pretrained(args.src, dtype=dt).to(args.device).eval()
    if args.offload_gb > 0:
        import tempfile
        b = AutoModelForCausalLM.from_pretrained(
            args.dst, dtype=dt, device_map="auto", max_memory={"cpu": f"{args.offload_gb}GiB"},
            offload_folder=tempfile.mkdtemp(prefix="offload-"), offload_state_dict=True).eval()
    else:
        b = AutoModelForCausalLM.from_pretrained(args.dst, dtype=dt).to(args.device).eval()
    PROMPTS[:] = PROMPTS[:args.prompts]
    na, nb = sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters())
    print(f"loaded in {time.time()-t0:.0f}s | src {na/1e9:.3f}B params | dst {nb/1e9:.3f}B params ({nb/na:.2f}x)")

    worst, agree, total = 0.0, 0, 0
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids.to(args.device)
            la = a(ids).logits.float()
            lb = b(ids).logits.float()
            diff = (la - lb).abs()
            worst = max(worst, diff.max().item())
            agree += (la.argmax(-1) == lb.argmax(-1)).sum().item()
            total += la.shape[1]
            ga = a.generate(ids, max_new_tokens=args.gen_tokens, do_sample=False)
            gb = b.generate(ids, max_new_tokens=args.gen_tokens, do_sample=False)
            same = torch.equal(ga, gb)
            print(f"\nprompt: {p!r}\n  max|Δlogit| {diff.max():.2e}  mean {diff.mean():.2e}  greedy-identical: {same}")
            print(f"  src: {tok.decode(ga[0, ids.shape[1]:])!r}")
            if not same:
                print(f"  dst: {tok.decode(gb[0, ids.shape[1]:])!r}")
    print(f"\nSUMMARY  max|Δlogit| = {worst:.2e}   argmax agreement = {agree}/{total}")


if __name__ == "__main__":
    main()
