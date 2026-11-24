#!/usr/bin/env python
"""Read the small model: per-neuron and per-head activity statistics on sample text.

Produces a JSON that a model-guided growth init can use (grow.py --init guided):
  mlp[layer]  : mean |activation| of each intermediate neuron (gate*up before down_proj)
  heads[layer]: mean output norm of each attention head (before o_proj)
Also prints a short summary per layer so a human or LLM can decide what to split.

    python analyze.py --model Qwen/Qwen2.5-0.5B --out runs/stats-qwen05.json --tokens 200000
"""
import argparse
import json

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=200_000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subset", default="sample-10BT")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    cfg = model.config
    L, I = cfg.num_hidden_layers, cfg.intermediate_size
    nh, hd = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
    mlp_sum = torch.zeros(L, I, dtype=torch.float64)
    head_sum = torch.zeros(L, nh, dtype=torch.float64)
    count = [0]

    hooks = []
    for li, layer in enumerate(model.model.layers):
        def mlp_hook(mod, inp, out, li=li):      # input of down_proj = act(gate) * up  -> [B, T, I]
            mlp_sum[li] += inp[0].abs().float().sum(dim=(0, 1)).cpu().double()   # MPS has no float64
        def attn_hook(mod, inp, out, li=li):     # input of o_proj = concat of heads -> [B, T, nh*hd]
            x = inp[0].float().view(inp[0].shape[0], inp[0].shape[1], nh, hd)
            head_sum[li] += x.norm(dim=-1).sum(dim=(0, 1)).cpu().double()
        hooks.append(layer.mlp.down_proj.register_forward_hook(mlp_hook))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(attn_hook))

    ds = load_dataset(args.dataset, name=args.subset, split="train", streaming=True)
    buf, seen = [], 0
    with torch.no_grad():
        batch = []
        for ex in ds:
            buf.extend(tok(ex["text"]).input_ids + [tok.eos_token_id])
            while len(buf) >= args.seq_len:
                batch.append(buf[:args.seq_len]); buf = buf[args.seq_len:]
                if len(batch) == args.batch:
                    x = torch.tensor(batch, device=args.device)
                    model(input_ids=x)
                    seen += x.numel(); batch = []
                    print(f"\r{seen/1e3:.0f}k tokens", end="", flush=True)
            if seen >= args.tokens:
                break
    print()
    mlp_mean = (mlp_sum / seen).float()
    head_mean = (head_sum / seen).float()

    stats = {"model": args.model, "tokens": seen, "intermediate_size": I, "num_heads": nh,
             "mlp": mlp_mean.tolist(), "heads": head_mean.tolist()}
    json.dump(stats, open(args.out, "w"))

    print(f"{'layer':>5} | {'mlp mean|act|':>13} {'p10':>7} {'p90':>7} {'max':>7} {'dead<1%':>8} | head norms (min .. max)")
    for li in range(L):
        m = mlp_mean[li]
        dead = (m < 0.01 * m.mean()).sum().item()
        h = head_mean[li]
        print(f"{li:>5} | {m.mean():13.4f} {m.quantile(0.1):7.4f} {m.quantile(0.9):7.4f} {m.max():7.3f} {dead:>8} | "
              + " ".join(f"{v:.2f}" for v in h.tolist()))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
