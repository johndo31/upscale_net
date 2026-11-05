#!/usr/bin/env python
"""Read the small model's weights and residual-stream behaviour, as input for a hand-written growth plan.

Adds to analyze.py (which measured activity):
  weights : per-neuron norms of gate/up/down, near-duplicate neurons (cosine of down columns),
            per-head q/k/v/o norms, within-group head similarity
  stream  : per layer, how much attention and the MLP change the residual (||delta|| / ||x||)
            and the fraction of tokens where each neuron fires (gate*up > 0)
Writes one JSON and prints a per-layer summary.

    python analyze2.py --model Qwen/Qwen2.5-0.5B --stats runs/stats-qwen05.json --out runs/stats2-qwen05.json
"""
import argparse
import json

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stats", required=True, help="analyze.py output (activity)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=100_000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
    a = ap.parse_args()

    act = json.load(open(a.stats))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(a.device).eval()
    cfg = model.config
    L, I, d = cfg.num_hidden_layers, cfg.intermediate_size, cfg.hidden_size
    nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd, g = d // nh, nh // nkv

    # ---- weights -------------------------------------------------------------------------
    W = {"layers": []}
    for li, layer in enumerate(model.model.layers):
        m, at = layer.mlp, layer.self_attn
        gate, up, down = m.gate_proj.weight.detach(), m.up_proj.weight.detach(), m.down_proj.weight.detach()   # [I,d],[I,d],[d,I]
        dn = F.normalize(down.T, dim=1)                                   # [I, d] unit columns of down
        sim = dn @ dn.T
        sim.fill_diagonal_(0)
        best, best_j = sim.max(dim=1)                                     # nearest other neuron by output direction
        q, k, v, o = (at.q_proj.weight.detach(), at.k_proj.weight.detach(), at.v_proj.weight.detach(), at.o_proj.weight.detach())
        qn = q.view(nh, hd, d).flatten(1).norm(dim=1)
        on = o.view(d, nh, hd).permute(1, 0, 2).flatten(1).norm(dim=1)
        kn = k.view(nkv, hd, d).flatten(1).norm(dim=1)
        vn = v.view(nkv, hd, d).flatten(1).norm(dim=1)
        # how similar the q heads of a kv group are (cosine of flattened q weights)
        qflat = F.normalize(q.view(nh, -1), dim=1)
        group_sim = []
        for kv in range(nkv):
            idx = list(range(kv * g, (kv + 1) * g))
            S = qflat[idx] @ qflat[idx].T
            group_sim.append(round(((S.sum() - g) / (g * g - g)).item(), 4))
        W["layers"].append({
            "mlp_in_norm": (gate.norm(dim=1) * up.norm(dim=1)).sqrt().cpu().tolist(),
            "mlp_out_norm": down.norm(dim=0).cpu().tolist(),
            "mlp_dup_cos": best.cpu().tolist(), "mlp_dup_of": best_j.cpu().tolist(),
            "q_norm": qn.cpu().tolist(), "o_norm": on.cpu().tolist(),
            "k_norm": kn.cpu().tolist(), "v_norm": vn.cpu().tolist(), "group_q_cos": group_sim,
        })

    # ---- stream --------------------------------------------------------------------------
    attn_ratio = torch.zeros(L, dtype=torch.float64); mlp_ratio = torch.zeros(L, dtype=torch.float64)
    fire = torch.zeros(L, I, dtype=torch.float64); n_tok = [0]
    resid_in = {}

    def pre_hook(mod, args, kwargs, li):
        resid_in[li] = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
    def attn_hook(mod, inp, out, li):
        x = resid_in[li]; delta = out[0] if isinstance(out, tuple) else out
        attn_ratio[li] += (delta.float().norm(dim=-1) / x.float().norm(dim=-1)).sum().cpu().double()
    def mlp_hook(mod, inp, out, li):
        x = inp[0]; delta = out
        mlp_ratio[li] += (delta.float().norm(dim=-1) / (resid_in[li].float().norm(dim=-1) + 1e-6)).sum().cpu().double()
    def down_hook(mod, inp, out, li):
        fire[li] += (inp[0] > 0).float().sum(dim=(0, 1)).cpu().double()

    for li, layer in enumerate(model.model.layers):
        layer.register_forward_pre_hook(lambda m, args, kwargs, li=li: pre_hook(m, args, kwargs, li), with_kwargs=True)
        layer.self_attn.register_forward_hook(lambda m, i, o, li=li: attn_hook(m, i, o, li))
        layer.mlp.register_forward_hook(lambda m, i, o, li=li: mlp_hook(m, i, o, li))
        layer.mlp.down_proj.register_forward_hook(lambda m, i, o, li=li: down_hook(m, i, o, li))

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    buf, batch = [], []
    with torch.no_grad():
        for ex in ds:
            buf.extend(tok(ex["text"]).input_ids + [tok.eos_token_id])
            while len(buf) >= a.seq_len:
                batch.append(buf[:a.seq_len]); buf = buf[a.seq_len:]
                if len(batch) == a.batch:
                    x = torch.tensor(batch, device=a.device)
                    model(input_ids=x)
                    n_tok[0] += x.numel(); batch = []
                    print(f"\r{n_tok[0]/1e3:.0f}k tokens", end="", flush=True)
            if n_tok[0] >= a.tokens:
                break
    print()
    W["attn_ratio"] = (attn_ratio / n_tok[0]).tolist()
    W["mlp_ratio"] = (mlp_ratio / n_tok[0]).tolist()
    W["fire_frac"] = (fire / n_tok[0]).float().tolist()
    W["tokens"] = n_tok[0]
    json.dump(W, open(a.out, "w"))

    print(f"{'L':>2} | {'attn/x':>6} {'mlp/x':>6} | {'in-norm p50':>11} {'p99':>6} | {'out p50':>7} {'p99':>6} | {'dup>.9':>6} {'fire<1%':>7} {'fire>50%':>8} | q-norm min..max | grp-q-cos")
    for li in range(L):
        w = W["layers"][li]
        inn, outn = torch.tensor(w["mlp_in_norm"]), torch.tensor(w["mlp_out_norm"])
        dup = (torch.tensor(w["mlp_dup_cos"]) > 0.9).sum().item()
        ff = torch.tensor(W["fire_frac"][li])
        qn = torch.tensor(w["q_norm"])
        print(f"{li:>2} | {W['attn_ratio'][li]:6.3f} {W['mlp_ratio'][li]:6.3f} | {inn.median():11.3f} {inn.quantile(0.99):6.3f} | "
              f"{outn.median():7.3f} {outn.quantile(0.99):6.3f} | {dup:6d} {(ff < 0.01).sum().item():7d} {(ff > 0.5).sum().item():8d} | "
              f"{qn.min():.2f}..{qn.max():.2f} | {' '.join(f'{c:.2f}' for c in w['group_q_cos'])}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
