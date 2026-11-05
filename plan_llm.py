#!/usr/bin/env python
"""The "llm" growth plan: how I (the language model running this project) chose to fill the new
capacity after reading Qwen2.5-0.5B's weights and activations (analyze.py, analyze2.py).

The code keeps every choice function-preserving; the choices are the content. Decisions:

1. Depth. The four new identity blocks go after layers 0, 12, 21 and 23 instead of evenly spaced.
   Layer 0's MLP changes the residual by 12x its norm (it builds the representation from the tiny
   embedding) and layers 20-23 carry the most skewed, specialist neurons (top 1% of neurons = 5-18%
   of all activation). Extra depth goes where the work is; one block stays in the middle.

2. MLP neurons (2048 new per layer). Fresh units (random reader, zero writer) learn something new
   but start blind; a Net2Net split starts alive but makes gradient twins that never separate.
   I use a third exact construction: copy the READER of a busy neuron, give the copy a zero writer.
   It fires as often and as hard as its donor from step 0, so its writer gets a strong gradient
   immediately, and because its writer differs from the donor's the pair specialises.
   Donors = the busiest neurons by mean|act| x ||down column||, skipping neurons that already have
   a near-duplicate (cos > 0.9 of output direction) and neurons that fire on < 1% of tokens.
   Share of reader copies: 50% in the skewed layers (0, 19-23), 25% elsewhere; the rest fresh.

3. Attention heads (6 new q heads per layer, 3 per old kv group). Same reader-copy idea: the new
   head gets the q of the strongest head of its group (by output norm) and a zero o_proj, so it
   attends like a proven head and learns a new use for that pattern. Exceptions: where the group's
   q heads are already near-identical (mean pairwise cos > 0.3: redundant pattern) or the group
   has a near-dead head (norm < 0.3: the group is not short of slots), the new heads stay fresh.

4. Residual width (+384 dims) and the init scale stay as in zero init: zero-padded, std 0.02,
   which matches the source's own reader norms (~0.6 per row).

    python plan_llm.py --stats runs/stats-qwen05.json --stats2 runs/stats2-qwen05.json --out runs/plan-llm.json
"""
import argparse
import json

import torch

NEW_LAYERS_AFTER = [0, 12, 21, 23]
SKEWED = {0, 19, 20, 21, 22, 23}
COPY_SHARE_SKEWED, COPY_SHARE = 0.5, 0.25
DUP_COS, MIN_FIRE = 0.9, 0.01
GROUP_COS_REDUNDANT, DEAD_HEAD_NORM = 0.3, 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="runs/stats-qwen05.json")
    ap.add_argument("--stats2", default="runs/stats2-qwen05.json")
    ap.add_argument("--out", default="runs/plan-llm.json")
    ap.add_argument("--new-intermediate", type=int, default=6912)
    ap.add_argument("--new-heads", type=int, default=20)
    ap.add_argument("--new-kv", type=int, default=4)
    a = ap.parse_args()
    A, B = json.load(open(a.stats)), json.load(open(a.stats2))
    L, I = len(A["mlp"]), A["intermediate_size"]
    nh = A["num_heads"]; nkv = 2; g = nh // nkv
    n_new = a.new_intermediate - I

    # new GQA layout: which new q slots are empty and which old kv group they belong to (see grow.head_plan)
    G = a.new_heads // a.new_kv
    import grow
    q_map, kv_map = grow.head_plan(nh, nkv, a.new_heads, a.new_kv)
    empty = [(h, kv_map[h // G]) for h in range(a.new_heads) if q_map[h] is None]

    layer_plan = []
    for li in range(L):
        layer_plan.append([li, False])
        if li in NEW_LAYERS_AFTER:
            layer_plan.append([li, True])

    plan = {"note": "see plan_llm.py", "layer_plan": layer_plan, "layers": {}}
    summary = []
    for li in range(L):
        act = torch.tensor(A["mlp"][li]); w = B["layers"][li]
        contrib = act * torch.tensor(w["mlp_out_norm"])
        dup = torch.tensor(w["mlp_dup_cos"]) > DUP_COS
        fire = torch.tensor(B["fire_frac"][li]) < MIN_FIRE
        contrib[dup | fire] = -1.0                                  # never a donor
        share = COPY_SHARE_SKEWED if li in SKEWED else COPY_SHARE
        n_copy = int(n_new * share)
        donors = contrib.argsort(descending=True)[:n_copy].tolist()
        mlp_reader = donors + [None] * (n_new - n_copy)

        heads_reader = {}
        onorm = torch.tensor(A["heads"][li])                        # mean output norm per old head
        for h, old_kv in empty:
            members = list(range(old_kv * g, (old_kv + 1) * g))
            redundant = w["group_q_cos"][old_kv] > GROUP_COS_REDUNDANT
            has_dead = min(onorm[m].item() for m in members) < DEAD_HEAD_NORM
            if redundant or has_dead:
                heads_reader[str(h)] = None
            else:
                heads_reader[str(h)] = max(members, key=lambda m: onorm[m].item())
        plan["layers"][str(li)] = {"mlp_reader": mlp_reader, "heads_reader": heads_reader}
        n_head_copies = sum(v is not None for v in heads_reader.values())
        summary.append((li, n_copy, n_new - n_copy, n_head_copies, 6 - n_head_copies, int((dup | fire).sum())))

    json.dump(plan, open(a.out, "w"))
    print(f"identity blocks after layers {NEW_LAYERS_AFTER}")
    print(f"{'layer':>5} | {'neuron reader-copies':>20} {'fresh':>6} | {'head reader-copies':>18} {'fresh':>6} | excluded donors")
    for li, c, f, hc, hf, ex in summary:
        print(f"{li:>5} | {c:>20} {f:>6} | {hc:>18} {hf:>6} | {ex}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
