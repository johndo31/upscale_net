#!/usr/bin/env python
"""Function-preserving growth of a Qwen2/Llama-style causal LM.

Takes a trained checkpoint and produces a wider and deeper model that computes
*exactly* the same function at initialisation (Net2Net-style), so continued
pre-training starts from the small model's knowledge instead of from scratch.

Growth operations
-----------------
Width (hidden size d -> D):
    * residual stream is zero-padded: embedding / o_proj / down_proj write 0
      into the new dims, so the new dims stay exactly 0 until training.
    * matrices that *read* the residual (q/k/v/gate/up/lm_head) get random
      columns for the new dims. They read zeros, so the output is unchanged,
      but their presence gives the writers a non-zero gradient.
    * RMSNorm: mean over D of a zero-padded vector is (d/D) * mean over d.
      Compensate exactly with  w' = w * sqrt(d/D)  and  eps' = eps * d/D.
Heads (n_h/n_kv -> N_H/N_KV, head_dim fixed):
    * old q heads are placed into new GQA groups whose kv head is a copy of
      their original kv head; their o_proj columns move with them.
    * new q heads / kv heads are random; their o_proj columns are 0.
MLP (intermediate i -> I):
    * new gate/up rows random, matching down_proj columns 0.
Depth (L -> L'):
    * new blocks are copies of a neighbouring block with o_proj and
      down_proj zeroed, i.e. the block is the identity on the residual.

Everything is streamed tensor-by-tensor into safetensors shards, so the
peak memory is roughly one layer, not the whole model.
"""
import argparse
import json
import math
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "generation_config.json", "added_tokens.json",
]


def resolve(src: str) -> Path:
    p = Path(src)
    if p.is_dir():
        return p
    return Path(snapshot_download(src, allow_patterns=["*.json", "*.safetensors", "*.txt"]))


class Source:
    """Lazy reader over one or more safetensors shards."""

    def __init__(self, root: Path):
        self.files = sorted(root.glob("*.safetensors"))
        self.index = {}
        for f in self.files:
            with safe_open(f, "pt") as h:
                for k in h.keys():
                    self.index[k] = f

    def __contains__(self, k):
        return k in self.index

    def get(self, k) -> torch.Tensor:
        with safe_open(self.index[k], "pt") as h:
            return h.get_tensor(k).to(torch.float32)


class ShardWriter:
    def __init__(self, out: Path, dtype: torch.dtype, max_shard_bytes=2 * 1024**3):
        self.out, self.dtype, self.max = out, dtype, max_shard_bytes
        self.buf, self.buf_bytes, self.shards, self.weight_map = {}, 0, [], {}
        self.total = 0

    def add(self, name, t: torch.Tensor):
        t = t.contiguous().to(self.dtype)
        self.total += t.numel()
        if self.buf_bytes + t.nbytes > self.max and self.buf:
            self.flush()
        self.buf[name] = t
        self.buf_bytes += t.nbytes

    def flush(self):
        if not self.buf:
            return
        fname = f"model-{len(self.shards) + 1:05d}.safetensors"
        save_file(self.buf, self.out / fname, metadata={"format": "pt"})
        for k in self.buf:
            self.weight_map[k] = fname
        self.shards.append(fname)
        self.buf, self.buf_bytes = {}, 0

    def finish(self):
        self.flush()
        # rename to the canonical model-xxxxx-of-yyyyy.safetensors
        n = len(self.shards)
        final = {}
        for i, s in enumerate(self.shards, 1):
            new = f"model-{i:05d}-of-{n:05d}.safetensors"
            os.rename(self.out / s, self.out / new)
            final[s] = new
        self.weight_map = {k: final[v] for k, v in self.weight_map.items()}
        with open(self.out / "model.safetensors.index.json", "w") as f:
            json.dump({"metadata": {"total_size": int(self.total * torch.tensor([], dtype=self.dtype).element_size())},
                       "weight_map": self.weight_map}, f, indent=2)


def head_plan(n_h, n_kv, N_H, N_KV):
    """Map old q heads and kv heads into the new GQA layout.

    Returns (q_map, kv_map) where q_map[new_q] = old_q or None and
    kv_map[new_kv] = old_kv or None.
    """
    g, G = n_h // n_kv, N_H // N_KV
    groups_per_old = math.ceil(g / G)
    if n_kv * groups_per_old > N_KV:
        raise ValueError(f"need {n_kv * groups_per_old} kv heads to host old heads, have {N_KV}")
    q_map, kv_map = [None] * N_H, [None] * N_KV
    new_kv = 0
    for old_kv in range(n_kv):
        old_qs = list(range(old_kv * g, (old_kv + 1) * g))
        for _ in range(groups_per_old):
            kv_map[new_kv] = old_kv
            for slot in range(G):
                if old_qs:
                    q_map[new_kv * G + slot] = old_qs.pop(0)
            new_kv += 1
    return q_map, kv_map


def place_rows(old: torch.Tensor, mapping, block, rng, std):
    """Rows of `old` are grouped in blocks of `block`; produce a new tensor with
    len(mapping) blocks, copying block mapping[j] into block j (random if None)."""
    out = torch.randn(len(mapping) * block, *old.shape[1:], generator=rng) * std
    for j, i in enumerate(mapping):
        if i is not None:
            out[j * block:(j + 1) * block] = old[i * block:(i + 1) * block]
    return out


def widen_reader(w: torch.Tensor, D, rng, std):
    """[rows, d] -> [rows, D]; new columns random (they read zeros)."""
    rows, d = w.shape
    out = torch.randn(rows, D, generator=rng) * std
    out[:, :d] = w
    return out


def widen_writer_rows(w: torch.Tensor, D):
    """[d, cols] -> [D, cols]; new rows zero (write nothing into new dims)."""
    d, cols = w.shape
    out = torch.zeros(D, cols)
    out[:d] = w
    return out


def grow_norm(w: torch.Tensor, D, d):
    out = torch.full((D,), math.sqrt(d / D))
    out[:d] = w * math.sqrt(d / D)
    return out


def grow_layer(src: Source, li, cfg, new, rng, std, identity=False, init="zero", stats=None, plan=None):
    """Yield (name, tensor) for new layer index `new_li` from source layer `li`."""
    d, D = cfg["hidden_size"], new["hidden_size"]
    i, I = cfg["intermediate_size"], new["intermediate_size"]
    hd = cfg["hidden_size"] // cfg["num_attention_heads"]
    q_map, kv_map = head_plan(cfg["num_attention_heads"], cfg["num_key_value_heads"],
                              new["num_attention_heads"], new["num_key_value_heads"])
    p = f"model.layers.{li}."
    out = {}

    out["input_layernorm.weight"] = grow_norm(src.get(p + "input_layernorm.weight"), D, d)
    out["post_attention_layernorm.weight"] = grow_norm(src.get(p + "post_attention_layernorm.weight"), D, d)

    # attention: q/k/v read the residual, o writes it
    for name, mapping in (("q_proj", q_map), ("k_proj", kv_map), ("v_proj", kv_map)):
        w = widen_reader(src.get(p + f"self_attn.{name}.weight"), D, rng, std)
        out[f"self_attn.{name}.weight"] = place_rows(w, mapping, hd, rng, std)
        bk = p + f"self_attn.{name}.bias"
        if bk in src:
            out[f"self_attn.{name}.bias"] = place_rows(src.get(bk), mapping, hd, rng, 0.0)
    o = src.get(p + "self_attn.o_proj.weight")                 # [d, n_h*hd]
    o = widen_writer_rows(o, D)                                 # [D, n_h*hd]
    o_new = torch.zeros(D, len(q_map) * hd)
    for j, qi in enumerate(q_map):
        if qi is not None:
            o_new[:, j * hd:(j + 1) * hd] = o[:, qi * hd:(qi + 1) * hd]
    out["self_attn.o_proj.weight"] = torch.zeros_like(o_new) if identity else o_new

    # MLP: gate/up read, down writes
    for name in ("gate_proj", "up_proj"):
        w = widen_reader(src.get(p + f"mlp.{name}.weight"), D, rng, std)   # [i, D]
        out[f"mlp.{name}.weight"] = place_rows(w, list(range(i)) + [None] * (I - i), 1, rng, std)
    down = widen_writer_rows(src.get(p + "mlp.down_proj.weight"), D)      # [D, i]
    down_new = torch.zeros(D, I)
    down_new[:, :i] = down
    out["mlp.down_proj.weight"] = torch.zeros_like(down_new) if identity else down_new

    if init in ("split", "guided") and not identity:
        # Donor order: index order for split; busiest first for guided (mean |activation| from analyze.py).
        if init == "guided":
            order = sorted(range(i), key=lambda j: -stats["mlp"][li][j])
            head_rank = sorted(range(cfg["num_attention_heads"]), key=lambda h: -stats["heads"][li][h])
        else:
            order = list(range(i))
            head_rank = list(range(cfg["num_attention_heads"]))
        # New MLP neurons: copy existing neuron j, halve down_proj for both copies -> same function.
        for n in range(i, I):
            j = order[(n - i) % i]
            for name in ("gate_proj", "up_proj"):
                out[f"mlp.{name}.weight"][n] = out[f"mlp.{name}.weight"][j]
            out["mlp.down_proj.weight"][:, j] *= 0.5
            out["mlp.down_proj.weight"][:, n] = out["mlp.down_proj.weight"][:, j]
        # New q heads: copy an existing q head from the same kv group (so the copied kv head
        # matches), halve o_proj columns of both. Groups with no old head stay zero-output.
        G = new["num_attention_heads"] // new["num_key_value_heads"]
        for h, qi in enumerate(q_map):
            if qi is not None:
                continue
            group = [g for g in range(h // G * G, h // G * G + G) if q_map[g] is not None]
            if not group:
                continue
            # candidates ranked (busiest old head first for guided), cycling through them
            ranked = sorted(group, key=lambda g: head_rank.index(q_map[g]))
            donor = ranked[(h - h // G * G) % len(ranked)]
            for name in ("q_proj.weight", "q_proj.bias"):
                key = f"self_attn.{name}"
                if key in out:
                    out[key][h * hd:(h + 1) * hd] = out[key][donor * hd:(donor + 1) * hd]
            out["self_attn.o_proj.weight"][:, donor * hd:(donor + 1) * hd] *= 0.5
            out["self_attn.o_proj.weight"][:, h * hd:(h + 1) * hd] = out["self_attn.o_proj.weight"][:, donor * hd:(donor + 1) * hd]

    if init == "plan" and not identity:
        # Explicit per-layer choices (plan_llm.py): which old neuron / head each new slot copies
        # (copy + halve, exact) or null = fresh unit (random reader, zero writer, exact).
        # A donor used k times ends up with 1/2^k of its output and each copy gets the remainder: still exact.
        lp = plan["layers"].get(str(li), {})
        donors = lp.get("mlp", [])
        for n in range(i, I):
            j = donors[n - i] if n - i < len(donors) else None
            if j is None:
                continue
            for name in ("gate_proj", "up_proj"):
                out[f"mlp.{name}.weight"][n] = out[f"mlp.{name}.weight"][j]
            out["mlp.down_proj.weight"][:, j] *= 0.5
            out["mlp.down_proj.weight"][:, n] = out["mlp.down_proj.weight"][:, j]
        # "mlp_reader": copy only the READER (gate/up rows) of a busy neuron; the new writer stays zero.
        # Exact, and unlike a split the copy is not a gradient twin: its writer starts at zero while
        # the donor's is intact, so the pair specialises instead of staying identical.
        readers = lp.get("mlp_reader", [])
        for n in range(i, I):
            j = readers[n - i] if n - i < len(readers) else None
            if j is None:
                continue
            for name in ("gate_proj", "up_proj"):
                out[f"mlp.{name}.weight"][n] = out[f"mlp.{name}.weight"][j]
        G = new["num_attention_heads"] // new["num_key_value_heads"]
        g = cfg["num_attention_heads"] // cfg["num_key_value_heads"]
        for field, halve in (("heads", True), ("heads_reader", False)):
            for h_str, donor in lp.get(field, {}).items():
                h = int(h_str)
                if donor is None or q_map[h] is not None:
                    continue
                if kv_map[h // G] != donor // g:
                    raise ValueError(f"layer {li}: new head {h} is in kv group {kv_map[h // G]}, donor head {donor} belongs to group {donor // g}")
                src_slot = q_map.index(donor)                       # where the donor head sits in the new layout
                for name in ("q_proj.weight", "q_proj.bias"):
                    key = f"self_attn.{name}"
                    if key in out:
                        out[key][h * hd:(h + 1) * hd] = out[key][src_slot * hd:(src_slot + 1) * hd]
                if halve:                                           # split: both halves write; reader-copy: new head writes zero
                    out["self_attn.o_proj.weight"][:, src_slot * hd:(src_slot + 1) * hd] *= 0.5
                    out["self_attn.o_proj.weight"][:, h * hd:(h + 1) * hd] = out["self_attn.o_proj.weight"][:, src_slot * hd:(src_slot + 1) * hd]
    return out


def layer_plan(L, L_new):
    """Return list of (source_layer, is_new) of length L_new, inserting the
    extra identity blocks evenly, each copying the block before it."""
    extra = L_new - L
    if extra < 0:
        raise ValueError("cannot shrink depth")
    insert_after = set()
    for k in range(extra):
        insert_after.add(round((k + 1) * L / (extra + 1)) - 1)
    plan = []
    for li in range(L):
        plan.append((li, False))
        if li in insert_after:
            plan.append((li, True))
    # if rounding collapsed two insertions onto one slot, append at the end
    while len(plan) < L_new:
        plan.append((L - 1, True))
    return plan


def count_params(c):
    d, I, L, V = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"], c["vocab_size"]
    hd = d // c["num_attention_heads"] if "head_dim" not in c else c["head_dim"]
    nh, nkv = c["num_attention_heads"], c["num_key_value_heads"]
    attn = d * nh * hd + 2 * d * nkv * hd + nh * hd * d
    bias = nh * hd + 2 * nkv * hd
    mlp = 3 * d * I
    norms = 2 * d
    per_layer = attn + bias + mlp + norms
    emb = V * d * (1 if c.get("tie_word_embeddings") else 2)
    return per_layer * L + emb + d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="HF repo id or local dir")
    ap.add_argument("--dst", required=True, help="output dir")
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--intermediate", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--kv-heads", type=int, required=True)
    ap.add_argument("--dtype", choices=DTYPES, default="bf16")
    ap.add_argument("--init-std", type=float, default=None, help="std of random parts (default: initializer_range)")
    ap.add_argument("--stats", default=None, help="activity stats from analyze.py (required for --init guided)")
    ap.add_argument("--plan", default=None, help="JSON plan for --init plan (see plan_llm.py)")
    ap.add_argument("--init", choices=["zero", "split", "guided", "plan"], default="zero",
                    help="zero: new heads/neurons have random inputs and zero outputs (dead at init). "
                         "split: new heads/neurons are copies of existing ones with outputs halved (Net2Net), alive at init. "
                         "guided: like split, but the busiest units (by --stats) are the ones that get split. "
                         "plan: explicit per-layer donors / fresh units / layer duplication from --plan")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="only print the plan and parameter counts")
    args = ap.parse_args()

    root = resolve(args.src)
    cfg = json.load(open(root / "config.json"))
    stats = plan_json = None
    if args.init == "plan":
        if not args.plan:
            raise SystemExit("--init plan needs --plan <json>")
        plan_json = json.load(open(args.plan))
    if args.init == "guided":
        if not args.stats:
            raise SystemExit("--init guided needs --stats <analyze.py output>")
        stats = json.load(open(args.stats))
    assert cfg["model_type"] in ("qwen2", "llama"), f"unsupported model_type {cfg['model_type']}"
    d = cfg["hidden_size"]
    hd = d // cfg["num_attention_heads"]
    if args.hidden < d or args.intermediate < cfg["intermediate_size"] or args.layers < cfg["num_hidden_layers"]:
        raise SystemExit("can only grow, not shrink")

    new = dict(cfg)
    new.update(hidden_size=args.hidden, intermediate_size=args.intermediate,
               num_hidden_layers=args.layers, num_attention_heads=args.heads,
               num_key_value_heads=args.kv_heads, head_dim=hd,
               rms_norm_eps=cfg["rms_norm_eps"] * d / args.hidden,
               torch_dtype=args.dtype.replace("fp", "float").replace("bf16", "bfloat16"),
               max_window_layers=args.layers)
    q_map, kv_map = head_plan(cfg["num_attention_heads"], cfg["num_key_value_heads"], args.heads, args.kv_heads)
    plan = layer_plan(cfg["num_hidden_layers"], args.layers)
    if plan_json and plan_json.get("layer_plan"):
        plan = [(int(a), bool(b)) for a, b in plan_json["layer_plan"]]
        if len(plan) != args.layers:
            raise SystemExit(f"plan.layer_plan has {len(plan)} entries, need {args.layers}")
    if plan_json and plan_json.get("init_std") is not None:
        args.init_std = plan_json["init_std"]

    print(f"source : {args.src}  ({count_params(cfg)/1e9:.3f} B params)")
    print(f"  hidden {d} -> {args.hidden} | inter {cfg['intermediate_size']} -> {args.intermediate} | "
          f"layers {cfg['num_hidden_layers']} -> {args.layers} | heads {cfg['num_attention_heads']}/{cfg['num_key_value_heads']}"
          f" -> {args.heads}/{args.kv_heads} (head_dim {hd})")
    print(f"target : {count_params(new)/1e9:.3f} B params, {args.dtype}, "
          f"~{count_params(new) * torch.tensor([], dtype=DTYPES[args.dtype]).element_size() / 1e9:.1f} GB on disk")
    print(f"  q-head map : {q_map}")
    print(f"  kv-head map: {kv_map}")
    print(f"  layer plan : {[f'{s}{'+' if n else ''}' for s, n in plan]}   ('+' = identity copy)")
    print(f"  init       : {args.init}")
    if args.dry_run:
        return

    out = Path(args.dst)
    out.mkdir(parents=True, exist_ok=True)
    src = Source(root)
    rng = torch.Generator().manual_seed(args.seed)
    std = args.init_std if args.init_std is not None else cfg.get("initializer_range", 0.02)
    w = ShardWriter(out, DTYPES[args.dtype])

    emb = src.get("model.embed_tokens.weight")                       # [V, d]
    emb_new = torch.zeros(emb.shape[0], args.hidden)
    emb_new[:, :d] = emb
    w.add("model.embed_tokens.weight", emb_new)
    for new_li, (li, identity) in enumerate(plan):
        for k, t in grow_layer(src, li, cfg, new, rng, std, identity, args.init, stats, plan_json).items():
            w.add(f"model.layers.{new_li}.{k}", t)
        print(f"  layer {new_li:3d} <- {li}{' (identity)' if identity else ''}", flush=True)
    w.add("model.norm.weight", grow_norm(src.get("model.norm.weight"), args.hidden, d))
    if not cfg.get("tie_word_embeddings") and "lm_head.weight" in src:
        w.add("lm_head.weight", widen_reader(src.get("lm_head.weight"), args.hidden, rng, std))
    w.finish()
    print(f"wrote {w.total/1e9:.3f} B params in {len(w.shards)} shard(s)")

    json.dump(new, open(out / "config.json", "w"), indent=2)
    for f in TOKENIZER_FILES:
        if (root / f).exists():
            shutil.copy(root / f, out / f)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
