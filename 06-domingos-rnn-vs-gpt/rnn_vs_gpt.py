"""06 — Parameter-Matched Modernized RNN / LSTM vs. MicroGPT on Common Crawl (FineWeb-Edu).

Tests Pedro Domingos's claim:
  "An RNN with softmax layer, residual stream, layer norm and positional encoding does the same."

Three strictly parameter-matched (~10.6M non-embedding / ~30M total) 6-layer language models,
trained on the exact same Common Crawl token stream (seq_len=512, AdamW + cosine schedule):
  1. microgpt      : causal multi-head self-attention + Pre-LN + residual + pos enc + GELU MLP
  2. domingos_lstm : cuDNN LSTM + Pre-LN + residual + pos enc + GELU MLP
  3. domingos_rnn  : Elman RNN + softmax mixing/gating layer + Pre-LN + residual + pos enc + GELU MLP

Measures:
  - Held-out Common Crawl perplexity vs. tokens and vs. GPU wall-clock seconds
  - Held-out loss by token position bucket (0..31, 32..63, 64..127, 128..255, 256..511)
  - Zero-shot in-context associative recall at distances D = 32, 128, 320 tokens
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer


def make_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def make_rope_cache(max_len: int, d_head: int, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [T, d_head/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]
    T = x.shape[2]
    c = cos[:T].to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
    s = sin[:T].to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out1 = x1 * c - x2 * s
    out2 = x1 * s + x2 * c
    return torch.stack((out1, out2), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, use_rope: bool = False, max_len: int = 1024):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_rope = use_rope
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        if use_rope:
            cos, sin = make_rope_cache(max_len, self.d_head)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        if self.use_rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_mlp)
        self.fc2 = nn.Linear(d_mlp, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class GPTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int, use_rope: bool = False, max_len: int = 1024):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, use_rope=use_rope, max_len=max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DomingosLSTMBlock(nn.Module):
    """Pre-LN + residual stream + cuDNN LSTM + GELU MLP (parameter-matched to GPTBlock)."""
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.rnn = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.ln1(x))
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


class DomingosSoftmaxRNNBlock(nn.Module):
    """Pre-LN + residual stream + Elman RNN + softmax layer + GELU MLP (parameter-matched)."""
    def __init__(self, d_model: int, d_mlp: int, n_groups: int = 6):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.rnn = nn.RNN(d_model, d_model, num_layers=1, nonlinearity="tanh", batch_first=True)
        # Explicit softmax mixing/gating layer over multi-head feature groups
        self.n_groups = n_groups
        self.g_dim = d_model // n_groups
        self.gate = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        h, _ = self.rnn(self.ln1(x))
        # Group-wise softmax gate scaled by sqrt(g_dim) so variance stays O(1)
        g = self.gate(h).view(B, T, self.n_groups, self.g_dim)
        w = F.softmax(g, dim=-1) * math.sqrt(self.g_dim)
        h_soft = (h.view(B, T, self.n_groups, self.g_dim) * w).view(B, T, C)
        x = x + self.proj(h_soft)
        x = x + self.mlp(self.ln2(x))
        return x


class SeqLM(nn.Module):
    def __init__(self, arch: str, vocab_size: int, seq_len: int, d_model: int = 384, n_layers: int = 6, n_heads: int = 6):
        super().__init__()
        self.arch = arch
        self.d_model = d_model
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # Match sinusoidal PE scale (0.02) to tok_emb init std (0.02) so PE doesn't drown out tokens 35x!
        self.register_buffer("pos_emb", make_sinusoidal_pe(seq_len + 16, d_model) * 0.02)

        # Parameter-match non-embedding layer budget to 12 * d^2 per layer:
        # - microgpt      : attn (4 d^2) + MLP(4d -> 8 d^2) = 12 d^2 (sinusoidal PE)
        # - microgpt_rope : attn (4 d^2, RoPE) + MLP(4d -> 8 d^2) = 12 d^2 (zero extra params)
        # - domingos_lstm : LSTM (8 d^2) + MLP(2d -> 4 d^2) = 12 d^2
        # - domingos_rnn  : RNN (2 d^2) + gate+proj (2 d^2) + MLP(4d -> 8 d^2) = 12 d^2
        self.use_abs_pe = arch != "microgpt_rope"
        if arch == "microgpt":
            self.blocks = nn.ModuleList([GPTBlock(d_model, n_heads, d_model * 4, use_rope=False) for _ in range(n_layers)])
        elif arch == "microgpt_rope":
            self.blocks = nn.ModuleList([GPTBlock(d_model, n_heads, d_model * 4, use_rope=True, max_len=seq_len + 16) for _ in range(n_layers)])
        elif arch == "domingos_lstm":
            self.blocks = nn.ModuleList([DomingosLSTMBlock(d_model, d_model * 2) for _ in range(n_layers)])
        elif arch == "domingos_rnn":
            self.blocks = nn.ModuleList([DomingosSoftmaxRNNBlock(d_model, d_model * 4, n_groups=n_heads) for _ in range(n_layers)])
        else:
            raise ValueError(f"Unknown arch: {arch}")

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying (standard in GPT-2 / MicroGPT)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        x = self.tok_emb(ids)
        if self.use_abs_pe:
            x = x + self.pos_emb[:T].unsqueeze(0)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))

    def param_counts(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        emb = self.tok_emb.weight.numel()
        return {"total": total, "non_embed": total - emb, "embed": emb}


def stream_packed_tokens(tokenizer, seq_len: int, skip_docs: int = 0) -> Iterator[List[int]]:
    """Stream FineWeb-Edu (Common Crawl) packed into exact (seq_len + 1) token chunks."""
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    buf: List[int] = []
    eos = tokenizer.eos_token_id
    for idx, row in enumerate(ds):
        if idx < skip_docs:
            continue
        toks = tokenizer.encode(row["text"], add_special_tokens=False)
        buf.extend(toks)
        buf.append(eos)
        while len(buf) >= seq_len + 1:
            yield buf[: seq_len + 1]
            buf = buf[seq_len + 1 :]


@torch.no_grad()
def evaluate_val(model: SeqLM, val_chunks: torch.Tensor, batch_size: int, device: torch.device, amp_dtype) -> Dict:
    """Compute held-out loss, perplexity, and per-position-bucket loss."""
    model.eval()
    N, L1 = val_chunks.shape
    T = L1 - 1
    pos_loss_sum = torch.zeros(T, dtype=torch.float64, device=device)
    total_count = 0

    for s in range(0, N, batch_size):
        chunk = val_chunks[s : s + batch_size].to(device)
        x = chunk[:, :-1]
        y = chunk[:, 1:]
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(x)
        losses = F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none")  # [b, T]
        pos_loss_sum += losses.sum(dim=0).double()
        total_count += int(chunk.shape[0])

    pos_mean = (pos_loss_sum / max(1, total_count)).cpu()
    mean_loss = float(pos_mean.mean().item())
    ppl = float(math.exp(min(20.0, mean_loss)))
    buckets = {}
    for lo, hi in [(0, 32), (32, 64), (64, 128), (128, 256), (256, T)]:
        if lo < T:
            buckets[f"{lo}-{min(hi, T)}"] = round(float(pos_mean[lo : min(hi, T)].mean().item()), 4)
    return {"val_loss": round(mean_loss, 4), "val_ppl": round(ppl, 2), "pos_loss": buckets}


@torch.no_grad()
def evaluate_in_context_recall(model: SeqLM, tokenizer, val_chunks: torch.Tensor, device: torch.device, amp_dtype) -> Dict[str, float]:
    """Zero-shot in-context associative recall probe embedded in real Common Crawl text.
    Plants 4 novel 'name -> 2-word/token code' associations at the start, followed by real
    Common Crawl filler text of length D, then queries one of the keys!
    Measures exact next-token accuracy on the target value token across 64 trials per distance.
    """
    model.eval()
    rng = random.Random(12345)
    keys = ["alpha", "bravo", "delta", "gamma", "omega", "sigma", "theta", "kappa"]
    # Pick single-token capitalized words as unambiguous target values
    values = [" tiger", " ocean", " cedar", " comet", " pearl", " anvil", " lotus", " prism",
              " maple", " coral", " amber", " flint", " bison", " ivory", " topaz", " raven"]
    val_toks = [tokenizer.encode(v, add_special_tokens=False)[0] for v in values]
    filler_pool = val_chunks.flatten().tolist()

    results: Dict[str, float] = {}
    for dist in (32, 128, 320):
        hits = 0
        trials = 64
        for t in range(trials):
            chosen_k = rng.sample(keys, 4)
            chosen_v = rng.sample(values, 4)
            header = " ".join(f"Key {k} is{v}." for k, v in zip(chosen_k, chosen_v)) + " "
            q_idx = rng.randrange(4)
            query = f" Key {chosen_k[q_idx]} is"
            target_id = tokenizer.encode(chosen_v[q_idx], add_special_tokens=False)[0]

            h_ids = tokenizer.encode(header, add_special_tokens=False)
            q_ids = tokenizer.encode(query, add_special_tokens=False)
            f_start = rng.randrange(0, max(1, len(filler_pool) - dist - 1))
            f_ids = filler_pool[f_start : f_start + dist]
            prompt = (h_ids + f_ids + q_ids)[-model.seq_len :]
            x = torch.tensor([prompt], dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits = model(x)[0, -1].float()
            # Rank among the 4 candidate values in context (4-way associative recall, chance = 25%)
            cand_ids = [tokenizer.encode(v, add_special_tokens=False)[0] for v in chosen_v]
            cand_scores = logits[cand_ids]
            pred_idx = int(cand_scores.argmax().item())
            hits += int(pred_idx == q_idx)
        results[f"recall_D{dist}"] = round(hits / trials, 4)
    return results


def train_arch(
    arch: str,
    train_chunks: torch.Tensor,
    val_chunks: torch.Tensor,
    tokenizer,
    vocab_size: int,
    seq_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    steps: int,
    batch_size: int,
    lr: float,
    eval_every: int,
    seed: int,
    device: torch.device,
) -> Dict:
    torch.manual_seed(seed)
    model = SeqLM(arch, vocab_size, seq_len, d_model=d_model, n_layers=n_layers, n_heads=n_heads).to(device)
    pcounts = model.param_counts()
    print(
        f"\n=== Training {arch} | non_embed={pcounts['non_embed']/1e6:.2f}M total={pcounts['total']/1e6:.2f}M ===",
        flush=True,
    )

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95),
    )
    warmup = min(100, steps // 10)

    def lr_mult(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    if device.type == "cpu":
        amp_dtype = torch.float32

    N_tr = train_chunks.shape[0]
    tps = batch_size * seq_len
    history = []
    t0 = time.time()

    # Step 0 eval
    v0 = evaluate_val(model, val_chunks, batch_size, device, amp_dtype)
    history.append({"step": 0, "tokens": 0, "wall_s": 0.0, **v0})
    print(f"  step {0:4d}/{steps} | tok=  0.0M | t=  0.0s | val_loss={v0['val_loss']:.4f} ppl={v0['val_ppl']:6.2f}", flush=True)

    train_time = 0.0
    for step in range(1, steps + 1):
        t_s = time.time()
        model.train()
        # Deterministic identical batch sequence across all architectures
        start = ((step - 1) * batch_size) % max(1, N_tr - batch_size)
        chunk = train_chunks[start : start + batch_size].to(device)
        x = chunk[:, :-1]
        y = chunk[:, 1:]

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time += time.time() - t_s

        if step % eval_every == 0 or step == steps:
            ve = evaluate_val(model, val_chunks, batch_size, device, amp_dtype)
            toks_seen = step * tps
            tok_s = toks_seen / max(1e-6, train_time)
            history.append({
                "step": step,
                "tokens": toks_seen,
                "wall_s": round(train_time, 1),
                "tok_per_s": round(tok_s, 0),
                **ve,
            })
            print(
                f"  step {step:4d}/{steps} | tok={toks_seen/1e6:5.1f}M | t={train_time:5.1f}s ({tok_s/1e3:5.1f}k tok/s) | "
                f"val_loss={ve['val_loss']:.4f} ppl={ve['val_ppl']:6.2f} | pos={ve['pos_loss']}",
                flush=True,
            )

    recall = evaluate_in_context_recall(model, tokenizer, val_chunks, device, amp_dtype)
    print(f"  In-context 4-way associative recall (chance=25%): {recall}", flush=True)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "arch": arch,
        "params": pcounts,
        "total_train_s": round(train_time, 1),
        "final": history[-1],
        "recall": recall,
        "history": history,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--val-seqs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--archs", type=str, default="microgpt,microgpt_rope,domingos_lstm,domingos_rnn")
    ap.add_argument("--out", type=str, default="runs/domingos_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"06-domingos-rnn-vs-gpt | device={device} steps={args.steps} batch={args.batch_size} seq_len={args.seq_len}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = len(tokenizer)

    # Pre-buffer exact identical val + train token chunks from Common Crawl (FineWeb-Edu)
    needed_train = args.steps * args.batch_size
    total_needed = args.val_seqs + needed_train
    print(f"Streaming {total_needed} packed {args.seq_len}-tok Common Crawl chunks ({total_needed*args.seq_len/1e6:.1f}M tokens)...", flush=True)
    t_data = time.time()
    chunks = []
    for c in stream_packed_tokens(tokenizer, args.seq_len):
        chunks.append(c)
        if len(chunks) % 50000 == 0:
            print(f"  buffered {len(chunks)}/{total_needed} chunks ({time.time() - t_data:.0f}s)...", flush=True)
        if len(chunks) >= total_needed:
            break
    all_t = torch.tensor(chunks, dtype=torch.long)
    val_chunks = all_t[: args.val_seqs]
    train_chunks = all_t[args.val_seqs :]
    print(f"Buffered {len(train_chunks)} train + {len(val_chunks)} val chunks in {time.time() - t_data:.1f}s", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            results = prev.get("results", {})
            if results:
                print(f"Resuming from {out_path}: already completed {list(results.keys())}", flush=True)
        except Exception:
            results = {}

    archs = [s.strip() for s in args.archs.split(",") if s.strip()]
    for arch in archs:
        if arch in results:
            print(f"\n=== Skipping already-completed {arch} ===", flush=True)
            continue
        results[arch] = train_arch(
            arch, train_chunks, val_chunks, tokenizer, vocab_size,
            seq_len=args.seq_len, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
            steps=args.steps, batch_size=args.batch_size, lr=args.lr, eval_every=args.eval_every,
            seed=args.seed, device=device,
        )
        # Incremental save after every architecture
        out_path.write_text(json.dumps({"config": vars(args), "results": results}, indent=2))

    print("\n=== FINAL SUMMARY (Common Crawl / FineWeb-Edu, equal tokens) ===", flush=True)
    for arch, r in results.items():
        f = r["final"]
        print(
            f"  {arch:15s} | non_emb={r['params']['non_embed']/1e6:5.2f}M | "
            f"ppl={f['val_ppl']:6.2f} (loss={f['val_loss']:.4f}) | "
            f"t={r['total_train_s']:5.1f}s ({f['tok_per_s']/1e3:5.1f}k tok/s) | recall={r['recall']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
