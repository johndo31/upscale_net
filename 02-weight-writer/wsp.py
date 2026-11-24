"""Weight-writing self-improvement experiment: split, generation, LoRA fine-tuning, evaluation, stages.

Usage (run from this folder, with HF_HUB_OFFLINE=1):
  python wsp.py split                                  # create / show split.json
  python wsp.py check                                  # split + round-trip + LoRA + generation checks
  python wsp.py bench                                  # generation and fine-tuning speed
  python wsp.py pilot                                  # runs/base/round0.json
  python wsp.py teacher                                # runs/teacher/round1.json
  python wsp.py loop --arm pass --rounds 3             # runs/pass/round{1..3}.json  (also hindsight, unverified)
Common flags: --k --temperature --batch-size --limit --steps --out-dir --seed --heldout-k --engine --device --rank
Devices: --device auto picks cuda, then mps, then cpu. Defaults differ per device (see configure()).
Training saves <out-dir>/<arm>/round<r>.adapter.pt before merging/evaluating; a round whose JSON is missing
but whose adapter exists is evaluated from the adapter without retraining.
"""
import argparse
import contextlib
import copy
import gc
import hashlib
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
import torch.nn as nn

import tasks

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen2.5-0.5B"


def pick_device(name="auto"):
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda: CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        sys.exit("--device mps: MPS is not available")
    return name


DEVICE = pick_device()                  # overridden by configure() from --device
GEN_DTYPE = torch.float16               # dtype of the merged model used for generation (configure())
TRAIN_AMP = None                        # autocast dtype for training forwards (bf16 on cuda), None = plain fp32
TRAIN_ATTN = "eager" if DEVICE == "mps" else "sdpa"
SPLIT_PATH = os.path.join(HERE, "split.json")
SOL = json.load(open(os.path.join(HERE, "solutions.json")))
FEWSHOT_SET = {(n, f) for n, f, _ in tasks.FEWSHOT}
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def dtype_name(dt):
    return None if dt is None else str(dt).replace("torch.", "")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def free():
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()


def sync():
    if DEVICE == "mps":
        torch.mps.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()


def amp_ctx():
    """Autocast for training forwards (backward runs outside it)."""
    if TRAIN_AMP is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=DEVICE, dtype=TRAIN_AMP)


def rss_gb():
    """(current, peak) resident set size of this process in GB (Linux /proc)."""
    cur = peak = float("nan")
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    cur = int(line.split()[1]) / 2 ** 20
                elif line.startswith("VmHWM:"):
                    peak = int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return cur, peak


def footprint_gb():
    """(current, peak) physical memory footprint of this process in GB, as macOS accounts it (includes GPU memory)."""
    import re, subprocess
    try:
        out = subprocess.run(["footprint", str(os.getpid())], capture_output=True, text=True, timeout=20).stdout
        val = lambda key: next(float(m.group(1)) / (1024 if m.group(2) == "MB" else 1 if m.group(2) == "GB" else 1024 ** 2)
                               for m in [re.search(rf"{key}:\s*([\d.]+)\s*(KB|MB|GB)", out)] if m)
        return val("phys_footprint"), val("phys_footprint_peak")
    except Exception:
        return float("nan"), float("nan")


def log_mem(tag):
    if sys.platform == "darwin":
        cur, peak = footprint_gb()
        what = "footprint"
    else:
        cur, peak = rss_gb()
        what = "rss"
    if DEVICE == "cuda":
        gpu = (f"cuda allocated {torch.cuda.memory_allocated() / 2 ** 30:.2f} GB, "
               f"max allocated {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB")
    else:
        gpu = f"gpu {torch.mps.driver_allocated_memory() / 2 ** 30 if DEVICE == 'mps' else 0.0:.2f} GB"
    log(f"memory [{tag}]: {what} {cur:.2f} GB, peak {peak:.2f} GB, {gpu}, limit {MEM_GB} GB")
    return peak


def set_memory_limit(mem_gb, cpu_reserve_gb=1.5):
    """Hard cap: GPU allocations beyond (mem_gb - reserve) raise an out-of-memory error instead of swapping.
    mps: unified memory, so a CPU reserve is subtracted. cuda: the cap applies to device memory only."""
    global MEM_GB
    MEM_GB = mem_gb
    if DEVICE == "mps" and mem_gb:
        frac = max(0.05, (mem_gb - cpu_reserve_gb) * 2 ** 30 / torch.mps.recommended_max_memory())
        torch.mps.set_per_process_memory_fraction(frac)
        log(f"memory limit {mem_gb} GB: GPU allocations capped at {mem_gb - cpu_reserve_gb:.1f} GB")
    elif DEVICE == "cuda" and mem_gb:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, mem_gb * 2 ** 30 / total))
        log(f"memory limit {mem_gb} GB: CUDA allocations capped")


MEM_GB = None


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


# ===================================================================== split
def make_split(seed=0, frac=0.25):
    tasks_out = []
    for f in range(16):
        if (2, f) not in FEWSHOT_SET:
            tasks_out.append({"n": 2, "f": f, "level": SOL["2"][str(f)]["level"], "split": "train"})
    fs3 = [f for f in range(256) if (3, f) not in FEWSHOT_SET]
    classes = {}
    for f in fs3:
        classes.setdefault(tasks.p_class(f, 3), []).append(f)
    lvl = lambda f: SOL["3"][str(f)]["level"]
    for c, mem in classes.items():
        assert len({lvl(f) for f in mem}) == 1, "level must be constant within a p-class"
    forced_held, forced_train = tasks.p_class(150, 3), tasks.p_class(105, 3)
    rng = random.Random(seed)
    held = set()
    for L in sorted({lvl(f) for f in fs3}):
        cls = sorted(c for c in classes if lvl(classes[c][0]) == L)
        total = sum(len(classes[c]) for c in cls)
        target = frac * total
        cur = 0
        if forced_held in cls:
            held.add(forced_held); cur += len(classes[forced_held])
        rest = [c for c in cls if c not in (forced_held, forced_train)]
        rng.shuffle(rest)
        for c in rest:
            s = len(classes[c])
            if abs(cur + s - target) < abs(cur - target):
                held.add(c); cur += s
    for f in fs3:
        c = tasks.p_class(f, 3)
        tasks_out.append({"n": 3, "f": f, "level": lvl(f), "split": "heldout" if c in held else "train"})
    return {"seed": seed, "tasks": tasks_out}


def load_split():
    if not os.path.exists(SPLIT_PATH):
        dump(make_split(), SPLIT_PATH)
        log("wrote", SPLIT_PATH)
    return json.load(open(SPLIT_PATH))["tasks"]


def task_lists(limit=None):
    T = load_split()
    tr = [t for t in T if t["split"] == "train"]
    ho = [t for t in T if t["split"] == "heldout"]
    if limit:
        tr, ho = tr[:limit], ho[:limit]
    return tr, ho


def split_stats(T):
    out = {}
    for s in ("train", "heldout"):
        for n in (2, 3):
            sub = [t for t in T if t["split"] == s and t["n"] == n]
            if sub:
                out[f"{s} n={n}"] = {"total": len(sub), "by_level": {l: sum(t["level"] == l for t in sub) for l in range(4)}}
    return out


# ===================================================================== model utils
_TOK = None


def get_tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    return _TOK


def load_model(dtype=None, attn="sdpa"):
    """dtype None = GEN_DTYPE. Training uses attn=TRAIN_ATTN: "eager" on MPS, where the SDPA backward with a
    mask + GQA is ~20x slower than eager; "sdpa" on cuda/cpu."""
    from transformers import AutoModelForCausalLM
    dtype = dtype or GEN_DTYPE
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, attn_implementation=attn).to(DEVICE)
    m.eval()
    return m


def max_new_tokens_default(margin=20):
    tok = get_tok()
    return max(len(tok(s["answer"] + "\n\n").input_ids) for n in SOL for s in SOL[n].values()) + margin


def prefix_ids():
    return get_tok()(tasks.PREFIX).input_ids


def _vocab_flags(tok, V):
    """Per token id: contains a blank line, starts with newline, ends with newline (byte-level BPE: 'Ċ' = \\n)."""
    has, st, en = torch.zeros(V, dtype=torch.bool), torch.zeros(V, dtype=torch.bool), torch.zeros(V, dtype=torch.bool)
    toks = tok.convert_ids_to_tokens(list(range(len(tok))))
    special = set(tok.all_special_ids)
    for i, s in enumerate(toks):
        if s is None or i in special:
            continue
        has[i] = "ĊĊ" in s
        st[i] = s.startswith("Ċ")
        en[i] = s.endswith("Ċ")
    return has, st, en


_FLAGS = {}


class FastGen:
    """fp16/bf16 decoding (dtype of the model) for a batch of prompts that all start with the same token prefix.
    The prefix keys/values are computed once (batch 1) and shared by all rows ("cascade" attention);
    each row keeps only its own suffix + generated tokens in a preallocated buffer (no cache copying)."""

    def __init__(self, model, pref):
        from transformers import DynamicCache
        self.m = model
        cfg = model.config
        self.nh, self.nkv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.g = self.nh // self.nkv
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.nh
        self.scale = self.hd ** -0.5
        self.Lp = len(pref)
        with torch.no_grad():
            c = DynamicCache(config=cfg)
            model.model(input_ids=torch.tensor([pref], device=DEVICE), past_key_values=c, use_cache=True)
            self.Kp = [l.keys[0].contiguous() for l in c.layers]      # nkv x Lp x hd
            self.Vp = [l.values[0].contiguous() for l in c.layers]
        del c

    def forward(self, ids, kb, vb, t0):
        """ids: B x s new tokens at local positions t0..t0+s-1. Writes K/V into kb/vb. Returns last-position logits."""
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        m, B, s = self.m, ids.shape[0], ids.shape[1]
        nkv, g, hd, Lp = self.nkv, self.g, self.hd, self.Lp
        x = m.model.embed_tokens(ids)
        pos = torch.arange(Lp + t0, Lp + t0 + s, device=ids.device)[None]
        cos, sin = m.model.rotary_emb(x, pos)
        t = t0 + s
        mask = None
        if s > 1:
            qi = torch.arange(s, device=ids.device)[:, None] + t0
            kj = torch.arange(t, device=ids.device)[None]
            mask = torch.zeros(s, t, device=ids.device).masked_fill(kj > qi, float("-inf")).repeat(g, 1)  # (g*s) x t
        for li, layer in enumerate(m.model.layers):
            at = layer.self_attn
            h = layer.input_layernorm(x)
            q = at.q_proj(h).view(B, s, self.nh, hd).transpose(1, 2)
            k = at.k_proj(h).view(B, s, nkv, hd).transpose(1, 2)
            v = at.v_proj(h).view(B, s, nkv, hd).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            kb[li][:, :, t0:t] = k
            vb[li][:, :, t0:t] = v
            qg = (q * self.scale).reshape(B, nkv, g * s, hd)                       # head h -> kv head h // g
            sp = torch.matmul(qg.transpose(0, 1).reshape(nkv, B * g * s, hd), self.Kp[li].transpose(1, 2))
            sp = sp.view(nkv, B, g * s, Lp).transpose(0, 1)                          # B x nkv x gs x Lp
            sl = torch.matmul(qg, kb[li][:, :, :t].transpose(2, 3))                  # B x nkv x gs x t
            sc = torch.cat([sp, sl], -1).float()
            if mask is not None:
                sc[..., Lp:] += mask
            p = torch.softmax(sc, -1).to(x.dtype)
            pp = p[..., :Lp].transpose(0, 1).reshape(nkv, B * g * s, Lp)
            o = torch.matmul(pp, self.Vp[li]).view(nkv, B, g * s, hd).transpose(0, 1)
            o = o + torch.matmul(p[..., Lp:], vb[li][:, :, :t])
            o = o.reshape(B, self.nh, s, hd).transpose(1, 2).reshape(B, s, self.nh * hd)
            x = x + at.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return m.lm_head(m.model.norm(x[:, -1]))

    @torch.no_grad()
    def run(self, ids, temperature, max_new, flags, eos, chunk=64):
        """ids: B x S (all rows same length, starting after the shared prefix). Returns list of token lists."""
        B, S = ids.shape
        T = S + max_new
        dt = self.Kp[0].dtype
        kb = [torch.empty(B, self.nkv, T, self.hd, dtype=dt, device=DEVICE) for _ in self.Kp]
        vb = [torch.empty(B, self.nkv, T, self.hd, dtype=dt, device=DEVICE) for _ in self.Kp]
        logits = torch.cat([self.forward(ids[a:a + chunk], [k[a:a + chunk] for k in kb], [v[a:a + chunk] for v in vb], 0)
                            for a in range(0, B, chunk)])
        has, st, en = flags
        done = torch.zeros(B, dtype=torch.bool, device=DEVICE)
        prev_en = torch.zeros(B, dtype=torch.bool, device=DEVICE)
        out = []
        for step in range(max_new):
            if temperature and temperature > 0:
                nxt = torch.multinomial(torch.softmax(logits.float() / temperature, -1), 1)[:, 0]
            else:
                nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            out.append(nxt)
            done = done | has[nxt] | (prev_en & st[nxt]) | (nxt == eos)
            prev_en = en[nxt]
            if step == max_new - 1 or (step % 4 == 3 and bool(done.all())):
                break
            logits = self.forward(nxt[:, None], kb, vb, S + step)
        del kb, vb
        return torch.stack(out, 1).tolist()


def _hf_generate(model, tok, ids_list, temperature, max_new):
    """Reference path: transformers generate with stop_strings (left padding)."""
    L = max(len(x) for x in ids_list)
    pad = tok.pad_token_id
    inp = torch.tensor([[pad] * (L - len(x)) + x for x in ids_list], device=DEVICE)
    am = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids_list], device=DEVICE)
    kw = dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0) if temperature and temperature > 0 else dict(do_sample=False)
    with torch.no_grad():
        out = model.generate(input_ids=inp, attention_mask=am, max_new_tokens=max_new, stop_strings=["\n\n"],
                             tokenizer=tok, pad_token_id=pad, **kw)
    return out[:, L:].tolist()


ENGINE = "fast"
MICRO_BATCH = 4
PREFIX_GRAD = True


def generate(model, tok, prompts, k, temperature, batch_size, max_new_tokens, seed=0, progress=True):
    """k answer strings (first block) per prompt. temperature 0 -> greedy, k forced to 1."""
    if not temperature or temperature <= 0:
        k = 1
    torch.manual_seed(seed)
    ids = [tok(p).input_ids for p in prompts]
    V = model.get_output_embeddings().weight.shape[0]
    if V not in _FLAGS:
        _FLAGS[V] = tuple(x.to(DEVICE) for x in _vocab_flags(tok, V))
    flags = _FLAGS[V]
    pref = prefix_ids()
    use_fast = ENGINE == "fast" and all(x[:len(pref)] == pref and len(x) > len(pref) for x in ids)
    fg = FastGen(model, pref) if use_fast else None
    items = sorted(((len(ids[i]), i, j) for i in range(len(prompts)) for j in range(k)))
    groups = {}
    for L, i, j in items:
        groups.setdefault(L if use_fast else 0, []).append((i, j))
    res = [[None] * k for _ in prompts]
    t0, done_n, total = time.time(), 0, len(items)
    for L, its in groups.items():
        for a in range(0, len(its), batch_size):
            b = its[a:a + batch_size]
            if use_fast:
                x = torch.tensor([ids[i][len(pref):] for i, _ in b], device=DEVICE)
                gen = fg.run(x, temperature, max_new_tokens, flags, tok.eos_token_id)
            else:
                gen = _hf_generate(model, tok, [ids[i] for i, _ in b], temperature, max_new_tokens)
            for (i, j), g in zip(b, gen):
                res[i][j] = tasks.first_block(tok.decode(g, skip_special_tokens=True))
            done_n += len(b)
            if progress:
                el = time.time() - t0
                log(f"  gen {done_n}/{total}  {done_n / el:.1f} ans/s")
    del fg
    return res


# ===================================================================== LoRA
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, gen):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=base.weight.dtype))
        bound = 1.0 / math.sqrt(base.in_features)        # = kaiming_uniform(a=sqrt(5)), as in peft
        with torch.no_grad():
            self.A.copy_((torch.rand(self.A.shape, generator=gen) * 2 - 1) * bound)

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def add_lora(model, r=16, alpha=32, seed=0):
    for p in model.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(seed)
    params = []
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in LORA_TARGETS:
                if hasattr(parent, name) and isinstance(getattr(parent, name), nn.Linear):
                    w = LoRALinear(getattr(parent, name), r, alpha, gen)
                    setattr(parent, name, w)
                    params += [w.A, w.B]
    return params


@torch.no_grad()
def merge(model):
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in LORA_TARGETS:
                w = getattr(parent, name, None)
                if isinstance(w, LoRALinear):
                    w.base.weight += (w.B @ w.A) * w.scale
                    setattr(parent, name, w.base)
    return model


# ===================================================================== fine-tuning
def _prefix_kv(model, pref, grad=True):
    """Run the shared prompt prefix once (batch 1). Returns (graph K/V tensors, detached leaf copies).
    Micro-batches attend to the leaves; their K/V gradients accumulate on the leaves and are pushed
    through the prefix graph once per optimizer step (exact gradient, one prefix backward)."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, amp_ctx():
        pc = model.model(input_ids=torch.tensor([pref], device=DEVICE), use_cache=True).past_key_values
    kv = [t for l in pc.layers for t in (l.keys, l.values)]
    leaves = [t.detach().requires_grad_(grad) for t in kv]
    return kv, leaves


def _suffix_loss(model, batch, Lp, leaves, reduction="mean"):
    """Loss on answer tokens for a batch whose prompts all start with the same Lp-token prefix (K/V in leaves)."""
    from transformers import DynamicCache
    B = len(batch)
    seqs, labs = [], []
    for pids, aids in batch:
        s = pids[Lp:] + aids
        seqs.append(s)
        labs.append([-100] * (len(pids) - Lp) + aids)
    S = max(len(s) for s in seqs)
    pad = get_tok().pad_token_id
    ids = torch.tensor([s + [pad] * (S - len(s)) for s in seqs], device=DEVICE)
    lab = torch.tensor([l + [-100] * (S - len(l)) for l in labs], device=DEVICE)
    am = torch.tensor([[1] * (Lp + len(s)) + [0] * (S - len(s)) for s in seqs], device=DEVICE)    # right padding
    if Lp:
        cache = DynamicCache(ddp_cache_data=[(leaves[2 * i].expand(B, -1, -1, -1), leaves[2 * i + 1].expand(B, -1, -1, -1))
                                             for i in range(len(leaves) // 2)])
        pos = torch.arange(Lp, Lp + S, device=DEVICE)[None].expand(B, -1)
        h = model.model(input_ids=ids, attention_mask=am, position_ids=pos, past_key_values=cache, use_cache=True).last_hidden_state
    else:
        h = model.model(input_ids=ids, attention_mask=am).last_hidden_state
    tgt = lab[:, 1:]
    sel = tgt != -100
    logits = model.lm_head(h[:, :-1][sel])                 # logits only where the loss is taken
    return nn.functional.cross_entropy(logits.float(), tgt[sel], reduction=reduction)


def train_step_backward(model, batch, pref, micro_batch, prefix_grad=True):
    """Accumulates the gradient of the mean answer-token loss over `batch`. Returns the loss value.
    micro_batch >= len(batch) = one pass over the whole batch (no accumulation)."""
    Lp = len(pref)
    ntok = sum(len(a) for _, a in batch)
    kv, leaves = _prefix_kv(model, pref, prefix_grad) if Lp else ([], [])
    tot = 0.0
    for c in range(0, len(batch), micro_batch):
        with amp_ctx():
            part = _suffix_loss(model, batch[c:c + micro_batch], Lp, leaves, reduction="sum") / ntok
        part.backward()
        tot += part.item()
    if Lp and prefix_grad:
        torch.autograd.backward(kv, [l.grad for l in leaves])
    return tot


def _shared_prefix_loss(model, batch, pref):
    """Single-graph version (used by the check against the full-sequence loss)."""
    if not pref:
        return _suffix_loss(model, batch, 0, [])
    kv, _ = _prefix_kv(model, pref, True)
    return _suffix_loss(model, batch, len(pref), kv)


def _full_loss(model, batch):
    """Reference: plain full-sequence loss (used only by the check)."""
    S = max(len(p) + len(a) for p, a in batch)
    pad = get_tok().pad_token_id
    ids = torch.tensor([p + a + [pad] * (S - len(p) - len(a)) for p, a in batch], device=DEVICE)
    lab = torch.tensor([[-100] * len(p) + a + [-100] * (S - len(p) - len(a)) for p, a in batch], device=DEVICE)
    am = (lab != -100) | (torch.arange(S, device=DEVICE)[None] < torch.tensor([len(p) for p, _ in batch], device=DEVICE)[:, None])
    h = model.model(input_ids=ids, attention_mask=am.long()).last_hidden_state
    tgt = lab[:, 1:]
    sel = tgt != -100                                      # logits only where the loss is taken (memory)
    return nn.functional.cross_entropy(model.lm_head(h[:, :-1][sel]).float(), tgt[sel])


EPOCHS, MAX_STEPS, POOL_CAP = 10, 600, 4          # overridden by --epochs / --max-steps / --pool-cap


def default_steps(n_examples, batch_size=16):
    return min(MAX_STEPS, max(30, round(EPOCHS * n_examples / batch_size)))


def tokenize_examples(examples):
    tok = get_tok()
    return [(tok(p).input_ids, tok(a + "\n\n").input_ids) for p, a in examples]


def examples_sha(examples):
    return hashlib.sha256(json.dumps(examples).encode()).hexdigest()


def save_adapter(model, path, info):
    """LoRA A/B tensors (fp32, CPU) + info, written atomically."""
    lora = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            lora[name + ".A"] = mod.A.detach().float().cpu().clone()
            lora[name + ".B"] = mod.B.detach().float().cpu().clone()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"format": 1, "lora": lora, **info}, tmp)
    os.replace(tmp, path)
    log(f"saved adapter ({len(lora) // 2} LoRA modules, step {info.get('steps')}): {path}")


def model_from_adapter(ad):
    """Fresh base model + saved LoRA tensors, merged exactly as after training. Returns the eval model (GEN_DTYPE)."""
    free()
    model = load_model(torch.float32, attn=TRAIN_ATTN)
    add_lora(model, ad["r"], ad["alpha"], 0)
    mods = dict(model.named_modules())
    with torch.no_grad():
        for key, t in ad["lora"].items():
            name, which = key.rsplit(".", 1)
            getattr(mods[name], which).copy_(t.to(DEVICE))
    merge(model)
    model = model.to(GEN_DTYPE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    free()
    return model


def finetune(examples, steps=None, lr=2e-4, batch_size=16, r=16, alpha=32, seed=0, log_every=20, loss_log=None,
             micro_batch=None, save_path=None, save_info=None):
    """examples: list of (prompt_text, answer_text). Returns the merged model (GEN_DTYPE) in eval mode.
    micro_batch: split each batch for gradient accumulation (same gradient, less activation memory);
    0/None = MICRO_BATCH, which is 0 (= whole batch in one pass) except on mps.
    save_path: write the LoRA adapter there after training, before merging."""
    micro_batch = micro_batch or MICRO_BATCH or batch_size
    free()
    torch.manual_seed(seed)
    steps = steps or default_steps(len(examples), batch_size)
    data = tokenize_examples(examples)
    pref = prefix_ids()
    if not all(p[:len(pref)] == pref and len(p) > len(pref) for p, _ in data):
        pref = []
    t_start = time.time()
    model = load_model(torch.float32, attn=TRAIN_ATTN)
    params = add_lora(model, r, alpha, seed)
    model.train()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    warm = max(1, round(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    rng = random.Random(seed)
    order, pos = [], 0
    log(f"finetune: {len(examples)} examples, {steps} steps, batch {batch_size}, micro-batch {micro_batch}, "
        f"lr {lr}, r {r}, attn {TRAIN_ATTN}, amp {dtype_name(TRAIN_AMP)}, device {DEVICE}")
    t0, acc = time.time(), []
    for step in range(steps):
        batch = []
        while len(batch) < min(batch_size, len(data)):
            if pos >= len(order):
                order = list(range(len(data))); rng.shuffle(order); pos = 0
            batch.append(data[order[pos]]); pos += 1
        opt.zero_grad(set_to_none=True)
        loss = torch.tensor(train_step_backward(model, batch, pref, micro_batch, PREFIX_GRAD))
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        acc.append(loss.item())
        if (step + 1) % log_every == 0 or step == steps - 1:
            el = time.time() - t0
            msg = f"  step {step + 1}/{steps} loss {np.mean(acc):.4f} lr {sched.get_last_lr()[0]:.2e} {(step + 1) / el:.2f} it/s"
            log(msg)
            if (step + 1) % (5 * log_every) == 0 or step == steps - 1 or step + 1 == log_every:
                log_mem(f"train step {step + 1}")
            if loss_log is not None:
                loss_log.append({"step": step + 1, "loss": float(np.mean(acc))})
            acc = []
    if save_path:
        save_adapter(model, save_path, {**(save_info or {}), "r": r, "alpha": alpha, "steps": steps, "lr": lr,
                                        "seed": seed, "n_examples": len(examples), "examples_sha": examples_sha(examples),
                                        "loss_log": list(loss_log or []), "finetune_seconds": time.time() - t_start})
    merge(model)
    del opt, params
    model = model.to(GEN_DTYPE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    free()
    return model


# ===================================================================== evaluation
def score(text, n, f):
    net = tasks.parse(text, n)
    if net is None:
        return {"text": text, "valid": False, "tt": None, "passed": False, "size": None}
    tt = tasks.run(net, n)
    return {"text": text, "valid": True, "tt": tt, "passed": tt == f, "size": list(tasks.size(net))}


def _metrics(recs, k):
    if not recs:
        return {"n_tasks": 0}
    m = {"n_tasks": len(recs)}
    S = [r["samples"][:k] for r in recs]
    m["valid_rate"] = float(np.mean([s["valid"] for ss in S for s in ss]))
    m["mean_pass"] = float(np.mean([np.mean([s["passed"] for s in ss]) for ss in S]))
    m[f"solved@{k}"] = float(np.mean([any(s["passed"] for s in ss) for ss in S]))
    m["greedy_pass"] = float(np.mean([r["greedy"]["passed"] for r in recs]))
    m["greedy_valid"] = float(np.mean([r["greedy"]["valid"] for r in recs]))
    if all("extra" in r for r in recs):
        K = len(recs[0]["samples"]) + len(recs[0]["extra"])
        m[f"solved@{K}"] = float(np.mean([any(s["passed"] for s in r["samples"] + r["extra"]) for r in recs]))
        m[f"mean_pass@{K}"] = float(np.mean([np.mean([s["passed"] for s in r["samples"] + r["extra"]]) for r in recs]))
    return m


def metrics(records, k):
    out = {}
    for s in ("train", "heldout"):
        R = [r for r in records if r["split"] == s]
        out[s] = _metrics(R, k)
        out[s]["by_level"] = {str(L): _metrics([r for r in R if r["level"] == L], k) for L in sorted({r["level"] for r in R})}
    return out


def evaluate(model, task_list, k=16, temperature=0.8, batch_size=256, seed=0, heldout_k=None):
    tok = get_tok()
    mnt = max_new_tokens_default()
    prompts = [tasks.prompt(t["f"], t["n"]) for t in task_list]
    t0 = time.time()
    log(f"evaluate: {len(task_list)} tasks, k={k}, T={temperature}, max_new={mnt}")
    samp = generate(model, tok, prompts, k, temperature, batch_size, mnt, seed=seed)
    greedy = generate(model, tok, prompts, 1, 0.0, batch_size, mnt, seed=seed, progress=False)
    extra = {}
    if heldout_k and heldout_k > k:
        hi = [i for i, t in enumerate(task_list) if t["split"] == "heldout"]
        if hi:
            log(f"extra held-out samples: {len(hi)} tasks x {heldout_k - k}")
            ex = generate(model, tok, [prompts[i] for i in hi], heldout_k - k, temperature, batch_size, mnt, seed=seed + 7919)
            extra = dict(zip(hi, ex))
    recs = []
    for i, t in enumerate(task_list):
        r = dict(t)
        r["samples"] = [score(x, t["n"], t["f"]) for x in samp[i]]
        r["greedy"] = score(greedy[i][0], t["n"], t["f"])
        if i in extra:
            r["extra"] = [score(x, t["n"], t["f"]) for x in extra[i]]
        recs.append(r)
    met = metrics(recs, k)
    if extra:
        met["heldout"] = {**met["heldout"], **_metrics([r for r in recs if "extra" in r], k)}
    log(f"evaluate done in {time.time() - t0:.0f}s")
    log_mem("after evaluate")
    return {"k": k, "temperature": temperature, "max_new_tokens": mnt, "eval_seconds": time.time() - t0,
            "metrics": met, "tasks": recs}


def summary(tag, ev):
    m = ev["metrics"]
    k = ev["k"]
    parts = []
    for s in ("train", "heldout"):
        x = m[s]
        if not x.get("n_tasks"):
            continue
        extra = "".join(f" {key} {v:.3f}" for key, v in x.items() if key.startswith("solved@") and key != f"solved@{k}")
        parts.append(f"{s}(n={x['n_tasks']}): valid {x['valid_rate']:.3f} pass {x['mean_pass']:.3f} "
                     f"solved@{k} {x[f'solved@{k}']:.3f} greedy {x['greedy_pass']:.3f}{extra}")
    return f"[{tag}] " + " | ".join(parts)


# ===================================================================== pools
def _ex_key(n, f):
    return f"{n}:{f}"


def update_pool(arm, pool, round_json, train_set):
    """pool: {"n:f": [{"text":..., "size":[...]}...]}; returns new pool built from the round's TRAIN-prompt samples."""
    if arm == "unverified":
        new = {}
        for r in round_json["tasks"]:
            if r["split"] != "train":
                continue
            new[_ex_key(r["n"], r["f"])] = [{"text": s["text"], "size": None} for s in r["samples"][:POOL_CAP]]
        return new
    pool = {key: list(v) for key, v in pool.items()}
    for r in round_json["tasks"]:
        if r["split"] != "train":
            continue
        n = r["n"]
        for s in r["samples"]:
            if not s["valid"]:
                continue
            if arm == "pass":
                if not s["passed"]:
                    continue
                g = r["f"]
            elif arm == "hindsight":
                g = s["tt"]
                if (n, g) not in train_set:          # held-out or few-shot target: never trained on
                    continue
            else:
                raise ValueError(arm)
            net = tasks.parse(s["text"], n)
            pool.setdefault(_ex_key(n, g), []).append({"text": tasks.render(net), "size": list(tasks.size(net))})
    out = {}
    for key, lst in pool.items():
        seen, uniq = set(), []
        for e in sorted(lst, key=lambda e: (tuple(e["size"]), e["text"])):
            if e["text"] not in seen:
                seen.add(e["text"]); uniq.append(e)
        out[key] = uniq[:POOL_CAP]
    return out


def pool_examples(pool):
    ex = []
    for key in sorted(pool, key=lambda s: tuple(map(int, s.split(":")))):
        n, f = map(int, key.split(":"))
        for e in pool[key]:
            ex.append((tasks.prompt(f, n), e["text"]))
    return ex


# ===================================================================== stages
def run_config(a):
    """Settings recorded under "config" in every round JSON."""
    import transformers
    return {"device": DEVICE, "gen_dtype": dtype_name(GEN_DTYPE), "train_amp": dtype_name(TRAIN_AMP),
            "train_attn": TRAIN_ATTN, "rank": a.rank, "alpha": 2 * a.rank, "lr": a.lr, "epochs": a.epochs,
            "max_steps": a.max_steps, "steps": a.steps, "pool_cap": a.pool_cap, "train_batch": a.train_batch,
            "micro_batch": MICRO_BATCH or a.train_batch, "k": a.k, "heldout_k": a.heldout_k, "rounds": a.rounds,
            "temperature": a.temperature, "batch_size": a.batch_size, "limit": a.limit, "engine": ENGINE,
            "out_dir": a.out_dir, "seed": a.seed, "mem_gb": MEM_GB, "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None, "argv": sys.argv[1:]}


def train_round(a, ex, steps, seed, adapter_path):
    """Fine-tune on ex, saving the adapter before merging; or, if a matching adapter is already on disk,
    skip training and rebuild the merged model from it. Returns (model, info for the round JSON)."""
    ad = None
    if os.path.exists(adapter_path):
        try:
            ad = torch.load(adapter_path, map_location="cpu", weights_only=True)
        except Exception as e:
            log(f"cannot load adapter {adapter_path} ({type(e).__name__}: {str(e)[:200]}); retraining")
    if ad is not None:
        want = {"examples_sha": examples_sha(ex), "steps": steps, "r": a.rank, "alpha": 2 * a.rank, "lr": a.lr, "seed": seed}
        bad = {key: (ad.get(key), v) for key, v in want.items() if ad.get(key) != v}
        if not bad:
            log(f"adapter found, skipping training ({ad['steps']} steps were trained): {adapter_path}")
            model = model_from_adapter(ad)
            return model, {"train_steps": ad["steps"], "loss_log": ad["loss_log"],
                           "finetune_seconds": ad["finetune_seconds"], "resumed_from_adapter": True}
        log(f"adapter {adapter_path} does not match this round (saved vs wanted: {bad}); retraining")
    losses = []
    t0 = time.time()
    model = finetune(ex, steps=steps, lr=a.lr, batch_size=a.train_batch, r=a.rank, alpha=2 * a.rank, seed=seed,
                     loss_log=losses, save_path=adapter_path, save_info={"config": run_config(a)})
    return model, {"train_steps": steps, "loss_log": losses, "finetune_seconds": time.time() - t0}


def stage_pilot(a):
    path = os.path.join(a.out_dir, "base", "round0.json")
    if os.path.exists(path):
        log("exists, skipping:", path); print(summary("pilot", json.load(open(path))), flush=True); return
    tr, ho = task_lists(a.limit)
    model = load_model(GEN_DTYPE)
    ev = evaluate(model, tr + ho, a.k, a.temperature, a.batch_size, a.seed, heldout_k=a.heldout_k)
    ev.update({"stage": "pilot", "arm": "base", "round": 0, "pool_size": 0, "pool_functions": 0, "config": run_config(a)})
    dump(ev, path)
    del model; free()
    print(summary("pilot", ev), flush=True)


def stage_teacher(a):
    path = os.path.join(a.out_dir, "teacher", "round1.json")
    if os.path.exists(path):
        log("exists, skipping:", path); print(summary("teacher", json.load(open(path))), flush=True); return
    tr, ho = task_lists(a.limit)
    ex = [(tasks.prompt(t["f"], t["n"]), SOL[str(t["n"])][str(t["f"])]["answer"]) for t in tr]
    steps = a.steps or default_steps(len(ex), a.train_batch)
    model, info = train_round(a, ex, steps, a.seed, os.path.join(a.out_dir, "teacher", "round1.adapter.pt"))
    ev = evaluate(model, tr + ho, a.k, a.temperature, a.batch_size, a.seed)
    ev.update({"stage": "teacher", "arm": "teacher", "round": 1, "pool_size": len(ex), "pool_functions": len(ex),
               **info, "config": run_config(a)})
    dump(ev, path)
    del model; free()
    print(summary("teacher", ev), flush=True)


def stage_loop(a):
    base0 = os.path.join(a.out_dir, "base", "round0.json")
    if not os.path.exists(base0):
        sys.exit(f"missing {base0}: run the pilot first")
    tr, ho = task_lists(a.limit)
    train_set = {(t["n"], t["f"]) for t in load_split() if t["split"] == "train"}
    pool = {}
    for r in range(1, a.rounds + 1):
        path = os.path.join(a.out_dir, a.arm, f"round{r}.json")
        if os.path.exists(path):
            prev = json.load(open(path))
            if prev.get("empty_pool"):
                log(f"{path}: empty pool, arm stopped"); return
            pool = prev["pool"]
            log("exists, skipping:", path); print(summary(f"{a.arm} r{r}", prev), flush=True)
            continue
        src = base0 if r == 1 else os.path.join(a.out_dir, a.arm, f"round{r - 1}.json")
        last = json.load(open(src))
        pool = update_pool(a.arm, pool, last, train_set)
        ex = pool_examples(pool)
        nfun = sum(1 for v in pool.values() if v)
        log(f"[{a.arm} r{r}] pool {len(ex)} examples over {nfun} functions (from {src})")
        if not ex:
            dump({"arm": a.arm, "round": r, "empty_pool": True, "pool_size": 0, "pool_functions": 0,
                  "config": run_config(a)}, path)
            print(f"[{a.arm} r{r}] empty pool, stopping arm", flush=True)
            return
        steps = a.steps or default_steps(len(ex), a.train_batch)
        model, info = train_round(a, ex, steps, a.seed + r, os.path.join(a.out_dir, a.arm, f"round{r}.adapter.pt"))
        ev = evaluate(model, tr + ho, a.k, a.temperature, a.batch_size, a.seed + 1000 * r)
        ev.update({"stage": "loop", "arm": a.arm, "round": r, "pool_size": len(ex), "pool_functions": nfun,
                   "pool": pool, **info, "config": run_config(a)})
        dump(ev, path)
        del model; free()
        print(summary(f"{a.arm} r{r}", ev) + f" | pool {len(ex)} ({nfun} fns)", flush=True)


# ===================================================================== checks and benchmark
def stage_split(a):
    T = load_split()
    print(json.dumps(split_stats(T), indent=1))


def stage_check(a):
    T = load_split()
    print(json.dumps(split_stats(T)))
    tr = {(t["n"], t["f"]) for t in T if t["split"] == "train"}
    ho = {(t["n"], t["f"]) for t in T if t["split"] == "heldout"}
    assert len(T) == 14 + 254 and not (tr & ho) and not ((tr | ho) & FEWSHOT_SET)
    assert (3, 150) in ho and (3, 105) in tr
    pc_tr = {tasks.p_class(f, 3) for n, f in tr if n == 3}
    pc_ho = {tasks.p_class(f, 3) for n, f in ho if n == 3}
    assert not (pc_tr & pc_ho), "shared p_class"
    for t in T:
        n, f = t["n"], t["f"]
        ans = SOL[str(n)][str(f)]["answer"]
        p = tasks.prompt(f, n)
        assert p.endswith(tasks.task_text(f, n))
        net = tasks.parse(ans + "\n\nTask: junk", n)
        assert net is not None and tasks.run(net, n) == f and tasks.render(net) == ans
    print(f"split ok: {len(tr)} train, {len(ho)} held-out, p-classes train/held-out {len(pc_tr)}/{len(pc_ho)}, none shared; "
          f"all {len(T)} teacher answers round-trip through prompt/parse/run", flush=True)

    tok = get_tok()
    # --- LoRA checks (fp32)
    model = load_model(torch.float32)
    prompts = [tasks.prompt(f, 3) + SOL["3"][str(f)]["answer"] for f in (3, 150, 77, 200)]
    enc = tok(prompts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        base = model(**enc, logits_to_keep=96).logits
        add_lora(model, 16, 32, 0)
        wrapped0 = model(**enc, logits_to_keep=96).logits
        for mod in model.modules():
            if isinstance(mod, LoRALinear):
                mod.B.normal_(0, 0.02)
        wrapped = model(**enc, logits_to_keep=96).logits
        merge(model)
        merged = model(**enc, logits_to_keep=96).logits
        merged16 = model.to(GEN_DTYPE)(**enc, logits_to_keep=96).logits.float()
    gname = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}[GEN_DTYPE]
    print(f"LoRA B=0 vs base: max|dlogit| {(wrapped0 - base).abs().max().item():.3e}", flush=True)
    print(f"LoRA random B: wrapped vs base {(wrapped - base).abs().max().item():.3e} (adapter has effect); "
          f"wrapped vs merged fp32 {(wrapped - merged).abs().max().item():.3e}; merged {gname} vs fp32 "
          f"{(merged16 - merged).abs().max().item():.3e} (logit scale {merged.abs().max().item():.1f})", flush=True)
    del model, base, wrapped0, wrapped, merged, merged16; free()

    # --- shared-prefix training loss equals full-sequence loss (fp32, no autocast; attention as in training).
    # Run on the first CHECK_LAYERS layers: the fp32 full-sequence reference needs ~9 GB of activations with
    # all 24 layers, which does not fit the 8 GB cap on the Mac; the equivalence does not depend on depth.
    global TRAIN_AMP
    amp = TRAIN_AMP
    TRAIN_AMP = None
    CHECK_LAYERS = 8
    model = load_model(torch.float32, attn=TRAIN_ATTN)
    model.model.layers = model.model.layers[:CHECK_LAYERS]
    model.config.num_hidden_layers = CHECK_LAYERS
    if getattr(model.config, "layer_types", None):
        model.config.layer_types = model.config.layer_types[:CHECK_LAYERS]
    free()
    print(f"gradient checks on the first {CHECK_LAYERS} layers, attention {TRAIN_ATTN}", flush=True)
    add_lora(model, 16, 32, 0)
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            with torch.no_grad():
                mod.B.normal_(0, 0.02)
    ex = tokenize_examples([(tasks.prompt(f, n), SOL[str(n)][str(f)]["answer"]) for n, f in ((2, 3), (3, 77), (3, 150), (2, 9))])
    l1 = _shared_prefix_loss(model, ex, prefix_ids()); l1.backward()
    g1 = torch.cat([mod.A.grad.flatten() for mod in model.modules() if isinstance(mod, LoRALinear)][:8])
    for p in model.parameters():
        p.grad = None
    l2 = _full_loss(model, ex); l2.backward()
    g2 = torch.cat([mod.A.grad.flatten() for mod in model.modules() if isinstance(mod, LoRALinear)][:8])
    allg = lambda: torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    G2 = allg()
    for p in model.parameters():
        p.grad = None
    l3 = train_step_backward(model, ex, prefix_ids(), micro_batch=2, prefix_grad=True)
    G3 = allg()
    for p in model.parameters():
        p.grad = None
    l4 = train_step_backward(model, ex, prefix_ids(), micro_batch=len(ex), prefix_grad=True)
    G4 = allg()
    print(f"train loss shared-prefix {l1.item():.6f} vs full-sequence {l2.item():.6f} vs micro-batched step {l3:.6f} "
          f"vs whole-batch step {l4:.6f}; grad rel diff shared-vs-full {((g1 - g2).norm() / g2.norm()).item():.2e}, "
          f"micro-batched-vs-full (all LoRA params) {((G3 - G2).norm() / G2.norm()).item():.2e}, "
          f"whole-batch-vs-full {((G4 - G2).norm() / G2.norm()).item():.2e}", flush=True)
    TRAIN_AMP = amp
    if TRAIN_AMP is not None:
        for p in model.parameters():
            p.grad = None
        l5 = train_step_backward(model, ex, prefix_ids(), micro_batch=MICRO_BATCH or len(ex), prefix_grad=True)
        G5 = allg()
        print(f"{dtype_name(TRAIN_AMP)} autocast training step (micro-batch {MICRO_BATCH or len(ex)}): loss {l5:.6f} "
              f"vs fp32 {l2.item():.6f}; grad rel diff vs fp32 full-sequence {((G5 - G2).norm() / G2.norm()).item():.2e}",
              flush=True)
    del model; free()

    # --- fast generation engine vs transformers.generate(stop_strings) greedy
    global ENGINE
    model = load_model(GEN_DTYPE)
    ps = [tasks.prompt(f, 3) for f in range(2, 34)] + [tasks.prompt(f, 2) for f in (0, 3, 9, 15)]
    mnt = max_new_tokens_default()
    ENGINE = "hf"; ref = generate(model, tok, ps, 1, 0.0, 64, mnt, progress=False)
    ENGINE = "fast"; fast = generate(model, tok, ps, 1, 0.0, 64, mnt, progress=False)
    same = sum(a == b for a, b in zip(ref, fast))
    print(f"greedy fast engine vs transformers.generate: {same}/{len(ps)} identical answers", flush=True)
    for a_, b_ in zip(ref, fast):
        if a_ != b_:
            print("  differs:", repr(a_[0]), "|", repr(b_[0]))
    del model; free()


def stage_bench(a):
    global ENGINE
    tok = get_tok()
    mnt = max_new_tokens_default()
    model = load_model(GEN_DTYPE)
    ps = [tasks.prompt(f, 3) for f in range(2, 256)]
    generate(model, tok, ps[:8], 1, 0.8, 8, mnt, progress=False)   # warm-up
    res = {}
    engines = [e for e in a.engines.split(",") if e]
    for eng in engines:
        ENGINE = eng
        for B in [int(x) for x in a.bench_batches.split(",")]:
            n_prompts = max(1, (B * a.bench_reps) // 16)
            t0 = time.time()
            out = generate(model, tok, ps[:n_prompts], 16, 0.8, B, mnt, progress=False)
            sync(); dt = time.time() - t0
            N = sum(len(x) for x in out)
            res[(eng, B)] = N / dt
            log(f"gen engine={eng} batch={B}: {N} answers in {dt:.1f}s = {N / dt:.2f} ans/s")
    del model; free()
    ex = [(tasks.prompt(t["f"], t["n"]), SOL[str(t["n"])][str(t["f"])]["answer"]) for t in task_lists()[0]]
    t0 = time.time()
    m = finetune(ex, steps=a.bench_steps, batch_size=a.train_batch, r=a.rank, alpha=2 * a.rank, log_every=5)
    dt = time.time() - t0
    del m; free()
    log(f"finetune {a.bench_steps} steps incl. load+merge: {dt:.1f}s")
    log_mem("bench end")
    print(json.dumps({f"{e}@{b}": round(v, 2) for (e, b), v in res.items()}), flush=True)


def configure(a):
    """Device and the device-dependent defaults. mps keeps the original Mac settings."""
    global DEVICE, GEN_DTYPE, TRAIN_AMP, TRAIN_ATTN, ENGINE, MICRO_BATCH, EPOCHS, MAX_STEPS, POOL_CAP
    DEVICE = pick_device(a.device)
    GEN_DTYPE = DTYPES[a.gen_dtype] if a.gen_dtype != "auto" else torch.float32 if DEVICE == "cpu" else torch.float16
    TRAIN_AMP = (torch.bfloat16 if (a.train_amp == "bf16" or (a.train_amp == "auto" and DEVICE == "cuda")) else None)
    TRAIN_ATTN = "eager" if DEVICE == "mps" else "sdpa"
    if a.micro_batch is None:
        a.micro_batch = 4 if DEVICE == "mps" else 0               # 0 = whole training batch in one pass
    if a.mem_gb is None:
        a.mem_gb = 8.0 if DEVICE == "mps" else 0.0
    if a.batch_size is None:
        a.batch_size = {"cuda": 512, "mps": 256}.get(DEVICE, 64)
    if a.bench_batches is None:
        a.bench_batches = "256,512,1024" if DEVICE == "cuda" else "64,128,256"
    ENGINE = a.engine
    MICRO_BATCH = a.micro_batch
    EPOCHS, MAX_STEPS, POOL_CAP = a.epochs, a.max_steps, a.pool_cap
    log(f"device {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else "") +
        f", generation {dtype_name(GEN_DTYPE)} batch {a.batch_size}, training attn {TRAIN_ATTN} "
        f"amp {dtype_name(TRAIN_AMP)} micro-batch {MICRO_BATCH or 'whole batch'}, LoRA r {a.rank}")
    set_memory_limit(a.mem_gb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["split", "check", "bench", "pilot", "teacher", "loop"])
    ap.add_argument("--arm", choices=["pass", "hindsight", "unverified"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--heldout-k", type=int, default=64, help="pilot only: total samples for held-out tasks")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--gen-dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto",
                    help="dtype of the merged model for generation (auto: fp16 on cuda/mps, fp32 on cpu)")
    ap.add_argument("--train-amp", choices=["auto", "bf16", "none"], default="auto",
                    help="autocast for training forwards (auto: bf16 on cuda, none elsewhere)")
    ap.add_argument("--batch-size", type=int, default=None, help="generation batch size (default: cuda 512, mps 256, cpu 64)")
    ap.add_argument("--train-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank r (alpha = 2r)")
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="gradient-accumulation chunk of the training batch (default: mps 4, else 0 = whole batch)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=10, help="training passes over the pool when --steps is not given")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--pool-cap", type=int, default=4, help="answers kept per function in a training pool")
    ap.add_argument("--mem-gb", type=float, default=None,
                    help="hard memory limit for this process (0 = no limit; default: 8 on mps, none on cuda/cpu)")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", choices=["fast", "hf"], default="fast")
    ap.add_argument("--engines", default="fast,hf", help="bench only")
    ap.add_argument("--bench-batches", default=None, help="default: cuda 256,512,1024, else 64,128,256")
    ap.add_argument("--bench-reps", type=int, default=2, help="bench: batches per setting")
    ap.add_argument("--bench-steps", type=int, default=10)
    a = ap.parse_args()
    configure(a)
    if not os.path.isabs(a.out_dir):
        a.out_dir = os.path.join(HERE, a.out_dir)
    if a.cmd == "loop" and not a.arm:
        ap.error("loop needs --arm")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    {"split": stage_split, "check": stage_check, "bench": stage_bench, "pilot": stage_pilot,
     "teacher": stage_teacher, "loop": stage_loop}[a.cmd](a)


if __name__ == "__main__":
    main()
