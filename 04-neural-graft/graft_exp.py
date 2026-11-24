"""04 — Neural Organ Grafting into Qwen2.5-1.5B-Instruct.

Tests whether transplanting a frozen, exact sparse 2-layer Transformer organ (from
03-transformer-writer) into a frozen real LLM via a tiny zero-init linear socket
(W_in: d_llm -> 16, W_out: 4 -> d_llm, ~30K params = 0.002% of Qwen) unlocks OOD
length generalization (L=4..8 -> L=16, 32, 64) where standard LoRA AND an identically
sized SGD-trained graft collapse — while measuring collateral drift on English text NLL.

Arms:
  1. base           : frozen Qwen2.5-1.5B-Instruct (no training)
  2. lora           : Qwen2.5-1.5B-Instruct + LoRA (r=16) trained on L=4..8
  3. sgd_graft      : frozen Qwen + trainable socket + trainable random TinyTransformer organ
  4. compiled_graft : frozen Qwen + trainable socket + FROZEN compiled TinyTransformer organ
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse target_tf, tasks, and compiled circuits from 03-transformer-writer
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "03-transformer-writer"))
sys.path.insert(0, "/content/tw")

from target_tf import D_HEAD, D_MODEL, N_HEADS, N_LAYERS, N_OUT, TinyTransformer, parse_sparse_weights
from tasks import TASKS_BY_NAME, Task
from verify_realizability import REFERENCE_PROOF_CIRCUITS, WORKED_EXAMPLES
from writer_loop import attach_lora, unwrap_and_reset_lora


# Actual 8-weight inverse-key veto circuit discovered by Qwen2.5-Coder-3B in 03-transformer-writer!
ALL_0_DISCOVERED = """WEIGHTS:
L0.H0.Q[6,0] = 1
L0.H0.K[1,0] = 40
L0.H0.K[2,0] = 40
L0.H0.K[3,0] = 40
L0.H0.V[0,0] = 1
L0.H0.O[0,7] = 1
OUT.W[7,1] = 200
OUT.b[0] = 0.5"""

BIGRAM_1_2_CIRCUIT = """WEIGHTS:
L0.H0.alibi = -10
L0.H0.V[1,0] = 1
L0.H0.O[0,7] = 1
L0.M.W1[7,0] = 1
L0.M.W1[2,0] = 1
L0.M.b1[0] = -1.5
L0.M.W2[0,11] = 2
L1.H0.Q[6,0] = 1
L1.H0.K[11,0] = 40
L1.H0.V[11,0] = 1
L1.H0.O[0,12] = 1
OUT.W[12,1] = 2
OUT.b[0] = 1"""

ORGAN_SPECS: Dict[str, str] = {
    "all_0": ALL_0_DISCOVERED,
    "bigram_1_2": BIGRAM_1_2_CIRCUIT,
    "sorted_asc": REFERENCE_PROOF_CIRCUITS["sorted_asc"],
    "dyck1": REFERENCE_PROOF_CIRCUITS["dyck1"],
}

PROMPT_PREFIX: Dict[str, str] = {
    "all_0": "Check all zero. Input: ",
    "bigram_1_2": "Check pair one-two. Input: ",
    "sorted_asc": "Check non-decreasing. Input: ",
    "dyck1": "Check balanced parens. Input: ",
}

# Short held-out English prose snippets to measure collateral NLL drift on normal language
ENGLISH_PROBES = [
    "In 1936 Alan Turing defined a mathematical model of computation consisting of an infinite tape divided into cells.",
    "The northern lights occur when charged particles from the solar wind collide with gases in Earth's upper atmosphere.",
    "After a long winter, the mountain passes finally cleared in late May, allowing supply trucks to reach the valley.",
    "Modern compilers translate high-level source code into machine instructions through several optimization passes.",
    "Coffee plants grow best in tropical highlands with steady rainfall, volcanic soil, and moderate daytime temperatures.",
    "Although the experiment produced a clear signal in small trials, the effect vanished once a proper control was added.",
]


def organ_forward_from_states(organ: TinyTransformer, x_in: torch.Tensor) -> torch.Tensor:
    """Run TinyTransformer starting from continuous projected states x_in [B, T, 16] -> [B, 4]."""
    B, T, _ = x_in.shape
    device = x_in.device
    x = x_in
    pos = torch.arange(T, device=device)
    dist = (pos[:, None] - pos[None, :]).to(x.dtype)
    mask = pos[None, :] < pos[:, None]
    mask[0, 0] = True
    attn_mask = torch.where(mask, torch.zeros_like(dist), torch.full_like(dist, float("-inf")))
    scale = 1.0 / math.sqrt(D_HEAD)
    for l in range(N_LAYERS):
        attn_delta = torch.zeros_like(x)
        for h in range(N_HEADS):
            q = x @ organ.Q[l, h]
            k = x @ organ.K[l, h]
            v = x @ organ.V[l, h]
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            scores = scores + organ.alibi[l, h] * dist.unsqueeze(0) + attn_mask.unsqueeze(0)
            probs = F.softmax(scores, dim=-1)
            attn_delta = attn_delta + torch.matmul(probs, v) @ organ.O[l, h]
        x = x + attn_delta
        x = x + F.relu(x @ organ.W1[l] + organ.b1[l]) @ organ.W2[l]
    return x[:, -1, :] @ organ.OUT_W + organ.OUT_b


def calibrate_linear_socket(base_llm: nn.Module, tokenizer, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Closed-form ridge regression mapping frozen LLM token embeddings -> 16-dim organ port.
    Uses ZERO task data: maps '0'..'3' -> ch 0..3, '=' -> ch 5, and 2048 background vocab tokens -> ch 4 (BOS/other),
    with ch 6 = 1.0 everywhere.
    """
    E = base_llm.get_input_embeddings().weight.detach().float()  # [V, d_llm]
    V, d_llm = E.shape
    tok_digits = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(4)]
    tok_eq = tokenizer.encode("=", add_special_tokens=False)[0]
    special = set(tok_digits + [tok_eq])

    # Sample 2048 background tokens (plus prompt tokens) as "other / BOS"
    rng = random.Random(0)
    bg = [i for i in rng.sample(range(min(V, 50000)), 2048) if i not in special]
    # Upweight the 5 port tokens so ridge regression fits them to <1e-4 error
    idxs = tok_digits * 64 + [tok_eq] * 64 + bg
    X = E[idxs]  # [N, d_llm]
    Y = torch.zeros(len(idxs), D_MODEL, device=E.device, dtype=torch.float32)
    pos = 0
    for _ in range(64):
        for d in range(4):
            Y[pos, d] = 1.0
            pos += 1
    for _ in range(64):
        Y[pos, 5] = 1.0
        pos += 1
    Y[pos:, 4] = 1.0
    Y[:, 6] = 1.0

    X_aug = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
    # Ridge regression: (X^T X + lam I) W = X^T Y
    XtX = X_aug.T @ X_aug
    XtX.diagonal().add_(1e-2)
    XtY = X_aug.T @ Y
    sol = torch.linalg.solve(XtX, XtY)  # [d_llm+1, 16]
    W = sol[:-1].T.contiguous().to(device)  # [16, d_llm]
    b = sol[-1].contiguous().to(device)     # [16]
    return W, b


class GraftedLLM(nn.Module):
    """Wraps a frozen HF CausalLM with a grafted TinyTransformer organ + linear residual socket.

    At initialization, W_out = 0 so the grafted model is mathematically identical to base_llm
    on every input (exact function preservation).
    """

    def __init__(
        self,
        base_llm: nn.Module,
        organ: TinyTransformer,
        freeze_organ: bool = True,
        calibrated_in: Tuple[torch.Tensor, torch.Tensor] | None = None,
        socket_deadzone: float = 0.0,
    ):
        super().__init__()
        self.base_llm = base_llm
        for p in self.base_llm.parameters():
            p.requires_grad_(False)

        self.organ = organ
        for p in self.organ.parameters():
            p.requires_grad_(not freeze_organ)

        self.socket_deadzone = float(socket_deadzone)
        d_llm = base_llm.config.hidden_size
        self.W_in = nn.Linear(d_llm, D_MODEL, bias=True, dtype=torch.float32)
        self.W_out = nn.Linear(N_OUT, d_llm, bias=False, dtype=torch.float32)

        if calibrated_in is not None:
            W_cal, b_cal = calibrated_in
            with torch.no_grad():
                self.W_in.weight.copy_(W_cal)
                self.W_in.bias.copy_(b_cal)
            for p in self.W_in.parameters():
                p.requires_grad_(False)
        else:
            nn.init.normal_(self.W_in.weight, mean=0.0, std=0.25)
            nn.init.zeros_(self.W_in.bias)
            with torch.no_grad():
                self.W_in.bias[6] = 1.0

        nn.init.zeros_(self.W_out.weight)  # exact identity at step 0!

    def _project_in(self, h0: torch.Tensor) -> torch.Tensor:
        z_in = self.W_in(h0)  # [B, T, 16]
        if self.socket_deadzone > 0.0:
            dz = self.socket_deadzone
            z_sym = (F.relu(z_in[:, :, :6] - dz) / (1.0 - dz)).clamp(0.0, 1.0)
            z_in = torch.cat([z_sym, z_in[:, :, 6:]], dim=-1)
        return z_in

    def forward_last_logits(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (lm_logits_at_last_pos [B, V], organ_logits [B, 4])."""
        with torch.no_grad():
            embed_layer = self.base_llm.get_input_embeddings()
            h0 = embed_layer(input_ids).float()  # [B, T, d_llm]
            out = self.base_llm.model(input_ids=input_ids, use_cache=False)
            h_last = out.last_hidden_state[:, -1, :].float()  # [B, d_llm]

        z_in = self._project_in(h0)  # [B, T, 16]
        organ_logits = organ_forward_from_states(self.organ, z_in)  # [B, 4]
        # Differentiable neural gate from organ's own EOS port channel ch5 (0 on normal text, 1 on '=')
        gate = F.relu(z_in[:, -1, 5:6]).clamp(0.0, 1.0)
        h_mod = h_last + gate * self.W_out(torch.tanh(organ_logits / 10.0) * 10.0)
        lm_head = self.base_llm.get_output_embeddings()
        logits = lm_head(h_mod.to(lm_head.weight.dtype)).float()
        return logits, organ_logits

    @torch.no_grad()
    def english_nll(self, tokenizer, device: torch.device) -> float:
        """Compute next-token NLL on ENGLISH_PROBES (adding gated graft contribution at every position)."""
        total_loss = 0.0
        total_toks = 0
        embed_layer = self.base_llm.get_input_embeddings()
        lm_head = self.base_llm.get_output_embeddings()
        for text in ENGLISH_PROBES:
            ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            h0 = embed_layer(ids).float()
            h_all = self.base_llm.model(input_ids=ids, use_cache=False).last_hidden_state.float()
            z_in = self._project_in(h0)
            B, T, _ = z_in.shape
            x = z_in
            pos = torch.arange(T, device=device)
            dist = (pos[:, None] - pos[None, :]).to(x.dtype)
            mask = pos[None, :] < pos[:, None]
            mask[0, 0] = True
            attn_mask = torch.where(mask, torch.zeros_like(dist), torch.full_like(dist, float("-inf")))
            scale = 1.0 / math.sqrt(D_HEAD)
            for l in range(N_LAYERS):
                ad = torch.zeros_like(x)
                for h in range(N_HEADS):
                    scores = torch.matmul(x @ self.organ.Q[l, h], (x @ self.organ.K[l, h]).transpose(-2, -1)) * scale
                    scores = scores + self.organ.alibi[l, h] * dist.unsqueeze(0) + attn_mask.unsqueeze(0)
                    ad = ad + torch.matmul(F.softmax(scores, dim=-1), x @ self.organ.V[l, h]) @ self.organ.O[l, h]
                x = x + ad
                x = x + F.relu(x @ self.organ.W1[l] + self.organ.b1[l]) @ self.organ.W2[l]
            org_all = x @ self.organ.OUT_W + self.organ.OUT_b
            gate_all = F.relu(z_in[:, :, 5:6]).clamp(0.0, 1.0)
            h_mod = h_all + gate_all * self.W_out(torch.tanh(org_all / 10.0) * 10.0)
            logits = lm_head(h_mod.to(lm_head.weight.dtype)).float()
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum")
            total_loss += float(loss.item())
            total_toks += int(ids.shape[1] - 1)
        return total_loss / max(1, total_toks)


@torch.no_grad()
def plain_llm_english_nll(llm: nn.Module, tokenizer, device: torch.device) -> float:
    llm.eval()
    total_loss = 0.0
    total_toks = 0
    for text in ENGLISH_PROBES:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        out = llm(input_ids=ids, use_cache=False)
        logits = out.logits.float()
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum")
        total_loss += float(loss.item())
        total_toks += int(ids.shape[1] - 1)
    return total_loss / max(1, total_toks)


def make_prompt_ids(tokenizer, task: Task, seq: List[int]) -> List[int]:
    txt = PROMPT_PREFIX[task.name] + "".join(map(str, seq)) + "="
    return tokenizer.encode(txt, add_special_tokens=False)


def build_eval_batches(tokenizer, task: Task, device: torch.device) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Fixed deterministic eval splits at ID (L=4..8) and OOD (L=16, 32, 64)."""
    h = sum((i + 1) * ord(c) for i, c in enumerate(task.name))
    splits = {
        "id": ([4, 6, 8], 40, 1000 + h),
        "L16": ([16], 128, 2000 + h),
        "L32": ([32], 128, 3000 + h),
        "L64": ([64], 128, 4000 + h),
    }
    out: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
    for name, (lens, n_per, seed) in splits.items():
        rng = random.Random(seed)
        by_len: Dict[int, Tuple[List[List[int]], List[int]]] = {}
        for L in lens:
            for _ in range(n_per):
                seq, y = task.gen(rng, L)
                ids = make_prompt_ids(tokenizer, task, seq)
                by_len.setdefault(len(ids), ([], []))
                by_len[len(ids)][0].append(ids)
                by_len[len(ids)][1].append(y)
        out[name] = [
            (torch.tensor(xs, dtype=torch.long, device=device), torch.tensor(ys, dtype=torch.long, device=device))
            for xs, ys in by_len.values()
        ]
    return out


@torch.no_grad()
def eval_binary_accuracy(forward_fn, batches: List[Tuple[torch.Tensor, torch.Tensor]], tok0: int, tok1: int, batch_size: int = 32) -> float:
    correct = 0
    total = 0
    for ids, ys in batches:
        for s in range(0, ids.shape[0], batch_size):
            sub_ids = ids[s : s + batch_size]
            sub_ys = ys[s : s + batch_size]
            logits = forward_fn(sub_ids)  # [b, V]
            pair = torch.stack([logits[:, tok0], logits[:, tok1]], dim=-1)
            preds = pair.argmax(dim=-1)
            correct += int((preds == sub_ys).sum().item())
            total += int(sub_ys.numel())
    return correct / max(1, total)


def train_graft_arm(
    base_llm: nn.Module,
    tokenizer,
    task: Task,
    freeze_organ: bool,
    compiled_spec: str | None,
    calibrated_in: Tuple[torch.Tensor, torch.Tensor] | None,
    eval_splits,
    tok0: int,
    tok1: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    socket_deadzone: float = 0.0,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    if compiled_spec is not None:
        pr = parse_sparse_weights(compiled_spec, device=device)
        assert pr.model is not None
        organ = pr.model
    else:
        organ = TinyTransformer(learnable_embed=False).to(device)
        organ.init_random_for_sgd(seed=seed)

    graft = GraftedLLM(
        base_llm, organ, freeze_organ=freeze_organ, calibrated_in=calibrated_in, socket_deadzone=socket_deadzone,
    ).to(device)
    trainable = [p for p in graft.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)

    # Check exact step-0 function preservation on English NLL!
    nll_step0 = graft.english_nll(tokenizer, device)

    ans_ids = torch.tensor([tok0, tok1], dtype=torch.long, device=device)
    for step in range(1, steps + 1):
        L = rng.choice([4, 5, 6, 7, 8])
        xs, ys = [], []
        for _ in range(batch_size):
            seq, y = task.gen(rng, L)
            xs.append(make_prompt_ids(tokenizer, task, seq))
            ys.append(y)
        ids = torch.tensor(xs, dtype=torch.long, device=device)
        labs = torch.tensor(ys, dtype=torch.long, device=device)
        target_toks = ans_ids[labs]

        logits, _ = graft.forward_last_logits(ids)
        loss = F.cross_entropy(logits, target_toks)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()

    graft.eval()
    res = {
        k: eval_binary_accuracy(lambda x: graft.forward_last_logits(x)[0], b, tok0, tok1)
        for k, b in eval_splits.items()
    }
    res["english_nll_step0"] = round(nll_step0, 4)
    res["english_nll_final"] = round(graft.english_nll(tokenizer, device), 4)
    res["n_trainable_params"] = sum(p.numel() for p in trainable)
    return res


def train_lora_arm(
    base_llm: nn.Module,
    tokenizer,
    task: Task,
    eval_splits,
    tok0: int,
    tok1: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    lora_params = attach_lora(base_llm, r=16, alpha=32.0)
    opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)

    base_llm.train()
    base_llm.gradient_checkpointing_enable()
    base_llm.enable_input_require_grads()
    ans_ids = torch.tensor([tok0, tok1], dtype=torch.long, device=device)
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    micro = min(8, batch_size)
    accum = max(1, batch_size // micro)

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            L = rng.choice([4, 5, 6, 7, 8])
            xs, ys = [], []
            for _ in range(micro):
                seq, y = task.gen(rng, L)
                xs.append(make_prompt_ids(tokenizer, task, seq))
                ys.append(y)
            ids = torch.tensor(xs, dtype=torch.long, device=device)
            labs = torch.tensor(ys, dtype=torch.long, device=device)
            target_toks = ans_ids[labs]

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                out = base_llm(input_ids=ids, use_cache=False)
                logits = out.logits[:, -1, :]
                loss = F.cross_entropy(logits.float(), target_toks) / accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        sched.step()

    base_llm.gradient_checkpointing_disable()
    base_llm.eval()
    res = {
        k: eval_binary_accuracy(lambda x: base_llm(input_ids=x, use_cache=False).logits[:, -1, :].float(), b, tok0, tok1)
        for k, b in eval_splits.items()
    }
    res["english_nll_final"] = round(plain_llm_english_nll(base_llm, tokenizer, device), 4)
    res["n_trainable_params"] = sum(p.numel() for p in lora_params)
    unwrap_and_reset_lora(base_llm, merge=False)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tasks", type=str, default="all_0,bigram_1_2,sorted_asc,dyck1")
    ap.add_argument("--arms", type=str, default="all",
                    help="Comma-separated arms to run, or 'all' (base,lora,sgd_graft,compiled_graft_blind,compiled_graft_calibrated,compiled_graft_deadzone)")
    ap.add_argument("--graft-steps", type=int, default=800)
    ap.add_argument("--lora-steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="runs/graft_results.json")
    args = ap.parse_args()

    all_arm_names = [
        "base", "lora", "sgd_graft", "compiled_graft_blind",
        "compiled_graft_calibrated", "compiled_graft_deadzone",
    ]
    arms_to_run = set(all_arm_names if args.arms.strip() == "all" else [s.strip() for s in args.arms.split(",") if s.strip()])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    print(f"04-neural-graft | device={device} dtype={dtype} model={args.model} arms={sorted(arms_to_run)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        base_llm = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    except TypeError:
        base_llm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    base_llm.eval()

    tok0 = tokenizer.encode("0", add_special_tokens=False)[0]
    tok1 = tokenizer.encode("1", add_special_tokens=False)[0]
    base_nll = plain_llm_english_nll(base_llm, tokenizer, device)
    calibrated_in = calibrate_linear_socket(base_llm, tokenizer, device)
    print(f"Base English probe NLL: {base_nll:.4f} (tok0={tok0}, tok1={tok1}) | calibrated linear socket ready", flush=True)

    task_names = [s.strip() for s in args.tasks.split(",") if s.strip()]
    all_results: Dict[str, Dict[str, Dict]] = {}
    t0 = time.time()

    for idx, tname in enumerate(task_names):
        task = TASKS_BY_NAME[tname]
        splits = build_eval_batches(tokenizer, task, device)
        print(f"\n=== Task: {tname} ===", flush=True)
        task_res: Dict[str, Dict] = {}

        if "base" in arms_to_run:
            r_base = {
                k: eval_binary_accuracy(lambda x: base_llm(input_ids=x, use_cache=False).logits[:, -1, :].float(), b, tok0, tok1)
                for k, b in splits.items()
            }
            r_base["english_nll_final"] = round(base_nll, 4)
            print(f"  base                : ID={r_base['id']*100:5.1f}% L16={r_base['L16']*100:5.1f}% L32={r_base['L32']*100:5.1f}% L64={r_base['L64']*100:5.1f}% | NLL={r_base['english_nll_final']:.4f}", flush=True)
            task_res["base"] = r_base

        if "lora" in arms_to_run:
            r_lora = train_lora_arm(
                base_llm, tokenizer, task, splits, tok0, tok1,
                steps=args.lora_steps, batch_size=args.batch_size, lr=2e-4, seed=args.seed + idx, device=device,
            )
            print(f"  lora (r=16)         : ID={r_lora['id']*100:5.1f}% L16={r_lora['L16']*100:5.1f}% L32={r_lora['L32']*100:5.1f}% L64={r_lora['L64']*100:5.1f}% | NLL={r_lora['english_nll_final']:.4f}", flush=True)
            task_res["lora"] = r_lora

        if "sgd_graft" in arms_to_run:
            r_sgd_g = train_graft_arm(
                base_llm, tokenizer, task, freeze_organ=False, compiled_spec=None, calibrated_in=calibrated_in,
                eval_splits=splits, tok0=tok0, tok1=tok1,
                steps=args.graft_steps, batch_size=args.batch_size, lr=1e-2, seed=args.seed + idx, device=device,
            )
            print(f"  sgd_graft (cal)     : ID={r_sgd_g['id']*100:5.1f}% L16={r_sgd_g['L16']*100:5.1f}% L32={r_sgd_g['L32']*100:5.1f}% L64={r_sgd_g['L64']*100:5.1f}% | NLL={r_sgd_g['english_nll_final']:.4f}", flush=True)
            task_res["sgd_graft"] = r_sgd_g

        if "compiled_graft_blind" in arms_to_run:
            r_comp_blind = train_graft_arm(
                base_llm, tokenizer, task, freeze_organ=True, compiled_spec=ORGAN_SPECS[tname], calibrated_in=None,
                eval_splits=splits, tok0=tok0, tok1=tok1,
                steps=args.graft_steps, batch_size=args.batch_size, lr=1e-2, seed=args.seed + idx, device=device,
            )
            print(f"  compiled_graft_blind: ID={r_comp_blind['id']*100:5.1f}% L16={r_comp_blind['L16']*100:5.1f}% L32={r_comp_blind['L32']*100:5.1f}% L64={r_comp_blind['L64']*100:5.1f}% | NLL={r_comp_blind['english_nll_final']:.4f}", flush=True)
            task_res["compiled_graft_blind"] = r_comp_blind

        if "compiled_graft_calibrated" in arms_to_run:
            r_comp_cal = train_graft_arm(
                base_llm, tokenizer, task, freeze_organ=True, compiled_spec=ORGAN_SPECS[tname], calibrated_in=calibrated_in,
                eval_splits=splits, tok0=tok0, tok1=tok1,
                steps=300, batch_size=args.batch_size, lr=2e-2, seed=args.seed + idx, device=device,
                socket_deadzone=0.0,
            )
            print(f"  compiled_graft_cal  : ID={r_comp_cal['id']*100:5.1f}% L16={r_comp_cal['L16']*100:5.1f}% L32={r_comp_cal['L32']*100:5.1f}% L64={r_comp_cal['L64']*100:5.1f}% | NLL={r_comp_cal['english_nll_final']:.4f}", flush=True)
            task_res["compiled_graft_calibrated"] = r_comp_cal

        if "compiled_graft_deadzone" in arms_to_run:
            r_comp_dz = train_graft_arm(
                base_llm, tokenizer, task, freeze_organ=True, compiled_spec=ORGAN_SPECS[tname], calibrated_in=calibrated_in,
                eval_splits=splits, tok0=tok0, tok1=tok1,
                steps=300, batch_size=args.batch_size, lr=2e-2, seed=args.seed + idx, device=device,
                socket_deadzone=0.3,
            )
            print(f"  compiled_graft_dz   : ID={r_comp_dz['id']*100:5.1f}% L16={r_comp_dz['L16']*100:5.1f}% L32={r_comp_dz['L32']*100:5.1f}% L64={r_comp_dz['L64']*100:5.1f}% | NLL={r_comp_dz['english_nll_final']:.4f}", flush=True)
            task_res["compiled_graft_deadzone"] = r_comp_dz

        all_results[tname] = task_res

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": vars(args),
        "base_english_nll": round(base_nll, 4),
        "elapsed_s": round(time.time() - t0, 1),
        "results": all_results,
    }, indent=2))
    print(f"\nSaved -> {out_path} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
