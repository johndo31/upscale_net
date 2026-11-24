"""Verifier-bootstrapped Transformer Writer loop (Expert Iteration / STaR + Hindsight).

A language model reads an algorithmic sequence specification (+ 6 short L<=6 examples),
writes a brief PLAN and sparse PyTorch weights for TinyTransformer, and gets checked
ONLY on short sequences (L=3..8) by an exact verifier.

Zero human-written solutions beyond the 4 worked examples in the prompt.
Evaluated on:
  1. Training tasks (22): ID (L=3..8) and OOD length generalization (L=16, 32, 64)
  2. Held-out unseen tasks (10): zero-shot compilation at ID (L=3..8) and OOD (L=64)

Arms:
  - hindsight : keep exact-passing circuits for their target task AND relabel any valid
                circuit that solves another TRAINING task (never held-out tasks).
  - pass      : keep exact-passing circuits only for their own target task.
  - unverified: control — keep parsed circuits without checking accuracy.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from target_tf import ParseResult, parse_sparse_weights, serialize_sparse_weights
from tasks import TASKS, TASKS_BY_NAME, Task, evaluate_model, make_eval_suite
from verify_realizability import WORKED_EXAMPLES


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# ---------- Minimal pure-PyTorch LoRA (zero external peft dependency) ----------

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(x.float(), self.A), self.B) * self.scale
        return out + delta.to(out.dtype)

    @torch.no_grad()
    def merge_into_base(self) -> nn.Linear:
        delta = (self.B @ self.A) * self.scale
        self.base.weight.add_(delta.to(self.base.weight.dtype))
        return self.base


def attach_lora(model: nn.Module, r: int = 16, alpha: float = 32.0) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in LORA_TARGETS and isinstance(child, nn.Linear):
                wrapped = LoRALinear(child, r=r, alpha=alpha)
                setattr(module, child_name, wrapped)
                params.extend([wrapped.A, wrapped.B])
    return params


def unwrap_and_reset_lora(model: nn.Module, merge: bool = False) -> None:
    for _, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                base = child.merge_into_base() if merge else child.base
                setattr(module, child_name, base)


def get_lora_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.named_parameters() if k.endswith(".A") or k.endswith(".B")}


def load_lora_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for k, p in model.named_parameters():
            if k in state:
                p.copy_(state[k].to(p.device, p.dtype))


# ---------- Prompt construction ----------

SYSTEM_DOC = """You compile variable-length sequence algorithms into exact sparse PyTorch weights for a 2-layer Transformer (no LayerNorm).

ARCHITECTURE:
- Vocab V=6: data tokens 0, 1, 2, 3; BOS=4 at pos 0; EOS=5 at final pos L+1.
- Input embedding E (fixed, d_model=16):
    ch 0..3 : one-hot data token (0..3)
    ch 4    : is_BOS
    ch 5    : is_EOS
    ch 6    : constant 1.0 (bias channel)
    ch 7..15: 9 zero-initialized working channels
- 2 layers l in {0, 1}. Each layer has:
  1) 2 strictly-causal attention heads h in {0, 1}, d_head=4, per-head ALiBi slope:
       scores[i,j] = (Q_i . K_j) / 2 + alibi * (i - j),  mask: j < i (for i>0), j==0 (for i==0)
       x = x + sum_h softmax(scores_h) @ V_h @ O_h
     Tips:
     * alibi = -10 attends to j = i - 1 (previous token; at EOS this is x_L).
     * alibi = +10 with Q[6,0]=1, K[4,0]=-200 masks BOS and attends to j = 1 (first token x_1).
     * alibi = 0 with Q=K=0 computes the uniform average over j < i.
     * alibi = 0 with Q[6,0]=1, K[ch,0]=40 attends strongly to any past position with ch > 0.
  2) ReLU MLP (d_mlp=8): x = x + relu(x @ W1 + b1) @ W2
- Readout at EOS (final position): logits = x[EOS] @ OUT.W + OUT.b (argmax over output classes).

FORMAT:
Output a short PLAN: (1-3 sentences), then WEIGHTS: followed by sparse assignments (all omitted weights are 0), then END.
Valid line formats:
  L{0|1}.H{0|1}.alibi = <val>
  L{0|1}.H{0|1}.{Q|K|V}[r,c] = <val>   (r in 0..15, c in 0..3)
  L{0|1}.H{0|1}.O[r,c] = <val>         (r in 0..3,  c in 0..15)
  L{0|1}.M.W1[r,c] = <val>             (r in 0..15, c in 0..7)
  L{0|1}.M.b1[c] = <val>               (c in 0..7)
  L{0|1}.M.W2[r,c] = <val>             (r in 0..7,  c in 0..15)
  OUT.W[r,c] = <val>                   (r in 0..15, c in 0..3)
  OUT.b[c] = <val>                     (c in 0..3)"""


def format_task_block(task: Task) -> str:
    rng = random.Random(777 + sum((i + 1) * ord(c) for i, c in enumerate(task.name)))
    ex_lines = []
    for L in (3, 4, 4, 5, 5, 6):
        seq, y = task.gen(rng, L)
        ex_lines.append(f"  {' '.join(map(str, seq))} -> {y}")
    cls_str = "{0, 1}" if task.n_classes == 2 else "{0, 1, 2, 3}"
    return (
        f"TASK: {task.spec}\n"
        f"Output classes: {cls_str}\n"
        f"Examples:\n" + "\n".join(ex_lines)
    )


def build_messages(task: Task) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_DOC}]
    for ex_name in ("last_tok", "has_0", "majority_1", "bigram_0_1"):
        ex_task = TASKS_BY_NAME[ex_name]
        msgs.append({"role": "user", "content": format_task_block(ex_task)})
        msgs.append({"role": "assistant", "content": WORKED_EXAMPLES[ex_name].strip() + "\nEND"})
    msgs.append({"role": "user", "content": format_task_block(task)})
    return msgs


def render_prompt(tokenizer, task: Task) -> str:
    msgs = build_messages(task)
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ---------- Generation & Verification ----------

@torch.no_grad()
def generate_completions(
    model,
    tokenizer,
    prompts: List[str],
    n_samples: int,
    temperature: float,
    max_new_tokens: int,
    batch_size: int,
    device: torch.device,
) -> List[List[str]]:
    """Generate `n_samples` completions per prompt (greedy if temperature <= 0)."""
    model.eval()
    flat_prompts: List[str] = []
    for p in prompts:
        flat_prompts.extend([p] * n_samples)

    flat_out: List[str] = []
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for start in range(0, len(flat_prompts), batch_size):
        chunk = flat_prompts[start : start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        in_len = enc["input_ids"].shape[1]
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        if temperature <= 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95
        out_ids = model.generate(**enc, **gen_kwargs)
        for row in out_ids:
            txt = tokenizer.decode(row[in_len:], skip_special_tokens=True)
            if "\nEND" in txt:
                txt = txt.split("\nEND", 1)[0] + "\nEND"
            flat_out.append(txt.strip())

    grouped: List[List[str]] = []
    for i in range(len(prompts)):
        grouped.append(flat_out[i * n_samples : (i + 1) * n_samples])
    return grouped


def normalize_completion(raw_text: str, pr: ParseResult) -> str:
    """Keep the model's PLAN (if any) + canonicalized WEIGHTS block + END."""
    plan = ""
    if "PLAN:" in raw_text and "WEIGHTS:" in raw_text:
        plan = "PLAN:" + raw_text.split("PLAN:", 1)[1].split("WEIGHTS:", 1)[0].strip()
    assert pr.model is not None
    canon_w = serialize_sparse_weights(pr.model)
    if plan:
        return f"{plan}\nWEIGHTS:\n{canon_w}\nEND"
    return f"WEIGHTS:\n{canon_w}\nEND"


def verify_and_score_split(
    tasks_list: List[Task],
    suites: Dict[str, Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]],
    greedy_texts: List[str],
    sample_texts: List[List[str]],
    eval_device: torch.device,
) -> Dict:
    """Evaluate greedy (pass@1) and sampled (pass@K) on ID (verif/id) and OOD (L16/32/64)."""
    per_task: Dict[str, Dict] = {}
    parsed_cache: List[Tuple[Task, str, ParseResult]] = []

    for task, g_txt, s_txts in zip(tasks_list, greedy_texts, sample_texts):
        suite = suites[task.name]

        # Greedy pass@1
        g_pr = parse_sparse_weights(g_txt, device=eval_device)
        if g_pr.model is not None:
            g_verif = evaluate_model(g_pr.model, task, suite["verif"])
            g_id = evaluate_model(g_pr.model, task, suite["id"])
            g_l16 = evaluate_model(g_pr.model, task, suite["L16"])
            g_l32 = evaluate_model(g_pr.model, task, suite["L32"])
            g_l64 = evaluate_model(g_pr.model, task, suite["L64"])
            g_nnz = g_pr.n_entries
        else:
            g_verif = g_id = g_l16 = g_l32 = g_l64 = 0.0
            g_nnz = -1

        # Sampled pass@K
        best_v = g_verif
        best_pass_pr: Optional[Tuple[str, ParseResult, Dict[str, float]]] = None
        if g_pr.model is not None and g_verif >= 0.999:
            best_pass_pr = (g_txt, g_pr, {"verif": g_verif, "id": g_id, "L16": g_l16, "L32": g_l32, "L64": g_l64})

        n_parsed = 0
        n_passed = 0
        for txt in s_txts:
            pr = parse_sparse_weights(txt, device=eval_device)
            if pr.model is None:
                continue
            n_parsed += 1
            parsed_cache.append((task, txt, pr))
            v = evaluate_model(pr.model, task, suite["verif"])
            if v > best_v:
                best_v = v
            if v >= 0.999:
                n_passed += 1
                if best_pass_pr is None or pr.n_entries < best_pass_pr[1].n_entries:
                    best_pass_pr = (
                        txt,
                        pr,
                        {
                            "verif": v,
                            "id": evaluate_model(pr.model, task, suite["id"]),
                            "L16": evaluate_model(pr.model, task, suite["L16"]),
                            "L32": evaluate_model(pr.model, task, suite["L32"]),
                            "L64": evaluate_model(pr.model, task, suite["L64"]),
                        },
                    )

        per_task[task.name] = {
            "tier": task.tier,
            "greedy_parsed": g_pr.model is not None,
            "greedy_nnz": g_nnz,
            "greedy_verif": g_verif,
            "greedy_id": g_id,
            "greedy_L16": g_l16,
            "greedy_L32": g_l32,
            "greedy_L64": g_l64,
            "greedy_pass": g_verif >= 0.999,
            "greedy_L64_pass": g_l64 >= 0.99,
            "sample_parsed_rate": n_parsed / max(1, len(s_txts)),
            "sample_pass_count": n_passed,
            "best_verif": best_v,
            "pass_at_k": best_pass_pr is not None,
            "best_pass_nnz": best_pass_pr[1].n_entries if best_pass_pr else -1,
            "best_pass_L64": best_pass_pr[2]["L64"] if best_pass_pr else 0.0,
            "best_pass_L64_pass": (best_pass_pr[2]["L64"] >= 0.99) if best_pass_pr else False,
            "best_pass_text": normalize_completion(best_pass_pr[0], best_pass_pr[1]) if best_pass_pr else None,
        }

    n = len(tasks_list)
    summary = {
        "n_tasks": n,
        "greedy_parsed": sum(1 for r in per_task.values() if r["greedy_parsed"]),
        "greedy_pass": sum(1 for r in per_task.values() if r["greedy_pass"]),
        "greedy_L64_pass": sum(1 for r in per_task.values() if r["greedy_L64_pass"]),
        "pass_at_k": sum(1 for r in per_task.values() if r["pass_at_k"]),
        "pass_at_k_L64": sum(1 for r in per_task.values() if r["best_pass_L64_pass"]),
        "mean_greedy_id": sum(r["greedy_id"] for r in per_task.values()) / max(1, n),
        "mean_greedy_L64": sum(r["greedy_L64"] for r in per_task.values()) / max(1, n),
        "mean_best_verif": sum(r["best_verif"] for r in per_task.values()) / max(1, n),
        "per_task": per_task,
    }
    return summary, parsed_cache


# ---------- LoRA Fine-tuning on Verified Pool ----------

def train_lora_on_pool(
    base_model,
    tokenizer,
    task_buckets: Dict[str, List[Tuple[str, str]]],
    lora_rank: int,
    steps: int,
    lr: float,
    grad_accum: int,
    seed: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Attach fresh LoRA to base_model, train task-balanced on (prompt, completion) pairs."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    lora_params = attach_lora(base_model, r=lora_rank, alpha=float(lora_rank * 2))

    # Pre-tokenize dataset grouped by task key for balanced sampling
    encoded_by_task: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
    total_ex = 0
    for _, pairs in task_buckets.items():
        if not pairs:
            continue
        bucket_enc = []
        for prompt, completion in pairs:
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tokenizer(completion + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            ids = p_ids + c_ids
            labels = [-100] * len(p_ids) + c_ids
            bucket_enc.append((torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)))
        encoded_by_task.append(bucket_enc)
        total_ex += len(bucket_enc)

    opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=lr * 0.1)

    base_model.train()
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    losses = []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum):
            bucket_enc = rng.choice(encoded_by_task)
            ids, labels = rng.choice(bucket_enc)
            ids = ids.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                out = base_model(input_ids=ids, labels=labels, use_cache=False)
                loss = out.loss / grad_accum
            loss.backward()
            step_loss += float(loss.item())
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        sched.step()
        losses.append(step_loss)
        if step % max(1, steps // 4) == 0 or step == steps:
            avg_l = sum(losses[-min(20, len(losses)) :]) / min(20, len(losses))
            print(f"    [train step {step:4d}/{steps}] loss={avg_l:.4f} (tasks={len(encoded_by_task)}, pool={total_ex})", flush=True)

    base_model.gradient_checkpointing_disable()
    base_model.eval()
    state = get_lora_state(base_model)
    return state


# ---------- Main Self-Training Loop ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--arm", type=str, choices=["hindsight", "pass", "unverified"], default="hindsight")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--k", type=int, default=16, help="Samples per training task per round")
    ap.add_argument("--heldout-k", type=int, default=8, help="Samples per heldout task per round")
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--train-steps", type=int, default=120)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-per-task", type=int, default=6, help="Keep top-N sparsest unique circuits per task")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="runs/writer_hindsight")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    eval_device = torch.device("cpu")  # TinyTransformer eval is faster on CPU without kernel-launch overhead
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    if device.type == "cpu":
        dtype = torch.float32

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    train_tasks = [t for t in TASKS if t.split == "train"]
    held_tasks = [t for t in TASKS if t.split == "heldout"]
    print(f"Device={device} dtype={dtype} model={args.model} arm={args.arm}")
    print(f"Tasks: {len(train_tasks)} train, {len(held_tasks)} heldout")

    # Build deterministic verification + OOD suites
    all_suites = {t.name: make_eval_suite(t, device=eval_device) for t in train_tasks + held_tasks}
    # Quick screening batch per train task for fast cross-task hindsight
    quick_train = {}
    for t in train_tasks:
        for L, (toks, labs) in all_suites[t.name]["verif"].items():
            if L in (6, 7, 8):
                quick_train[t.name] = {L: (toks, labs)}
                break

    print("Loading tokenizer and model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    train_prompts = [render_prompt(tokenizer, t) for t in train_tasks]
    held_prompts = [render_prompt(tokenizer, t) for t in held_tasks]
    prompt_by_name = {t.name: p for t, p in zip(train_tasks, train_prompts)}

    # Verified pool: task_name -> dict[canonical_weights_str -> (nnz, full_completion_text)]
    pool: Dict[str, Dict[str, Tuple[int, str]]] = {t.name: {} for t in train_tasks}
    # Unverified pool (for control arm)
    unverif_pool: List[Tuple[str, str]] = []

    def add_to_pool(task_name: str, pr: ParseResult, raw_text: str, relabeled: bool) -> bool:
        assert pr.model is not None
        canon_w = serialize_sparse_weights(pr.model)
        # If relabeled via hindsight, drop the wrong-task PLAN and keep clean WEIGHTS
        comp = f"WEIGHTS:\n{canon_w}\nEND" if relabeled else normalize_completion(raw_text, pr)
        bucket = pool[task_name]
        if canon_w in bucket:
            return False
        bucket[canon_w] = (pr.n_entries, comp)
        # Occam pruning: keep up to max_per_task sparsest unique circuits per task
        if len(bucket) > args.max_per_task:
            sorted_keys = sorted(bucket.keys(), key=lambda k: bucket[k][0])
            for drop_k in sorted_keys[args.max_per_task :]:
                del bucket[drop_k]
        return canon_w in bucket

    for r in range(0, args.rounds + 1):
        t_round = time.time()
        print(f"\n================ ROUND {r}/{args.rounds} ({args.arm}) ================", flush=True)

        # 1. Generate greedy (pass@1) and K samples on train & heldout
        t_gen = time.time()
        tr_greedy = [xs[0] for xs in generate_completions(model, tokenizer, train_prompts, 1, 0.0, args.max_new_tokens, args.batch_size, device)]
        tr_samples = generate_completions(model, tokenizer, train_prompts, args.k, args.temperature, args.max_new_tokens, args.batch_size, device)
        he_greedy = [xs[0] for xs in generate_completions(model, tokenizer, held_prompts, 1, 0.0, args.max_new_tokens, args.batch_size, device)]
        he_samples = generate_completions(model, tokenizer, held_prompts, args.heldout_k, args.temperature, args.max_new_tokens, args.batch_size, device)
        gen_s = time.time() - t_gen
        print(f"  Generated in {gen_s:.1f}s", flush=True)

        # 2. Evaluate train & heldout
        tr_summary, tr_parsed = verify_and_score_split(train_tasks, all_suites, tr_greedy, tr_samples, eval_device)
        he_summary, _ = verify_and_score_split(held_tasks, all_suites, he_greedy, he_samples, eval_device)

        # 3. Update training pool (strictly from train_tasks verification on L<=8 only!)
        new_direct = 0
        new_hindsight = 0
        rng = random.Random(args.seed + r)

        # Also include greedy parses from train tasks
        for task, g_txt in zip(train_tasks, tr_greedy):
            g_pr = parse_sparse_weights(g_txt, device=eval_device)
            if g_pr.model is not None:
                tr_parsed.append((task, g_txt, g_pr))

        if args.arm == "unverified":
            for task, txt, pr in tr_parsed:
                if rng.random() < 0.25:
                    unverif_pool.append((prompt_by_name[task.name], normalize_completion(txt, pr)))
            # Keep bounded size
            if len(unverif_pool) > 256:
                unverif_pool = unverif_pool[-256:]
        else:
            for src_task, txt, pr in tr_parsed:
                assert pr.model is not None
                # Direct target check
                v_own = evaluate_model(pr.model, src_task, all_suites[src_task.name]["verif"])
                if v_own >= 0.999:
                    if add_to_pool(src_task.name, pr, txt, relabeled=False):
                        new_direct += 1
                # Cross-task hindsight (strictly across TRAIN tasks only — NEVER heldout!)
                if args.arm == "hindsight":
                    for other in train_tasks:
                        if other.name == src_task.name:
                            continue
                        if evaluate_model(pr.model, other, quick_train[other.name]) >= 0.95:
                            v_other = evaluate_model(pr.model, other, all_suites[other.name]["verif"])
                            if v_other >= 0.999:
                                if add_to_pool(other.name, pr, txt, relabeled=True):
                                    new_hindsight += 1

        pool_tasks_covered = sum(1 for v in pool.values() if v)
        pool_total = sum(len(v) for v in pool.values())

        print(
            f"  TRAIN   ({len(train_tasks)}): pass@1={tr_summary['greedy_pass']:2d} (L64={tr_summary['greedy_L64_pass']:2d}) | "
            f"pass@{args.k}={tr_summary['pass_at_k']:2d} (L64={tr_summary['pass_at_k_L64']:2d}) | "
            f"mean_id={tr_summary['mean_greedy_id']*100:5.1f}% mean_L64={tr_summary['mean_greedy_L64']*100:5.1f}%",
            flush=True,
        )
        print(
            f"  HELDOUT ({len(held_tasks)}): pass@1={he_summary['greedy_pass']:2d} (L64={he_summary['greedy_L64_pass']:2d}) | "
            f"pass@{args.heldout_k}={he_summary['pass_at_k']:2d} (L64={he_summary['pass_at_k_L64']:2d}) | "
            f"mean_id={he_summary['mean_greedy_id']*100:5.1f}% mean_L64={he_summary['mean_greedy_L64']*100:5.1f}%",
            flush=True,
        )
        print(
            f"  POOL: tasks={pool_tasks_covered}/{len(train_tasks)} circuits={pool_total} "
            f"(+{new_direct} direct, +{new_hindsight} hindsight)",
            flush=True,
        )

        # Print solved lists for quick visibility
        tr_p1 = sorted([k for k, v in tr_summary["per_task"].items() if v["greedy_pass"]])
        he_p1 = sorted([k for k, v in he_summary["per_task"].items() if v["greedy_pass"]])
        he_pk = sorted([k for k, v in he_summary["per_task"].items() if v["pass_at_k"]])
        print(f"  TRAIN pass@1:   {tr_p1}", flush=True)
        print(f"  HELDOUT pass@1: {he_p1}  |  pass@{args.heldout_k}: {he_pk}", flush=True)

        # Save round record
        record = {
            "round": r,
            "arm": args.arm,
            "gen_s": round(gen_s, 1),
            "pool_tasks_covered": pool_tasks_covered,
            "pool_total": pool_total,
            "new_direct": new_direct,
            "new_hindsight": new_hindsight,
            "train": tr_summary,
            "heldout": he_summary,
            "pool_tasks": sorted([k for k, v in pool.items() if v]),
        }
        (out_dir / f"round{r}.json").write_text(json.dumps(record, indent=2))

        if r == args.rounds:
            break

        # 4. Fine-tune fresh LoRA from clean base model on accumulated pool
        if args.arm == "unverified":
            task_buckets = {"unverified": list(unverif_pool)} if unverif_pool else {}
        else:
            task_buckets = {
                t_name: [(prompt_by_name[t_name], comp) for _, comp in bucket.values()]
                for t_name, bucket in pool.items()
                if bucket
            }

        if not task_buckets:
            print("  [skip train] Pool is empty; keeping base model for next round.", flush=True)
            continue

        # Reset base model cleanly (keep LoRA unmerged during generation so unwrap resets cleanly!)
        unwrap_and_reset_lora(model, merge=False)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        t_tr = time.time()
        lora_state = train_lora_on_pool(
            model,
            tokenizer,
            task_buckets,
            lora_rank=args.lora_rank,
            steps=args.train_steps,
            lr=args.lr,
            grad_accum=args.grad_accum,
            seed=args.seed + 100 * r,
            device=device,
        )
        torch.save(lora_state, out_dir / f"round{r + 1}.adapter.pt")
        print(f"  Trained LoRA in {time.time() - t_tr:.1f}s (total round {time.time() - t_round:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
