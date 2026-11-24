"""Arm 2: Verifier-Feedback Multi-Turn Weight Debugging + Single-Shot Distillation.

Compared directly against Arm 1 (blind single-shot hindsight loop in writer_loop.py):
  - Instead of throwing away every candidate circuit with verif < 100%, the verifier runs
    the candidate weights on L=3..8 and returns exact diagnostics:
      * accuracy on L=3..8
      * up to 4 failing L<=8 examples with predicted vs expected class, EOS logits, and
        non-zero working residual channels x[EOS, 7..15]
  - The writer gets up to `--debug-turns` (default 3) turns to inspect its own failing
    weights + EOS residual values and emit a corrected circuit.
  - Whenever a trajectory reaches verif >= 99.9% on L=3..8, BOTH:
      1. the clean single-shot (task -> final PLAN + WEIGHTS) pair, AND
      2. the debugging turn (failing attempt + verifier trace -> fixed WEIGHTS)
    are added to the verified pool and distilled via fresh LoRA from the base model.
  - Strictly evaluated on SINGLE-SHOT pass@1 (zero debugging at test time!) AND multi-turn
    debugging on BOTH the 22 train tasks and 10 held-out unseen tasks, at ID (L=3..8) and
    OOD (L=16, 32, 64).
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from target_tf import D_HEAD, N_HEADS, N_LAYERS, ParseResult, TinyTransformer, parse_sparse_weights, serialize_sparse_weights
from tasks import TASKS, TASKS_BY_NAME, Task, evaluate_model, make_eval_suite
from verify_realizability import WORKED_EXAMPLES
from writer_loop import (
    SYSTEM_DOC,
    build_messages,
    format_task_block,
    generate_completions,
    normalize_completion,
    render_prompt,
    train_lora_on_pool,
    unwrap_and_reset_lora,
    verify_and_score_split,
)


@torch.no_grad()
def forward_with_diagnostics(model: TinyTransformer, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run TinyTransformer and return (logits [B, 4], x_eos [B, 16], x_peak_abs [B, 16])."""
    B, T = tokens.shape
    device = tokens.device
    x = F.embedding(tokens, model.embed)
    pos = torch.arange(T, device=device)
    dist = (pos[:, None] - pos[None, :]).to(x.dtype)
    mask = pos[None, :] < pos[:, None]
    mask[0, 0] = True
    attn_mask = torch.where(mask, torch.zeros_like(dist), torch.full_like(dist, float("-inf")))
    scale = 1.0 / math.sqrt(D_HEAD)
    for l in range(N_LAYERS):
        attn_delta = torch.zeros_like(x)
        for h in range(N_HEADS):
            q = x @ model.Q[l, h]
            k = x @ model.K[l, h]
            v = x @ model.V[l, h]
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            scores = scores + model.alibi[l, h] * dist.unsqueeze(0) + attn_mask.unsqueeze(0)
            probs = F.softmax(scores, dim=-1)
            attn_delta = attn_delta + torch.matmul(probs, v) @ model.O[l, h]
        x = x + attn_delta
        x = x + F.relu(x @ model.W1[l] + model.b1[l]) @ model.W2[l]
    x_eos = x[:, -1, :]
    x_peak = x.abs().max(dim=1).values
    logits = x_eos @ model.OUT_W + model.OUT_b
    return logits, x_eos, x_peak


@torch.no_grad()
def build_verifier_feedback(
    pr: ParseResult,
    task: Task,
    verif_batches: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    max_examples: int = 4,
) -> Tuple[float, str]:
    """Return (verif_acc, concise diagnostic message strictly from L<=8 verif batches)."""
    if pr.model is None:
        return 0.0, (
            f"VERIFIER: Parse error ({pr.error}).\n"
            "Fix the syntax and output valid PLAN:, WEIGHTS:, and END lines."
        )

    model = pr.model
    correct = 0
    total = 0
    # Collect diverse failure types (by expected class) for crisp debugging
    failures_by_exp: Dict[int, List[str]] = {}

    for L in sorted(verif_batches.keys()):
        toks, labs = verif_batches[L]
        logits_full, x_eos, x_peak = forward_with_diagnostics(model, toks)
        logits = logits_full[:, : task.n_classes]
        preds = logits.argmax(dim=-1)
        eq = preds == labs
        correct += int(eq.sum().item())
        total += int(labs.numel())

        wrong_idxs = torch.nonzero(~eq, as_tuple=False).flatten().tolist()
        for idx in wrong_idxs:
            exp = int(labs[idx].item())
            if len(failures_by_exp.get(exp, [])) >= 2:
                continue
            pred = int(preds[idx].item())
            seq = toks[idx, 1:-1].tolist()
            lg = [round(float(v), 2) for v in logits[idx].tolist()]
            eos_parts = []
            peak_parts = []
            for ch in range(7, 16):
                val = float(x_eos[idx, ch].item())
                pk = float(x_peak[idx, ch].item())
                if abs(val) > 1e-3:
                    eos_parts.append(f"ch{ch}={val:.2f}")
                if pk > 1e-3 and abs(pk - abs(val)) > 0.05:
                    peak_parts.append(f"ch{ch}={pk:.2f}")
            res_str = ", ".join(eos_parts) if eos_parts else "all zero"
            if peak_parts:
                res_str += " | seq_peak: " + ", ".join(peak_parts)
            line = f"  {' '.join(map(str, seq))} -> pred {pred}, expected {exp} | logits={lg} | EOS: {res_str}"
            failures_by_exp.setdefault(exp, []).append(line)

    acc = correct / max(1, total)
    if acc >= 0.999:
        return acc, "VERIFIER: PASS (100.0% on L=3..8)."

    shown: List[str] = []
    # Round-robin across expected classes so model sees both false positives & false negatives
    keys = sorted(failures_by_exp.keys())
    pos = 0
    while len(shown) < max_examples and keys:
        added = False
        for k in keys:
            if pos < len(failures_by_exp[k]) and len(shown) < max_examples:
                shown.append(failures_by_exp[k][pos])
                added = True
        if not added:
            break
        pos += 1

    msg = (
        f"VERIFIER: FAILED (accuracy {acc * 100:.1f}% on L=3..8).\n"
        f"Failing examples:\n" + "\n".join(shown) + "\n"
        "Diagnose why the weights failed on these inputs and output corrected PLAN:, WEIGHTS:, and END."
    )
    return acc, msg


def run_debug_rollouts(
    model,
    tokenizer,
    tasks_list: List[Task],
    suites: Dict[str, Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]],
    n_trajectories: int,
    max_turns: int,
    temperature: float,
    max_new_tokens: int,
    batch_size: int,
    device: torch.device,
    eval_device: torch.device,
) -> Tuple[List[Tuple[Task, str, ParseResult, Optional[List[Dict[str, str]]]]], Dict[str, Dict]]:
    """Run batched multi-turn debugging rollouts on `tasks_list`.

    Returns:
      - all_attempts: list of (task, completion_text, ParseResult, debug_msgs_or_None)
      - debug_stats: per-task best accuracy & pass status across debug turns
    """
    # Active trajectories: list of dicts
    trajs = []
    for task in tasks_list:
        base_msgs = build_messages(task)
        for t_idx in range(n_trajectories):
            trajs.append({
                "task": task,
                "msgs": [dict(m) for m in base_msgs],
                "done": False,
                "best_acc": 0.0,
                "best_pass": None,  # (turn, text, pr, debug_history)
            })

    all_attempts: List[Tuple[Task, str, ParseResult, Optional[List[Dict[str, str]]]]] = []

    for turn in range(1, max_turns + 1):
        active = [tr for tr in trajs if not tr["done"]]
        if not active:
            break
        prompts = [
            tokenizer.apply_chat_template(tr["msgs"], tokenize=False, add_generation_prompt=True)
            for tr in active
        ]
        # Turn 1 trajectory 0 can be greedy if desired, rest sampled; here sample all with temperature
        outs = [
            xs[0]
            for xs in generate_completions(
                model,
                tokenizer,
                prompts,
                n_samples=1,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                device=device,
            )
        ]

        n_passed_this_turn = 0
        for tr, txt in zip(active, outs):
            task = tr["task"]
            pr = parse_sparse_weights(txt, device=eval_device)
            acc, fb = build_verifier_feedback(pr, task, suites[task.name]["verif"])
            if acc > tr["best_acc"]:
                tr["best_acc"] = acc

            # Record full debug prefix if this was turn >= 2
            debug_msgs = [dict(m) for m in tr["msgs"]] if turn >= 2 else None
            if pr.model is not None:
                all_attempts.append((task, txt, pr, debug_msgs))

            if pr.model is not None and acc >= 0.999:
                tr["done"] = True
                n_passed_this_turn += 1
                tr["best_pass"] = (turn, txt, pr, debug_msgs)
            else:
                tr["msgs"].append({"role": "assistant", "content": txt})
                tr["msgs"].append({"role": "user", "content": fb})

        print(
            f"    [debug turn {turn}/{max_turns}] active={len(active)} newly_passed={n_passed_this_turn}",
            flush=True,
        )

    # Aggregate per-task debug stats
    debug_stats: Dict[str, Dict] = {}
    for task in tasks_list:
        mine = [tr for tr in trajs if tr["task"].name == task.name]
        passes = [tr["best_pass"] for tr in mine if tr["best_pass"] is not None]
        best_acc = max((tr["best_acc"] for tr in mine), default=0.0)
        if passes:
            # Pick sparsest passing circuit
            best_p = min(passes, key=lambda item: item[2].n_entries)
            _, p_txt, p_pr, _ = best_p
            suite = suites[task.name]
            l64 = evaluate_model(p_pr.model, task, suite["L64"])
            debug_stats[task.name] = {
                "passed": True,
                "pass_count": len(passes),
                "min_turn": min(item[0] for item in passes),
                "best_acc": 1.0,
                "best_nnz": p_pr.n_entries,
                "best_L64": l64,
                "best_L64_pass": l64 >= 0.99,
                "best_text": normalize_completion(p_txt, p_pr),
            }
        else:
            debug_stats[task.name] = {
                "passed": False,
                "pass_count": 0,
                "min_turn": -1,
                "best_acc": best_acc,
                "best_nnz": -1,
                "best_L64": 0.0,
                "best_L64_pass": False,
                "best_text": None,
            }

    return all_attempts, debug_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--trajs", type=int, default=6, help="Parallel debug trajectories per train task")
    ap.add_argument("--debug-turns", type=int, default=3, help="Max verifier-feedback turns per trajectory")
    ap.add_argument("--heldout-trajs", type=int, default=4, help="Parallel debug trajectories per heldout task")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--train-steps", type=int, default=140)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-per-task", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="runs/coder3b_debug")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    eval_device = torch.device("cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    if device.type == "cpu":
        dtype = torch.float32

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    train_tasks = [t for t in TASKS if t.split == "train"]
    held_tasks = [t for t in TASKS if t.split == "heldout"]
    print(f"Device={device} dtype={dtype} model={args.model} mode=DEBUG+DISTILL")
    print(f"Tasks: {len(train_tasks)} train, {len(held_tasks)} heldout | trajs={args.trajs} x turns={args.debug_turns}")

    all_suites = {t.name: make_eval_suite(t, device=eval_device) for t in train_tasks + held_tasks}
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

    # Verified single-shot pool: task_name -> dict[canon_w -> (nnz, completion)]
    pool: Dict[str, Dict[str, Tuple[int, str]]] = {t.name: {} for t in train_tasks}
    # Verified multi-turn debug pool: task_name -> list[(rendered_debug_prompt, completion)]
    debug_pool: Dict[str, List[Tuple[str, str]]] = {t.name: [] for t in train_tasks}

    def add_to_pool(task_name: str, pr: ParseResult, raw_text: str, relabeled: bool, debug_msgs=None) -> bool:
        assert pr.model is not None
        canon_w = serialize_sparse_weights(pr.model)
        comp = f"WEIGHTS:\n{canon_w}\nEND" if relabeled else normalize_completion(raw_text, pr)
        bucket = pool[task_name]
        added = False
        if canon_w not in bucket:
            bucket[canon_w] = (pr.n_entries, comp)
            if len(bucket) > args.max_per_task:
                for drop_k in sorted(bucket.keys(), key=lambda k: bucket[k][0])[args.max_per_task :]:
                    del bucket[drop_k]
            added = canon_w in bucket
        if (not relabeled) and (debug_msgs is not None):
            dp = tokenizer.apply_chat_template(debug_msgs, tokenize=False, add_generation_prompt=True)
            db = debug_pool[task_name]
            if len(db) < 4:
                db.append((dp, comp))
        return added

    for r in range(0, args.rounds + 1):
        t_round = time.time()
        print(f"\n================ DEBUG ROUND {r}/{args.rounds} ================", flush=True)

        # 1. Pure SINGLE-SHOT greedy evaluation (no debugging!) on train & heldout
        t_g = time.time()
        tr_greedy = [xs[0] for xs in generate_completions(model, tokenizer, train_prompts, 1, 0.0, args.max_new_tokens, args.batch_size, device)]
        he_greedy = [xs[0] for xs in generate_completions(model, tokenizer, held_prompts, 1, 0.0, args.max_new_tokens, args.batch_size, device)]
        tr_ss, _ = verify_and_score_split(train_tasks, all_suites, tr_greedy, [[] for _ in train_tasks], eval_device)
        he_ss, _ = verify_and_score_split(held_tasks, all_suites, he_greedy, [[] for _ in held_tasks], eval_device)
        ss_s = time.time() - t_g

        # 2. Multi-turn verifier-feedback debugging rollouts on train tasks
        t_d = time.time()
        print(f"  Running train debug rollouts ({len(train_tasks)} tasks x {args.trajs} trajs x {args.debug_turns} turns)...", flush=True)
        tr_attempts, tr_dbg = run_debug_rollouts(
            model, tokenizer, train_tasks, all_suites,
            n_trajectories=args.trajs, max_turns=args.debug_turns,
            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size, device=device, eval_device=eval_device,
        )
        print(f"  Running heldout debug rollouts ({len(held_tasks)} tasks x {args.heldout_trajs} trajs x {args.debug_turns} turns)...", flush=True)
        _, he_dbg = run_debug_rollouts(
            model, tokenizer, held_tasks, all_suites,
            n_trajectories=args.heldout_trajs, max_turns=args.debug_turns,
            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size, device=device, eval_device=eval_device,
        )
        dbg_s = time.time() - t_d

        # 3. Update verified pool (from greedy + debug attempts; strictly TRAIN tasks on L<=8 only!)
        new_direct = 0
        new_hindsight = 0
        for task, g_txt in zip(train_tasks, tr_greedy):
            g_pr = parse_sparse_weights(g_txt, device=eval_device)
            if g_pr.model is not None:
                tr_attempts.append((task, g_txt, g_pr, None))

        for src_task, txt, pr, dbg_msgs in tr_attempts:
            assert pr.model is not None
            v_own = evaluate_model(pr.model, src_task, all_suites[src_task.name]["verif"])
            if v_own >= 0.999:
                if add_to_pool(src_task.name, pr, txt, relabeled=False, debug_msgs=dbg_msgs):
                    new_direct += 1
            for other in train_tasks:
                if other.name == src_task.name:
                    continue
                if evaluate_model(pr.model, other, quick_train[other.name]) >= 0.95:
                    if evaluate_model(pr.model, other, all_suites[other.name]["verif"]) >= 0.999:
                        if add_to_pool(other.name, pr, txt, relabeled=True, debug_msgs=None):
                            new_hindsight += 1

        pool_cov = sum(1 for v in pool.values() if v)
        pool_tot = sum(len(v) for v in pool.values())
        dbg_tr_pass = sum(1 for v in tr_dbg.values() if v["passed"])
        dbg_tr_l64 = sum(1 for v in tr_dbg.values() if v["best_L64_pass"])
        dbg_he_pass = sum(1 for v in he_dbg.values() if v["passed"])
        dbg_he_l64 = sum(1 for v in he_dbg.values() if v["best_L64_pass"])

        print(
            f"  TRAIN   ({len(train_tasks)}): single-shot p@1={tr_ss['greedy_pass']:2d} (L64={tr_ss['greedy_L64_pass']:2d}) | "
            f"debug={dbg_tr_pass:2d} (L64={dbg_tr_l64:2d}) | mean_ss_L64={tr_ss['mean_greedy_L64']*100:5.1f}%",
            flush=True,
        )
        print(
            f"  HELDOUT ({len(held_tasks)}): single-shot p@1={he_ss['greedy_pass']:2d} (L64={he_ss['greedy_L64_pass']:2d}) | "
            f"debug={dbg_he_pass:2d} (L64={dbg_he_l64:2d}) | mean_ss_L64={he_ss['mean_greedy_L64']*100:5.1f}%",
            flush=True,
        )
        print(
            f"  POOL: tasks={pool_cov}/{len(train_tasks)} circuits={pool_tot} (+{new_direct}d +{new_hindsight}h) "
            f"[ss={ss_s:.0f}s dbg={dbg_s:.0f}s]",
            flush=True,
        )
        print("  TRAIN solved (debug):  ", sorted([k for k, v in tr_dbg.items() if v["passed"]]), flush=True)
        print("  HELDOUT solved (ss):   ", sorted([k for k, v in he_ss["per_task"].items() if v["greedy_pass"]]), flush=True)
        print("  HELDOUT solved (debug):", sorted([k for k, v in he_dbg.items() if v["passed"]]), flush=True)

        record = {
            "round": r,
            "ss_s": round(ss_s, 1),
            "dbg_s": round(dbg_s, 1),
            "pool_tasks_covered": pool_cov,
            "pool_total": pool_tot,
            "new_direct": new_direct,
            "new_hindsight": new_hindsight,
            "train_single_shot": tr_ss,
            "heldout_single_shot": he_ss,
            "train_debug": tr_dbg,
            "heldout_debug": he_dbg,
            "pool_tasks": sorted([k for k, v in pool.items() if v]),
        }
        (out_dir / f"round{r}.json").write_text(json.dumps(record, indent=2))

        if r == args.rounds:
            break

        # 4. Distill single-shot circuits + debug trajectories into fresh LoRA
        task_buckets: Dict[str, List[Tuple[str, str]]] = {}
        for t_name, bucket in pool.items():
            if not bucket:
                continue
            pairs = [(prompt_by_name[t_name], comp) for _, comp in bucket.values()]
            # Include debug trajectories in separate sub-bucket so both single-shot & debugging improve
            task_buckets[t_name] = pairs
            if debug_pool[t_name]:
                task_buckets[f"{t_name}__dbg"] = list(debug_pool[t_name])

        if not task_buckets:
            print("  [skip train] Pool is empty.", flush=True)
            continue

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
