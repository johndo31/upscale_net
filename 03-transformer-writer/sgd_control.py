"""SGD / AdamW control baseline on the exact same TinyTransformer architecture.

For every task, trains a fresh TinyTransformer on sequences of length L in [3..8]
(with standard AdamW + cosine schedule), then evaluates:
  - ID  (L = 3..8, fresh test sequences)
  - OOD (L = 16, 32, 64)

This gives an apples-to-apples length-generalization baseline against Weight Writer.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from target_tf import BOS_ID, EOS_ID, TinyTransformer
from tasks import TASKS, TASKS_BY_NAME, Task, evaluate_model, make_eval_suite


def make_train_minibatch(task: Task, rng: random.Random, batch_size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    L = rng.choice([3, 4, 5, 6, 7, 8])
    seqs, ys = [], []
    for _ in range(batch_size):
        s, y = task.gen(rng, L)
        seqs.append([BOS_ID] + s + [EOS_ID])
        ys.append(y)
    return (
        torch.tensor(seqs, dtype=torch.long, device=device),
        torch.tensor(ys, dtype=torch.long, device=device),
    )


def train_one_task(
    task: Task,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    learnable_embed: bool,
    device: torch.device,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    suite = make_eval_suite(task, device=device)

    model = TinyTransformer(learnable_embed=learnable_embed).to(device)
    model.init_random_for_sgd(seed=seed)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)

    best_verif = -1.0
    best_state = None

    for step in range(1, steps + 1):
        model.train()
        toks, labs = make_train_minibatch(task, rng, batch_size, device)
        logits = model(toks)[:, : task.n_classes]
        loss = F.cross_entropy(logits, labs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 200 == 0 or step == steps:
            model.eval()
            v_acc = evaluate_model(model, task, suite["verif"])
            if v_acc >= best_verif:
                best_verif = v_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if best_verif >= 1.0 and step >= 1000:
                # Keep training a little past first 100% for margin, then allow early exit
                pass

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return {k: evaluate_model(model, task, b) for k, b in suite.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--learnable-embed", action="store_true")
    ap.add_argument("--tasks", type=str, default="", help="Comma-separated task names (default: all train+heldout)")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="runs/sgd_control.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.tasks:
        selected = [TASKS_BY_NAME[n.strip()] for n in args.tasks.split(",") if n.strip()]
    else:
        selected = [t for t in TASKS if t.split in ("train", "heldout")]

    results: Dict[str, Dict[str, float]] = {}
    t0 = time.time()
    print(f"{'task':18s} {'split':8s} {'T':>1s} {'verif':>7s} {'id':>7s} {'L16':>7s} {'L32':>7s} {'L64':>7s}")
    print("-" * 68)
    for idx, task in enumerate(selected):
        accs = train_one_task(
            task,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + idx,
            learnable_embed=args.learnable_embed,
            device=device,
        )
        results[task.name] = accs
        print(
            f"{task.name:18s} {task.split:8s} {task.tier:1d} "
            f"{accs['verif']*100:6.1f}% {accs['id']*100:6.1f}% "
            f"{accs['L16']*100:6.1f}% {accs['L32']*100:6.1f}% {accs['L64']*100:6.1f}%",
            flush=True,
        )

    means = {
        k: sum(r[k] for r in results.values()) / max(1, len(results))
        for k in ("verif", "id", "L16", "L32", "L64")
    }
    perfect = {
        k: sum(1 for r in results.values() if r[k] >= 0.99)
        for k in ("verif", "id", "L16", "L32", "L64")
    }
    print("-" * 68)
    print(
        f"{'MEAN':18s} {'all':8s}   "
        f"{means['verif']*100:6.1f}% {means['id']*100:6.1f}% "
        f"{means['L16']*100:6.1f}% {means['L32']*100:6.1f}% {means['L64']*100:6.1f}%"
    )
    print(
        f"{'SOLVED (>=99%)':18s} {'all':8s}   "
        f"{perfect['verif']:4d}/{len(results):<2d} {perfect['id']:4d}/{len(results):<2d} "
        f"{perfect['L16']:4d}/{len(results):<2d} {perfect['L32']:4d}/{len(results):<2d} {perfect['L64']:4d}/{len(results):<2d}"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": vars(args),
        "elapsed_s": round(time.time() - t0, 2),
        "means": means,
        "solved_99": perfect,
        "per_task": results,
    }, indent=2))
    print(f"Saved -> {out_path} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
