"""Random sparse & dense weight search baseline for 03-transformer-writer.

Tests the exact question that crushed the loop in 02-weight-writer:
"How many variable-length algorithmic Transformer tasks can random weights discover
at matched verifier budget (with full hindsight across all 32 non-example tasks)?"
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import torch

from target_tf import D_HEAD, D_MLP, D_MODEL, N_HEADS, N_LAYERS, N_OUT, TinyTransformer
from tasks import TASKS, evaluate_model, make_eval_suite


WEIGHT_PALETTE = [-200.0, -40.0, -20.0, -10.0, -4.0, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 10.0, 20.0, 40.0, 200.0]
ALIBI_PALETTE = [-10.0, -4.0, -1.0, 0.0, 0.0, 0.0, 1.0, 4.0, 10.0]


def sample_random_sparse(rng: random.Random, min_nnz: int = 5, max_nnz: int = 32) -> TinyTransformer:
    m = TinyTransformer(learnable_embed=False)
    nnz = rng.randint(min_nnz, max_nnz)
    with torch.no_grad():
        for l in range(N_LAYERS):
            for h in range(N_HEADS):
                m.alibi[l, h] = rng.choice(ALIBI_PALETTE)
        slots = [
            ("Q", (N_LAYERS, N_HEADS, D_MODEL, D_HEAD)),
            ("K", (N_LAYERS, N_HEADS, D_MODEL, D_HEAD)),
            ("V", (N_LAYERS, N_HEADS, D_MODEL, D_HEAD)),
            ("O", (N_LAYERS, N_HEADS, D_HEAD, D_MODEL)),
            ("W1", (N_LAYERS, D_MODEL, D_MLP)),
            ("b1", (N_LAYERS, D_MLP)),
            ("W2", (N_LAYERS, D_MLP, D_MODEL)),
            ("OUT_W", (D_MODEL, N_OUT)),
            ("OUT_b", (N_OUT,)),
        ]
        for _ in range(nnz):
            name, shape = rng.choice(slots)
            idx = tuple(rng.randrange(s) for s in shape)
            getattr(m, name)[idx] = rng.choice(WEIGHT_PALETTE)
    m.eval()
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=5000, help="Number of random circuits to evaluate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="runs/random_baseline.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    eval_tasks = [t for t in TASKS if t.split in ("train", "heldout")]
    suites = {t.name: make_eval_suite(t, device=device) for t in eval_tasks}

    # Pre-pack a quick 16-example screening batch per task so we don't waste time on dead circuits
    quick = {}
    for t in eval_tasks:
        # Use length 6 (32 examples) for instant screening
        for L, (toks, labs) in suites[t.name]["verif"].items():
            if L in (6, 7, 8):
                quick[t.name] = {L: (toks, labs)}
                break

    best_verif: Dict[str, float] = {t.name: 0.0 for t in eval_tasks}
    solved_verif: Dict[str, Dict[str, float]] = {}

    rng = random.Random(args.seed)
    t0 = time.time()

    for i in range(1, args.budget + 1):
        m = sample_random_sparse(rng).to(device)
        for t in eval_tasks:
            q_acc = evaluate_model(m, t, quick[t.name])
            if q_acc >= 0.90:
                v_acc = evaluate_model(m, t, suites[t.name]["verif"])
                if v_acc > best_verif[t.name]:
                    best_verif[t.name] = v_acc
                if v_acc >= 0.999 and t.name not in solved_verif:
                    solved_verif[t.name] = {
                        k: evaluate_model(m, t, b) for k, b in suites[t.name].items()
                    }
                    solved_verif[t.name]["found_at"] = i
            elif q_acc > best_verif[t.name]:
                # Track rough best even below threshold
                pass
        if i % 500 == 0 or i == args.budget:
            dt = time.time() - t0
            print(
                f"[{i:5d}/{args.budget}] {dt:5.1f}s ({i/dt:6.1f} circ/s) "
                f"solved_verif={len(solved_verif)}/{len(eval_tasks)}: {sorted(solved_verif.keys())}",
                flush=True,
            )

    train_names = [t.name for t in eval_tasks if t.split == "train"]
    held_names = [t.name for t in eval_tasks if t.split == "heldout"]
    train_solved = [n for n in train_names if n in solved_verif]
    held_solved = [n for n in held_names if n in solved_verif]
    train_l64 = [n for n in train_solved if solved_verif[n]["L64"] >= 0.99]

    print("\n=== Random Sparse Search Summary ===")
    print(f"Budget: {args.budget} random circuits (hindsight across all {len(eval_tasks)} tasks)")
    print(f"Train solved (verif=100%):   {len(train_solved)}/{len(train_names)}  | L64>=99%: {len(train_l64)}/{len(train_names)}")
    print(f"Heldout solved (verif=100%): {len(held_solved)}/{len(held_names)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "budget": args.budget,
        "seed": args.seed,
        "elapsed_s": round(time.time() - t0, 2),
        "train_solved": train_solved,
        "train_l64": train_l64,
        "heldout_solved": held_solved,
        "solved_details": solved_verif,
    }, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
