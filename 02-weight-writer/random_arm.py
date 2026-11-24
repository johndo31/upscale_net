"""Random-pool arm: the hindsight training recipe, but the pool is filled with RANDOM integer networks instead of
the model's own answers, at the same sampling budget.

For B round-equivalents, each round draws k random networks per TRAIN task (weights and biases uniform in
-R..R, shape of that task: n=2 -> 2 hidden, n=3 -> 3 hidden), renders them with tasks.render, scores the text
with wsp.score (the code path of model answers) and folds the round into the pool with
wsp.update_pool("hindsight", ...), one round at a time (same cap-per-function, smallest-first accumulation).
Then the base model is fine-tuned on the final pool exactly as a loop round (wsp.train_round: same steps rule,
adapter saved before merging, resume from a matching adapter) and evaluated on all tasks (wsp.evaluate).
Output: <out-dir>/random_R<R>/round<B>.json in the loop's round format (report.py shows it as one more arm).
Seeds match loop round B: networks from numpy default_rng([seed, R]), training seed seed+B, eval seed seed+1000*B.

Usage (pod):
  python random_arm.py --device cuda --seed 0 --rank 16 --lr 2e-4 --k 16 --epochs 40 --pool-cap 4 \
      --max-steps 2000 --rounds 10 --rand-range 2 --out-dir runs_gpu/main
  python random_arm.py --pool-only ...     # build and check the pool only (no model), print coverage per round
"""
import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

import tasks
import wsp


def parser():
    """wsp.main()'s flags that configure()/run_config()/train_round()/evaluate() read, with the same defaults."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rounds", type=int, default=10, help="sampling budget in round-equivalents (B)")
    ap.add_argument("--rand-range", type=int, default=2, help="R: integer weights and biases uniform in -R..R")
    ap.add_argument("--pool-only", action="store_true", help="build and verify the pool, print coverage, no model")
    ap.add_argument("--k", type=int, default=16, help="random networks per train task per round; also eval samples")
    ap.add_argument("--heldout-k", type=int, default=64, help="recorded in config only (the loop does not use it)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--gen-dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    ap.add_argument("--train-amp", choices=["auto", "bf16", "none"], default="auto")
    ap.add_argument("--batch-size", type=int, default=None, help="generation batch size (default: cuda 512, mps 256, cpu 64)")
    ap.add_argument("--train-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank r (alpha = 2r)")
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--pool-cap", type=int, default=4)
    ap.add_argument("--mem-gb", type=float, default=None)
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", choices=["fast", "hf"], default="fast")
    ap.add_argument("--bench-batches", default=None, help=argparse.SUPPRESS)   # read by wsp.configure only
    return ap


def random_net(rng, n, R):
    H = tasks.HIDDEN[n]
    W1 = rng.integers(-R, R + 1, size=(H, n)).astype(float)
    b1 = rng.integers(-R, R + 1, size=H).astype(float)
    W2 = rng.integers(-R, R + 1, size=H).astype(float)
    b2 = float(rng.integers(-R, R + 1))
    return W1, b1, W2, b2


def random_round(rng, train_tasks, k, R):
    """A fake loop round: k random networks per train task, scored exactly like model answers."""
    recs = []
    for t in train_tasks:
        n, f = t["n"], t["f"]
        recs.append({**t, "split": "train",
                     "samples": [wsp.score(tasks.render(random_net(rng, n, R)), n, f) for _ in range(k)]})
    return {"tasks": recs}


def build_pool(train_tasks, train_set, rounds, k, R, seed):
    """Pool after each of `rounds` random rounds, via wsp.update_pool("hindsight", ...). Returns
    (pool, coverage_by_round, list of fake rounds)."""
    rng = np.random.default_rng([seed, R])
    pool, cov, fakes, solved_ever = {}, [], [], set()
    found = set()
    for b in range(1, rounds + 1):
        fake = random_round(rng, train_tasks, k, R)
        fakes.append(fake)
        pool = wsp.update_pool("hindsight", pool, fake, train_set)
        S = [s for r in fake["tasks"] for s in r["samples"]]
        solved = [r for r in fake["tasks"] if any(s["passed"] for s in r["samples"])]
        solved_ever |= {(r["n"], r["f"]) for r in solved}
        found |= {(r["n"], s["tt"]) for r in fake["tasks"] for s in r["samples"] if (r["n"], s["tt"]) in train_set}
        nfun = sum(1 for v in pool.values() if v)
        assert nfun == len(found)
        cov.append({"round": b, "pool_functions": nfun, "pool_size": sum(len(v) for v in pool.values()),
                    "networks": len(S), "first_try": float(np.mean([s["passed"] for s in S])),
                    "solved_within_k": len(solved) / len(fake["tasks"]),
                    "solved_ever": len(solved_ever) / len(fake["tasks"])})
        wsp.log(f"[random R={R}] round-equivalent {b}: {len(S)} networks, pool {cov[-1]['pool_size']} examples over "
                f"{nfun} functions, solved within {k} {cov[-1]['solved_within_k']:.3f}")
    return pool, cov, fakes


def check_pool(pool, fakes, train_set):
    """The per-round pool equals one wsp.update_pool("hindsight") call on all rounds' samples at once, and an
    independent re-statement of the rule (keep the POOL_CAP smallest distinct rendered answers per train function)."""
    allr = {"tasks": [r for fk in fakes for r in fk["tasks"]]}
    once = wsp.update_pool("hindsight", {}, allr, train_set)
    cand = {}
    for r in allr["tasks"]:
        for s in r["samples"]:
            if s["valid"] and (r["n"], s["tt"]) in train_set:
                net = tasks.parse(s["text"], r["n"])
                cand.setdefault(f"{r['n']}:{s['tt']}", set()).add((tuple(tasks.size(net)), tasks.render(net)))
    naive = {key: [{"text": t, "size": list(sz)} for sz, t in sorted(v)[:wsp.POOL_CAP]] for key, v in cand.items()}
    return pool == once, pool == naive


def main():
    a = parser().parse_args()
    wsp.configure(a)
    if not os.path.isabs(a.out_dir):
        a.out_dir = os.path.join(wsp.HERE, a.out_dir)
    R, B = a.rand_range, a.rounds
    arm = f"random_R{R}"
    path = os.path.join(a.out_dir, arm, f"round{B}.json")
    if os.path.exists(path) and not a.pool_only:
        wsp.log("exists, skipping:", path)
        print(wsp.summary(f"{arm} r{B}", json.load(open(path))), flush=True)
        return
    tr, ho = wsp.task_lists(a.limit)
    train_set = {(t["n"], t["f"]) for t in wsp.load_split() if t["split"] == "train"}
    pool, cov, fakes = build_pool(tr, train_set, B, a.k, R, a.seed)
    same_once, same_naive = check_pool(pool, fakes, train_set)
    wsp.log(f"pool check: equals one-shot wsp.update_pool on all rounds: {same_once}; equals independent rule: {same_naive}")
    if not (same_once and same_naive):
        sys.exit("pool check failed")
    ex = wsp.pool_examples(pool)
    nfun = sum(1 for v in pool.values() if v)
    wsp.log(f"[{arm} r{B}] pool {len(ex)} examples over {nfun} functions (from {B} x {a.k} x {len(tr)} random networks)")
    if a.pool_only:
        print(json.dumps({"pool_functions": nfun, "pool_size": len(ex), "coverage_by_round": cov}), flush=True)
        return
    if not ex:
        wsp.dump({"stage": "random", "arm": arm, "round": B, "empty_pool": True, "pool_size": 0, "pool_functions": 0,
                  "coverage_by_round": cov, "config": {**wsp.run_config(a), "rand_range": R}}, path)
        print(f"[{arm} r{B}] empty pool", flush=True)
        return
    steps = a.steps or wsp.default_steps(len(ex), a.train_batch)
    model, info = wsp.train_round(a, ex, steps, a.seed + B, os.path.join(a.out_dir, arm, f"round{B}.adapter.pt"))
    ev = wsp.evaluate(model, tr + ho, a.k, a.temperature, a.batch_size, a.seed + 1000 * B)
    ev.update({"stage": "random", "arm": arm, "round": B, "pool_size": len(ex), "pool_functions": nfun,
               "pool": pool, "coverage_by_round": cov, "rand_range": R, **info,
               "config": {**wsp.run_config(a), "rand_range": R}})
    wsp.dump(ev, path)
    del model; wsp.free()
    print(wsp.summary(f"{arm} r{B}", ev) + f" | pool {len(ex)} ({nfun} fns)", flush=True)


if __name__ == "__main__":
    main()
