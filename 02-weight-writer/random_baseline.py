"""Random-guessing baseline for the weight-writing experiment (CPU, numpy only, no model).

Is the model's own guessing better than drawing integer weights at random? For weight range R, every weight and
bias is drawn uniformly and independently from -R..R. One "round" mimics the loop's sampling budget: 16 random
networks for every TRAIN task, of the shape that task needs (n=2: 2 hidden units, n=3: 3 hidden units), i.e.
16 x 205 = 3,280 networks per round.

  (a) discovery, hindsight-style: cumulative number of distinct TRAIN functions computed by at least one random
      network so far (a network counts for the function it computes, whatever task slot it was drawn for, only
      against functions of its own input count; held-out and few-shot functions never count)
  (b) per-task success, pass-only-style: per round, the fraction of train tasks for which one of its own 16
      networks computes the task's function ("solved within 16 tries"); and the cumulative fraction of train
      tasks solved at least once so far
  (c) first-try success: P(one random network of the right shape computes the target), averaged over train tasks,
      estimated from all networks of that shape drawn in the run

Mean and min/max over seeds. Then the real runs (runs_gpu/main by default) are read for comparison.
Alignment: the hindsight pool in round r (round<r>.json) is built from r sampling rounds (base round0 and loop
rounds 1..r-1), so it is compared with r random rounds.

Usage: python random_baseline.py [--rounds 30] [--seeds 20] [--out random_baseline.json] [--runs runs_gpu/main]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

import tasks

HERE = os.path.dirname(os.path.abspath(__file__))
FEWSHOT_SET = {(n, f) for n, f, _ in tasks.FEWSHOT}


def load_train():
    """Train tasks in split order, and the level of every (n, f) from solutions.json."""
    T = json.load(open(os.path.join(HERE, "split.json")))["tasks"]
    sol = json.load(open(os.path.join(HERE, "solutions.json")))
    level = {(int(n), int(f)): v["level"] for n in sol for f, v in sol[n].items()}
    train = [t for t in T if t["split"] == "train"]
    return train, level


# ------------------------------------------------------------------ vectorised network evaluation
def draw(rng, shape, n, R):
    """Integer networks, shape = leading dims. Returns (W1 [..,H,n], b1 [..,H], W2 [..,H], b2 [..])."""
    H = tasks.HIDDEN[n]
    W1 = rng.integers(-R, R + 1, size=shape + (H, n))
    b1 = rng.integers(-R, R + 1, size=shape + (H,))
    W2 = rng.integers(-R, R + 1, size=shape + (H,))
    b2 = rng.integers(-R, R + 1, size=shape)
    return W1, b1, W2, b2


def run_many(W1, b1, W2, b2, n):
    """Truth tables (int) of many integer networks, exactly as tasks.run (integer arithmetic: y > 0.5 <=> y >= 1)."""
    X = tasks.inputs(n)                                                   # 2^n x n
    h = np.maximum(0, np.einsum("rn,...hn->...rh", X, W1) + b1[..., None, :])
    y = np.einsum("...rh,...h->...r", h, W2) + b2[..., None]
    return ((y > 0.5).astype(np.int64) << np.arange(2 ** n)).sum(-1)


def verify(n_nets=400, seed=123):
    rng = np.random.default_rng(seed)
    for n in (2, 3):
        for R in (1, 2, 3):
            W1, b1, W2, b2 = draw(rng, (n_nets,), n, R)
            fv = run_many(W1, b1, W2, b2, n)
            for i in range(n_nets):
                net = (W1[i].astype(float), b1[i].astype(float), W2[i].astype(float), float(b2[i]))
                assert tasks.run(net, n) == int(fv[i]), (n, R, i)
                assert tasks.run(tasks.parse(tasks.render(net), n), n) == int(fv[i])
    return 2 * 3 * n_nets


# ------------------------------------------------------------------ one simulation
def simulate(train, level, R, seed, rounds, k=16):
    rng = np.random.default_rng([seed, R])
    out = {}
    disc_total = np.zeros(rounds, dtype=np.int64)
    disc_level = {}                                     # level -> per-round cumulative counts
    solved_round = []                                   # per n: rounds x tasks
    hit_count = 0.0
    for n in (2, 3):
        idx = [i for i, t in enumerate(train) if t["n"] == n]
        fs = np.array([train[i]["f"] for i in idx])
        W1, b1, W2, b2 = draw(rng, (rounds, len(idx), k), n, R)
        F = run_many(W1, b1, W2, b2, n)                                    # rounds x tasks x k
        nf = 2 ** (2 ** n)
        # (a) discovery: which functions appear in each round, cumulative over rounds, restricted to train fns
        seen = np.zeros((rounds, nf), dtype=bool)
        for r in range(rounds):
            seen[r, np.unique(F[r])] = True
        seen = np.logical_or.accumulate(seen, axis=0)
        is_train = np.zeros(nf, dtype=bool)
        is_train[fs] = True
        disc_total += (seen & is_train).sum(1)
        for f in fs:
            L = level[(n, int(f))]
            disc_level.setdefault(L, np.zeros(rounds, dtype=np.int64))
            disc_level[L] += seen[:, f]
        # (b) own-slot success
        solved_round.append((F == fs[None, :, None]).any(-1))               # rounds x tasks
        # (c) first-try: frequency of each function among all networks of this shape
        freq = np.bincount(F.ravel(), minlength=nf) / F.size
        hit_count += freq[fs].sum()
    S = np.concatenate(solved_round, 1)                                     # rounds x 205
    out["discovery"] = disc_total.tolist()
    out["discovery_by_level"] = {str(L): v.tolist() for L, v in sorted(disc_level.items())}
    out["solved16_per_round"] = S.mean(1).tolist()
    out["solved_cum"] = np.logical_or.accumulate(S, axis=0).mean(1).tolist()
    out["first_try"] = hit_count / len(train)
    return out


def stats(xs):
    a = np.asarray(xs, dtype=float)
    return {"mean": a.mean(0).tolist(), "min": a.min(0).tolist(), "max": a.max(0).tolist()}


# ------------------------------------------------------------------ real runs
def load_rounds(d):
    out = {}
    for p in glob.glob(os.path.join(d, "round*.json")):
        out[int(re.search(r"round(\d+)\.json$", p).group(1))] = p
    return out


def real_runs(runs, train, level):
    """Model results aligned to sampling rounds. Sampling round j (1-based) = the train samples of JSON round j-1
    (round 0 = base pilot). Hindsight pool at JSON round r was built from sampling rounds 1..r."""
    base = os.path.join(runs, "base", "round0.json")
    hs = load_rounds(os.path.join(runs, "hindsight"))
    if not os.path.exists(base):
        return None
    train_keys = [(t["n"], t["f"]) for t in train]
    res = {"runs": runs, "hindsight": {}, "sampling": {}}
    paths = {0: base, **hs}
    cum = None
    for j in range(0, max(paths) + 1):
        if j not in paths:
            break
        d = json.load(open(paths[j]))
        if d.get("empty_pool"):
            break
        if j > 0:                                             # pool of round j: built from sampling rounds 1..j
            lv = {}
            for key, v in d["pool"].items():
                if v:
                    n, f = map(int, key.split(":"))
                    lv[str(level[(n, f)])] = lv.get(str(level[(n, f)]), 0) + 1
            res["hindsight"][j] = {"pool_functions": d["pool_functions"], "pool_size": d["pool_size"],
                                   "by_level": lv}
        recs = {(r["n"], r["f"]): r for r in d["tasks"] if r["split"] == "train"}
        if set(recs) != set(train_keys):
            continue
        ok = np.array([any(s["passed"] for s in recs[key]["samples"][:16]) for key in train_keys])
        cum = ok if cum is None else (cum | ok)
        m = d["metrics"]["train"]
        res["sampling"][j + 1] = {"json_round": j, "first_try": m["mean_pass"], "solved16": float(ok.mean()),
                                  "solved_cum": float(cum.mean())}
    return res


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--ranges", default="1,2,3")
    ap.add_argument("--k", type=int, default=16, help="random networks per train task per round")
    ap.add_argument("--runs", default=os.path.join(HERE, "runs_gpu", "main"), help="real runs to compare with")
    ap.add_argument("--out", default=os.path.join(HERE, "random_baseline.json"))
    a = ap.parse_args()
    nver = verify()
    train, level = load_train()
    assert len(train) == 205 and not ({(t["n"], t["f"]) for t in train} & FEWSHOT_SET)
    Rs = [int(x) for x in a.ranges.split(",")]
    lv_total = {}
    for t in train:
        lv_total[str(level[(t["n"], t["f"])])] = lv_total.get(str(level[(t["n"], t["f"])]), 0) + 1
    res = {"config": {"rounds": a.rounds, "seeds": a.seeds, "k": a.k, "ranges": Rs, "n_train": len(train),
                      "nets_per_round": a.k * len(train), "verified_against_tasks_run": nver},
           "train_functions_by_level": lv_total, "random": {}}
    for R in Rs:
        sims = [simulate(train, level, R, s, a.rounds, a.k) for s in range(a.seeds)]
        res["random"][str(R)] = {
            "discovery": stats([x["discovery"] for x in sims]),
            "discovery_by_level": {L: stats([x["discovery_by_level"][L] for x in sims]) for L in sims[0]["discovery_by_level"]},
            "solved16_per_round": stats([x["solved16_per_round"] for x in sims]),
            "solved_cum": stats([x["solved_cum"] for x in sims]),
            "first_try": stats([x["first_try"] for x in sims]),
        }
    real = real_runs(a.runs, train, level) if a.runs else None
    res["real"] = real
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)

    # ---------------- tables
    rr = res["random"]
    mm = lambda st, i, fmt="{:.1f}": (fmt.format(st["mean"][i]) + " (" + fmt.format(st["min"][i]) + "-" +
                                      fmt.format(st["max"][i]) + ")")
    print(f"verified vectorised evaluation against tasks.run on {nver} random networks\n")
    print(f"## Discovery: distinct train functions found (of {len(train)}), cumulative, after r rounds of "
          f"{a.k * len(train)} networks; random = mean (min-max) over {a.seeds} seeds\n")
    print("| round | hindsight pool fns | " + " | ".join(f"random R={R}" for R in Rs) + " |")
    print("|---|---|" + "---|" * len(Rs))
    hsd = (real or {}).get("hindsight", {})
    for i in range(a.rounds):
        r = i + 1
        h = hsd.get(r, {}).get("pool_functions", "-")
        print(f"| {r} | {h} | " + " | ".join(mm(rr[str(R)]["discovery"], i) for R in Rs) + " |")
    print(f"\n## Discovery by level (train functions per level: {lv_total})\n")
    lvls = sorted(lv_total, key=int)
    print("| round | source | " + " | ".join(f"level {L} (/{lv_total[L]})" for L in lvls) + " | total |")
    print("|---|---|" + "---|" * (len(lvls) + 1))
    for r in [x for x in (1, 10, 30) if x <= a.rounds]:
        if r in hsd:
            h = hsd[r]
            print(f"| {r} | hindsight | " + " | ".join(str(h["by_level"].get(L, 0)) for L in lvls) +
                  f" | {h['pool_functions']} |")
        for R in Rs:
            st = rr[str(R)]
            print(f"| {r} | random R={R} | " + " | ".join(mm(st["discovery_by_level"].get(L, {"mean": [0] * a.rounds,
                  "min": [0] * a.rounds, "max": [0] * a.rounds}), r - 1) for L in lvls) +
                  f" | {mm(st['discovery'], r - 1)} |")
    print("\n## Per-task success (own 16 tries per task), train tasks\n")
    print("| R | first-try success | solved within 16 (per round) | solved at least once after 1 / 10 / 30 rounds |")
    print("|---|---|---|---|")
    for R in Rs:
        st = rr[str(R)]
        ft = st["first_try"]
        cum = " / ".join(f"{st['solved_cum']['mean'][r - 1]:.3f}" for r in (1, 10, 30) if r <= a.rounds)
        s16 = np.mean(st["solved16_per_round"]["mean"])
        lo, hi = min(st["solved16_per_round"]["min"]), max(st["solved16_per_round"]["max"])
        print(f"| {R} | {ft['mean']:.4f} ({ft['min']:.4f}-{ft['max']:.4f}) | {s16:.3f} ({lo:.3f}-{hi:.3f}) | {cum} |")
    if real and real["sampling"]:
        print("\n## Model (hindsight run) per sampling round vs random (mean over seeds)\n")
        print("sampling round j = train samples of JSON round j-1 (j=1: base model)\n")
        print("| sampling round | model first-try | model solved@16 | model solved-ever | " +
              " | ".join(f"R={R} solved@16 | R={R} solved-ever" for R in Rs) + " |")
        print("|---|---|---|---|" + "---|---|" * len(Rs))
        for j, s in sorted(real["sampling"].items()):
            if j > a.rounds:
                break
            cells = " | ".join(f"{rr[str(R)]['solved16_per_round']['mean'][j - 1]:.3f} | "
                               f"{rr[str(R)]['solved_cum']['mean'][j - 1]:.3f}" for R in Rs)
            print(f"| {j} | {s['first_try']:.4f} | {s['solved16']:.3f} | {s['solved_cum']:.3f} | {cells} |")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
