"""Self-chosen exploration temperature: the hindsight loop, but every answer is preceded by a control line
"T = <value>" that sets the temperature used to sample that answer, and the model learns to write that line.

Prompt format: tasks.py's header and task text, with a "T = <v>" line between the truth table and W1 in each
few-shot example (T = 0.8, 1.1, 1.4, 0.5 in tasks.FEWSHOT order) and in each prompt. This script installs that
prefix as tasks.PREFIX at runtime (no file is edited), so wsp.prefix_ids() / wsp.generate (FastGen) /
wsp.finetune share it; every prompt passed to wsp is asserted to start with those prefix tokens.

One sample for task t:
  1. choose T by --policy
       self       with prob. --epsilon pick uniformly from --menu (forced); otherwise generate from
                  prompt(t) + "T =" at temperature 1.0, read up to the first newline, parse a float, snap to the
                  nearest menu value (unparseable -> uniform pick, invalid_choice)
       uniform    always a uniform pick from the menu (recorded as forced: the model did not write it)
       fixed:<v>  always v (recorded as forced)
  2. generate the answer from prompt(t) + "T = <v>\\n" at temperature v (wsp.generate, one call per v), score it.
  Greedy answer: T line = the model's greedy choice (self; 1.1 if unparseable), 1.1 (uniform), v (fixed:v);
  the answer itself is greedy.
Training set of round r (identical rule for all policies):
  answer pool   wsp.update_pool("hindsight", ...) on round r-1's TRAIN-task samples; a pooled answer for g is
                trained as (prompt(g) + "T = <T it was first generated with>\\n", answer)
  controller    a sample is a progress event if its rendered answer entered the pool in this update; each event
                gives (prompt(t) + "T =", " <v>") (loss only on the temperature); at most 2 per task per round
                (first in sample order); events of the last --window source rounds are kept.
  Steps wsp.default_steps(len(set)) unless --steps; wsp.train_round from the base model (adapter saved, resumable).
Output: <out-dir>/<arm>/round<r>.json (r = 0..R) in the loop's round format plus policy/menu/epsilon/window,
t_stats, prev_round_progress_by_T, new_functions, controller_examples, pool. Rounds are resume-safe: existing
round JSONs are replayed (the pool and controller set are rebuilt from the stored samples and checked).

Usage (pod):
  python selfexplore.py --policy self      --device cuda --seed 0 --rank 16 --lr 2e-4 --k 16 --epochs 40 \\
      --pool-cap 4 --max-steps 2000 --rounds 10 --out-dir runs_gpu/selfexplore
  python selfexplore.py --policy uniform   ...same flags...
  python selfexplore.py --policy fixed:1.4 ...same flags...
  python selfexplore.py --selftest         # model-free unit checks (tokenizer only)
"""
import argparse
import collections
import json
import math
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

import tasks
import wsp

FEWSHOT_T = (0.8, 1.1, 1.4, 0.5)          # T lines of the four few-shot examples, tasks.FEWSHOT order
CHOICE_TOKENS = 8                          # " 1.35\n" is 6 tokens (Qwen splits digits)
GREEDY_FALLBACK_T = 1.1
ORIG_PREFIX = tasks.PREFIX


def fmt_t(v):
    """0.8 -> "0.8", 1.0 -> "1.0", 1.25 -> "1.25"."""
    s = f"{float(v):.4f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def t_line(v):
    return f"T = {fmt_t(v)}\n"


SE_PREFIX = tasks.HEADER + "".join(tasks.task_text(f, n) + t_line(T) + tasks.render(w) + "\n\n"
                                   for (n, f, w), T in zip(tasks.FEWSHOT, FEWSHOT_T))


def install_prefix():
    """Make wsp use this script's few-shot prefix in this process (wsp.prefix_ids() reads tasks.PREFIX)."""
    tasks.PREFIX = SE_PREFIX


def prompt(f, n):
    return SE_PREFIX + tasks.task_text(f, n)


_PREF_IDS = {}


def check_prompts(prompts):
    """Assert that every prompt tokenizes to wsp's shared prefix tokens + a non-empty suffix (else wsp.generate
    would silently fall back to the slow HF path and wsp.finetune to no shared prefix)."""
    assert tasks.PREFIX == SE_PREFIX, "tasks.PREFIX is not the self-explore prefix (install_prefix not called?)"
    if "ids" not in _PREF_IDS:
        _PREF_IDS["ids"] = wsp.prefix_ids()
    pref = _PREF_IDS["ids"]
    assert pref == wsp.prefix_ids()
    tok = wsp.get_tok()
    for p in set(prompts):
        ids = tok(p).input_ids
        assert ids[:len(pref)] == pref and len(ids) > len(pref), f"prompt does not start with the prefix: ...{p[-80:]!r}"


def snap(v, menu):
    return min(menu, key=lambda m: (abs(m - v), m))


def parse_choice(text, menu):
    """Generated continuation of "... T =" -> menu value, or None if it does not parse as a finite float."""
    s = text.split("\n")[0].strip()
    try:
        v = float(s)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return snap(v, menu)


class Policy:
    def __init__(self, spec, menu, epsilon):
        self.spec, self.menu, self.epsilon = spec, list(menu), epsilon
        if spec == "self":
            self.kind, self.value, self.arm = "self", None, "selfT"
        elif spec == "uniform":
            self.kind, self.value, self.arm = "uniform", None, "uniformT"
        elif spec.startswith("fixed:"):
            self.kind, self.value = "fixed", float(spec.split(":", 1)[1])
            self.arm = f"fixedT{fmt_t(self.value)}"
        else:
            raise ValueError(f"--policy must be self, uniform or fixed:<v>, got {spec!r}")


# ===================================================================== sampling (one round)
def sample_round(model, task_list, a, pol, rnd, seed):
    """k samples per task with a per-sample temperature chosen by the policy, plus one greedy answer per task.
    Returns a dict in wsp.evaluate's format (metrics via wsp.metrics); samples carry T / forced / invalid_choice."""
    tok = wsp.get_tok()
    mnt = wsp.max_new_tokens_default()
    N, K, menu = len(task_list), a.k, pol.menu
    rng = np.random.default_rng([a.seed, rnd, 4242])
    base = [prompt(t["f"], t["n"]) for t in task_list]
    T = [[None] * K for _ in range(N)]
    forced = [[True] * K for _ in range(N)]
    invalid = [[False] * K for _ in range(N)]
    ttext = [[None] * K for _ in range(N)]
    gT, gtext = [None] * N, [None] * N
    t0 = time.time()
    wsp.log(f"sample round {rnd}: {N} tasks, k={K}, policy {pol.spec}, menu {menu}, epsilon {pol.epsilon}, max_new={mnt}")
    coin = rng.random((N, K))
    pick = rng.integers(len(menu), size=(N, K))
    if pol.kind == "self":
        cp = [b + "T =" for b in base]
        check_prompts(cp)
        ch = wsp.generate(model, tok, cp, K, 1.0, a.batch_size, CHOICE_TOKENS, seed=seed + 1, progress=False)
        for i in range(N):
            for j in range(K):
                if coin[i, j] < pol.epsilon:
                    T[i][j] = menu[pick[i, j]]
                    continue
                forced[i][j] = False
                ttext[i][j] = ch[i][j]
                v = parse_choice(ch[i][j], menu)
                if v is None:
                    invalid[i][j] = True
                    v = menu[pick[i, j]]
                T[i][j] = v
        gch = wsp.generate(model, tok, cp, 1, 0.0, a.batch_size, CHOICE_TOKENS, seed=seed, progress=False)
        for i in range(N):
            gtext[i] = gch[i][0]
            v = parse_choice(gch[i][0], menu)
            gT[i] = GREEDY_FALLBACK_T if v is None else v
    elif pol.kind == "uniform":
        T = [[menu[pick[i, j]] for j in range(K)] for i in range(N)]
        gT = [GREEDY_FALLBACK_T] * N
    else:
        T = [[pol.value] * K for _ in range(N)]
        gT = [pol.value] * N
    texts = [[None] * K for _ in range(N)]
    for v in sorted({x for row in T for x in row}):
        items = [(i, j) for i in range(N) for j in range(K) if T[i][j] == v]
        ps = [base[i] + t_line(v) for i, _ in items]
        check_prompts(ps)
        wsp.log(f"  T={fmt_t(v)}: {len(items)} samples")
        out = wsp.generate(model, tok, ps, 1, v, a.batch_size, mnt, seed=seed + 100 + round(100 * v), progress=False)
        for (i, j), o in zip(items, out):
            texts[i][j] = o[0]
    gp = [base[i] + t_line(gT[i]) for i in range(N)]
    check_prompts(gp)
    gout = wsp.generate(model, tok, gp, 1, 0.0, a.batch_size, mnt, seed=seed, progress=False)
    recs = []
    for i, t in enumerate(task_list):
        n, f = t["n"], t["f"]
        r = dict(t)
        r["samples"] = [{**wsp.score(texts[i][j], n, f), "T": T[i][j], "forced": forced[i][j],
                         "invalid_choice": invalid[i][j], "T_text": ttext[i][j]} for j in range(K)]
        r["greedy"] = {**wsp.score(gout[i][0], n, f), "T": gT[i], "T_text": gtext[i]}
        recs.append(r)
    met = wsp.metrics(recs, K)
    wsp.log(f"sample round {rnd} done in {time.time() - t0:.0f}s")
    wsp.log_mem("after evaluate")
    return {"k": K, "temperature": pol.value, "max_new_tokens": mnt, "eval_seconds": time.time() - t0,
            "metrics": met, "tasks": recs}


# ===================================================================== pool, progress events, controller set
def _rendered(s, n):
    return tasks.render(tasks.parse(s["text"], n))


def _pool_set(pool):
    return {(key, e["text"]) for key, lst in pool.items() for e in lst}


def progress_events(old, new, round_json, train_set, src_round, cap=2):
    """Samples (TRAIN-task prompts) whose rendered answer is in `new` and not in `old`.
    Returns (all events, events capped at `cap` per task in sample order)."""
    olds, news = _pool_set(old), _pool_set(new)
    ev_all, ev_cap = [], []
    for r in round_json["tasks"]:
        if r["split"] != "train":
            continue
        n, kept = r["n"], 0
        for j, s in enumerate(r["samples"]):
            if not s["valid"] or (n, s["tt"]) not in train_set:
                continue
            item = (wsp._ex_key(n, s["tt"]), _rendered(s, n))
            if item in news and item not in olds:
                e = {"n": n, "f": r["f"], "j": j, "T": s["T"], "forced": s.get("forced"), "fn": s["tt"],
                     "text": item[1], "round": src_round}
                ev_all.append(e)
                if kept < cap:
                    ev_cap.append(e); kept += 1
    return ev_all, ev_cap


def by_t(events):
    c = collections.Counter(fmt_t(e["T"]) for e in events)
    return dict(sorted(c.items(), key=lambda x: float(x[0])))


class State:
    """Pool, answer -> first temperature side table, and per-source-round controller events, built by folding in
    the rounds' samples one at a time (replayed from the JSONs on resume)."""

    def __init__(self, train_set, window):
        self.train_set, self.window = train_set, window
        self.pool, self.T_of, self.events, self.solved = {}, {}, {}, set()

    def consume(self, round_json, src_round):
        old = self.pool
        new = wsp.update_pool("hindsight", old, round_json, self.train_set)
        for r in round_json["tasks"]:
            if r["split"] != "train":
                continue
            n = r["n"]
            for s in r["samples"]:
                if s["valid"] and (n, s["tt"]) in self.train_set:
                    self.T_of.setdefault((n, _rendered(s, n)), s["T"])
        ev_all, ev_cap = progress_events(old, new, round_json, self.train_set, src_round)
        self.events[src_round] = ev_cap
        # an answer that is not in the pool now can never enter it later (the cap keeps the smallest), so only
        # pool members need their temperature
        keep = {(int(key.split(":")[0]), text) for key, text in _pool_set(new)}
        self.T_of = {k: v for k, v in self.T_of.items() if k in keep}
        self.pool = new
        self.solved |= {(r["n"], r["f"]) for r in round_json["tasks"]
                        if r["split"] == "train" and any(s["passed"] for s in r["samples"])}
        return {"new_function_keys": sorted(set(new) - set(old), key=lambda s: tuple(map(int, s.split(":")))),
                "progress_by_T": by_t(ev_all), "progress_capped_by_T": by_t(ev_cap),
                "n_progress": len(ev_all), "n_progress_capped": len(ev_cap)}

    def pool_T(self):
        return {key: [self.T_of[(int(key.split(":")[0]), e["text"])] for e in lst] for key, lst in self.pool.items()}

    def answer_examples(self):
        ex = []
        for key in sorted(self.pool, key=lambda s: tuple(map(int, s.split(":")))):
            n, f = map(int, key.split(":"))
            for e in self.pool[key]:
                ex.append((prompt(f, n) + t_line(self.T_of[(n, e["text"])]), e["text"]))
        return ex

    def controller_events(self, rnd):
        """Events from source rounds rnd-window .. rnd-1 (training set of round rnd)."""
        return [e for src in range(max(0, rnd - self.window), rnd) for e in self.events.get(src, [])]

    @staticmethod
    def controller_examples(events):
        return [(prompt(e["f"], e["n"]) + "T =", " " + fmt_t(e["T"])) for e in events]


def t_stats(round_json, pool_before, solved_before, train_set):
    """Temperature statistics over the round's TRAIN-task samples. Progress events are computed against the pool
    before this round's samples are added (= the events these samples produce in the next round's update)."""
    TR = [r for r in round_json["tasks"] if r["split"] == "train"]
    S = [s for r in TR for s in r["samples"]]
    new = wsp.update_pool("hindsight", pool_before, round_json, train_set)
    ev_all, ev_cap = progress_events(pool_before, new, round_json, train_set, None)
    pa, pc = by_t(ev_all), by_t(ev_cap)
    hist = lambda xs: dict(sorted(collections.Counter(fmt_t(x) for x in xs).items(), key=lambda x: float(x[0])))
    mean = lambda xs: float(np.mean(xs)) if xs else None
    own = lambda s: not s["forced"] and not s["invalid_choice"]
    attempts = [s for s in S if not s["forced"]]
    own_T = [s["T"] for s in S if own(s)]
    own_solved = [s["T"] for r in TR if (r["n"], r["f"]) in solved_before for s in r["samples"] if own(s)]
    own_unsolved = [s["T"] for r in TR if (r["n"], r["f"]) not in solved_before for s in r["samples"] if own(s)]
    per_T = {}
    for v in sorted({s["T"] for s in S}):
        sv = [s for s in S if s["T"] == v]
        key = fmt_t(v)
        per_T[key] = {"n": len(sv), "valid_rate": mean([s["valid"] for s in sv]), "pass_rate": mean([s["passed"] for s in sv]),
                      "progress": pa.get(key, 0), "progress_capped": pc.get(key, 0)}
    return {"n_samples": len(S), "hist_all": hist([s["T"] for s in S]),
            "n_forced": sum(s["forced"] for s in S), "n_model_choices": len(attempts),
            "invalid_rate": mean([s["invalid_choice"] for s in attempts]),
            "own_hist": hist(own_T), "own_mean": mean(own_T),
            "own_mean_solved_before": mean(own_solved), "own_n_solved_before": len(own_solved),
            "own_mean_never_solved": mean(own_unsolved), "own_n_never_solved": len(own_unsolved),
            "n_tasks_solved_before": len({(r["n"], r["f"]) for r in TR} & solved_before),
            "per_T": per_T, "n_progress": len(ev_all), "n_progress_capped": len(ev_cap)}


# ===================================================================== loop
def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--policy", default="self", help="self | uniform | fixed:<v>")
    ap.add_argument("--menu", default="0.5,0.8,1.1,1.4,1.7", help="comma-separated temperature menu")
    ap.add_argument("--epsilon", type=float, default=0.25, help="self: probability of a forced uniform pick")
    ap.add_argument("--window", type=int, default=3, help="source rounds whose controller events are kept")
    ap.add_argument("--arm-name", default=None, help="default selfT / uniformT / fixedT<v>")
    ap.add_argument("--selftest", action="store_true", help="model-free unit checks, then exit")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--heldout-k", type=int, default=64, help="recorded in config only (not used)")
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


def summary_line(arm, r, ev, nfun, new_fns):
    om = ev["t_stats"]["own_mean"]
    return (wsp.summary(f"{arm} r{r}", ev) + f" | pool {ev['pool_size']} ({nfun} fns)"
            + f" | T own-mean {'-' if om is None else f'{om:.3f}'} | new fns {new_fns}")


def main():
    a = parser().parse_args()
    menu = [float(x) for x in a.menu.split(",") if x.strip()]
    pol = Policy(a.policy, menu, a.epsilon)
    install_prefix()
    if a.selftest:
        wsp.POOL_CAP = a.pool_cap
        selftest(a, pol)
        return
    wsp.configure(a)
    a.temperature = pol.value                  # run_config records it (None = per-sample temperature)
    if not os.path.isabs(a.out_dir):
        a.out_dir = os.path.join(wsp.HERE, a.out_dir)
    arm = a.arm_name or pol.arm
    extra_cfg = {"policy": pol.spec, "menu": menu, "epsilon": a.epsilon, "window": a.window,
                 "fewshot_T": list(FEWSHOT_T), "choice_tokens": CHOICE_TOKENS}
    head = {"policy": pol.spec, "menu": menu, "epsilon": a.epsilon, "window": a.window}
    tr, ho = wsp.task_lists(a.limit)
    train_set = {(t["n"], t["f"]) for t in wsp.load_split() if t["split"] == "train"}
    st = State(train_set, a.window)
    pth = lambda r: os.path.join(a.out_dir, arm, f"round{r}.json")

    if os.path.exists(pth(0)):
        last = json.load(open(pth(0)))
        wsp.log("exists, skipping:", pth(0))
    else:
        model = wsp.load_model(wsp.GEN_DTYPE)
        last = sample_round(model, tr + ho, a, pol, 0, a.seed)
        last.update({"stage": "selfexplore", "arm": arm, "round": 0, **head,
                     "t_stats": t_stats(last, {}, set(), train_set), "prev_round_progress_by_T": {},
                     "new_functions": 0, "new_function_keys": [], "controller_examples": 0, "controller_set": [],
                     "pool_size": 0, "pool_functions": 0, "pool": {}, "pool_T": {}, "train_examples": 0,
                     "config": {**wsp.run_config(a), **extra_cfg}})
        wsp.dump(last, pth(0))
        del model; wsp.free()
    print(summary_line(arm, 0, last, 0, 0), flush=True)

    for r in range(1, a.rounds + 1):
        info = st.consume(last, r - 1)
        want = {key: v["progress"] for key, v in last["t_stats"]["per_T"].items() if v["progress"]}
        assert info["progress_by_T"] == want, ("progress events differ from the previous round's t_stats",
                                               info["progress_by_T"], want)
        nfun = sum(1 for v in st.pool.values() if v)
        ex_pool = st.answer_examples()
        ctrl = st.controller_events(r)
        ex = ex_pool + State.controller_examples(ctrl)
        path = pth(r)
        if os.path.exists(path):
            d = json.load(open(path))
            if d.get("empty_pool"):
                wsp.log(f"{path}: empty pool, arm stopped"); return
            assert d["pool"] == st.pool and d["pool_T"] == st.pool_T(), f"{path}: stored pool differs from the replay"
            assert d["train_examples"] == len(ex), f"{path}: stored training set size differs from the replay"
            wsp.log("exists, skipping:", path)
            print(summary_line(arm, r, d, nfun, len(info["new_function_keys"])), flush=True)
            last = d
            continue
        wsp.log(f"[{arm} r{r}] pool {len(ex_pool)} examples over {nfun} functions, {len(ctrl)} controller examples "
                f"(window {a.window}), {len(info['new_function_keys'])} new functions, progress by T "
                f"{info['progress_by_T']} (from round {r - 1})")
        if not ex_pool:
            wsp.dump({"stage": "selfexplore", "arm": arm, "round": r, "empty_pool": True, "pool_size": 0,
                      "pool_functions": 0, **head, "config": {**wsp.run_config(a), **extra_cfg}}, path)
            print(f"[{arm} r{r}] empty pool, stopping arm", flush=True)
            return
        check_prompts([p for p, _ in ex])
        steps = a.steps or wsp.default_steps(len(ex), a.train_batch)
        model, tinfo = wsp.train_round(a, ex, steps, a.seed + r, os.path.join(a.out_dir, arm, f"round{r}.adapter.pt"))
        ev = sample_round(model, tr + ho, a, pol, r, a.seed + 1000 * r)
        del model; wsp.free()
        ev.update({"stage": "selfexplore", "arm": arm, "round": r, **head,
                   "t_stats": t_stats(ev, st.pool, st.solved, train_set),
                   "prev_round_progress_by_T": info["progress_by_T"],
                   "prev_round_progress_capped_by_T": info["progress_capped_by_T"],
                   "new_functions": len(info["new_function_keys"]), "new_function_keys": info["new_function_keys"],
                   "controller_examples": len(ctrl),
                   "controller_set": [{key: e[key] for key in ("round", "n", "f", "j", "T", "forced", "fn")} for e in ctrl],
                   "pool_size": len(ex_pool), "pool_functions": nfun, "pool": st.pool, "pool_T": st.pool_T(),
                   "train_examples": len(ex), **tinfo, "config": {**wsp.run_config(a), **extra_cfg}})
        wsp.dump(ev, path)
        print(summary_line(arm, r, ev, nfun, len(info["new_function_keys"])), flush=True)
        last = ev


# ===================================================================== model-free checks
def selftest(a, pol):
    menu = pol.menu
    ok = lambda cond, msg: (print(("ok   " if cond else "FAIL ") + msg, flush=True), cond)[1]
    good = True
    tok = wsp.get_tok()
    # snapping / parsing of the temperature line
    cases = {" 1.4": 1.4, " 1.35": 1.4, " abc": None, "": None, " 0.95": 0.8, " 9": 1.7, " 0.5\nW1 = [[1]]": 0.5,
             " nan": None, " inf": None, "1.1": 1.1, " -3": 0.5}
    got = {s: parse_choice(s, menu) for s in cases}
    good &= ok(got == cases, f"parse_choice/snap {got}")
    # prefix: only the T lines differ from tasks.py's prefix; wsp sees it
    stripped = SE_PREFIX
    for T in FEWSHOT_T:
        stripped = stripped.replace(t_line(T), "", 1)
    good &= ok(stripped == ORIG_PREFIX and tasks.PREFIX == SE_PREFIX and wsp.prefix_ids() == tok(SE_PREFIX).input_ids,
               "SE_PREFIX = tasks.py prefix + T lines (0.8, 1.1, 1.4, 0.5); tasks.PREFIX patched; wsp.prefix_ids() uses it")
    good &= ok(all(ln in SE_PREFIX for ln in ("1 -> 1\nT = 0.8\nW1", "1 -> 0\nT = 1.1\nW1", "1 -> 1\nT = 1.4\nW1",
                                                  "1 -> 0\nT = 0.5\nW1")), "T line sits between truth table and W1")
    # answers generated after a T line still parse (continuation text as wsp.generate returns it)
    T_all = wsp.load_split()
    nparse = 0
    for t in T_all:
        n, f = t["n"], t["f"]
        ans = wsp.SOL[str(n)][str(f)]["answer"]
        cont = ans + "\n\nTask: 3 inputs, 3 hidden units.\n000 -> 1"
        net = tasks.parse(tasks.first_block(cont), n)
        net2 = tasks.parse("T = 1.4\n" + ans, n)
        nparse += net is not None and tasks.run(net, n) == f and net2 is not None and tasks.run(net2, n) == f
    good &= ok(nparse == len(T_all), f"tasks.parse on answers after a T line: {nparse}/{len(T_all)}")
    # token boundaries: separate prompt/target tokenization == joint tokenization (train/inference consistency)
    p = prompt(77, 3)
    j1 = tok(p + "T =").input_ids + tok(" 1.4\n\n").input_ids == tok(p + "T = 1.4\n\n").input_ids
    ans = wsp.SOL["3"]["77"]["answer"]
    j2 = tok(p + t_line(1.4)).input_ids + tok(ans + "\n\n").input_ids == tok(p + t_line(1.4) + ans + "\n\n").input_ids
    good &= ok(j1 and j2, f"tokenization of controller target / answer after T line equals joint tokenization: {j1}, {j2}")
    try:
        check_prompts([prompt(t["f"], t["n"]) + s for t in T_all for s in ("T =", t_line(0.5), t_line(1.7))])
        good &= ok(True, f"all {3 * len(T_all)} choice/answer prompts start with wsp.prefix_ids() ({len(wsp.prefix_ids())} tokens)")
    except AssertionError as e:
        good &= ok(False, f"prefix check: {e}")

    # synthetic rounds: random networks (like random_arm.py) with random temperatures and forced flags
    train_set = {(t["n"], t["f"]) for t in T_all if t["split"] == "train"}
    rng = np.random.default_rng(1)
    W = 2
    st = State(train_set, W)
    ref_pool, rounds = {}, []
    def rnet(n):
        H = tasks.HIDDEN[n]
        return (rng.integers(-1, 2, (H, n)).astype(float), rng.integers(-1, 2, H).astype(float),
                rng.integers(-1, 2, H).astype(float), float(rng.integers(-1, 2)))
    for rd in range(5):
        recs = []
        for t in T_all:
            ss = []
            for j in range(6):
                txt = tasks.render(rnet(t["n"])) if rng.random() > 0.15 else "W1 = garbage"
                ss.append({**wsp.score(txt, t["n"], t["f"]), "T": menu[rng.integers(len(menu))],
                           "forced": bool(rng.random() < 0.25), "invalid_choice": False, "T_text": None})
            recs.append({**t, "samples": ss, "greedy": ss[0]})
        rj = {"tasks": recs}
        rounds.append(rj)
        before = {k: list(v) for k, v in st.pool.items()}
        stats = t_stats(rj, st.pool, st.solved, train_set)
        info = st.consume(rj, rd)
        ref_pool = wsp.update_pool("hindsight", ref_pool, rj, train_set)
        good &= ok(st.pool == ref_pool, f"round {rd}: pool == chained wsp.update_pool('hindsight') "
                                        f"({sum(map(len, st.pool.values()))} answers, {len(st.pool)} fns)")
        ev_all, ev_cap = progress_events(before, st.pool, rj, train_set, rd)
        added = _pool_set(st.pool) - _pool_set(before)
        good &= ok({(wsp._ex_key(e["n"], e["fn"]), e["text"]) for e in ev_all} == added,
                   f"round {rd}: progress events == newly added pool answers ({len(added)} answers, {len(ev_all)} events)")
        per_task = collections.Counter((e["n"], e["f"]) for e in ev_all)
        cap_task = collections.Counter((e["n"], e["f"]) for e in ev_cap)
        first2 = all([e["j"] for e in ev_cap if (e["n"], e["f"]) == key] ==
                     sorted(e["j"] for e in ev_all if (e["n"], e["f"]) == key)[:2] for key in per_task)
        good &= ok(all(cap_task[key] == min(2, c) for key, c in per_task.items()) and first2 and st.events[rd] == ev_cap,
                   f"round {rd}: 2-per-task cap keeps the first two in sample order "
                   f"({len(ev_all)} -> {len(ev_cap)}, max per task {max(per_task.values(), default=0)})")
        good &= ok(info["progress_by_T"] == {k: v["progress"] for k, v in stats["per_T"].items() if v["progress"]},
                   f"round {rd}: t_stats progress by T (before) == events at the next update {info['progress_by_T']}")
    # window
    ctrl = st.controller_events(5)
    good &= ok({e["round"] for e in ctrl} == {3, 4} and len(ctrl) == len(st.events[3]) + len(st.events[4]),
               f"window {W}: round-5 controller set uses source rounds {sorted({e['round'] for e in ctrl})}, {len(ctrl)} examples")
    # T side table: first temperature an answer was generated with (all rounds, sample order)
    first = {}
    for rj in rounds:
        for r in rj["tasks"]:
            if r["split"] == "train":
                for s in r["samples"]:
                    if s["valid"] and (r["n"], s["tt"]) in train_set:
                        first.setdefault((r["n"], _rendered(s, r["n"])), s["T"])
    good &= ok(all(st.T_of[(int(k.split(':')[0]), e["text"])] == first[(int(k.split(':')[0]), e["text"])]
                   for k, lst in st.pool.items() for e in lst), "pool answers carry the T they were first generated with")
    exs = st.answer_examples() + State.controller_examples(ctrl)
    try:
        check_prompts([p for p, _ in exs])
        good &= ok(True, f"{len(exs)} training prompts start with the prefix")
    except AssertionError as e:
        good &= ok(False, str(e))
    print("example answer-pool training pair:", repr(exs[0][0][-60:]), "->", repr(exs[0][1]))
    print("example controller training pair:", repr(exs[-1][0][-40:]), "->", repr(exs[-1][1]))
    print("t_stats of the last synthetic round:", json.dumps({k: v for k, v in stats.items() if k != "per_T"}))
    print("metrics via wsp.metrics:", json.dumps(wsp.metrics(rounds[-1]["tasks"], 6)["train"])[:200])
    print("SELFTEST", "PASSED" if good else "FAILED", flush=True)
    if not good:
        sys.exit(1)


if __name__ == "__main__":
    main()
