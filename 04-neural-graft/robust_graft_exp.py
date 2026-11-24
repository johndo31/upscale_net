"""04c - Robustness follow-up to 04b (real-BPE organ grafting into frozen Qwen2.5-1.5B-Instruct).

Removes the limits of the 04b results (one seed, 128 test cases, untuned baselines, an English-damage
probe without any "=", circuits written by hand for the tasks).

Parts (select with --parts; every finished cell is merged into one JSON (--out) right away, so a
rerun resumes and skips finished cells; failed cells store an error string and are retried on rerun):

  A  Seeds and sample size. Every arm x seeds on add_carry_64 / real_parens_128 / decimal_sorted_64,
     --n-test cases per length. Test sets are generated from a fixed eval seed and are identical for
     all arms and seeds. Per-seed results plus mean/min/max per arm, task and split.
  B  Tuned baselines. LoRA lr in --lora-lr-grid, trained-circuit lr in --sgd-lr-grid, selected ONLY on a
     held-out in-range validation set (lengths 4..8) at the first seed; the selection is used for all
     seeds in A. Every grid cell's full results, the selected setting and the oracle (best at the
     longest length) are stored.
  C  Damage to ordinary text: prose (wikitext test), Python source (stdlib via inspect.getsource),
     templated maths lines. Loss overall and at positions whose input token is "=" / " =", for base,
     LoRA, trained circuit, written circuit (all A seed-0 models), plus written circuit x learned gate
     (logistic probe on Qwen's own last hidden state).
  D  Circuits written by a model (03-transformer-writer round JSON, tasks with best_pass_L64_pass),
     grafted with the fitted read-in + clamp wiring; reference circuits and LoRA (<=4 tasks) for comparison.
  E  Prefix robustness (eval only): longest length with an English sentence before the prompt.

Run order is always B, A, C, E, D (A uses B's selection; C and E reuse A's seed-0 models through
checkpoints in --ckpt-dir, so nothing is trained twice within one machine).
"""

from __future__ import annotations

import argparse
import functools
import gc
import importlib
import inspect
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "03-transformer-writer"), "/content/tw", str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import real_graft_exp as rg  # noqa: E402
from real_graft_exp import (  # noqa: E402
    RealGraftedLLM,
    SparseOrgan,
    build_add_carry_organ,
    build_decimal_sorted_organ,
    build_real_parens_organ,
    calibrate_port_socket,
    gen_add_carry,
    gen_decimal_sorted,
    gen_real_parens,
    verify_organs_cpu,
)
from graft_exp import calibrate_linear_socket  # noqa: E402
from target_tf import BOS_ID, TinyTransformer, parse_sparse_weights  # noqa: E402
from tasks import TASKS_BY_NAME, evaluate_model, make_eval_suite  # noqa: E402
from verify_realizability import REFERENCE_PROOF_CIRCUITS, WORKED_EXAMPLES  # noqa: E402
from writer_loop import attach_lora, get_lora_state, load_lora_state, unwrap_and_reset_lora  # noqa: E402


T0 = time.time()
DEFAULT_OUT = str(HERE / "runs" / "robust_graft_results.json")
SMOKE_OUT = str(HERE / "runs" / "robust_graft_smoke_results.json")
DEFAULT_ROUND = str(HERE.parent / "03-transformer-writer" / "runs" / "coder3b_hindsight" / "round5.json")
REAL_TASKS = ("add_carry_64", "real_parens_128", "decimal_sorted_64")
WRITTEN = "compiled_graft_deadzone"
GATED = "compiled_graft_deadzone_gated"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:6.1f}m] {msg}", flush=True)


def lr_key(lr: float) -> str:
    return f"lr={lr:g}"


def is_done(cell: Any) -> bool:
    return isinstance(cell, dict) and "error" not in cell


# =============================================================================
# FIXED TEXT DATA
# =============================================================================

# 40 fixed English sentences with no digits, brackets or "=" (Part E prefixes).
PREFIX_SENTENCES = [
    "The old lighthouse keeper watched the storm roll in from the west.",
    "Fresh bread was cooling on the windowsill when the guests arrived.",
    "Most migrating birds follow coastlines and river valleys on their long journeys.",
    "She folded the letter carefully and slipped it into her coat pocket.",
    "The museum reopened after a long renovation of its eastern wing.",
    "A gentle rain fell over the vineyard throughout the quiet afternoon.",
    "Engineers inspected the bridge before allowing heavy trucks to cross it.",
    "The library keeps its rarest maps in a cool room without windows.",
    "He tuned the guitar slowly while the audience found their seats.",
    "Volcanic islands often have black sand beaches and steep green hills.",
    "The committee postponed its decision until more evidence was available.",
    "Children chased each other across the park as the sun went down.",
    "Good soup starts with onions cooked slowly in a little butter.",
    "The train left the station on time despite the heavy snowfall.",
    "Astronomers noticed a faint comet moving across the northern sky.",
    "Her grandmother taught her how to knit during the long winter evenings.",
    "The river flooded the lower fields but spared the village itself.",
    "Several startups are building tools that help farmers track soil moisture.",
    "The detective noticed that the clock on the mantel had stopped.",
    "Wolves were reintroduced to the valley and the elk changed their habits.",
    "Our neighbours painted their fence a bright shade of yellow last week.",
    "The orchestra rehearsed the final movement until everyone was exhausted.",
    "Desert plants store water in thick leaves and waxy stems.",
    "The editor asked for a shorter introduction and a clearer conclusion.",
    "A small fishing boat drifted past the harbour lights at dawn.",
    "The new policy encourages employees to work from home on Fridays.",
    "Bees communicate the location of flowers through a kind of dance.",
    "The castle walls were built from stone quarried in the nearby hills.",
    "He forgot his umbrella and arrived at the meeting soaking wet.",
    "Glaciers carve wide valleys as they move slowly down the mountains.",
    "The bakery on the corner sells the best croissants in town.",
    "Students gathered in the hall to hear the visiting professor speak.",
    "The captain ordered the crew to lower the sails before the squall.",
    "Fog covered the harbour so thickly that the ferries stayed in port.",
    "The gardener planted tulips along the path leading to the front door.",
    "Many languages borrow words for food from their neighbours.",
    "The mountain hut offered warm tea and a dry place to sleep.",
    "A stray cat followed us home and decided to stay for good.",
    "The festival ended with fireworks reflected in the dark lake.",
    "Researchers interviewed nurses about their longest and hardest shifts.",
]
assert len(PREFIX_SENTENCES) == 40
assert not any(ch in s for s in PREFIX_SENTENCES for ch in "0123456789()[]{}=")

# Python source for the code set: evaluation and gate-training modules are disjoint.
CODE_EVAL_MODULES = [
    "argparse", "difflib", "textwrap", "configparser", "csv", "calendar", "fractions", "statistics",
    "heapq", "shlex", "pprint", "string", "ipaddress", "dataclasses", "enum", "functools", "optparse",
]
CODE_TRAIN_MODULES = [
    "json.decoder", "json.encoder", "json.scanner", "bisect", "tokenize", "copy", "glob", "fnmatch",
    "base64", "quopri", "colorsys", "sched", "queue", "graphlib", "netrc", "plistlib", "gettext", "zipapp",
]


def maths_line(r: random.Random) -> str:
    k = r.randrange(12)
    a, b, c = r.randint(2, 40), r.randint(2, 40), r.randint(2, 20)
    if k == 0:
        return f"The area is A = {a} * {b} = {a * b} square metres."
    if k == 1:
        return f"The perimeter is P = 2 * ({a} + {b}) = {2 * (a + b)} cm."
    if k == 2:
        return f"If x = {a}, then 3x + {b} = {3 * a + b}."
    if k == 3:
        return f"The total cost is C = {a} * {c} = {a * c} dollars."
    if k == 4:
        return f"With m = {c} kg and g = 10 m/s^2, the weight is W = m * g = {10 * c} N."
    if k == 5:
        return f"The speed is v = d / t = {a * c} / {c} = {a} km/h."
    if k == 6:
        return f"The sum is S = {a} + {b} + {c} = {a + b + c}."
    if k == 7:
        return f"Let n = {c}; then n^2 = {c * c} and 2n = {2 * c}."
    if k == 8:
        return f"The difference is d = {a} - {b} = {a - b}."
    if k == 9:
        return f"The average is ({a} + {b}) / 2 = {(a + b) / 2:g}."
    if k == 10:
        return f"Since y = {c}x + {b}, at x = {a} we get y = {c * a + b}."
    return f"A rectangle with sides {a} and {b} has D^2 = {a}^2 + {b}^2 = {a * a + b * b}."


def make_maths_lines(seed: int, n: int, exclude: set) -> List[str]:
    r = random.Random(seed)
    out: List[str] = []
    seen = set(exclude)
    while len(out) < n:
        s = maths_line(r)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def dyck_ok(seq: List[str]) -> bool:
    """Honest checker: balanced and depth never below zero."""
    bal = 0
    for ch in seq:
        bal += 1 if ch == "(" else -1
        if bal < 0:
            return False
    return bal == 0


def random_dyck(rng: random.Random, n_pairs: int) -> List[str]:
    seq: List[str] = []
    opened = closed = 0
    while len(seq) < 2 * n_pairs:
        can_open, can_close = opened < n_pairs, closed < opened
        if can_open and (not can_close or rng.random() < 0.5):
            seq.append("(")
            opened += 1
        else:
            seq.append(")")
            closed += 1
    return seq


HARD_PARENS_MIN_LEN = 6  # at length 4 no sequence is balanced, starts "(", ends ")" and is invalid


def gen_hard_parens(rng: random.Random, length: int) -> Tuple[str, int]:
    """Hard bracket split (evaluation only). 50% valid. Every negative has equal counts of "(" and ")",
    starts with "(" and ends with ")": it is A + ")(" + B with A, B non-empty valid sequences, i.e. a
    valid sequence A + "()" + B whose "()" at an interior depth-0 point is swapped to ")(", so the depth
    goes to -1 once in the middle. Count-only and first-token rules cannot separate the classes."""
    L = max(HARD_PARENS_MIN_LEN, length if length % 2 == 0 else length + 1)
    n = L // 2
    if rng.random() < 0.5:
        seq = random_dyck(rng, n)
    else:
        a = rng.randint(1, n - 2)
        seq = random_dyck(rng, a) + [")", "("] + random_dyck(rng, n - 1 - a)
    y = 1 if dyck_ok(seq) else 0
    assert seq.count("(") == seq.count(")") and seq[0] == "(" and seq[-1] == ")"
    return "Balanced brackets: " + " ".join(seq) + " =", y


def parens_rule_stats(prompts_labels: List[Tuple[str, int]]) -> Dict[str, Any]:
    """Share of negatives a count-only rule ("invalid iff counts differ") and a first-token rule
    ("invalid iff it starts with ')'") would classify correctly."""
    negs = [[c for c in p.split(": ", 1)[1] if c in "()"] for p, y in prompts_labels if y == 0]
    n = len(negs)
    cnt = sum(1 for s in negs if s.count("(") != s.count(")"))
    first = sum(1 for s in negs if s and s[0] == ")")
    either = sum(1 for s in negs if s.count("(") != s.count(")") or (s and s[0] == ")"))
    return {"n_cases": len(prompts_labels), "n_negatives": n,
            "count_rule_neg_correct": round(cnt / n, 4) if n else None,
            "first_token_rule_neg_correct": round(first / n, 4) if n else None,
            "either_rule_neg_correct": round(either / n, 4) if n else None,
            "positive_share": round(1 - n / max(1, len(prompts_labels)), 4)}


def verify_hard_parens_cpu(n: int = 200) -> Dict[str, Any]:
    """Written bracket circuit on the hard splits (same token simulation as verify_organs_cpu).
    Reports accuracy; does not assert."""
    organ = build_real_parens_organ(torch.device("cpu"))
    rng = random.Random(7)
    out = {}
    for L in (6, 8, 32, 128):
        ok = neg_ok = n_neg = 0
        for _ in range(n):
            prompt, y = gen_hard_parens(rng, L)
            toks = ["P0", "P1", "P2"] + [c for c in prompt.split(": ")[1] if c in "()="]
            x = torch.zeros(1, len(toks), 16)
            x[0, :, 4] = 1.0
            for i, ch in enumerate(toks):
                x[0, i, {"(": 0, ")": 1, "=": 2}.get(ch, 3)] = 1.0
            with torch.no_grad():
                pred = int(organ.forward_states(x)[0, -1].argmax())
            ok += int(pred == y)
            if y == 0:
                n_neg += 1
                neg_ok += int(pred == 0)
        out[f"hard_L{L}"] = {"acc": round(ok / n, 4), "neg_acc": round(neg_ok / max(1, n_neg), 4), "n": n}
        print(f"  [CPU verify] real_parens HARD | L={L:3d} | acc={ok}/{n} | negatives={neg_ok}/{n_neg}", flush=True)
    return out


def evenly_spaced(items: List[Any], n: int) -> List[Any]:
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return list(items)
    if n == 1:
        return [items[0]]
    idx = sorted({round(k * (len(items) - 1) / (n - 1)) for k in range(n)})
    return [items[i] for i in idx]


# =============================================================================
# CIRCUIT HELPERS
# =============================================================================

def tiny_to_sparse(tiny: TinyTransformer) -> SparseOrgan:
    """Copy a 03-transformer-writer TinyTransformer into the SparseOrgan used by the graft harness
    (same maths: no LayerNorm, ALiBi, strictly causal attention, ReLU MLP)."""
    org = SparseOrgan(d_model=16, d_mlp=8, n_out=4, n_layers=2, n_heads=2, d_head=4)
    with torch.no_grad():
        for name in ("Q", "K", "V", "O", "alibi", "W1", "b1", "W2", "OUT_W", "OUT_b"):
            getattr(org, name).copy_(getattr(tiny, name).detach().cpu())
    return org


def load_round_tasks(round_json: str) -> List[Dict[str, Any]]:
    rj = json.loads(Path(round_json).read_text())
    out = []
    for split in ("train", "heldout"):
        for name, r in rj.get(split, {}).get("per_task", {}).items():
            if r.get("best_pass_L64_pass") and r.get("best_pass_text"):
                out.append({
                    "name": name, "split": split, "text": r["best_pass_text"],
                    "nnz": int(r.get("best_pass_nnz", -1)), "recorded_L64": float(r.get("best_pass_L64", 0.0)),
                })
    return out


def reference_circuit(name: str) -> Optional[str]:
    return REFERENCE_PROOF_CIRCUITS.get(name) or WORKED_EXAMPLES.get(name)


def cpu_verify_written(round_json: str) -> Dict[str, Any]:
    """Parse every model-written circuit, re-run it on the tasks.py suites (same seeds as the writer
    loop) and check it reaches its recorded L64 accuracy. Also simulates the graft's prompt prefix
    (k extra 'other' positions mapped onto the BOS channel) to flag circuits that depend on BOS being
    a single position."""
    out: Dict[str, Any] = {}
    for e in load_round_tasks(round_json):
        task = TASKS_BY_NAME[e["name"]]
        pr = parse_sparse_weights(e["text"])
        if pr.model is None:
            out[e["name"]] = {"parsed": False, "error": pr.error}
            continue
        suite = make_eval_suite(task)
        accs = {k: evaluate_model(pr.model, task, b) for k, b in suite.items()}
        sim = {}
        for k_bos in (3, 6, 10):
            correct = total = 0
            with torch.no_grad():
                for _, (toks, labs) in suite["L64"].items():
                    pre = torch.full((toks.shape[0], k_bos - 1), BOS_ID, dtype=toks.dtype)
                    logits = pr.model(torch.cat([pre, toks], dim=1))[:, : task.n_classes]
                    correct += int((logits.argmax(-1) == labs).sum())
                    total += int(labs.numel())
            sim[f"L64_with_{k_bos}_bos"] = correct / max(1, total)
        # Equivalence of the SparseOrgan copy used in the graft
        org = tiny_to_sparse(pr.model)
        toks, _ = next(iter(suite["L16"].values()))
        with torch.no_grad():
            a = pr.model(toks)
            b = org.forward_states(F.embedding(toks, pr.model.embed))[:, -1, :]
        out[e["name"]] = {
            "parsed": True, "split": e["split"], "n_entries": pr.n_entries, "recorded_L64": e["recorded_L64"],
            **{f"cpu_{k}": round(v, 4) for k, v in accs.items()},
            "matches_recorded_L64": abs(accs["L64"] - e["recorded_L64"]) < 1e-9,
            "prefix_sim": sim,
            "sparse_copy_max_abs_diff": float((a - b).abs().max()),
            "has_reference_circuit": reference_circuit(e["name"]) is not None,
        }
    return out


# =============================================================================
# RESULT STORE (one JSON, written atomically after every cell)
# =============================================================================

CRITICAL_KEYS = ("model", "n_test", "n_val", "eval_seed", "lora_steps", "sgd_steps", "written_steps", "batch_size", "smoke")


class Store:
    def __init__(self, path: Path, config: Dict[str, Any], fresh: bool):
        self.path = path
        self.data: Dict[str, Any] = {}
        if path.exists() and not fresh:
            try:
                self.data = json.loads(path.read_text())
            except Exception as e:  # noqa: BLE001
                log(f"could not read {path} ({e}); starting a new file")
                self.data = {}
        old = self.data.get("config", {})
        if old and bool(old.get("smoke")) != bool(config.get("smoke")):
            stale = path.with_suffix(".stale.json")
            path.replace(stale)
            log(f"existing {path.name} was written with smoke={old.get('smoke')}; moved to {stale.name}, starting fresh")
            self.data = {}
            old = {}
        diffs = {k: (old.get(k), config.get(k)) for k in CRITICAL_KEYS if old and old.get(k) != config.get(k)}
        if diffs:
            log(f"WARNING: config differs from the stored run on {diffs}; finished cells are kept (use --fresh to restart)")
            self.data.setdefault("config_history", []).append(old)
        self.data["config"] = config
        self.save()

    def get(self, *keys):
        d: Any = self.data
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d

    def set(self, *keys, value) -> None:
        d = self.data
        for k in keys[:-1]:
            nxt = d.get(k)
            if not isinstance(nxt, dict):
                nxt = {}
                d[k] = nxt
            d = nxt
        d[keys[-1]] = value
        self.save()

    def note(self, msg: str) -> None:
        self.data.setdefault("notes", []).append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}")
        self.save()

    def save(self) -> None:
        self.data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.data["elapsed_s_this_process"] = round(time.time() - T0, 1)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str))
        os.replace(tmp, self.path)


# =============================================================================
# GATE PROBE
# =============================================================================

class GateProbe(nn.Module):
    """g = sigmoid(w . h_last + b): d_llm + 1 parameters on Qwen's own frozen last hidden state."""

    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.lin(h.float()))


# =============================================================================
# RUNNER
# =============================================================================

class Runner:
    def __init__(self, args, store: Store):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.args = args
        self.store = store
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        log(f"device={self.device} dtype={self.dtype} model={args.model}")
        self.tok = AutoTokenizer.from_pretrained(args.model)
        try:
            base = AutoModelForCausalLM.from_pretrained(args.model, dtype=self.dtype)
        except TypeError:
            base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=self.dtype)
        self.base = base.to(self.device)
        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lm_head = self.base.get_output_embeddings()
        self.emb = self.base.get_input_embeddings()
        self.d_llm = int(self.base.config.hidden_size)

        meta = {
            "python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(),
            "device": str(self.device), "dtype": str(self.dtype),
            "gpu": torch.cuda.get_device_name(0) if self.device.type == "cuda" else None,
        }
        try:
            import transformers
            meta["transformers"] = transformers.__version__
        except Exception:  # noqa: BLE001
            pass
        store.set("meta", value=meta)

        enc1 = self.enc
        self.tok_digits = [enc1(str(d))[0] for d in range(10)]
        self.tok_eq = sorted({enc1("=")[0], enc1(" =")[0]})
        tok_yes, tok_no = enc1("Yes")[0], enc1("No")[0]
        tok_lp = sorted({enc1("(")[0], enc1(" (")[0]})
        tok_rp = sorted({enc1(")")[0], enc1(" )")[0]})
        self.eq_t = torch.tensor(self.tok_eq, dtype=torch.long, device=self.device)
        log(f"token ids digits={self.tok_digits} eq={self.tok_eq} lp={tok_lp} rp={tok_rp} No={tok_no} Yes={tok_yes}")

        dev = self.device
        cal_add = calibrate_port_socket(self.base, 32, [[t] for t in self.tok_digits] + [self.tok_eq], other_ch=11, const_ch=12, device=dev)
        cal_par = calibrate_port_socket(self.base, 16, [tok_lp, tok_rp, self.tok_eq], other_ch=3, const_ch=4, device=dev)
        cal_srt = calibrate_port_socket(self.base, 16, [[t] for t in self.tok_digits] + [self.tok_eq], other_ch=11, const_ch=12, device=dev)

        def gen_add(rng, L, casc=False):
            p, _sums, msb = gen_add_carry(rng, L, cascade=casc)
            return p, msb

        self.specs: Dict[str, Dict[str, Any]] = {
            "add_carry_64": dict(
                name="add_carry_64", cand=self.tok_digits, build=lambda: build_add_carry_organ(dev), cal=cal_add,
                n_sym=12, gate_ch=10, train_lens=[4, 5, 6, 7, 8],
                test_splits=[("id", [4, 6, 8], False), ("L16", [16], False), ("L32", [32], False),
                             ("L64", [64], False), ("cascade_L16", [16], True), ("cascade_L32", [32], True),
                             ("cascade_L64", [64], True)],
                longest="L64", gen=gen_add, full_sum=True,
            ),
            "real_parens_128": dict(
                name="real_parens_128", cand=[tok_no, tok_yes], build=lambda: build_real_parens_organ(dev), cal=cal_par,
                n_sym=4, gate_ch=2, train_lens=[4, 6, 8],
                # "casc" flag = hard split for brackets (evaluation only; training always passes False)
                test_splits=[("id", [4, 6, 8], False), ("L16", [16], False), ("L32", [32], False),
                             ("L64", [64], False), ("L128", [128], False),
                             ("hard_id", [6, 8], True), ("hard_L32", [32], True), ("hard_L128", [128], True)],
                longest="L128",
                gen=lambda rng, L, casc=False: gen_hard_parens(rng, L) if casc else gen_real_parens(rng, L),
                full_sum=False,
            ),
            "decimal_sorted_64": dict(
                name="decimal_sorted_64", cand=[tok_no, tok_yes], build=lambda: build_decimal_sorted_organ(dev), cal=cal_srt,
                n_sym=12, gate_ch=10, train_lens=[4, 5, 6, 7, 8],
                test_splits=[("id", [4, 6, 8], False), ("L16", [16], False), ("L32", [32], False), ("L64", [64], False)],
                longest="L64", gen=lambda rng, L, casc=False: gen_decimal_sorted(rng, L), full_sum=False,
            ),
        }
        self.tasks = [t for t in args.tasks.split(",") if t.strip()]
        for t in self.tasks:
            assert t in self.specs, f"unknown task {t}"
        self.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        self.lora_seeds = [int(s) for s in args.lora_seeds.split(",") if s.strip()]
        self.lora_grid = [float(x) for x in args.lora_lr_grid.split(",") if x.strip()]
        self.sgd_grid = [float(x) for x in args.sgd_lr_grid.split(",") if x.strip()]
        self.parts = set(p.strip().lower() for p in args.parts.split(",") if p.strip())
        self.ckpt_dir = Path(args.ckpt_dir) / args.model.replace("/", "__")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._batch_cache: Dict[Tuple, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        self.dsets: Optional[Dict[str, List[List[int]]]] = None
        self.gate_neg_seqs: List[List[int]] = []
        self._neg_feats: Optional[torch.Tensor] = None
        self._timing_logged = {"lora": False, "graft": False}

        n_t = len(self.tasks)
        planned = 0
        if "b" in self.parts:
            planned += len(self.lora_grid) * n_t
        if "a" in self.parts:
            planned += max(0, len(self.lora_seeds) - 1) * n_t
        if "d" in self.parts:
            planned += args.d_lora_max
        self.planned_lora_steps = planned * args.lora_steps
        self.base_probe_nll = rg.plain_english_nll(self.base, self.tok, self.device)
        log(f"base English-probe NLL (6 sentences, as in 04b) = {self.base_probe_nll:.4f}")

    # ------------------------------------------------------------------ utils
    def enc(self, s: str) -> List[int]:
        return self.tok.encode(s, add_special_tokens=False)

    def free(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def cell(self, keys: Tuple[str, ...], fn: Callable[[], Dict[str, Any]]) -> Any:
        label = "/".join(keys)
        cur = self.store.get(*keys)
        if is_done(cur):
            log(f"skip (done) {label}")
            return cur
        log(f"start {label}")
        t = time.time()
        try:
            res = fn()
            res["_seconds"] = round(time.time() - t, 1)
            log(f"done  {label} ({res['_seconds']}s) {self.brief(res)}")
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            print(tb, flush=True)
            res = {"error": f"{type(e).__name__}: {e}", "traceback": tb[-3000:], "_seconds": round(time.time() - t, 1)}
            log(f"FAIL  {label}: {res['error']}")
            self.store.data.setdefault("errors", []).append({"cell": label, "error": res["error"], "time": time.strftime("%H:%M:%S")})
        finally:
            try:
                unwrap_and_reset_lora(self.base, merge=False)
            except Exception:  # noqa: BLE001
                pass
            self.base.eval()
            self.free()
        self.store.set(*keys, value=res)
        return res

    @staticmethod
    def brief(res: Dict[str, Any]) -> str:
        keys = [k for k, v in res.items() if isinstance(v, float) and not k.startswith("_") and k not in ("lr",)]
        return " ".join(f"{k}={res[k]:.3f}" for k in keys[:9])

    def ckpt_path(self, arm: str, key: str, lr: float, seed: int, steps: int) -> Path:
        return self.ckpt_dir / f"{arm}__{key}__lr{lr:g}__s{seed}__n{steps}.pt"

    def hp(self, key: str, arm: str) -> Tuple[float, str]:
        if arm == WRITTEN:
            return self.args.written_lr, "fixed"
        sel = self.store.get("part_b", key, "selected", arm)
        if sel is not None:
            return float(sel), "part_b_validation"
        return (self.args.lora_lr_default if arm == "lora" else self.args.sgd_lr_default), "default_no_part_b"

    def steps_for(self, arm: str) -> int:
        return {"lora": self.args.lora_steps, "sgd_graft": self.args.sgd_steps, WRITTEN: self.args.written_steps}[arm]

    # --------------------------------------------------------------- batches
    def batches(self, key: str, split: str, lens: List[int], casc: bool, n_per: int, tag: str = "test",
                prefix_fn: Optional[Callable[[int], str]] = None, cache_tag: str = "") -> List[Tuple[torch.Tensor, torch.Tensor]]:
        ck = (key, split, tuple(lens), casc, n_per, tag, cache_tag)
        if ck in self._batch_cache:
            return self._batch_cache[ck]
        spec = self.specs[key]
        rng = random.Random(f"{tag}|{self.args.eval_seed}|{key}|{split}")
        by_len: Dict[int, Tuple[List[List[int]], List[int]]] = {}
        i = 0
        for L in lens:
            for _ in range(n_per):
                p, y = spec["gen"](rng, L, casc)
                if prefix_fn is not None:
                    p = prefix_fn(i) + p
                i += 1
                ids = self.enc(p)
                by_len.setdefault(len(ids), ([], []))
                by_len[len(ids)][0].append(ids)
                by_len[len(ids)][1].append(int(y))
        out = [
            (torch.tensor(xs, dtype=torch.long, device=self.device), torch.tensor(ys, dtype=torch.long, device=self.device))
            for _, (xs, ys) in sorted(by_len.items())
        ]
        self._batch_cache[ck] = out
        return out

    def record_split_stats(self) -> None:
        """For every bracket split: share of negatives a count-only / first-token rule gets right.
        Regenerates exactly the examples of self.batches (same rng string, same call order)."""
        key = "real_parens_128"
        if key not in self.tasks:
            return
        spec = self.specs[key]
        splits = [(n, l, c, self.args.n_test, "test") for n, l, c in spec["test_splits"]]
        splits.append(("val", spec["train_lens"], False, self.args.n_val, "val"))
        stats = {}
        for name, lens, casc, n_per, tag in splits:
            rng = random.Random(f"{tag}|{self.args.eval_seed}|{key}|{name}")
            pl = [spec["gen"](rng, L, casc) for L in lens for _ in range(n_per)]
            stats[name] = parens_rule_stats(pl)
        self.store.set("split_stats", key, value=stats)
        log("split_stats real_parens_128: " + ", ".join(
            f"{k}: count={v['count_rule_neg_correct']} first={v['first_token_rule_neg_correct']}" for k, v in stats.items()))

    def eval_sets(self, key: str, with_val: bool = True) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]:
        spec = self.specs[key]
        out = {name: self.batches(key, name, lens, casc, self.args.n_test) for name, lens, casc in spec["test_splits"]}
        if with_val:
            out["val"] = self.batches(key, "val", spec["train_lens"], False, self.args.n_val, tag="val")
        return out

    def train_batch(self, key: str, rng: random.Random, bs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        spec = self.specs[key]
        L = rng.choice(spec["train_lens"])
        by_l: Dict[int, Tuple[List[List[int]], List[int]]] = {}
        for _ in range(bs):
            p, y = spec["gen"](rng, L, False)
            ids = self.enc(p)
            by_l.setdefault(len(ids), ([], []))
            by_l[len(ids)][0].append(ids)
            by_l[len(ids)][1].append(int(y))
        xs, ys = max(by_l.values(), key=lambda kv: len(kv[0]))
        return (torch.tensor(xs, dtype=torch.long, device=self.device), torch.tensor(ys, dtype=torch.long, device=self.device))

    # -------------------------------------------------------------- forwards
    @torch.no_grad()
    def fwd_plain(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.base.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1, :]
        return self.lm_head(h).float()

    @torch.no_grad()
    def fwd_graft(self, graft: RealGraftedLLM, ids: torch.Tensor, probe: Optional[GateProbe] = None,
                  gate_log: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Same computation as RealGraftedLLM.forward_last_logits, optionally times a learned gate."""
        h0 = self.emb(ids).float()
        h_last = self.base.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1, :].float()
        z = graft._project_in(h0)
        org = graft.organ.forward_states(z)[:, -1, :]
        gate = F.relu(z[:, -1, graft.gate_ch: graft.gate_ch + 1]).clamp(0.0, 1.0)
        if probe is not None:
            g = probe(h_last)
            if gate_log is not None:
                gate_log.append(g.squeeze(-1).detach().cpu())
            gate = gate * g
        h_mod = h_last + gate * graft.W_out(torch.tanh(org / 10.0) * 10.0)
        return self.lm_head(h_mod.to(self.lm_head.weight.dtype)).float()

    @torch.no_grad()
    def hidden_plain(self, ids: torch.Tensor):
        return self.base.model(input_ids=ids, use_cache=False).last_hidden_state.float(), None

    def hidden_graft(self, graft: RealGraftedLLM, probe: Optional[GateProbe] = None):
        @torch.no_grad()
        def f(ids: torch.Tensor):
            h0 = self.emb(ids).float()
            h = self.base.model(input_ids=ids, use_cache=False).last_hidden_state.float()
            z = graft._project_in(h0)
            org = graft.organ.forward_states(z)
            gate = F.relu(z[:, :, graft.gate_ch: graft.gate_ch + 1]).clamp(0.0, 1.0)
            if probe is not None:
                gate = gate * probe(h)
            return h + gate * graft.W_out(torch.tanh(org / 10.0) * 10.0), gate.squeeze(-1)
        return f

    # ------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, fwd: Callable[[torch.Tensor], torch.Tensor], sets: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]],
                 cand: List[int]) -> Dict[str, Any]:
        cand_t = torch.tensor(cand, dtype=torch.long, device=self.device)
        out: Dict[str, Any] = {}
        nll: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        bs = self.args.eval_bs
        for name, batches in sets.items():
            correct = total = 0
            nsum = 0.0
            for ids, ys in batches:
                for s in range(0, ids.shape[0], bs):
                    logits = fwd(ids[s: s + bs])
                    yb = ys[s: s + bs]
                    pred = logits.index_select(-1, cand_t).argmax(-1)
                    correct += int((pred == yb).sum())
                    total += int(yb.numel())
                    nsum += float(F.cross_entropy(logits, cand_t[yb], reduction="sum"))
            out[name] = correct / max(1, total)
            nll[name] = round(nsum / max(1, total), 5)
            counts[name] = total
        out["answer_nll"] = nll
        out["n_cases"] = counts
        return out

    @torch.no_grad()
    def damage(self, hidden_fn, seqs: List[List[int]]) -> Dict[str, Any]:
        """Next-token loss overall, at positions whose INPUT token is "=" / " =" (where the write-out
        fires), elsewhere, and at positions where the (effective) gate is > 0."""
        by_len: Dict[int, List[List[int]]] = {}
        for s in seqs:
            if len(s) >= 2:
                by_len.setdefault(len(s), []).append(s)
        tot = n = 0.0
        eq_sum = eq_n = 0.0
        act_sum = act_n = 0.0
        gate_eq_sum = 0.0
        has_gate = False
        bs = self.args.damage_bs
        for _, group in sorted(by_len.items()):
            for s in range(0, len(group), bs):
                ids = torch.tensor(group[s: s + bs], dtype=torch.long, device=self.device)
                h, gate = hidden_fn(ids)
                logits = self.lm_head(h[:, :-1, :].to(self.lm_head.weight.dtype)).float()
                V = logits.shape[-1]
                ce = F.cross_entropy(logits.reshape(-1, V), ids[:, 1:].reshape(-1), reduction="none").view(ids.shape[0], -1)
                del logits
                eqm = torch.isin(ids[:, :-1], self.eq_t)
                tot += float(ce.sum())
                n += ce.numel()
                eq_sum += float(ce[eqm].sum())
                eq_n += int(eqm.sum())
                if gate is not None:
                    has_gate = True
                    g = gate[:, :-1]
                    act = g > 1e-6
                    act_sum += float(ce[act].sum())
                    act_n += int(act.sum())
                    gate_eq_sum += float(g[eqm].sum())
        r: Dict[str, Any] = {
            "nll": round(tot / max(1, n), 5), "n_tokens": int(n),
            "nll_at_eq": round(eq_sum / eq_n, 5) if eq_n else None, "n_eq": int(eq_n),
            "nll_not_eq": round((tot - eq_sum) / max(1, n - eq_n), 5),
        }
        if has_gate:
            r["n_gate_active"] = int(act_n)
            r["nll_at_gate_active"] = round(act_sum / act_n, 5) if act_n else None
            r["mean_gate_at_eq"] = round(gate_eq_sum / eq_n, 5) if eq_n else None
        return r

    def damage_all(self, hidden_fn) -> Dict[str, Any]:
        self.build_damage_sets()
        assert self.dsets is not None
        return {name: self.damage(hidden_fn, seqs) for name, seqs in self.dsets.items() if seqs}

    # -------------------------------------------------------------- training
    def _log_timing(self, kind: str, sps: float) -> None:
        if self._timing_logged[kind]:
            return
        self._timing_logged[kind] = True
        if kind == "lora":
            est = sps * self.planned_lora_steps / 3600.0
            log(f"TIMING LoRA: {sps:.3f} s/step at batch {self.args.batch_size}; ~{self.planned_lora_steps} LoRA steps planned in this run -> ~{est:.2f} h of LoRA training")
            self.store.set("timing", "lora_s_per_step", value=round(sps, 4))
            self.store.set("timing", "lora_planned_steps", value=self.planned_lora_steps)
            self.store.set("timing", "lora_est_hours", value=round(est, 3))
        else:
            log(f"TIMING graft: {sps:.3f} s/step at batch {self.args.batch_size}")
            self.store.set("timing", "graft_s_per_step", value=round(sps, 4))

    def train_graft(self, key: str, arm: str, steps: int, lr: float, seed: int) -> Tuple[RealGraftedLLM, Dict[str, Any]]:
        spec = self.specs[key]
        freeze = arm != "sgd_graft"
        torch.manual_seed(seed)
        rng = random.Random(f"train|{arm}|{key}|{seed}")
        organ = spec["build"]()
        if not freeze:
            organ.init_random_for_sgd(seed=seed)
        graft = RealGraftedLLM(self.base, organ, n_sym_ch=spec["n_sym"], gate_ch=spec["gate_ch"], calibrated_in=spec["cal"],
                               freeze_organ=freeze, socket_deadzone=0.3).to(self.device)
        trainable = [p for p in graft.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=lr * 0.05)
        ans = torch.tensor(spec["cand"], dtype=torch.long, device=self.device)
        losses: List[float] = []
        t = time.time()
        for step in range(steps):
            ids, ys = self.train_batch(key, rng, self.args.batch_size)
            logits, _ = graft.forward_last_logits(ids)
            loss = F.cross_entropy(logits, ans[ys])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss))
            if step == min(19, steps - 1):
                self._log_timing("graft", (time.time() - t) / (step + 1))
            if steps >= 4 and (step + 1) % max(1, steps // 4) == 0:
                log(f"    {arm} {key} lr={lr:g} seed={seed} step {step + 1}/{steps} loss={sum(losses[-50:]) / len(losses[-50:]):.4f}")
        graft.eval()
        info = {
            "lr": lr, "steps": steps, "seed": seed, "train_s": round(time.time() - t, 1),
            "s_per_step": round((time.time() - t) / max(1, steps), 4),
            "final_train_loss": round(sum(losses[-50:]) / max(1, len(losses[-50:])), 5) if losses else None,
            "n_trainable_params": int(sum(p.numel() for p in trainable)),
            "organ_nnz": int(organ.nnz()),
        }
        return graft, info

    def load_graft(self, key: str, path: Path) -> RealGraftedLLM:
        spec = self.specs[key]
        organ = spec["build"]()
        graft = RealGraftedLLM(self.base, organ, n_sym_ch=spec["n_sym"], gate_ch=spec["gate_ch"], calibrated_in=spec["cal"],
                               freeze_organ=True, socket_deadzone=0.3).to(self.device)
        sd = torch.load(path, map_location=self.device)
        graft.W_out.load_state_dict(sd["W_out"])
        graft.organ.load_state_dict(sd["organ"])
        graft.eval()
        return graft

    def train_lora(self, key: str, steps: int, lr: float, seed: int) -> Tuple[List[nn.Parameter], Dict[str, Any]]:
        spec = self.specs[key]
        torch.manual_seed(seed)
        rng = random.Random(f"lora|{key}|{seed}")
        params = attach_lora(self.base, r=16, alpha=32.0)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=lr * 0.1)
        ans = torch.tensor(spec["cand"], dtype=torch.long, device=self.device)
        micro = max(1, min(self.args.lora_micro, self.args.batch_size))
        accum = max(1, self.args.batch_size // micro)
        use_amp = self.device.type == "cuda" and self.dtype == torch.bfloat16
        self.base.train()
        losses: List[float] = []
        t = time.time()
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            sl = 0.0
            for _ in range(accum):
                ids, ys = self.train_batch(key, rng, micro)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp):
                    h = self.base.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1, :]
                    logits = self.lm_head(h).float()
                loss = F.cross_entropy(logits, ans[ys]) / accum
                loss.backward()
                sl += float(loss)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            losses.append(sl)
            if step == min(9, steps - 1):
                self._log_timing("lora", (time.time() - t) / (step + 1))
            if steps >= 4 and (step + 1) % max(1, steps // 4) == 0:
                log(f"    lora {key} lr={lr:g} seed={seed} step {step + 1}/{steps} loss={sum(losses[-50:]) / len(losses[-50:]):.4f}")
        self.base.eval()
        info = {
            "lr": lr, "steps": steps, "seed": seed, "train_s": round(time.time() - t, 1),
            "s_per_step": round((time.time() - t) / max(1, steps), 4),
            "final_train_loss": round(sum(losses[-50:]) / max(1, len(losses[-50:])), 5) if losses else None,
            "n_trainable_params": int(sum(p.numel() for p in params)), "rank": 16, "micro_batch": micro, "accum": accum,
        }
        return params, info

    # ------------------------------------------------------------ arm cells
    def full_sum_eval(self, graft: RealGraftedLLM) -> Dict[str, Any]:
        dig = set(self.tok_digits)
        out = {}
        n = self.args.full_sum_n
        for tag, L, casc in (("full_sum_L16", 16, False), ("full_sum_L64", 64, False), ("full_sum_cascade64", 64, True)):
            rng = random.Random(f"fullsum|{self.args.eval_seed}|{tag}")
            exact = dig_ok = dig_tot = 0
            for _ in range(n):
                p, sums, _ = gen_add_carry(rng, L, cascade=casc)
                ids = self.enc(p)
                b_pos = [i for i in range(1, len(ids)) if ids[i] in dig and ids[i - 1] in dig]
                with torch.no_grad():
                    _, org_all = graft.forward_last_logits(torch.tensor([ids], dtype=torch.long, device=self.device))
                preds = [int(org_all[0, q].argmax()) for q in b_pos]
                exact += int(preds == sums)
                dig_ok += sum(int(a == b) for a, b in zip(preds, sums))
                dig_tot += len(sums)
            out[tag] = {"exact_all_digits": round(exact / max(1, n), 4), "per_digit": round(dig_ok / max(1, dig_tot), 4), "n": n}
        return out

    def graft_cell(self, key: str, arm: str, lr: float, seed: int, extra: Optional[Callable] = None) -> Dict[str, Any]:
        spec = self.specs[key]
        steps = self.steps_for(arm)
        graft, info = self.train_graft(key, arm, steps, lr, seed)
        sets = self.eval_sets(key)
        # consistency check: our forward == RealGraftedLLM.forward_last_logits
        ids0 = sets["val"][0][0][:4]
        with torch.no_grad():
            diff = float((self.fwd_graft(graft, ids0) - graft.forward_last_logits(ids0)[0]).abs().max())
        res = self.evaluate(lambda x: self.fwd_graft(graft, x), sets, spec["cand"])
        res.update(info)
        res["forward_consistency_max_abs_diff"] = diff
        res["english_probe_nll"] = round(graft.english_nll(self.tok, self.device), 5)
        if spec.get("full_sum") and arm == WRITTEN:
            res["full_sum_eval"] = self.full_sum_eval(graft)
        path = self.ckpt_path(arm, key, lr, seed, steps)
        torch.save({"W_out": graft.W_out.state_dict(), "organ": graft.organ.state_dict()}, path)
        if extra is not None:
            res.update(extra(graft))
        del graft
        return res

    def lora_cell(self, key: str, lr: float, seed: int, save: bool, extra: Optional[Callable] = None) -> Dict[str, Any]:
        spec = self.specs[key]
        steps = self.args.lora_steps
        try:
            _params, info = self.train_lora(key, steps, lr, seed)
            res = self.evaluate(self.fwd_plain, self.eval_sets(key), spec["cand"])
            res.update(info)
            res["english_probe_nll"] = round(rg.plain_english_nll(self.base, self.tok, self.device), 5)
            if save:
                torch.save(get_lora_state(self.base), self.ckpt_path("lora", key, lr, seed, steps))
            if extra is not None:
                res.update(extra())
        finally:
            unwrap_and_reset_lora(self.base, merge=False)
            self.base.eval()
            self.free()
        return res

    def base_cell(self, key: str) -> Dict[str, Any]:
        res = self.evaluate(self.fwd_plain, self.eval_sets(key), self.specs[key]["cand"])
        res["english_probe_nll"] = round(self.base_probe_nll, 5)
        return res

    # seed-0 models shared by Parts A, C, E ---------------------------------
    def ensure_seed0_ckpt(self, key: str, arm: str) -> Tuple[Path, Dict[str, Any]]:
        lr, src = self.hp(key, arm)
        seed = self.lora_seeds[0] if arm == "lora" else self.seeds[0]
        steps = self.steps_for(arm)
        path = self.ckpt_path(arm, key, lr, seed, steps)
        if not path.exists():
            log(f"checkpoint {path.name} missing -> training it now (seed-0 model for {arm}/{key})")
            if arm == "lora":
                res = self.lora_cell(key, lr, seed, save=True)
            else:
                res = self.graft_cell(key, arm, lr, seed)
            res["hp_source"] = src
            cur = self.store.get("part_a", key, arm, f"seed={seed}")
            if not is_done(cur):
                self.store.set("part_a", key, arm, f"seed={seed}", value=res)
            else:
                self.store.note(f"retrained {arm}/{key}/seed={seed} only to recreate its checkpoint (not on this machine)")
        return path, {"lr": lr, "seed": seed, "steps": steps, "hp_source": src}

    def with_seed0_lora(self, key: str, fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        path, info = self.ensure_seed0_ckpt(key, "lora")
        state = torch.load(path, map_location=self.device)
        attach_lora(self.base, r=16, alpha=32.0)
        load_lora_state(self.base, state)
        self.base.eval()
        try:
            out = fn()
        finally:
            unwrap_and_reset_lora(self.base, merge=False)
            self.free()
        out["model"] = info
        return out

    def seed0_graft(self, key: str, arm: str) -> Tuple[RealGraftedLLM, Dict[str, Any]]:
        path, info = self.ensure_seed0_ckpt(key, arm)
        return self.load_graft(key, path), info

    # =========================================================== PART B
    def part_b(self) -> None:
        log("=== PART B: tuned baselines (selection on in-range validation only) ===")
        s0, ls0 = self.seeds[0], self.lora_seeds[0]
        for key in self.tasks:
            spec = self.specs[key]
            for lr in self.lora_grid:
                self.cell(("part_b", key, "lora", lr_key(lr)), functools.partial(self.lora_cell, key, lr, ls0, True))
            for lr in self.sgd_grid:
                self.cell(("part_b", key, "sgd_graft", lr_key(lr)), functools.partial(self.graft_cell, key, "sgd_graft", lr, s0))
            selected, oracle = {}, {}
            for arm, grid in (("lora", self.lora_grid), ("sgd_graft", self.sgd_grid)):
                cands = [(lr, self.store.get("part_b", key, arm, lr_key(lr))) for lr in grid]
                cands = [(lr, c) for lr, c in cands if is_done(c)]
                if not cands:
                    continue
                best = max(cands, key=lambda lc: (lc[1]["val"], -lc[1]["answer_nll"]["val"]))
                selected[arm] = best[0]
                orc = max(cands, key=lambda lc: (lc[1][spec["longest"]], lc[1]["val"]))
                oracle[arm] = {"lr": orc[0], "longest_split": spec["longest"], "acc_longest": orc[1][spec["longest"]],
                               "selected_acc_longest": best[1][spec["longest"]]}
            self.store.set("part_b", key, "selected", value=selected)
            self.store.set("part_b", key, "oracle", value=oracle)
            self.store.set("part_b", key, "rule", value=(
                f"selected = highest accuracy on 'val' (lengths {spec['train_lens']}, {self.args.n_val} per length, "
                f"separate seed from test), ties -> lower answer NLL on val; seed {s0} (circuit) / {ls0} (LoRA). "
                f"oracle = highest accuracy on '{spec['longest']}'."))
            log(f"PART B {key}: selected={selected} oracle={oracle}")

    # =========================================================== PART A
    def part_a(self) -> None:
        log("=== PART A: seeds and sample size ===")
        for key in self.tasks:
            self.cell(("part_a", key, "base"), functools.partial(self.base_cell, key))
            for arm, seeds in (("lora", self.lora_seeds), ("sgd_graft", self.seeds), (WRITTEN, self.seeds)):
                lr, src = self.hp(key, arm)
                for s in seeds:
                    self.cell(("part_a", key, arm, f"seed={s}"), functools.partial(self._a_cell, key, arm, lr, src, s))
            self.summarize_a(key)

    def _a_cell(self, key: str, arm: str, lr: float, src: str, seed: int) -> Dict[str, Any]:
        steps = self.steps_for(arm)
        first = self.lora_seeds[0] if arm == "lora" else self.seeds[0]
        if arm in ("lora", "sgd_graft") and seed == first:
            b = self.store.get("part_b", key, arm, lr_key(lr))
            if is_done(b) and b.get("steps") == steps and b.get("seed") == seed:
                log(f"  reusing Part B cell {arm}/{lr_key(lr)} for seed {seed} (identical setting)")
                res = {k: v for k, v in b.items() if k != "_seconds"}
                res["copied_from_part_b"] = True
                res["hp_source"] = src
                return res
        if arm == "lora":
            res = self.lora_cell(key, lr, seed, save=(seed == first))
        else:
            res = self.graft_cell(key, arm, lr, seed)
        res["hp_source"] = src
        return res

    def summarize_a(self, key: str) -> None:
        spec = self.specs[key]
        split_names = [s[0] for s in spec["test_splits"]] + ["val"]
        summ: Dict[str, Any] = {}
        base = self.store.get("part_a", key, "base")
        if is_done(base):
            summ["base"] = {sp: base.get(sp) for sp in split_names}
        for arm in ("lora", "sgd_graft", WRITTEN):
            cells = self.store.get("part_a", key, arm) or {}
            good = {k: c for k, c in cells.items() if k.startswith("seed=") and is_done(c)}
            if not good:
                continue
            arm_s: Dict[str, Any] = {"seeds": sorted(good.keys()), "lr": sorted({c.get("lr") for c in good.values()}),
                                     "hp_source": sorted({str(c.get("hp_source")) for c in good.values()})}
            for sp in split_names:
                vals = [c[sp] for c in good.values() if isinstance(c.get(sp), (int, float))]
                if vals:
                    arm_s[sp] = {"mean": round(sum(vals) / len(vals), 5), "min": round(min(vals), 5),
                                 "max": round(max(vals), 5), "n_seeds": len(vals)}
            summ[arm] = arm_s
        self.store.set("part_a", key, "summary", value=summ)

    # =========================================================== PART C
    def build_damage_sets(self) -> None:
        if self.dsets is not None:
            return
        a = self.args
        T = a.passage_len
        meta: Dict[str, Any] = {}
        prose: List[List[int]] = []
        try:
            prose, meta["prose"] = self._load_prose(a.n_passages, T)
        except Exception as e:  # noqa: BLE001
            meta["prose"] = {"error": f"{type(e).__name__}: {e}"}
            log(f"prose set unavailable: {e}")
        code_eval, m_eval = self._load_code(CODE_EVAL_MODULES, a.n_code, T)
        code_train, m_train = self._load_code(CODE_TRAIN_MODULES, a.n_code_train, T)
        meta["code"] = m_eval
        meta["code_gate_train"] = m_train
        math_eval_txt = make_maths_lines(1234, a.n_math, set())
        math_train_txt = make_maths_lines(5678, a.n_math_train, set(math_eval_txt))
        math_eval = [self.enc(s) for s in math_eval_txt]
        math_train = [self.enc(s) for s in math_train_txt]
        meta["maths"] = {"source": "templates, seed 1234", "n_lines": len(math_eval), "examples": math_eval_txt[:3]}
        meta["maths_gate_train"] = {"source": "templates, seed 5678, disjoint from eval lines", "n_lines": len(math_train)}
        self.dsets = {"prose": prose, "code": code_eval, "maths": math_eval}
        self.gate_neg_seqs = code_train + math_train
        for name, seqs in list(self.dsets.items()) + [("gate_negatives", self.gate_neg_seqs)]:
            n_eq = sum(int(t in self.tok_eq) for s in seqs for t in s[:-1])
            m = meta.setdefault(name, {}) if name != "gate_negatives" else meta.setdefault("gate_negatives", {})
            m.update({"n_seqs": len(seqs), "n_predicted_tokens": sum(max(0, len(s) - 1) for s in seqs), "n_eq_positions": n_eq})
        self.store.set("part_c", "sets", value=meta)
        log(f"damage sets: " + ", ".join(f"{k}: {len(v)} seqs" for k, v in self.dsets.items()) + f", gate negatives: {len(self.gate_neg_seqs)} seqs")

    def _load_prose(self, n: int, T: int) -> Tuple[List[List[int]], Dict[str, Any]]:
        from datasets import load_dataset

        tries = [("Salesforce/wikitext", "wikitext-103-raw-v1"), ("wikitext", "wikitext-103-raw-v1"),
                 ("Salesforce/wikitext", "wikitext-2-raw-v1"), ("wikitext", "wikitext-2-raw-v1"),
                 ("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1")]
        errs = []
        for name, cfg in tries:
            try:
                ds = load_dataset(name, cfg, split="test")
                col = "text" if "text" in ds.column_names else ds.column_names[0]
                texts = ds[col]
                break
            except Exception as e:  # noqa: BLE001
                errs.append(f"{name}/{cfg}: {type(e).__name__}: {str(e)[:200]}")
        else:
            raise RuntimeError("no wikitext variant could be loaded: " + " | ".join(errs))
        lines = []
        n_head = 0
        for t in texts:
            for line in t.split("\n"):
                s = line.strip()
                if not s:
                    continue
                if s.startswith("=") and s.endswith("="):
                    n_head += 1  # wikitext section headings ("= = Title = =") are dropped: prose without "=" markup
                    continue
                lines.append(s)
        text = "\n".join(lines)[:8_000_000]
        ids = self.enc(text)
        chunks = [ids[i: i + T] for i in range(0, len(ids) - T + 1, T)]
        picked = evenly_spaced(chunks, n)
        return picked, {"source": f"{name}/{cfg} test", "headings_dropped": n_head, "n_chunks_available": len(chunks),
                        "passage_len": T, "load_errors_before_success": errs, "example": self.tok.decode(picked[0][:60]) if picked else ""}

    def _load_code(self, modules: List[str], n: int, T: int) -> Tuple[List[List[int]], Dict[str, Any]]:
        chunks: List[List[int]] = []
        used, failed = [], []
        for m in modules:
            try:
                src = inspect.getsource(importlib.import_module(m))
            except Exception as e:  # noqa: BLE001
                failed.append(f"{m}: {type(e).__name__}")
                continue
            used.append(m)
            ids = self.enc(src)
            chunks.extend(ids[i: i + T] for i in range(0, len(ids) - T + 1, T))
        picked = evenly_spaced(chunks, n)
        return picked, {"source": "inspect.getsource of stdlib modules", "python": sys.version.split()[0],
                        "modules": used, "failed_modules": failed, "n_chunks_available": len(chunks), "chunk_len": T}

    def gate_negative_features(self) -> torch.Tensor:
        if self._neg_feats is not None:
            return self._neg_feats
        self.build_damage_sets()
        feats = []
        by_len: Dict[int, List[List[int]]] = {}
        for s in self.gate_neg_seqs:
            by_len.setdefault(len(s), []).append(s)
        with torch.no_grad():
            for _, group in sorted(by_len.items()):
                for i in range(0, len(group), self.args.damage_bs):
                    ids = torch.tensor(group[i: i + self.args.damage_bs], dtype=torch.long, device=self.device)
                    h = self.base.model(input_ids=ids, use_cache=False).last_hidden_state.float()
                    m = torch.isin(ids, self.eq_t)
                    if m.any():
                        feats.append(h[m])
        self._neg_feats = torch.cat(feats, 0) if feats else torch.zeros(0, self.d_llm, device=self.device)
        return self._neg_feats

    def train_probe(self, key: str) -> Tuple[GateProbe, Dict[str, Any]]:
        neg = self.gate_negative_features()
        rng = random.Random(f"gate|{key}")
        by_len: Dict[int, List[List[int]]] = {}
        spec = self.specs[key]
        for _ in range(self.args.gate_pos):
            p, _y = spec["gen"](rng, rng.choice(spec["train_lens"]), False)
            ids = self.enc(p)
            by_len.setdefault(len(ids), []).append(ids)
        pos_l = []
        with torch.no_grad():
            for _, group in sorted(by_len.items()):
                for i in range(0, len(group), self.args.eval_bs):
                    ids = torch.tensor(group[i: i + self.args.eval_bs], dtype=torch.long, device=self.device)
                    pos_l.append(self.base.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1, :].float())
        pos = torch.cat(pos_l, 0)
        if neg.shape[0] == 0:
            raise RuntimeError("no '=' positions in the gate-training text")
        X = torch.cat([pos, neg], 0)
        y = torch.cat([torch.ones(pos.shape[0]), torch.zeros(neg.shape[0])]).to(self.device)
        mu = X.mean(0)
        sd = X.std(0) + 1e-4
        Xn = (X - mu) / sd
        torch.manual_seed(0)
        lin = nn.Linear(self.d_llm, 1).to(self.device)
        nn.init.zeros_(lin.weight)
        nn.init.zeros_(lin.bias)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
        pw = torch.tensor([neg.shape[0] / max(1, pos.shape[0])], device=self.device)
        for _ in range(self.args.gate_steps):
            logit = lin(Xn).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw) + 1e-4 * lin.weight.pow(2).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        probe = GateProbe(self.d_llm).to(self.device)
        with torch.no_grad():
            w = lin.weight / sd
            probe.lin.weight.copy_(w)
            probe.lin.bias.copy_(lin.bias - (w * mu).sum())
            g = probe(X).squeeze(-1)
            fold_diff = float((g - torch.sigmoid(lin(Xn).squeeze(-1))).abs().max())
        info = {
            "n_pos": int(pos.shape[0]), "n_neg": int(neg.shape[0]), "steps": self.args.gate_steps, "n_params": self.d_llm + 1,
            "train_loss": round(float(loss), 5), "fold_max_abs_diff": fold_diff,
            "train_acc_pos": round(float((g[: pos.shape[0]] > 0.5).float().mean()), 4),
            "train_acc_neg": round(float((g[pos.shape[0]:] <= 0.5).float().mean()), 4),
            "positives": f"final token of {self.args.gate_pos} training-generator prompts (lengths {spec['train_lens']})",
            "negatives": "'='/' =' input positions in code_gate_train + maths_gate_train (disjoint from eval)",
        }
        return probe, info

    def part_c(self) -> None:
        log("=== PART C: damage to ordinary text (incl. '=') ===")
        self.build_damage_sets()
        self.cell(("part_c", "damage", "base"), lambda: self.damage_all(self.hidden_plain))
        for key in self.tasks:
            self.cell(("part_c", "damage", f"lora|{key}"),
                      functools.partial(self.with_seed0_lora, key, lambda: self.damage_all(self.hidden_plain)))
            for arm in ("sgd_graft", WRITTEN):
                self.cell(("part_c", "damage", f"{arm}|{key}"), functools.partial(self._c_graft_damage, key, arm))
            self.cell(("part_c", "damage", f"{GATED}|{key}"), functools.partial(self._c_gated, key))
        self.summarize_c()

    def _c_graft_damage(self, key: str, arm: str) -> Dict[str, Any]:
        graft, info = self.seed0_graft(key, arm)
        out = self.damage_all(self.hidden_graft(graft))
        out["model"] = info
        return out

    def _c_gated(self, key: str) -> Dict[str, Any]:
        spec = self.specs[key]
        graft, info = self.seed0_graft(key, WRITTEN)
        probe, pinfo = self.train_probe(key)
        torch.save(probe.state_dict(), self.ckpt_dir / f"gate_probe__{key}__s{info['seed']}.pt")
        sets = self.eval_sets(key)
        acc: Dict[str, Any] = {}
        gate_stats: Dict[str, Any] = {}
        for name, b in sets.items():
            gl: List[torch.Tensor] = []
            r = self.evaluate(lambda x: self.fwd_graft(graft, x, probe, gl), {name: b}, spec["cand"])
            acc[name] = r[name]
            acc.setdefault("answer_nll", {})[name] = r["answer_nll"][name]
            g = torch.cat(gl) if gl else torch.zeros(0)
            gate_stats[name] = {"mean": round(float(g.mean()), 4) if g.numel() else None,
                                "frac_open_gt_0.5": round(float((g > 0.5).float().mean()), 4) if g.numel() else None}
        out = {"task_accuracy": acc, "probe_on_test_prompts": gate_stats, "probe": pinfo, "model": info}
        out.update(self.damage_all(self.hidden_graft(graft, probe)))
        return out

    def summarize_c(self) -> None:
        dmg = self.store.get("part_c", "damage") or {}
        base = dmg.get("base")
        if not is_done(base):
            return
        summ: Dict[str, Any] = {}
        for model, r in dmg.items():
            if not is_done(r):
                continue
            row = {}
            for sname, b in base.items():
                if not isinstance(b, dict) or sname not in r or not isinstance(r[sname], dict) or "nll" not in b:
                    continue
                m = r[sname]
                row[sname] = {
                    "nll": m["nll"], "delta_nll": round(m["nll"] - b["nll"], 5),
                    "nll_at_eq": m.get("nll_at_eq"),
                    "delta_nll_at_eq": (round(m["nll_at_eq"] - b["nll_at_eq"], 5)
                                        if m.get("nll_at_eq") is not None and b.get("nll_at_eq") is not None else None),
                    "n_eq": m.get("n_eq"),
                }
            summ[model] = row
        self.store.set("part_c", "summary", value=summ)

    # =========================================================== PART E
    def part_e(self) -> None:
        log("=== PART E: prefix robustness (eval only) ===")
        for key in self.tasks:
            spec = self.specs[key]
            longest = spec["longest"]
            lens, casc = next((l, c) for n, l, c in spec["test_splits"] if n == longest)
            plain = self.batches(key, longest, lens, casc, self.args.n_test)
            pref = self.batches(key, longest, lens, casc, self.args.n_test,
                                prefix_fn=lambda i: PREFIX_SENTENCES[i % len(PREFIX_SENTENCES)] + " ", cache_tag="prefixed")
            sets = {longest: plain, f"{longest}_prefixed": pref}
            cand = spec["cand"]
            self.cell(("part_e", key, "base"), lambda: self.evaluate(self.fwd_plain, sets, cand))
            self.cell(("part_e", key, "lora"), functools.partial(self.with_seed0_lora, key, lambda: self.evaluate(self.fwd_plain, sets, cand)))
            for arm in (WRITTEN, "sgd_graft"):
                self.cell(("part_e", key, arm), functools.partial(self._e_graft, key, arm, sets))
            probe_path = self.ckpt_dir / f"gate_probe__{key}__s{self.seeds[0]}.pt"
            if probe_path.exists():
                self.cell(("part_e", key, GATED), functools.partial(self._e_gated, key, sets, probe_path))

    def _e_graft(self, key: str, arm: str, sets) -> Dict[str, Any]:
        graft, info = self.seed0_graft(key, arm)
        r = self.evaluate(lambda x: self.fwd_graft(graft, x), sets, self.specs[key]["cand"])
        r["model"] = info
        return r

    def _e_gated(self, key: str, sets, probe_path: Path) -> Dict[str, Any]:
        graft, info = self.seed0_graft(key, WRITTEN)
        probe = GateProbe(self.d_llm).to(self.device)
        probe.load_state_dict(torch.load(probe_path, map_location=self.device))
        r = self.evaluate(lambda x: self.fwd_graft(graft, x, probe), sets, self.specs[key]["cand"])
        r["model"] = info
        return r

    # =========================================================== PART D
    def part_d(self) -> None:
        log("=== PART D: circuits written by a model (03-transformer-writer) ===")
        entries = load_round_tasks(self.args.round_json)
        if self.args.d_max_tasks > 0:
            entries = entries[: self.args.d_max_tasks]
        self.store.set("part_d", "source", value={"round_json": Path(self.args.round_json).name,
                                                 "tasks": [e["name"] for e in entries],
                                                 "prompt_format": "'Input: ' + digits + '='",
                                                 "wiring": "graft_exp.calibrate_linear_socket read-in (digits 0..3 -> ch0..3, '=' -> ch5, "
                                                           "other -> ch4, const ch6) + RealGraftedLLM clamp (deadzone 0.3 on ch0..5), gate on ch5"})
        self.store.set("part_d", "cpu_verify", value=cpu_verify_written(self.args.round_json))
        cal16 = calibrate_linear_socket(self.base, self.tok, self.device)
        dev = self.device
        for e in entries:
            task = TASKS_BY_NAME[e["name"]]

            def gen(rng, L, casc=False, _t=task):
                seq, y = _t.gen(rng, L)
                return "Input: " + "".join(map(str, seq)) + "=", y

            for variant, text in (("written", e["text"]), ("reference", reference_circuit(e["name"]))):
                if text is None:
                    continue
                key = f"d_{variant}_{e['name']}"

                def build(_text=text):
                    pr = parse_sparse_weights(_text)
                    assert pr.model is not None, pr.error
                    return tiny_to_sparse(pr.model).to(dev)

                self.specs[key] = dict(
                    name=key, cand=self.tok_digits[: task.n_classes], build=build, cal=cal16, n_sym=6, gate_ch=5,
                    train_lens=[4, 5, 6, 7, 8],
                    test_splits=[("id", [4, 6, 8], False), ("L16", [16], False), ("L32", [32], False), ("L64", [64], False)],
                    longest="L64", gen=gen, full_sum=False,
                )
        for e in entries:
            name = e["name"]
            kw = f"d_written_{name}"
            self.cell(("part_d", "tasks", name, "base"), functools.partial(self.base_cell, kw))
            self.cell(("part_d", "tasks", name, "written"), functools.partial(self._d_graft, kw, e))
            kr = f"d_reference_{name}"
            if kr in self.specs:
                self.cell(("part_d", "tasks", name, "reference"), functools.partial(self._d_graft, kr, None))
        lora_tasks = self.choose_d_lora_tasks(entries)
        self.store.set("part_d", "lora_tasks", value={"tasks": lora_tasks, "lr": self.args.d_lora_lr, "seed": self.lora_seeds[0],
                                                     "steps": self.args.lora_steps,
                                                     "rule": "one task per family (largest circuit, ties by name), families ordered by circuit size, first k"})
        for name in lora_tasks:
            kw = f"d_written_{name}"
            self.cell(("part_d", "tasks", name, "lora"), functools.partial(
                self.lora_cell, kw, self.args.d_lora_lr, self.lora_seeds[0], False,
                lambda: {"damage": self.damage_all(self.hidden_plain)}))
        self.summarize_d(entries)

    def _d_graft(self, key: str, e: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        res = self.graft_cell(key, WRITTEN, self.args.written_lr, self.seeds[0],
                              extra=lambda g: {"damage": self.damage_all(self.hidden_graft(g))})
        if e is not None:
            res["circuit_nnz"] = e["nnz"]
            res["circuit_text"] = e["text"]
        return res

    def choose_d_lora_tasks(self, entries: List[Dict[str, Any]]) -> List[str]:
        if self.args.d_lora_tasks.strip():
            return [t.strip() for t in self.args.d_lora_tasks.split(",") if t.strip()]
        best: Dict[str, Dict[str, Any]] = {}
        for e in sorted(entries, key=lambda x: (-x["nnz"], x["name"])):
            fam = TASKS_BY_NAME[e["name"]].family
            best.setdefault(fam, e)
        ordered = sorted(best.values(), key=lambda x: (-x["nnz"], x["name"]))
        return [e["name"] for e in ordered[: self.args.d_lora_max]]

    def summarize_d(self, entries: List[Dict[str, Any]]) -> None:
        rows = {}
        base_dmg = self.store.get("part_c", "damage", "base")
        for e in entries:
            r = {}
            for variant in ("base", "written", "reference", "lora"):
                c = self.store.get("part_d", "tasks", e["name"], variant)
                if not is_done(c):
                    continue
                v = {sp: c.get(sp) for sp in ("id", "L16", "L32", "L64")}
                v["n_trainable_params"] = c.get("n_trainable_params", 0)
                dm = c.get("damage") or {}
                if "prose" in dm:
                    v["prose_nll"] = dm["prose"]["nll"]
                    if is_done(base_dmg) and "prose" in base_dmg:
                        v["prose_delta_nll"] = round(dm["prose"]["nll"] - base_dmg["prose"]["nll"], 5)
                r[variant] = v
            rows[e["name"]] = r
        self.store.set("part_d", "summary", value=rows)


# =============================================================================
# MAIN
# =============================================================================

def apply_smoke(args) -> None:
    args.seeds = args.seeds.split(",")[0]
    args.lora_seeds = args.lora_seeds.split(",")[0]
    args.lora_steps, args.sgd_steps, args.written_steps = 3, 4, 4
    args.n_test, args.n_val = 8, 8
    args.n_passages, args.n_code, args.n_code_train = 4, 4, 4
    args.n_math, args.n_math_train = 8, 8
    args.gate_pos, args.gate_steps = 16, 30
    args.full_sum_n = 4
    args.d_max_tasks, args.d_lora_max = 2, 1
    args.batch_size, args.lora_micro, args.eval_bs, args.damage_bs = 8, 8, 16, 2
    if args.out == DEFAULT_OUT:
        args.out = SMOKE_OUT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default="a,b,c,d,e", help="subset of a,b,c,d,e (always run in order b,a,c,e,d)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tasks", default=",".join(REAL_TASKS))
    ap.add_argument("--seeds", default="0,1,2,3,4", help="seeds for the circuit arms (first = selection seed)")
    ap.add_argument("--lora-seeds", default="0,1,2")
    ap.add_argument("--n-test", type=int, default=512, help="test cases per length")
    ap.add_argument("--n-val", type=int, default=256, help="validation cases per training length (Part B selection)")
    ap.add_argument("--eval-seed", type=int, default=20261124)
    ap.add_argument("--lora-steps", type=int, default=1000)
    ap.add_argument("--sgd-steps", type=int, default=3000)
    ap.add_argument("--written-steps", type=int, default=350)
    ap.add_argument("--written-lr", type=float, default=2e-2)
    ap.add_argument("--lora-lr-grid", default="1e-4,3e-4")
    ap.add_argument("--sgd-lr-grid", default="1e-3,3e-3,1e-2")
    ap.add_argument("--lora-lr-default", type=float, default=2e-4, help="used by A only if no Part B selection exists")
    ap.add_argument("--sgd-lr-default", type=float, default=1e-2, help="used by A only if no Part B selection exists")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lora-micro", type=int, default=32, help="LoRA micro-batch (grad accumulation fills --batch-size)")
    ap.add_argument("--eval-bs", type=int, default=64)
    ap.add_argument("--damage-bs", type=int, default=8)
    ap.add_argument("--passage-len", type=int, default=256)
    ap.add_argument("--n-passages", type=int, default=200)
    ap.add_argument("--n-code", type=int, default=200)
    ap.add_argument("--n-code-train", type=int, default=100)
    ap.add_argument("--n-math", type=int, default=120)
    ap.add_argument("--n-math-train", type=int, default=120)
    ap.add_argument("--gate-pos", type=int, default=512)
    ap.add_argument("--gate-steps", type=int, default=1500)
    ap.add_argument("--full-sum-n", type=int, default=128)
    ap.add_argument("--round-json", default=DEFAULT_ROUND)
    ap.add_argument("--d-max-tasks", type=int, default=0, help="0 = all tasks with best_pass_L64_pass")
    ap.add_argument("--d-lora-max", type=int, default=4)
    ap.add_argument("--d-lora-tasks", default="", help="override the deterministic choice (comma list)")
    ap.add_argument("--d-lora-lr", type=float, default=2e-4)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--ckpt-dir", default=str(HERE / "runs" / "robust_ckpt"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing --out file")
    ap.add_argument("--verify-only", action="store_true", help="CPU self-tests only (no LLM)")
    args = ap.parse_args()
    if args.smoke:
        apply_smoke(args)
    torch.manual_seed(0)
    random.seed(0)

    store = Store(Path(args.out), dict(vars(args)), fresh=args.fresh)
    log(f"robust_graft_exp | parts={args.parts} smoke={args.smoke} out={args.out}")

    log("CPU self-test of the written circuits (04b organs)")
    t = time.time()
    verify_organs_cpu()
    selftest: Dict[str, Any] = {"organs_04b": "pass", "seconds": round(time.time() - t, 1)}
    log("CPU self-test of the written bracket circuit on the HARD splits (reported, not asserted)")
    selftest["real_parens_hard"] = verify_hard_parens_cpu(n=40 if args.smoke else 200)
    if Path(args.round_json).exists():
        cv = cpu_verify_written(args.round_json)
        selftest["model_written_circuits"] = {
            "n": len(cv), "all_parse": all(v.get("parsed") for v in cv.values()),
            "all_match_recorded_L64": all(v.get("matches_recorded_L64") for v in cv.values()),
            "max_sparse_copy_diff": max((v.get("sparse_copy_max_abs_diff", 0.0) for v in cv.values()), default=0.0),
        }
        for name, v in cv.items():
            log(f"  [CPU verify written] {name:14s} parsed={v.get('parsed')} L64={v.get('cpu_L64')} recorded={v.get('recorded_L64')} "
                f"match={v.get('matches_recorded_L64')} prefix-sim={v.get('prefix_sim')}")
    else:
        selftest["model_written_circuits"] = f"round json not found: {args.round_json}"
    store.set("selftest", value=selftest)
    if args.verify_only:
        log("verify-only: done")
        return

    runner = Runner(args, store)
    try:
        runner.record_split_stats()
    except Exception as e:  # noqa: BLE001
        log(f"split_stats failed: {type(e).__name__}: {e}")
    order =[p for p in ("b", "a", "c", "e", "d") if p in runner.parts]
    for p in order:
        try:
            getattr(runner, f"part_{p}")()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            log(f"PART {p.upper()} aborted: {type(e).__name__}: {e}")
            store.data.setdefault("errors", []).append({"cell": f"part_{p}", "error": f"{type(e).__name__}: {e}"})
            store.save()
    n_err = sum(1 for _ in store.data.get("errors", []))
    store.set("finished", value=time.strftime("%Y-%m-%d %H:%M:%S"))
    log(f"ROBUST_DONE (errors recorded so far: {n_err}) -> {args.out}")


if __name__ == "__main__":
    main()
