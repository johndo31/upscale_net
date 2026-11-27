"""07-think-calc: Grafting a compiled 2-layer variable-length '+'/'-' calculator organ
+ a 1,537-param <think>-trained gate onto frozen DeepSeek-R1-Distill-Qwen-1.5B.

Zero tool-calling syntax. Zero RL. Zero edits to base model weights.
Only trainable parameters:
  - W_out:     nn.Linear(12, 1536, bias=False) -> 18,432 params (0..9, '-', STOP_SPACE; trained on 3-6 digit +/-)
  - GateProbe: nn.Linear(1536, 1, bias=True)   ->  1,537 params (trained on harvested <think> '=')

Steps (each saved atomically to --out; reruns skip completed steps):
  1. Tokenizer & closed-form ridge socket calibration + read-in verification
  2. Train W_out on short 3-6 digit '+' and '-' expressions
  3. Eval 1: Standalone variable-length '+'/'-' OOD (ID 3-6, OOD-12/24/48, Chain-48, Mixed, Neg)
  4. Harvest real <think> traces & train 1,537-param GateProbe + Python/prose NLL damage eval
  5. Eval 2: 80 multi-step big-number (9-14 digit) word problems inside <think> (Base vs Grafted)
  6. Eval 3: 80 standard GSM8K test problems no-regression check (Base vs Grafted)
"""

from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import json
import math
import os
import platform
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from calc_organ import (  # noqa: E402
    ActiveExprSpan,
    CH_CONST,
    CH_DIG,
    CH_EQ,
    CH_MINUS,
    CH_OTHER,
    CH_PLUS,
    N_READ,
    N_SYM_OUT,
    SparseOrgan,
    assemble_bus,
    build_varlen_calc_organ,
    build_window_structural_features,
    determine_p_top_neural,
    parse_active_expr_symbols,
)

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:7.0f}s] {msg}", flush=True)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# SHARED CONSTANTS & TEXT SETS
# =============================================================================

ENGLISH_PROBES = [
    "In 1936 Alan Turing defined a mathematical model of computation consisting of an infinite tape divided into cells.",
    "The northern lights occur when charged particles from the solar wind collide with gases in Earth's upper atmosphere.",
    "After a long winter, the mountain passes finally cleared in late May, allowing supply trucks to reach the valley.",
    "Modern compilers translate high-level source code into machine instructions through several optimization passes.",
    "Coffee plants grow best in tropical highlands with steady rainfall, volcanic soil, and moderate daytime temperatures.",
    "Although the experiment produced a clear signal in small trials, the effect vanished once a proper control was added.",
]

CODE_EVAL_MODULES = [
    "argparse", "difflib", "textwrap", "configparser", "csv", "calendar", "fractions", "statistics",
    "heapq", "shlex", "pprint", "string", "ipaddress", "dataclasses", "enum", "functools",
]
CODE_TRAIN_MODULES = [
    "json.decoder", "json.encoder", "bisect", "tokenize", "copy", "glob", "fnmatch",
    "base64", "quopri", "colorsys", "sched", "queue", "graphlib", "netrc", "plistlib", "gettext",
    "http.client", "urllib.parse", "tarfile", "zipfile", "logging", "ast", "unittest.case", "doctest",
]


# =============================================================================
# ATOMIC RESUMABLE RESULT STORE
# =============================================================================

class Results:
    def __init__(self, path: str, config: dict, fresh: bool = False):
        self.path = Path(path)
        self.data: Dict[str, Any] = {"config": config, "meta": {}, "steps": {}}
        if self.path.exists() and not fresh:
            try:
                old = json.loads(self.path.read_text())
                if bool(old.get("config", {}).get("smoke")) == bool(config.get("smoke")):
                    self.data["meta"] = old.get("meta", {})
                    self.data["steps"] = old.get("steps", {})
                    log(f"Resuming from {self.path} ({len(self.data['steps'])} steps present)")
                else:
                    log(f"Existing {self.path} has different smoke flag; starting fresh")
            except Exception as e:  # noqa: BLE001
                log(f"Could not read {self.path}: {e}; starting fresh")
        self.data["config"] = config
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.data["elapsed_s_this_run"] = round(time.time() - T0, 1)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str))
        os.replace(tmp, self.path)

    def done(self, key: str) -> bool:
        v = self.data["steps"].get(key)
        return isinstance(v, dict) and "error" not in v

    def get(self, key: str) -> Any:
        return self.data["steps"].get(key)

    def set(self, key: str, val: dict) -> None:
        self.data["steps"][key] = val
        self.save()


# =============================================================================
# SOCKET CALIBRATION & GATE PROBE
# =============================================================================

def dual_rail_clamp(z: torch.Tensor, dz: float = 0.3) -> torch.Tensor:
    """Deadzone hard-sigmoid clamp: <=0.3 -> 0.0, >=0.85 -> 1.0."""
    return (F.relu(z - dz) / (0.85 - dz)).clamp(0.0, 1.0)


def calibrate_port_socket(
    base_llm: nn.Module,
    d_read: int,
    port_token_ids: List[List[int]],
    other_ch: int,
    const_ch: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Closed-form ridge regression mapping frozen LLM token embeddings -> organ read-in bus.
    Zero task labels. Background vocab tokens map to other_ch; const_ch = 1.0 everywhere.
    """
    E = base_llm.get_input_embeddings().weight.detach().float()
    V, _ = E.shape
    special = set()
    for grp in port_token_ids:
        special.update(grp)

    rng = random.Random(0)
    bg_pool = min(V, 100000)
    bg_n = min(16384, bg_pool)
    bg = [i for i in rng.sample(range(bg_pool), bg_n) if i not in special]
    spec_idxs: List[int] = []
    spec_targets: List[int] = []
    for ch, grp in enumerate(port_token_ids):
        for tid in grp:
            spec_idxs.append(tid)
            spec_targets.append(ch)

    Xs = E[spec_idxs]
    Xb = E[bg]
    Ys = torch.zeros(len(spec_idxs), d_read, device=E.device, dtype=torch.float32)
    for row_i, ch in enumerate(spec_targets):
        Ys[row_i, ch] = 1.0
    Ys[:, const_ch] = 1.0
    Yb = torch.zeros(len(bg), d_read, device=E.device, dtype=torch.float32)
    Yb[:, other_ch] = 1.0
    Yb[:, const_ch] = 1.0

    Xs_aug = torch.cat([Xs, torch.ones(Xs.shape[0], 1, device=E.device)], dim=1)
    Xb_aug = torch.cat([Xb, torch.ones(Xb.shape[0], 1, device=E.device)], dim=1)
    special_w = 256.0
    XtX = special_w * (Xs_aug.T @ Xs_aug) + (Xb_aug.T @ Xb_aug)
    XtX.diagonal().add_(1e-2)
    XtY = special_w * (Xs_aug.T @ Ys) + (Xb_aug.T @ Yb)
    sol = torch.linalg.solve(XtX, XtY)
    return sol[:-1].T.contiguous().to(device), sol[-1].contiguous().to(device)


class GateProbe(nn.Module):
    """1,537-param logistic gate on frozen LLM last_hidden_state: g = sigmoid(w . h + b)."""

    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.lin(h.float()))


# =============================================================================
# TOKEN-LEVEL EXPRESSION VALIDATOR (STRICT BINARY INTEGER +/- ONLY)
# =============================================================================

class TokenVocabMap:
    """Maps BPE token IDs to symbol categories for active-expression parsing and socket calibration."""

    def __init__(self, tok):
        self.tok = tok
        self.digit_ids: List[int] = []
        for d in range(10):
            ids = tok.encode(str(d), add_special_tokens=False)
            assert len(ids) == 1, f"Digit '{d}' is not a single token: {ids}"
            self.digit_ids.append(ids[0])
        self.digit_map: Dict[int, str] = {tid: str(d) for d, tid in enumerate(self.digit_ids)}

        def single_ids(variants: Sequence[str]) -> List[int]:
            out = []
            for v in variants:
                ids = tok.encode(v, add_special_tokens=False)
                if len(ids) == 1 and ids[0] not in out:
                    out.append(ids[0])
            return out

        self.plus_ids = single_ids(["+", " +", "\n+", "\n +"])
        self.minus_ids = single_ids(["-", " -", "\n-", "\n -"])
        self.eq_ids = single_ids(["=", " =", "\n="])
        self.space_ids = single_ids([" ", "  ", "\t", "\n", "\n\n", " \n", "\r\n"])
        self.comma_ids = single_ids([","])
        self.minus_emit_id = tok.encode("-", add_special_tokens=False)[0]
        self.sink_id = self.space_ids[0] if self.space_ids else tok.encode(":", add_special_tokens=False)[0]
        # 11 output symbol token IDs: '0'..'9' + '-'
        self.sym_out_ids = list(self.digit_ids) + [self.minus_emit_id]

        self.plus_set = set(self.plus_ids)
        self.minus_set = set(self.minus_ids)
        self.eq_set = set(self.eq_ids)
        self.space_set = set(self.space_ids)
        self.comma_set = set(self.comma_ids)
        self.sym_cache: Dict[int, str] = {}

    def sym_of(self, tid: int) -> str:
        if tid in self.digit_map:
            return self.digit_map[tid]
        if tid in self.plus_set:
            return "+"
        if tid in self.minus_set:
            return "-"
        if tid in self.eq_set:
            return "="
        if tid in self.space_set:
            return "SPACE"
        if tid in self.comma_set:
            return "COMMA"
        return "OTHER"

    def piece(self, tid: int) -> str:
        s = self.sym_cache.get(tid)
        if s is None:
            s = self.tok.decode([tid])
            self.sym_cache[tid] = s
        return s

    def parse_strict_binary_expr(
        self,
        ids: Sequence[int],
        eq_pos: int,
        max_idx: int = 64,
    ) -> Optional[ActiveExprSpan]:
        """Strict validator: returns ActiveExprSpan iff ids[:eq_pos+1] ends in an isolated
        evaluable binary integer `a + b =` or `a - b =` expression (with optional thousands-separator
        commas inside integer operands, e.g. `325,263,952 + 905,359,078 =`).

        Rejects:
          - decimals (`3.5 + 2 =`, `1 + 2.5 =`)
          - variables / identifiers (`x + 5 =`, `2x + 3 =`, `a1 + 2 =`)
          - chained operators (`1 + 2 + 3 =`, `4 * 5 + 2 =`, `10 / 2 - 1 =`, `-5 + 3 =`)
          - malformed commas (`,123 + 4 =`, `3,5 + 2 =`, `123, + 4 =`)
          - assignment / comparison (`==`, `!=`, `<=`, `>=`, `+=`, `-=`)
          - exponent / parenthesized RHS or LHS (`(1 + 2) + 3 =`, `2^3 + 1 =`)
        """
        if eq_pos <= 0 or eq_pos >= len(ids) or ids[eq_pos] not in self.eq_set:
            return None

        # Only look back up to 3*max_idx + 24 tokens (leaves headroom for thousands commas)
        win_start = max(0, eq_pos - (3 * max_idx + 24))
        sub = ids[win_start:eq_pos + 1]
        syms = [self.sym_of(t) for t in sub]
        local_eq = len(sub) - 1

        span = parse_active_expr_symbols(syms, local_eq, max_idx=max_idx)
        if span is None:
            return None

        # Check token immediately before operand `a`
        before_a = span.start_idx - 1
        # Check raw piece immediately touching a (no space)
        if before_a >= 0 and syms[before_a] != "SPACE":
            prev_txt = self.piece(sub[before_a])
            # Reject if directly attached to decimal point, comma, letter, digit, or operator
            if prev_txt and prev_txt[-1] in ".,*/^%+-_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789)":
                return None

        # Also skip spaces before `a` to reject chained operators like `10 + 20 + 30 =` or `5 * 4 + 2 =`
        j = before_a
        while j >= 0 and syms[j] == "SPACE":
            j -= 1
        if j >= 0:
            prev_nonspace = self.piece(sub[j]).rstrip()
            if prev_nonspace and prev_nonspace[-1] in "+-*/^%":
                return None

        # Check no weird characters inside the window between a_start and eq_pos (only digits, spaces, op, =)
        # parse_active_expr_symbols already guarantees only SPACE between a, op, b, =.
        # Convert span indices back to global `ids` indices
        return ActiveExprSpan(
            start_idx=span.start_idx + win_start,
            a_positions=[p + win_start for p in span.a_positions],
            op_idx=span.op_idx + win_start,
            op=span.op,
            b_positions=[p + win_start for p in span.b_positions],
            eq_idx=eq_pos,
        )


# =============================================================================
# GRAFT MODULE (FROZEN LLM + FROZEN ORGAN + TRAINABLE W_OUT + GATE PROBE)
# =============================================================================

N_WOUT_CH = N_SYM_OUT + 1  # 0..9 digits, 10 = '-', 11 = STOP_SPACE (' ')
STOP_CH_VAL = math.tanh(6.0 / 10.0) * 10.0


class ThinkCalcGraft(nn.Module):
    """Frozen DeepSeek-R1 + closed-form ridge read-in + compiled 2-layer SparseOrgan + W_out."""

    def __init__(
        self,
        base_llm: nn.Module,
        organ: SparseOrgan,
        cal_in: Tuple[torch.Tensor, torch.Tensor],
        vmap: TokenVocabMap,
        max_idx: int = 64,
    ):
        super().__init__()
        self.base_llm = base_llm
        self.organ = organ
        self.vmap = vmap
        self.max_idx = max_idx
        self.register_buffer("W_in", cal_in[0].float().clone(), persistent=False)
        self.register_buffer("b_in", cal_in[1].float().clone(), persistent=False)
        d_llm = int(base_llm.config.hidden_size)
        self.W_out = nn.Linear(N_WOUT_CH, d_llm, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.W_out.weight)

    @torch.no_grad()
    def read_in(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [B, T] -> clamped read-in [B, T, N_READ]."""
        h0 = self.base_llm.get_input_embeddings()(ids).float()
        return dual_rail_clamp(h0 @ self.W_in.T + self.b_in)

    def organ_delta_on_window(
        self,
        win_ids: torch.Tensor,
        a_local: Sequence[int],
        b_local: Sequence[int],
        eq_local: int,
        answer_target_places: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run frozen organ on a single [1, T_win] window -> (delta [1, T_win, d_llm], gate [1, T_win]).

        Channel 11 (STOP_SPACE) fires for 1 step at the first post-units position (target_place == 0),
        emitting a trailing space (' ') before the structural gate shuts off.
        """
        z = self.read_in(win_ids)
        T_win = win_ids.shape[1]
        seg_a, seg_b, places = build_window_structural_features(
            T_win, a_local, b_local, eq_local, answer_target_places, self.max_idx, device=win_ids.device
        )
        x = assemble_bus(z, seg_a.unsqueeze(0), seg_b.unsqueeze(0), places.unsqueeze(0), self.max_idx)
        with torch.no_grad():
            o = self.organ.forward_states(x)
            sym_dig = torch.tanh(o[..., :N_SYM_OUT] / 10.0) * 10.0
            gate_dig = o[..., 11].clamp(0.0, 1.0)
            stop_mask = torch.zeros(1, T_win, dtype=torch.float32, device=win_ids.device)
            for off, p_val in enumerate(answer_target_places):
                if p_val == 0 and (eq_local + off) < T_win:
                    stop_mask[0, eq_local + off] = 1.0
                    break
            sym_stop = (stop_mask * STOP_CH_VAL).unsqueeze(-1)
            sym = torch.cat([sym_dig, sym_stop], dim=-1)  # [1, T_win, 12]
            gate = (gate_dig + stop_mask).clamp(0.0, 1.0)
        delta = gate.unsqueeze(-1) * self.W_out(sym)
        return delta, gate


# =============================================================================
# DATA GENERATORS (STANDALONE OOD + BIG-NUMBER WORD PROBLEMS)
# =============================================================================

def rand_int_digits(rng: random.Random, n: int) -> int:
    if n <= 1:
        return rng.randint(0, 9)
    return rng.randint(10 ** (n - 1), 10 ** n - 1)


def gen_expr_pair(rng: random.Random, bucket: str) -> Tuple[int, str, int, int]:
    """Generate (a, op, b, answer) for a standalone eval or training bucket."""
    if bucket == "train_3_6":
        la = rng.randint(3, 6)
        lb = rng.randint(3, 6)
        a = rand_int_digits(rng, la)
        b = rand_int_digits(rng, lb)
        op = "+" if rng.random() < 0.45 else "-"
        if op == "-" and rng.random() < 0.5 and a >= b:
            a, b = b, a
        return a, op, b, (a + b if op == "+" else a - b)

    if bucket == "id_3_6":
        a = rand_int_digits(rng, rng.randint(3, 6))
        b = rand_int_digits(rng, rng.randint(3, 6))
        op = "+" if rng.random() < 0.5 else "-"
        return a, op, b, (a + b if op == "+" else a - b)

    if bucket in ("ood_12", "ood_24", "ood_48"):
        L = int(bucket.split("_")[1])
        a = rand_int_digits(rng, L)
        b = rand_int_digits(rng, L)
        op = "+" if rng.random() < 0.5 else "-"
        return a, op, b, (a + b if op == "+" else a - b)

    if bucket == "chain_48":
        L = 48
        if rng.random() < 0.5:
            # 48-digit carry cascade
            a_digs = [rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(L - 2)]
            b_digs = [9 - d for d in a_digs]
            if b_digs[0] == 0:
                b_digs[0] = 1; a_digs[0] = 8
            a0 = rng.randint(1, 9); b0 = rng.randint(10 - a0, 9)
            a = int("".join(map(str, a_digs + [a0])))
            b = int("".join(map(str, b_digs + [b0])))
            return a, "+", b, a + b
        else:
            # 48-digit borrow cascade
            a = 10 ** (L - 1)
            b = rng.randint(10 ** 3, 10 ** 8 - 1)
            return a, "-", b, a - b

    if bucket == "mixed_len_24_4":
        big = rand_int_digits(rng, 24)
        small = rand_int_digits(rng, 4)
        op = "+" if rng.random() < 0.5 else "-"
        if rng.random() < 0.5:
            a, b = big, small
        else:
            a, b = small, big
        return a, op, b, (a + b if op == "+" else a - b)

    if bucket == "negative":
        L = rng.randint(12, 32)
        a = rand_int_digits(rng, L)
        b = rand_int_digits(rng, L)
        if a >= b:
            a, b = b, a + rng.randint(1, 10 ** min(6, L))
        return a, "-", b, a - b

    raise ValueError(f"Unknown bucket: {bucket}")


def make_big_number_word_problems(n: int, seed: int = 202601) -> List[Dict[str, Any]]:
    """Deterministic benchmark of multi-step (2-4 step) +/- word problems with 9-14 digit integers.
    Numbers are formatted without commas so the model naturally writes raw integer equations.
    """
    rng = random.Random(seed)
    problems: List[Dict[str, Any]] = []
    cities = ["Northbridge", "Valoria", "Kestrel", "Solhaven", "Alderbrook", "Maris", "Oakhaven", "Vesper"]
    orgs = ["Apex Energy", "Helios Logistics", "Meridian Rail", "Orion Systems", "Vertex Bio", "Cobalt Mining"]
    items = ["microchips", "solar cells", "battery packs", "fiber cables", "sensor modules", "turbine blades"]

    for i in range(n):
        tpl = i % 5
        L1 = rng.randint(9, 14)
        L2 = rng.randint(9, 14)
        L3 = rng.randint(9, 13)
        L4 = rng.randint(9, 12)

        if tpl == 0:
            # 2-step national/org budget (add then subtract)
            org = rng.choice(orgs)
            a = rand_int_digits(rng, L1)
            b = rand_int_digits(rng, L2)
            c = rng.randint(10 ** (min(L1, L2) - 1), a + b - 1)
            ans = a + b - c
            q = (
                f"{org} started the year with a reserve fund of {a} dollars. "
                f"During the year, it received {b} dollars in revenue and spent {c} dollars on operations. "
                f"How many dollars remain in the reserve fund at the end of the year?"
            )
            steps = 2
        elif tpl == 1:
            # 3-step population / census tracking
            city = rng.choice(cities)
            a = rand_int_digits(rng, L1)
            b = rand_int_digits(rng, L2)
            c = rand_int_digits(rng, L3)
            d = rng.randint(10 ** 8, min(a + b + c - 1, 10 ** L4 - 1))
            ans = a + b + c - d
            q = (
                f"A regional network around {city} recorded {a} data packets on Monday, "
                f"{b} data packets on Tuesday, and {c} data packets on Wednesday. "
                f"If {d} corrupted packets were discarded across the three days, "
                f"how many valid data packets remained?"
            )
            steps = 3
        elif tpl == 2:
            # 3-step warehouse inventory (subtract, add, subtract)
            item = rng.choice(items)
            a = rand_int_digits(rng, max(L1, 11))
            b = rng.randint(10 ** 8, a // 2)
            c = rand_int_digits(rng, L2)
            d = rng.randint(10 ** 8, (a - b + c) // 2)
            ans = a - b + c - d
            q = (
                f"A global distributor held {a} {item} in stock. "
                f"It shipped out {b} {item} in Q1, received a new delivery of {c} {item} in Q2, "
                f"and shipped out {d} {item} in Q3. "
                f"How many {item} are left in stock?"
            )
            steps = 3
        elif tpl == 3:
            # 2-step astronomical distance difference
            a = rand_int_digits(rng, L1)
            b = rand_int_digits(rng, L2)
            c = rand_int_digits(rng, L3)
            # Ensure positive difference or allow negative? Keep positive for clear word-problem semantics
            total_ab = a + b
            if c > total_ab:
                a, c = c, a
                total_ab = a + b
            ans = total_ab - c
            q = (
                f"Probe Alpha traveled {a} kilometers in its first leg and {b} kilometers in its second leg. "
                f"Probe Beta traveled a total of {c} kilometers. "
                f"How many more kilometers did Probe Alpha travel than Probe Beta?"
            )
            steps = 2
        else:
            # 4-step bank ledger (3 deposits + 2 withdrawals or 2+2)
            a = rand_int_digits(rng, max(L1, 11))
            b = rand_int_digits(rng, L2)
            c = rng.randint(10 ** 8, a // 2)
            d = rand_int_digits(rng, L3)
            e = rng.randint(10 ** 8, (a + b - c + d) // 2)
            ans = a + b - c + d - e
            q = (
                f"A clearinghouse account opened with {a} cents. "
                f"It processed a deposit of {b} cents, a withdrawal of {c} cents, "
                f"a second deposit of {d} cents, and a final withdrawal of {e} cents. "
                f"What is the closing balance in cents?"
            )
            steps = 4

        problems.append({
            "id": i,
            "template": tpl,
            "n_steps": steps,
            "question": q,
            "answer": ans,
        })
    return problems


def make_2step_8_10_word_problems(n: int, seed: int = 202602) -> List[Dict[str, Any]]:
    """Deterministic benchmark of clean 1-2 step +/- word problems with 8-10 digit integers.
    Covers `a + b - c`, `a - b + c`, and `a + b` with raw unformatted integers.
    """
    rng = random.Random(seed)
    problems: List[Dict[str, Any]] = []
    orgs = ["Apex Energy", "Helios Logistics", "Meridian Rail", "Orion Systems", "Vertex Bio", "Cobalt Mining"]
    items = ["microchips", "solar cells", "battery packs", "fiber cables", "sensor modules", "turbine blades"]
    regions = ["Northbridge", "Valoria", "Kestrel", "Solhaven", "Alderbrook", "Maris"]

    for i in range(n):
        tpl = i % 3
        L1 = rng.randint(8, 10)
        L2 = rng.randint(8, 10)
        L3 = rng.randint(8, 10)

        if tpl == 0:
            # 2-step: a + b - c
            org = rng.choice(orgs)
            a = rand_int_digits(rng, L1)
            b = rand_int_digits(rng, L2)
            c = rand_int_digits(rng, L3)
            if c >= a + b:
                c = rng.randint(10 ** 7, a + b - 1)
            ans = a + b - c
            q = (
                f"{org} started the year with {a} dollars in reserves. "
                f"It received {b} dollars in revenue and spent {c} dollars on operations. "
                f"How many dollars remain in reserves?"
            )
            steps = 2
        elif tpl == 1:
            # 2-step: a - b + c
            item = rng.choice(items)
            a = rand_int_digits(rng, max(L1, 9))
            b = rand_int_digits(rng, min(L2, len(str(a))))
            if b >= a:
                b = rng.randint(10 ** 7, a - 1)
            c = rand_int_digits(rng, L3)
            ans = a - b + c
            q = (
                f"A warehouse held {a} {item} in stock. "
                f"It shipped out {b} {item} and later received a new delivery of {c} {item}. "
                f"How many {item} are in stock now?"
            )
            steps = 2
        else:
            # 1-step: a + b
            reg = rng.choice(regions)
            a = rand_int_digits(rng, L1)
            b = rand_int_digits(rng, L2)
            ans = a + b
            q = (
                f"Two facilities in {reg} produced {a} units in the first half of the year "
                f"and {b} units in the second half of the year. "
                f"How many total units did they produce?"
            )
            steps = 1

        problems.append({
            "id": i,
            "template": tpl,
            "n_steps": steps,
            "question": q,
            "answer": ans,
        })
    return problems


def make_word_problems(n: int, problem_set: str = "default") -> List[Dict[str, Any]]:
    if problem_set == "2step_8_10":
        return make_2step_8_10_word_problems(n, seed=202602)
    if problem_set == "default":
        return make_big_number_word_problems(n, seed=202601)
    raise ValueError(f"Unknown problem_set: {problem_set!r}")


def make_hard_negative_think_prompts(n: int, seed: int = 999) -> List[str]:
    """Prompts that encourage <think> traces with algebra, multiplication, division, and decimals
    so GateProbe sees abundant in-distribution hard negatives inside <think>."""
    rng = random.Random(seed)
    prompts = []
    for i in range(n):
        k = i % 4
        if k == 0:
            a, b = rng.randint(2, 19), rng.randint(5, 80)
            prompts.append(f"Solve for x: {a}x + {b} = {a * rng.randint(2, 30) + b}.")
        elif k == 1:
            a, b = rng.randint(12, 999), rng.randint(12, 99)
            prompts.append(f"Compute the product {a} * {b} and divide the result by 2.")
        elif k == 2:
            a, b = round(rng.uniform(1.5, 99.5), 2), round(rng.uniform(1.5, 99.5), 2)
            prompts.append(f"A rectangle has length {a} meters and width {b} meters. Find its area and perimeter.")
        else:
            a, b, c = rng.randint(10, 99), rng.randint(10, 99), rng.randint(10, 99)
            prompts.append(f"What is the average of {a}, {b}, and {c}? Show each step.")
    return prompts


def make_direct_arith_think_prompts(n: int, seed: int = 33333) -> List[str]:
    """Concise 2-step +/- prompts that quickly produce explicit binary integer '=' equations
    inside <think> (even within 256-320 tokens)."""
    rng = random.Random(seed)
    prompts = []
    for i in range(n):
        a = rand_int_digits(rng, rng.randint(2, 5))
        b = rand_int_digits(rng, rng.randint(2, 5))
        c = rand_int_digits(rng, rng.randint(2, 4))
        k = i % 3
        if k == 0:
            prompts.append(f"Compute {a} + {b}, then subtract {c} from the result.")
        elif k == 1:
            prompts.append(f"Start with {a + b}, subtract {b}, and then add {c}.")
        else:
            prompts.append(f"Find the value of ({a} + {b}) - {c}. Show each arithmetic step.")
    return prompts


def make_synth_think_gate_examples(n_pos: int, n_neg: int, seed: int) -> Tuple[List[str], List[str]]:
    """Build diverse <think>-wrapped positive binary integer `a +/- b =` examples and matched
    <think>-wrapped hard-negative examples (algebra, mul/div, chained ops, decimals, assignments).
    Every returned string ends with '=' (or ' =') so its final token is the target '=' position."""
    rng = random.Random(seed)
    pos_prefixes = [
        "<think>\nLet's compute ",
        "<think>\nFirst, we calculate ",
        "<think>\nStep 1: we need to find ",
        "<think>\nNext, adding the two values gives ",
        "<think>\nSo the intermediate result is ",
        "<think>\nLet's check the units: ",
        "<think>\nStep 2: total = ",
        "<think>\nNow, subtracting the expenses: ",
    ]
    neg_prefixes = [
        "<think>\nLet's compute ",
        "<think>\nFirst, we set up ",
        "<think>\nStep 1: we have ",
        "<think>\nNext, multiplying gives ",
        "<think>\nSo the equation is ",
        "<think>\nLet's check the formula: ",
        "<think>\nStep 2: ",
        "<think>\nNow, simplifying gives ",
    ]

    pos_texts: List[str] = []
    for i in range(n_pos):
        la = rng.choice([1, 2, 3, 4, 5, 6, 9, 12])
        lb = rng.choice([1, 2, 3, 4, 5, 6, 9, 12])
        a = rand_int_digits(rng, la)
        b = rand_int_digits(rng, lb)
        op = "+" if rng.random() < 0.5 else "-"
        sp = " " if rng.random() < 0.8 else ""
        pref = pos_prefixes[i % len(pos_prefixes)]
        # Optionally prepend a completed prior step so length & context match real <think> traces
        if i % 3 == 0:
            p1, p2 = rand_int_digits(rng, 3), rand_int_digits(rng, 3)
            pref = f"<think>\nWe start with {p1} and {p2}. First, {p1} + {p2} = {p1 + p2}.\nNext, "
        use_commas = (a >= 1000 or b >= 1000) and (rng.random() < 0.35)
        a_s = f"{a:,}" if use_commas else str(a)
        b_s = f"{b:,}" if use_commas else str(b)
        pos_texts.append(f"{pref}{a_s}{sp}{op}{sp}{b_s}{sp}=")

    neg_texts: List[str] = []
    vars_list = ["x", "y", "n", "total", "balance", "diff", "result", "area"]
    for i in range(n_neg):
        pref = neg_prefixes[i % len(neg_prefixes)]
        k = i % 7
        a = rand_int_digits(rng, rng.randint(1, 5))
        b = rand_int_digits(rng, rng.randint(1, 5))
        c = rand_int_digits(rng, rng.randint(1, 4))
        v = rng.choice(vars_list)
        if k == 0:
            # Multiplication
            neg_texts.append(f"{pref}{a} * {b} =")
        elif k == 1:
            # Division
            neg_texts.append(f"{pref}{a * rng.randint(2, 9)} / {max(2, b % 97)} =")
        elif k == 2:
            # Variable assignment
            neg_texts.append(f"{pref}{v} =")
        elif k == 3:
            # Algebra
            neg_texts.append(f"{pref}{rng.randint(2, 15)}{v} + {b} =")
        elif k == 4:
            # Chained operator (3 operands)
            op1, op2 = rng.choice(["+", "-"]), rng.choice(["+", "-"])
            neg_texts.append(f"{pref}{a} {op1} {b} {op2} {c} =")
        elif k == 5:
            # Decimal arithmetic
            da = round(rng.uniform(1.1, 99.9), 2)
            db = round(rng.uniform(1.1, 99.9), 2)
            op = rng.choice(["+", "-"])
            neg_texts.append(f"{pref}{da} {op} {db} =")
        else:
            # Parenthesized or mixed-op expression
            if i % 2 == 0:
                neg_texts.append(f"{pref}({a} + {b}) * {c} =")
            else:
                neg_texts.append(f"{pref}{a} * {b} + {c} =")

    return pos_texts, neg_texts


# =============================================================================
# ANSWER EXTRACTION & ARITHMETIC-STEP AUDIT
# =============================================================================

def extract_final_int(text: str) -> Optional[int]:
    """Extract final integer answer from a DeepSeek-R1 response (prefers \\boxed{...}, then post-</think>)."""
    # 1. Look for \boxed{...} (last occurrence)
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        cand = boxed[-1].replace(",", "").replace("$", "").replace(" ", "").strip()
        m = re.search(r"-?\d+", cand)
        if m:
            try:
                return int(m.group(0))
            except ValueError:
                pass
    # 2. Look after </think> if present
    tail = text.split("</think>")[-1] if "</think>" in text else text
    # Strip commas inside numbers
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", tail)
    nums = re.findall(r"-?\d+", cleaned)
    if nums:
        try:
            return int(nums[-1])
        except ValueError:
            pass
    return None


def parse_gsm8k_answer(ans_str: str) -> Optional[int]:
    m = re.search(r"####\s*(-?[\d,]+)", ans_str)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


RECHECK_PAT = re.compile(
    r"\b(wait|hold on|double[- ]check|let me check|recalculate|re-check|recheck|let's check|actually)\b",
    re.IGNORECASE,
)


def default_tag_for_model(model_name: str) -> str:
    """Derive a clean filename-safe tag from a HuggingFace model ID."""
    known = {
        "deepseek-ai/deepseek-r1-distill-qwen-1.5b": "r1-1.5b",
        "qwen/qwen3-4b": "qwen3-4b",
        "xiaomimimo/mimo-7b-rl": "mimo-7b-rl",
    }
    low = model_name.strip().lower()
    if low in known:
        return known[low]
    base = model_name.strip().split("/")[-1].lower()
    return re.sub(r"[^a-z0-9._-]+", "-", base).strip("-") or "model"


BIN_EQ_IN_THINK_PAT = re.compile(r"(?<![\w.,])(\d+(?:,\d+)*)\s*([+-])\s*(\d+(?:,\d+)*)\s*=(?!=)")
MULTI_TERM_SUM_PAT = re.compile(r"(?<![\w.])\d[\d,]*\s*[+-]\s*\d[\d,]*\s*[+-]\s*\d[\d,]*(?![\w.])")
BOXED_NUM_PAT = re.compile(r"\\boxed\{[^}]*\d[^}]*\}")

FAILURE_TAG_KEYS = (
    "truncated_think",
    "hit_token_limit",
    "hit_limit_before_boxed",
    "no_final_answer",
    "wrote_big_binary_eq",
    "column_by_column_only",
    "wrote_multi_term_sum",
    "missing_input_numbers",
)


def diagnose_failure(
    question: str,
    text: str,
    pred: Optional[int],
    n_gen_tokens: int,
    max_new_tokens: int,
) -> Dict[str, bool]:
    """Compute per-problem diagnostic tags for a thinking trace."""
    has_close = "</think>" in text
    think_part = text.split("</think>", 1)[0] if has_close else text
    if think_part.startswith("<think>"):
        think_part = think_part[len("<think>"):]

    truncated_think = not has_close
    hit_token_limit = int(n_gen_tokens) >= int(max_new_tokens)
    has_boxed = BOXED_NUM_PAT.search(text) is not None
    hit_limit_before_boxed = bool(hit_token_limit and not has_boxed)
    no_final_answer = bool(has_close and (pred is None))

    # Inspect isolated binary `a +/- b =` equations inside <think> (raw or thousands-comma formatted)
    big_bin_count = 0
    small_bin_count = 0
    for m in BIN_EQ_IN_THINK_PAT.finditer(think_part):
        # Reject chained left-hand operators (e.g. `10 + 20 + 30 =`)
        j = m.start() - 1
        while j >= 0 and think_part[j] in " \t":
            j -= 1
        if j >= 0 and think_part[j] in "+-*/^%":
            continue
        a_s = m.group(1).replace(",", "")
        b_s = m.group(3).replace(",", "")
        if len(a_s) >= 6 or len(b_s) >= 6:
            big_bin_count += 1
        elif len(a_s) <= 2 and len(b_s) <= 2:
            small_bin_count += 1

    wrote_big_binary_eq = big_bin_count >= 1
    column_by_column_only = (small_bin_count >= 3) and (big_bin_count == 0)
    wrote_multi_term_sum = MULTI_TERM_SUM_PAT.search(think_part) is not None

    # Check if any >=6-digit input number from the question never appears in <think> (raw or comma-formatted)
    big_inputs = re.findall(r"(?<!\d)\d{6,}(?!\d)", question)
    think_no_commas = re.sub(r"(?<=\d),(?=\d)", "", think_part)
    missing_input_numbers = False
    for num_s in big_inputs:
        if re.search(rf"(?<!\d){re.escape(num_s)}(?!\d)", think_no_commas) is None:
            missing_input_numbers = True
            break

    return {
        "truncated_think": truncated_think,
        "hit_token_limit": hit_token_limit,
        "hit_limit_before_boxed": hit_limit_before_boxed,
        "no_final_answer": no_final_answer,
        "wrote_big_binary_eq": wrote_big_binary_eq,
        "column_by_column_only": column_by_column_only,
        "wrote_multi_term_sum": wrote_multi_term_sum,
        "missing_input_numbers": missing_input_numbers,
    }


def aggregate_failure_breakdown(
    tags_list: Sequence[Dict[str, bool]],
    correct_list: Sequence[bool],
    gen_lens: Sequence[int],
) -> Dict[str, Any]:
    """Aggregate per-problem diagnostic tag counts and generation length stats."""
    n = len(tags_list)
    out: Dict[str, Any] = {
        "n": n,
        "n_wrong": sum(1 for c in correct_list if not c),
        "mean_gen_tokens": round(sum(gen_lens) / max(1, n), 2),
    }
    for k in FAILURE_TAG_KEYS:
        out[k] = sum(1 for t in tags_list if t.get(k))
    return out


def audit_think_arithmetic(text: str) -> Dict[str, int]:
    """Audit all binary integer `<num> [+-] <num> = <num>` equations written inside <think>...</think>
    (supports raw, thousands-comma, mis-grouped comma, and newline-separated integers)."""
    think_part = text.split("</think>")[0] if "</think>" in text else text
    pat = re.compile(
        r"(?<![\w.,])(\d+(?:,\d+)*)\s*([+-])\s*(\d+(?:,\d+)*)\s*=\s*(-?\d+(?:,\d+)*)(?![\w]|\.\d)"
    )
    n_tot = 0
    n_ok = 0
    for m in pat.finditer(think_part):
        a = int(m.group(1).replace(",", ""))
        op = m.group(2)
        b = int(m.group(3).replace(",", ""))
        c = int(m.group(4).replace(",", ""))
        exp = a + b if op == "+" else a - b
        n_tot += 1
        if c == exp:
            n_ok += 1
    return {"n_eq": n_tot, "n_correct": n_ok}


def grade_thinking(
    traces: Sequence[Dict[str, Any]],
    tokenizer: Any,
    max_new_tokens: int,
    is_grafted: bool = False,
) -> Dict[str, Any]:
    """Grade thinking traces for Base or Grafted arm.

    Each element of `traces` is a dict:
      {"text": str, "gen_ids": List[int], "graft_events": Optional[List[Dict[str, Any]]]}
    where each graft event has {"a": str, "b": str, "op": str, "end_gen_idx": int}.

    Returns:
      - mean_think_tokens, median_think_tokens
      - truncated_rate (hit max_new_tokens without closing </think>)
      - mean_rechecks (count of RECHECK_PAT matches inside <think>)
      - in_think_arith_acc, n_equations
      - graft_recheck_rate, n_graft_events (when is_grafted=True)
    """
    n = len(traces)
    if n == 0:
        out: Dict[str, Any] = {
            "mean_think_tokens": 0.0,
            "median_think_tokens": 0.0,
            "truncated_rate": 0.0,
            "mean_rechecks": 0.0,
            "in_think_arith_acc": None,
            "n_equations": 0,
        }
        if is_grafted:
            out["graft_recheck_rate"] = None
            out["n_graft_events"] = 0
        return out

    think_lens: List[int] = []
    truncated = 0
    recheck_counts: List[int] = []
    tot_eq = 0
    ok_eq = 0
    graft_ev_total = 0
    graft_ev_rechecked = 0

    for tr in traces:
        txt = tr["text"]
        gen_ids = tr["gen_ids"]
        has_close = "</think>" in txt
        think_part = txt.split("</think>", 1)[0] if has_close else txt
        if think_part.startswith("<think>"):
            think_part = think_part[len("<think>"):]

        t_toks = len(tokenizer.encode(think_part, add_special_tokens=False))
        think_lens.append(t_toks)

        if (not has_close) and len(gen_ids) >= max_new_tokens:
            truncated += 1

        recheck_counts.append(len(RECHECK_PAT.findall(think_part)))

        aud = audit_think_arithmetic(txt)
        tot_eq += aud["n_eq"]
        ok_eq += aud["n_correct"]

        if is_grafted:
            events = tr.get("graft_events") or []
            for e_idx, ev in enumerate(events):
                graft_ev_total += 1
                end_i = int(ev["end_gen_idx"])
                # Stop lookahead before the next grafted equation starts (if present) so its rechecks don't bleed back
                next_cap = end_i + 80
                if e_idx + 1 < len(events) and "start_gen_idx" in events[e_idx + 1]:
                    next_cap = min(next_cap, int(events[e_idx + 1]["start_gen_idx"]))
                after_txt = tokenizer.decode(gen_ids[end_i:max(end_i, next_cap)], skip_special_tokens=True)[:160]
                has_doubt = RECHECK_PAT.search(after_txt) is not None
                a_s, b_s = str(ev["a"]), str(ev["b"])
                repeats_ops = (a_s in after_txt) and (b_s in after_txt)
                if has_doubt or repeats_ops:
                    graft_ev_rechecked += 1

    sorted_lens = sorted(think_lens)
    mid = n // 2
    median_len = float(sorted_lens[mid]) if n % 2 == 1 else 0.5 * (sorted_lens[mid - 1] + sorted_lens[mid])

    res: Dict[str, Any] = {
        "mean_think_tokens": round(sum(think_lens) / n, 2),
        "median_think_tokens": round(median_len, 2),
        "truncated_rate": round(truncated / n, 4),
        "mean_rechecks": round(sum(recheck_counts) / n, 3),
        "in_think_arith_acc": round(ok_eq / tot_eq, 4) if tot_eq > 0 else None,
        "n_equations": int(tot_eq),
    }
    if is_grafted:
        res["graft_recheck_rate"] = round(graft_ev_rechecked / graft_ev_total, 4) if graft_ev_total > 0 else None
        res["n_graft_events"] = int(graft_ev_total)
    return res


DIRECT_EQ_INSTRUCTION = (
    "Write each addition or subtraction directly on one line as a + b = or a - b = "
    "without breaking numbers into digit-by-digit columns or chunks."
)

CALCULATOR_PROMPT_INSTRUCTION = (
    "Inside your thinking, any result after '=' on a whole-number addition or subtraction comes "
    "from an exact calculator. Write each full addition or subtraction directly on one line as "
    "'a + b =' or 'a - b =' and rely on the result without redoing it by hand."
)


def format_r1_prompt(
    tokenizer,
    question: str,
    direct_eq_prompt: bool = False,
    calculator_prompt: bool = False,
) -> str:
    """Format a question for thinking models ending with '<think>\\n'."""
    if calculator_prompt:
        prefix = f"{CALCULATOR_PROMPT_INSTRUCTION}\n"
    elif direct_eq_prompt:
        prefix = f"{DIRECT_EQ_INSTRUCTION}\n"
    else:
        prefix = ""
    user_msg = (
        f"{prefix}{question}\n"
        "Please reason step by step, and put your final integer answer within \\boxed{}."
    )
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            s = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,
            )
            if "<think>" not in s:
                s = s + "<think>\n"
            return s
        except Exception:  # noqa: BLE001
            pass
    return f"User: {user_msg}\nAssistant: <think>\n"


# =============================================================================
# MID-LAYER TRUST STEERING VECTOR (ANTI-DISTRUST)
# =============================================================================

TRUST_STEER_PAIRS: List[Tuple[str, str, str]] = [
    (
        "<think>\n483,753,062 + 53,623,403,272,946 = 53623887026008",
        ". Now, let's",
        " (Wait, is that right? Let me check.)",
    ),
    (
        "<think>\n325,263,952 + 905,359,078 = 1230623030",
        ". Next, we subtract",
        ". Wait, let me double-check",
    ),
    (
        "<think>\n970,901,515,336 - 637,910,139,374 = 332991375962",
        ". So the total so far is",
        ". Hold on, let me verify",
    ),
    (
        "<think>\n48219384721 + 39571284905 = 87790669626",
        ". Moving to the next step,",
        ". Wait, let me add digit by digit",
    ),
    (
        "<think>\n100000000000 - 48219 = 99999951781",
        ". That gives us",
        ". Wait, did I borrow correctly? Let me check",
    ),
    (
        "<think>\n8,421,903,441 - 3,928,109,057 = 4493794384",
        ". So the remaining balance is",
        ". Hold on, let's recalculate",
    ),
    (
        "<think>\n6,286,917,586,52 + 3,137,930,031 = 631829688683",
        ". Proceeding to subtract",
        ". Wait, that seems off, let me check",
    ),
    (
        "<think>\n31,652,159,587,559 + 7,961,942,11 = 31652955781770",
        ". Finally, we compute",
        ". Actually, let me re-add",
    ),
    (
        "<think>\n741852963012 + 258147036988 = 1000000000000",
        ". Great, now let's",
        ". Wait, let me check each column",
    ),
    (
        "<think>\n509,112,847,300 - 194,883,219,150 = 314229628150",
        ". Therefore, the next step is",
        ". Hold on, is this subtraction correct?",
    ),
    (
        "<think>\n91234567890 + 12345678901 = 103580246791",
        ". Now we subtract the expenses:",
        ". Wait, let me double-check this sum",
    ),
    (
        "<think>\n600,000,000,000 - 123,456,789 = 599876543211",
        ". So the final answer is",
        ". Wait, hold on, let me verify",
    ),
]


@torch.no_grad()
def compute_trust_steering_vector(
    base_llm: nn.Module,
    tokenizer: Any,
    trust_layer: int,
    trust_alpha: float,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute a 1-vector (d_llm params) mid-layer trust steering vector `v_trust`
    at `base_llm.model.layers[trust_layer]` from balanced confident vs. self-doubt
    synthetic `<think>` continuations right after `= <exact_ans>`.

    Scaled to `mean_norm_at_layer * 0.08 * trust_alpha` so magnitude adapts cleanly
    across model sizes (1.5B, 4B, 7B) without overwhelming the residual stream.
    """
    layers = base_llm.model.layers
    n_layers = len(layers)
    layer_idx = max(0, min(n_layers - 1, int(trust_layer)))
    captured: List[torch.Tensor] = []

    def _capture_hook(_module, _inp, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h.detach().float())

    handle = layers[layer_idx].register_forward_hook(_capture_hook)
    pos_vecs: List[torch.Tensor] = []
    neg_vecs: List[torch.Tensor] = []
    try:
        for prefix, pos_cont, neg_cont in TRUST_STEER_PAIRS:
            pref_len = len(tokenizer.encode(prefix, add_special_tokens=False))
            for cont, bucket in ((pos_cont, pos_vecs), (neg_cont, neg_vecs)):
                ids = tokenizer.encode(prefix + cont, add_special_tokens=False)
                ids_t = torch.tensor([ids], dtype=torch.long, device=device)
                captured.clear()
                base_llm.model(input_ids=ids_t, use_cache=False)
                h = captured[0][0]  # [T, d_llm]
                start_i = min(pref_len, h.shape[0] - 1)
                bucket.append(h[start_i:].mean(dim=0))
    finally:
        handle.remove()

    H_pos = torch.stack(pos_vecs, dim=0)
    H_neg = torch.stack(neg_vecs, dim=0)
    diff = H_pos.mean(dim=0) - H_neg.mean(dim=0)
    raw_diff_norm = float(diff.norm().item())
    unit_dir = diff / max(1e-8, raw_diff_norm)
    mean_norm = float(torch.cat([H_pos, H_neg], dim=0).norm(dim=-1).mean().item())
    v_trust = (unit_dir * (mean_norm * 0.08 * float(trust_alpha))).to(device)

    pos_proj = float((H_pos @ unit_dir).mean().item())
    neg_proj = float((H_neg @ unit_dir).mean().item())
    info = {
        "trust_layer": layer_idx,
        "n_layers": n_layers,
        "trust_alpha": float(trust_alpha),
        "n_pairs": len(TRUST_STEER_PAIRS),
        "mean_norm_at_layer": round(mean_norm, 4),
        "raw_diff_norm": round(raw_diff_norm, 4),
        "v_trust_norm": round(float(v_trust.norm().item()), 4),
        "proj_sep": round(pos_proj - neg_proj, 4),
    }
    return v_trust, info


# =============================================================================
# INLINE GATED GENERATION ENGINE (FOR <think> AND STANDALONE)
# =============================================================================

@dataclass
class InlineCalcState:
    active: bool = False
    win_prefix_ids: Optional[List[int]] = None  # [sink] + expr tokens up to '='
    a_local: Optional[List[int]] = None
    b_local: Optional[List[int]] = None
    eq_local: int = 0
    p_top: int = 1
    answer_ids: Optional[List[int]] = None
    target_places: Optional[List[int]] = None
    a_str: str = ""
    b_str: str = ""
    op: str = "+"
    start_gen_idx: int = 0


@torch.no_grad()
def generate_with_inline_calc(
    graft: ThinkCalcGraft,
    prompt_ids: List[int],
    max_new_tokens: int,
    eos_token_id: int,
    probe: Optional[GateProbe] = None,
    use_graft: bool = True,
    force_open_at_eq: bool = False,
    gate_threshold: float = 0.5,
    trust_vec: Optional[torch.Tensor] = None,
    trust_layer: Optional[int] = None,
    trust_steer_steps: int = 3,
    handoff_cue_ids: Optional[List[int]] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """Greedy generation (`do_sample=False`) with optional gated inline calculator organ.

    When `use_graft=True`:
      - At every generated/prompt-final '=' token, checks `probe(h_last) > gate_threshold` (or `force_open_at_eq`).
      - If open and `parse_strict_binary_expr` finds a valid binary integer `a +/- b =`,
        activates `InlineCalcState` (determining `p_top` purely via `organ.forward_states`),
        and injects `gate * W_out(organ_out)` into `h_last` autoregressively until `organ`'s
        structural gate shuts off (`<= 0.5`).
      - Optionally streams `handoff_cue_ids` (e.g. `".\\nNow, "`) right after emitting a >=4-digit
        answer instead of a single space, and/or applies mid-layer `trust_vec` steering on the
        next `trust_steer_steps` (default 3) host-generated tokens after each completed organ emission.
    """
    dev = next(graft.W_out.parameters()).device
    lm_head = graft.base_llm.get_output_embeddings()
    vmap = graft.vmap

    full_ids = list(prompt_ids)
    cur_in = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
    past = None
    generated: List[int] = []
    state = InlineCalcState()
    pending_cue_ids: List[int] = []
    steer_box = [0]  # post_graft_steer_steps counter for mid-layer trust hook

    hook_handle = None
    if use_graft and trust_vec is not None and trust_layer is not None:
        tv = trust_vec.to(device=dev)

        def _trust_hook(_module, _inp, output):
            if steer_box[0] <= 0:
                return output
            if isinstance(output, tuple):
                h = output[0].clone()
                h[:, -1, :] = h[:, -1, :] + tv.to(dtype=h.dtype)
                return (h,) + output[1:]
            h = output.clone()
            h[:, -1, :] = h[:, -1, :] + tv.to(dtype=h.dtype)
            return h

        hook_handle = graft.base_llm.model.layers[int(trust_layer)].register_forward_hook(_trust_hook)

    stats: Dict[str, Any] = {
        "total_eq_seen": 0,
        "valid_expr_eq": 0,
        "probe_fires": 0,
        "true_fires": 0,
        "false_probe_fires": 0,
        "missed_exprs": 0,
        "organ_tokens_emitted": 0,
        "graft_events": [],
    }

    try:
        for _step_i in range(max_new_tokens):
            was_steering = steer_box[0] > 0
            out = graft.base_llm.model(input_ids=cur_in, past_key_values=past, use_cache=True)
            past = out.past_key_values
            h_last = out.last_hidden_state[:, -1, :].float()  # [1, d_llm]
            if was_steering:
                steer_box[0] = max(0, steer_box[0] - 1)

            forced_nxt: Optional[int] = None

            if use_graft and pending_cue_ids:
                forced_nxt = pending_cue_ids.pop(0)
                if not pending_cue_ids and trust_vec is not None:
                    steer_box[0] = trust_steer_steps
            elif use_graft:
                last_pos = len(full_ids) - 1
                last_tid = full_ids[-1]

                # If currently inactive and we just landed on '=', check gate!
                if not state.active and last_tid in vmap.eq_set:
                    stats["total_eq_seen"] += 1
                    span = vmap.parse_strict_binary_expr(full_ids, last_pos, max_idx=graft.max_idx)
                    is_valid = span is not None
                    if is_valid:
                        stats["valid_expr_eq"] += 1

                    if force_open_at_eq:
                        probe_open = True
                    elif probe is not None:
                        probe_open = bool(float(probe(h_last).item()) > gate_threshold)
                    else:
                        probe_open = True

                    if probe_open:
                        stats["probe_fires"] += 1
                        if is_valid:
                            stats["true_fires"] += 1
                        else:
                            stats["false_probe_fires"] += 1
                    elif is_valid:
                        stats["missed_exprs"] += 1

                    if probe_open and span is not None:
                        win_expr = [vmap.sink_id] + full_ids[span.start_idx:span.eq_idx + 1]
                        offset = 1 - span.start_idx
                        a_loc = [p + offset for p in span.a_positions]
                        b_loc = [p + offset for p in span.b_positions]
                        eq_loc = span.eq_idx + offset
                        z_expr = graft.read_in(torch.tensor([win_expr], dtype=torch.long, device=dev))
                        p_top, _is_neg = determine_p_top_neural(
                            graft.organ, z_expr, a_loc, b_loc, eq_loc, span.op, graft.max_idx
                        )
                        a_str = "".join(vmap.digit_map[full_ids[p]] for p in span.a_positions)
                        b_str = "".join(vmap.digit_map[full_ids[p]] for p in span.b_positions)
                        state = InlineCalcState(
                            active=True,
                            win_prefix_ids=win_expr,
                            a_local=a_loc,
                            b_local=b_loc,
                            eq_local=eq_loc,
                            p_top=p_top,
                            answer_ids=[],
                            target_places=[p_top],
                            a_str=a_str,
                            b_str=b_str,
                            op=span.op,
                            start_gen_idx=len(generated),
                        )

                # If active, run organ on current window and add delta if structural gate > 0.5
                if state.active:
                    assert state.win_prefix_ids is not None
                    assert state.a_local is not None and state.b_local is not None
                    assert state.answer_ids is not None and state.target_places is not None
                    use_handoff = (
                        state.target_places[-1] == 0
                        and bool(handoff_cue_ids)
                        and (len(state.a_str) >= 4 or len(state.b_str) >= 4)
                    )
                    if use_handoff:
                        assert handoff_cue_ids is not None
                        forced_nxt = handoff_cue_ids[0]
                        pending_cue_ids = list(handoff_cue_ids[1:])
                        stats["graft_events"].append({
                            "a": state.a_str,
                            "b": state.b_str,
                            "op": state.op,
                            "start_gen_idx": state.start_gen_idx,
                            "end_gen_idx": len(generated) + len(handoff_cue_ids),
                        })
                        state.active = False
                        if not pending_cue_ids and trust_vec is not None:
                            steer_box[0] = trust_steer_steps
                    else:
                        win_ids = torch.tensor(
                            [state.win_prefix_ids + state.answer_ids], dtype=torch.long, device=dev
                        )
                        delta, gate = graft.organ_delta_on_window(
                            win_ids, state.a_local, state.b_local, state.eq_local, state.target_places
                        )
                        g_last = float(gate[0, -1].item())
                        if g_last > 0.5:
                            h_last = h_last + delta[:, -1, :]
                            stats["organ_tokens_emitted"] += 1
                        else:
                            stats["graft_events"].append({
                                "a": state.a_str,
                                "b": state.b_str,
                                "op": state.op,
                                "start_gen_idx": state.start_gen_idx,
                                "end_gen_idx": len(generated),
                            })
                            state.active = False
                            if trust_vec is not None:
                                steer_box[0] = trust_steer_steps

            if forced_nxt is not None:
                nxt = forced_nxt
            else:
                logits = lm_head(h_last.to(lm_head.weight.dtype)).float()
                nxt = int(logits[0].argmax().item())
            generated.append(nxt)
            full_ids.append(nxt)

            # Advance inline calc state based on emitted token
            if use_graft and state.active:
                assert state.answer_ids is not None and state.target_places is not None
                if state.target_places[-1] == 0:
                    # Just emitted the 1-step trailing STOP_SPACE (' ') after units digit -> cleanly shut off
                    stats["graft_events"].append({
                        "a": state.a_str,
                        "b": state.b_str,
                        "op": state.op,
                        "start_gen_idx": state.start_gen_idx,
                        "end_gen_idx": len(generated),
                    })
                    state.active = False
                    if trust_vec is not None:
                        steer_box[0] = trust_steer_steps
                else:
                    state.answer_ids.append(nxt)
                    if nxt in vmap.minus_set and len(state.answer_ids) == 1:
                        state.target_places.append(state.p_top)
                    elif nxt in vmap.digit_map:
                        state.target_places.append(max(0, state.target_places[-1] - 1))
                    else:
                        # Unexpected non-digit/non-minus token -> safely deactivate
                        stats["graft_events"].append({
                            "a": state.a_str,
                            "b": state.b_str,
                            "op": state.op,
                            "start_gen_idx": state.start_gen_idx,
                            "end_gen_idx": len(generated),
                        })
                        state.active = False

            if nxt == eos_token_id:
                break
            cur_in = torch.tensor([[nxt]], dtype=torch.long, device=dev)
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    return generated, stats


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class Experiment:
    def __init__(self, args: argparse.Namespace):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        self.ckpt_dir = Path(args.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        mode_tag = "smoke" if args.smoke else "full"
        self.w_out_path = self.ckpt_dir / f"w_out_{args.tag}_{mode_tag}.pt"
        self.probe_path = self.ckpt_dir / f"gate_probe_{args.tag}_{mode_tag}.pt"
        out_p = Path(args.out)
        if out_p.name.endswith("_results.json"):
            self.traces_path = out_p.with_name(out_p.name[:-len("_results.json")] + "_traces.jsonl")
        else:
            suf = "_smoke" if args.smoke else ""
            self.traces_path = out_p.with_name(f"think_calc_{args.tag}{suf}_traces.jsonl")
        if args.fresh and self.traces_path.exists():
            self.traces_path.unlink()

        config = {
            "model": args.model,
            "tag": args.tag,
            "smoke": args.smoke,
            "problem_set": args.problem_set,
            "base_only_calib": args.base_only_calib,
            "max_idx": args.max_idx,
            "w_out_steps": args.w_out_steps,
            "eval1_n": args.eval1_n,
            "eval2_n": args.eval2_n,
            "eval3_n": args.eval3_n,
            "max_new_tokens": args.max_new_tokens,
            "direct_eq_prompt": args.direct_eq_prompt,
            "calculator_prompt": args.calculator_prompt,
            "skip_step3": args.skip_step3,
            "skip_base_eval2": args.skip_base_eval2,
            "skip_step6": args.skip_step6,
            "trust_steer": args.trust_steer,
            "trust_alpha": args.trust_alpha,
            "trust_layer": args.trust_layer,
            "handoff_cue": args.handoff_cue,
        }
        self.store = Results(args.out, config, fresh=args.fresh)
        log(f"Loading {args.model} on {self.device} ({self.dtype})...")
        self.tok = AutoTokenizer.from_pretrained(args.model)
        try:
            llm = AutoModelForCausalLM.from_pretrained(args.model, dtype=self.dtype)
        except TypeError:
            llm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=self.dtype)
        self.llm = llm.to(self.device)
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad_(False)
        self.d_llm = int(self.llm.config.hidden_size)
        self.eos_id = int(self.tok.eos_token_id)

        # Optional graft-side anti-distrust mechanisms
        self.trust_vec: Optional[torch.Tensor] = None
        self.trust_layer: Optional[int] = None
        self.trust_info: Optional[Dict[str, Any]] = None
        if args.trust_steer:
            n_layers = len(self.llm.model.layers)
            self.trust_layer = int(args.trust_layer) if args.trust_layer is not None else (n_layers // 3)
            self.trust_vec, self.trust_info = compute_trust_steering_vector(
                self.llm, self.tok, self.trust_layer, args.trust_alpha, self.device
            )
            self.store.data["config"]["trust_layer"] = self.trust_layer
            log(f"  Computed mid-layer trust steering vector: {self.trust_info}")

        self.handoff_cue_str: str = args.handoff_cue.replace("\\n", "\n") if args.handoff_cue else ""
        self.handoff_cue_ids: Optional[List[int]] = (
            self.tok.encode(self.handoff_cue_str, add_special_tokens=False)
            if self.handoff_cue_str
            else None
        )
        if self.handoff_cue_ids:
            log(f"  Handoff cue enabled: {self.handoff_cue_str!r} -> {self.handoff_cue_ids}")

        self.store.data["meta"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "gpu": torch.cuda.get_device_name(0) if self.device.type == "cuda" else None,
            "d_llm": self.d_llm,
        }
        self.store.save()

        self.vmap = TokenVocabMap(self.tok)
        self.organ: Optional[SparseOrgan] = None
        if not args.base_only_calib:
            self.organ = build_varlen_calc_organ(max_idx=args.max_idx, device=self.device)
            self.organ.eval()
            for p in self.organ.parameters():
                p.requires_grad_(False)

        self.cal_in: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.graft: Optional[ThinkCalcGraft] = None
        self.probe: Optional[GateProbe] = None

    # ------------------------------------------------------------------ Step 1
    def step1_socket_calibration(self) -> None:
        log("=== STEP 1: Tokenizer & Closed-Form Ridge Socket Calibration ===")
        vmap = self.vmap
        other_explicit = list(dict.fromkeys(vmap.space_ids + vmap.comma_ids + [vmap.sink_id]))
        port_ids = (
            [[d] for d in vmap.digit_ids]
            + [vmap.plus_ids, vmap.minus_ids, vmap.eq_ids, other_explicit]
        )
        self.cal_in = calibrate_port_socket(
            self.llm, N_READ, port_ids, other_ch=CH_OTHER, const_ch=CH_CONST, device=self.device
        )
        self.graft = ThinkCalcGraft(self.llm, self.organ, self.cal_in, vmap, max_idx=self.args.max_idx).to(self.device)

        if self.store.done("step1_socket"):
            log("  step1_socket already recorded; socket recalibrated in memory.")
            return

        # Verify read-in accuracy on active-window tokens (digits, +/-/=, spaces/sink)
        win_check_ids = list(dict.fromkeys(
            vmap.digit_ids + vmap.plus_ids + vmap.minus_ids + vmap.eq_ids + other_explicit
        ))
        win_t = torch.tensor([win_check_ids], dtype=torch.long, device=self.device)
        z_win = self.graft.read_in(win_t)[0]
        exp_win = torch.zeros_like(z_win)
        for t, tid in enumerate(win_check_ids):
            sym = vmap.sym_of(tid)
            if len(sym) == 1 and sym.isdigit():
                exp_win[t, CH_DIG + int(sym)] = 1.0
            elif sym == "+":
                exp_win[t, CH_PLUS] = 1.0
            elif sym == "-":
                exp_win[t, CH_MINUS] = 1.0
            elif sym == "=":
                exp_win[t, CH_EQ] = 1.0
            else:
                exp_win[t, CH_OTHER] = 1.0
        exp_win[:, CH_CONST] = 1.0
        win_max_err = float((z_win - exp_win).abs().max().item())
        assert win_max_err < 0.05, f"Active-window read-in error too high: {win_max_err:.6f}"

        # Verify read-in accuracy on real '+' and '-' prompts
        test_prompts = [
            "Compute: 48219384721 + 39571284905 =87790669626",
            "Compute: 325,263,952 + 905,359,078 =1230623030",
            "Compute: 6,286,917,586,52\n+ 3,137,930,031 =631829688683",
            "Compute: 100000000000 - 48219 =99999951781",
            "Compute: 48219 - 100000000000 =-99999951781",
            "Step 1: 999+1=1000",
        ]
        max_err = 0.0
        n_checked = 0
        n_exact = 0
        for p in test_prompts:
            ids = torch.tensor([self.tok.encode(p, add_special_tokens=False)], dtype=torch.long, device=self.device)
            z = self.graft.read_in(ids)[0]
            exp = torch.zeros_like(z)
            for t, tid in enumerate(ids[0].tolist()):
                sym = vmap.sym_of(tid)
                if len(sym) == 1 and sym.isdigit():
                    exp[t, CH_DIG + int(sym)] = 1.0
                elif sym == "+":
                    exp[t, CH_PLUS] = 1.0
                elif sym == "-":
                    exp[t, CH_MINUS] = 1.0
                elif sym == "=":
                    exp[t, CH_EQ] = 1.0
                else:
                    exp[t, CH_OTHER] = 1.0
            exp[:, CH_CONST] = 1.0
            diff = (z - exp).abs()
            max_err = max(max_err, float(diff.max().item()))
            n_exact += int((diff.max(dim=-1).values < 1e-4).sum().item())
            n_checked += int(ids.shape[1])

        # Mandatory end-to-end read_in + compiled organ self-test across all buckets
        st_rng = random.Random(2026)
        st_cases: List[Tuple[int, str, int, int]] = []
        for bname in ("id_3_6", "ood_12", "ood_24", "ood_48", "chain_48", "mixed_len_24_4", "negative"):
            for _ in range(4):
                st_cases.append(gen_expr_pair(st_rng, bname))
        # Add edge cases: a == b, single-digit carry, single-digit no-carry, negative 1-digit, comma-formatted
        st_cases.extend([
            (5, "+", 3, 8), (5, "+", 4, 9), (123456, "-", 123456, 0), (3, "-", 8, -5),
            (325263952, "+", 905359078, 325263952 + 905359078),
            (1000000, "-", 1, 999999),
            (39, "-", 4821, 39 - 4821),
        ])
        st_ok = 0
        sym_to_tid = list(vmap.digit_ids) + [vmap.minus_emit_id]
        with torch.no_grad():
            for ci, (a, op, b, ans) in enumerate(st_cases):
                sp = " " if ci % 2 == 0 else ""
                use_comma = ci >= len(st_cases) - 3
                a_s = f"{a:,}" if use_comma else str(a)
                b_s = f"{b:,}" if use_comma else str(b)
                p_ids = self.tok.encode(f"Compute: {a_s}{sp}{op}{sp}{b_s}{sp}=", add_special_tokens=False)
                span = vmap.parse_strict_binary_expr(p_ids, len(p_ids) - 1, max_idx=self.args.max_idx)
                assert span is not None
                win_expr = [vmap.sink_id] + p_ids[span.start_idx:span.eq_idx + 1]
                off = 1 - span.start_idx
                a_loc = [p + off for p in span.a_positions]
                b_loc = [p + off for p in span.b_positions]
                eq_loc = span.eq_idx + off
                z_expr = self.graft.read_in(torch.tensor([win_expr], dtype=torch.long, device=self.device))
                p_top, _ = determine_p_top_neural(self.organ, z_expr, a_loc, b_loc, eq_loc, span.op, self.args.max_idx)
                seq_ids = list(win_expr)
                t_places = [p_top]
                emitted: List[str] = []
                for _ in range(self.args.max_idx + 4):
                    z_cur = self.graft.read_in(torch.tensor([seq_ids], dtype=torch.long, device=self.device))
                    sa, sb, pl = build_window_structural_features(
                        len(seq_ids), a_loc, b_loc, eq_loc, t_places, self.args.max_idx, device=self.device
                    )
                    o = self.organ.forward_states(assemble_bus(z_cur, sa.unsqueeze(0), sb.unsqueeze(0), pl.unsqueeze(0), self.args.max_idx))[0, -1]
                    if float(o[11].clamp(0.0, 1.0).item()) <= 0.5:
                        break
                    sym_i = int(o[:N_SYM_OUT].argmax().item())
                    emitted.append("-" if sym_i == 10 else str(sym_i))
                    seq_ids.append(sym_to_tid[sym_i])
                    t_places.append(p_top if sym_i == 10 else max(0, t_places[-1] - 1))
                if "".join(emitted) == str(ans):
                    st_ok += 1
        assert st_ok == len(st_cases), f"Step 1 read_in+organ self-test failed: {st_ok}/{len(st_cases)}"

        # Vocab leakage over 4096 random non-symbol tokens
        V = self.llm.get_input_embeddings().weight.shape[0]
        sample = torch.tensor(random.Random(5).sample(range(min(V, 150000)), 4096), device=self.device)
        with torch.no_grad():
            zb = dual_rail_clamp(self.llm.get_input_embeddings()(sample).float() @ self.cal_in[0].T + self.cal_in[1])
        special = set(vmap.digit_ids + vmap.plus_ids + vmap.minus_ids + vmap.eq_ids)
        mask = torch.tensor([int(t) not in special for t in sample.tolist()], device=self.device)
        leak = float((zb[mask][:, :CH_OTHER] > 0).any(-1).float().mean().item())

        res = {
            "digit_ids": vmap.digit_ids,
            "plus_ids": vmap.plus_ids,
            "minus_ids": vmap.minus_ids,
            "eq_ids": vmap.eq_ids,
            "space_ids": vmap.space_ids,
            "comma_ids": vmap.comma_ids,
            "organ_nnz": self.organ.nnz(),
            "organ_d_model": self.organ.d_model,
            "active_window_readin_max_err": round(win_max_err, 6),
            "organ_readin_selftest": f"{st_ok}/{len(st_cases)}",
            "readin_exact_acc": round(n_exact / max(1, n_checked), 6),
            "readin_max_abs_err": round(max_err, 6),
            "vocab_leak_rate_4096": round(leak, 6),
        }
        log(f"  Step 1 done: win_err={res['active_window_readin_max_err']} selftest={res['organ_readin_selftest']} "
            f"readin_acc={res['readin_exact_acc']} max_err={res['readin_max_abs_err']} "
            f"leak={res['vocab_leak_rate_4096']} organ_nnz={res['organ_nnz']}")
        self.store.set("step1_socket", res)

    # ------------------------------------------------------------------ Step 2
    def step2_train_w_out(self) -> None:
        assert self.graft is not None
        log(f"=== STEP 2: Train Write-Out W_out ({N_WOUT_CH * self.d_llm:,} params) on 3-6 Digit +/- + STOP_SPACE ===")
        if self.store.done("step2_w_out") and self.w_out_path.exists():
            self.graft.W_out.load_state_dict(torch.load(self.w_out_path, map_location=self.device))
            log(f"  Loaded cached W_out from {self.w_out_path}")
            return

        t0 = time.time()
        rng = random.Random(42)
        torch.manual_seed(42)
        nn.init.zeros_(self.graft.W_out.weight)

        steps = self.args.w_out_steps
        bs = self.args.w_out_bs
        lr = self.args.w_out_lr
        # Zero weight decay so W_out retains full override margin once loss drops on strong hosts (e.g. Qwen3-4B)
        opt = torch.optim.AdamW([self.graft.W_out.weight], lr=lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=lr * 0.05)
        lm_head = self.llm.get_output_embeddings()
        vmap = self.vmap

        last_loss = 0.0
        last_adv_loss = 0.0
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            total_loss = 0.0
            total_adv_loss = 0.0
            n_toks = 0

            # Accumulate over bs variable-length 3-6 digit examples (digits + 1-step trailing ' ')
            prefixes = ("Compute: ", "Calculate: ", "<think>\nFirst, ", "<think>\nSo ")
            for _ in range(bs):
                a, op, b, ans = gen_expr_pair(rng, "train_3_6")
                sp = " " if rng.random() < 0.7 else ""
                pref = rng.choice(prefixes)
                use_comma = (a >= 1000 or b >= 1000) and (rng.random() < 0.25)
                a_s = f"{a:,}" if use_comma else str(a)
                b_s = f"{b:,}" if use_comma else str(b)
                prompt = f"{pref}{a_s}{sp}{op}{sp}{b_s}{sp}="
                ans_str = str(ans)
                p_ids = self.tok.encode(prompt, add_special_tokens=False)
                a_ids = [vmap.minus_emit_id if c == "-" else vmap.digit_ids[int(c)] for c in ans_str]
                tgt_ids = a_ids + [vmap.sink_id]

                eq_pos = len(p_ids) - 1
                span = vmap.parse_strict_binary_expr(p_ids, eq_pos, max_idx=self.args.max_idx)
                assert span is not None, f"Failed to parse training prompt: {prompt!r}"

                # Full sequence for teacher forcing: prompt + ans (targets = ans + trailing ' ')
                full = p_ids + a_ids
                ids_t = torch.tensor([full], dtype=torch.long, device=self.device)
                tgt_t = torch.tensor(tgt_ids, dtype=torch.long, device=self.device)

                with torch.no_grad():
                    h_full = self.llm.model(input_ids=ids_t, use_cache=False).last_hidden_state.float()
                    h = h_full[:, eq_pos:, :]  # [1, L_ans + 1, d_llm]
                    # Mismatched on-manifold host states (confidently predicting other tokens) so strong
                    # hosts that already solve 3-6 digit sums still train W_out with full OOD override margin
                    perm_idx = torch.randint(0, h_full.shape[1], (h.shape[1],), device=self.device)
                    h_adv = h_full[:, perm_idx, :]

                # Active window: [sink] + expr + ans
                win_ids = [vmap.sink_id] + full[span.start_idx:]
                offset = 1 - span.start_idx
                a_loc = [p + offset for p in span.a_positions]
                b_loc = [p + offset for p in span.b_positions]
                eq_loc = span.eq_idx + offset

                if ans_str.startswith("-"):
                    n_digs = len(ans_str) - 1
                    t_places = [n_digs] + list(range(n_digs, 0, -1)) + [0]
                else:
                    n_digs = len(ans_str)
                    t_places = list(range(n_digs, 0, -1)) + [0]
                assert len(t_places) == len(tgt_ids)

                win_t = torch.tensor([win_ids], dtype=torch.long, device=self.device)
                delta, _gate = self.graft.organ_delta_on_window(win_t, a_loc, b_loc, eq_loc, t_places)
                delta_eq = delta[:, eq_loc:, :]
                logits_clean = lm_head((h + delta_eq).to(lm_head.weight.dtype)).float()[0]  # [L_ans + 1, V]
                logits_adv = lm_head((h_adv + delta_eq).to(lm_head.weight.dtype)).float()[0]
                loss_clean = F.cross_entropy(logits_clean, tgt_t, reduction="sum")
                loss_adv = F.cross_entropy(logits_adv, tgt_t, reduction="sum")
                loss = 0.5 * (loss_clean + loss_adv)
                (loss / (bs * max(1, len(tgt_ids)))).backward()
                total_loss += float(loss_clean.item())
                total_adv_loss += float(loss_adv.item())
                n_toks += len(tgt_ids)

            torch.nn.utils.clip_grad_norm_([self.graft.W_out.weight], 1.0)
            opt.step()
            sched.step()
            last_loss = total_loss / max(1, n_toks)
            last_adv_loss = total_adv_loss / max(1, n_toks)
            if (step + 1) in (1, 10, 50) or (step + 1) % max(1, steps // 10) == 0 or (step + 1) == steps:
                log(f"    W_out step {step + 1}/{steps} loss={last_loss:.4f} adv_loss={last_adv_loss:.4f} "
                    f"({(time.time() - t0) / (step + 1):.2f}s/step)")

        torch.save(self.graft.W_out.state_dict(), self.w_out_path)
        col_norm = float(self.graft.W_out.weight.detach().float().norm(dim=0).mean().item())
        res = {
            "n_params": int(self.graft.W_out.weight.numel()),
            "steps": steps,
            "batch_size": bs,
            "lr": lr,
            "final_loss": round(last_loss, 5),
            "final_adv_loss": round(last_adv_loss, 5),
            "w_out_col_norm_mean": round(col_norm, 4),
            "seconds": round(time.time() - t0, 1),
        }
        log(f"  Step 2 done: {res}")
        self.store.set("step2_w_out", res)
        cleanup_cuda()

    # ------------------------------------------------------------------ Step 3
    def step3_eval_standalone_ood(self) -> None:
        assert self.graft is not None
        log("=== STEP 3: Eval 1 — Un-gated Circuit + Socket Accuracy (+ Fair Chat Base) ===")
        if self.store.done("step3_eval1_ood"):
            log("  step3_eval1_ood already done; skipping.")
            return

        t0 = time.time()
        buckets = ["id_3_6", "ood_12", "ood_24", "ood_48", "chain_48", "mixed_len_24_4", "negative"]
        n_per = self.args.eval1_n
        res: Dict[str, Any] = {
            "eval_mode": "ungated_circuit_and_socket",
            "n_per_bucket": n_per,
            "buckets": {},
        }

        for bi, bname in enumerate(buckets):
            rng = random.Random(1000 + 37 * bi)
            cases = [gen_expr_pair(rng, bname) for _ in range(n_per)]
            graft_ok = graft_str_ok = 0
            sample_failures: List[Dict[str, Any]] = []

            for a, op, b, ans in cases:
                prompt = f"Compute: {a} {op} {b} ="
                p_ids = self.tok.encode(prompt, add_special_tokens=False)
                ans_str = str(ans)
                max_tok = len(ans_str) + 4

                # Un-gated circuit + socket check (organ forced active at '=', autonomous stop at p=1)
                g_gen, _ = generate_with_inline_calc(
                    self.graft, p_ids, max_new_tokens=max_tok, eos_token_id=self.eos_id,
                    use_graft=True, force_open_at_eq=True
                )
                g_txt = self.tok.decode(g_gen, skip_special_tokens=True).strip()
                m_g = re.match(r"^(-?\d+)", g_txt)
                ok_exact = bool(m_g and int(m_g.group(1)) == ans)
                ok_prefix = g_txt.startswith(ans_str)
                if ok_exact:
                    graft_ok += 1
                elif len(sample_failures) < 3:
                    sample_failures.append({
                        "expr": f"{a} {op} {b} =",
                        "gold": ans_str,
                        "got": g_txt[:80],
                        "prefix_exact": ok_prefix,
                    })
                if ok_prefix:
                    graft_str_ok += 1

            b_res: Dict[str, Any] = {
                "circuit_ungated_exact": round(graft_ok / n_per, 4),
                "circuit_ungated_prefix_exact": round(graft_str_ok / n_per, 4),
                "n_exact": graft_ok,
                "n": n_per,
            }
            if sample_failures:
                b_res["sample_failures"] = sample_failures
            res["buckets"][bname] = b_res
            log(f"    {bname:16s} | circuit_ungated={b_res['circuit_ungated_exact']:.4f} "
                f"(prefix={b_res['circuit_ungated_prefix_exact']:.4f}, {graft_ok}/{n_per})")

        # Fair chat-template Base check (apples-to-apples thinking prompt, max_new_tokens=512)
        fair_buckets = [("id_3_6", 0), ("ood_12", 1)]
        n_fair = min(n_per, 8 if self.args.smoke else 16)
        fair_max_tok = 512
        fair_res: Dict[str, Any] = {
            "n_per_bucket": n_fair,
            "max_new_tokens": fair_max_tok,
            "buckets": {},
        }
        log(f"  Running fair chat-template Base check (n={n_fair}, max_new_tokens={fair_max_tok})...")
        for bname, bi in fair_buckets:
            rng = random.Random(1000 + 37 * bi)
            cases = [gen_expr_pair(rng, bname) for _ in range(n_fair)]
            base_ok = 0
            n_trunc = 0
            for a, op, b, ans in cases:
                p_str = format_r1_prompt(self.tok, f"Compute {a} {op} {b}.")
                p_ids = self.tok.encode(p_str, add_special_tokens=False)
                b_gen, _ = generate_with_inline_calc(
                    self.graft, p_ids, max_new_tokens=fair_max_tok, eos_token_id=self.eos_id, use_graft=False
                )
                b_txt = self.tok.decode(b_gen, skip_special_tokens=True)
                if "</think>" not in b_txt:
                    n_trunc += 1
                pred = extract_final_int(b_txt)
                if pred == ans:
                    base_ok += 1
            fair_res["buckets"][bname] = {
                "base_chat_exact": round(base_ok / max(1, n_fair), 4),
                "n_correct": base_ok,
                "n_truncated_think": n_trunc,
                "n": n_fair,
            }
            log(f"    base_chat_fair {bname:8s} | exact={base_ok}/{n_fair} ({base_ok / max(1, n_fair):.4f}) "
                f"| truncated_think={n_trunc}/{n_fair}")

        res["base_chat_fair"] = fair_res
        res["seconds"] = round(time.time() - t0, 1)
        self.store.set("step3_eval1_ood", res)
        cleanup_cuda()

    # ------------------------------------------------------------------ Step 4
    def _load_code_chunks(self, modules: Sequence[str], max_chunks: int, chunk_len: int = 256) -> List[List[int]]:
        """Load Python stdlib code chunks of <= chunk_len tokens without ever tokenizing
        an entire multi-thousand-line file in a single call."""
        chunks: List[List[int]] = []
        target_pool = max_chunks * 2
        for m in modules:
            if len(chunks) >= target_pool:
                break
            try:
                src = inspect.getsource(importlib.import_module(m))
            except Exception:  # noqa: BLE001
                continue
            lines = src.splitlines(keepends=True)
            buf: List[int] = []
            # Tokenize in small 40-line blocks so we never exceed model_max_length
            for li in range(0, len(lines), 40):
                block = "".join(lines[li:li + 40])
                if not block.strip():
                    continue
                buf.extend(self.tok.encode(block, add_special_tokens=False))
                while len(buf) >= chunk_len:
                    chunks.append(buf[:chunk_len])
                    buf = buf[chunk_len:]
                    if len(chunks) >= target_pool:
                        break
                if len(chunks) >= target_pool:
                    break
        if len(chunks) > max_chunks:
            step = max(1, len(chunks) // max_chunks)
            chunks = chunks[::step][:max_chunks]
        return chunks

    @torch.no_grad()
    def _eval_nll_damage(self, seqs: List[List[int]], use_probe: bool, threshold: float = 0.5) -> Dict[str, Any]:
        """Evaluate next-token NLL overall and at '=' positions with base vs gated graft."""
        assert self.graft is not None and self.probe is not None
        lm_head = self.llm.get_output_embeddings()
        eq_t = torch.tensor(self.vmap.eq_ids, dtype=torch.long, device=self.device)
        tot = n_tok = eq_tot = eq_n = 0.0
        gate_sum = 0.0
        n_fired = 0

        for ids_list in seqs:
            if len(ids_list) < 2:
                continue
            ids = torch.tensor([ids_list], dtype=torch.long, device=self.device)
            h = self.llm.model(input_ids=ids, use_cache=False).last_hidden_state.float()  # [1, T, d]
            eq_mask = torch.isin(ids[0, :-1], eq_t)
            if use_probe and eq_mask.any():
                eq_positions = eq_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                g_vals = self.probe(h[0, eq_positions]).squeeze(-1)  # [K]
                gate_sum += float(g_vals.sum().item())
                for idx_k, pos in enumerate(eq_positions):
                    gv = float(g_vals[idx_k].item())
                    if gv > threshold:
                        n_fired += 1
                        # Honest probe-only damage: if probe fires, inject W_out on a dummy/parsed organ state
                        span = self.vmap.parse_strict_binary_expr(ids_list, pos, max_idx=self.args.max_idx)
                        if span is not None:
                            win = [self.vmap.sink_id] + ids_list[span.start_idx:pos + 1]
                            off = 1 - span.start_idx
                            a_loc = [p + off for p in span.a_positions]
                            b_loc = [p + off for p in span.b_positions]
                            eq_loc = pos + off
                            delta, _ = self.graft.organ_delta_on_window(
                                torch.tensor([win], dtype=torch.long, device=self.device),
                                a_loc, b_loc, eq_loc, [1]
                            )
                            h[0, pos] = h[0, pos] + gv * delta[0, -1]
                        else:
                            # Even if no span parsed, simulate worst-case probe-only injection of W_out(0-digit)
                            fake = torch.zeros(1, N_WOUT_CH, device=self.device)
                            fake[0, 0] = STOP_CH_VAL
                            h[0, pos] = h[0, pos] + gv * self.graft.W_out(fake)[0]

            logits = lm_head(h[:, :-1, :].to(lm_head.weight.dtype)).float()[0]
            ce = F.cross_entropy(logits, ids[0, 1:], reduction="none")
            tot += float(ce.sum().item())
            n_tok += float(ce.numel())
            if eq_mask.any():
                eq_tot += float(ce[eq_mask].sum().item())
                eq_n += float(eq_mask.sum().item())

        return {
            "nll": round(tot / max(1.0, n_tok), 5),
            "nll_at_eq": round(eq_tot / eq_n, 5) if eq_n > 0 else None,
            "n_tokens": int(n_tok),
            "n_eq": int(eq_n),
            "mean_probe_at_eq": round(gate_sum / eq_n, 5) if (use_probe and eq_n > 0) else None,
            "n_probe_fires_at_eq": int(n_fired) if use_probe else None,
            "threshold": threshold if use_probe else None,
        }

    def step4_harvest_and_train_gate(self) -> None:
        assert self.graft is not None
        log(f"=== STEP 4: Harvest <think> Traces & Train {self.d_llm + 1:,}-param GateProbe ===")
        if self.store.done("step4_gate_probe") and self.probe_path.exists():
            self.probe = GateProbe(self.d_llm).to(self.device)
            self.probe.load_state_dict(torch.load(self.probe_path, map_location=self.device))
            self.probe.eval()
            log(f"  Loaded cached GateProbe from {self.probe_path}")
            return

        t0 = time.time()
        # 1. Build harvest prompt set with source labels for problem-level stratified 80/20 split
        n_dir = 6 if self.args.smoke else 32
        n_gsm = 6 if self.args.smoke else 40
        n_big = 4 if self.args.smoke else 32
        n_neg = 4 if self.args.smoke else 32

        harvest_items: List[Tuple[str, str]] = [
            ("direct", q) for q in make_direct_arith_think_prompts(n_dir, seed=33333)
        ]
        try:
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main", split="train")
            for i in range(n_gsm):
                harvest_items.append(("gsm8k", ds[i]["question"]))
            log(f"  Loaded {n_gsm} GSM8K train questions for <think> harvesting")
        except Exception as e:  # noqa: BLE001
            log(f"  GSM8K train load failed ({e}); using extra direct arithmetic prompts")
            for q in make_direct_arith_think_prompts(n_gsm, seed=44444):
                harvest_items.append(("gsm8k", q))

        for p in make_big_number_word_problems(n_big, seed=77777):
            harvest_items.append(("big_wp", p["question"]))
        for q in make_hard_negative_think_prompts(n_neg, seed=88888):
            harvest_items.append(("hard_neg", q))

        # 2. Generate real <think> traces with frozen LLM and harvest '=' hidden states
        # Each entry: (source_type, X [K, d], y [K]) — one entry per distinct problem/trace
        trace_feats: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        eq_t = torch.tensor(self.vmap.eq_ids, dtype=torch.long, device=self.device)
        max_h_tok = 320 if self.args.smoke else 448

        for qi, (src_type, q) in enumerate(harvest_items):
            p_str = format_r1_prompt(self.tok, q)
            p_ids = self.tok.encode(p_str, add_special_tokens=False)
            gen_ids, _ = generate_with_inline_calc(
                self.graft, p_ids, max_new_tokens=max_h_tok, eos_token_id=self.eos_id, use_graft=False
            )
            full_ids = p_ids + gen_ids
            ids_t = torch.tensor([full_ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                h = self.llm.model(input_ids=ids_t, use_cache=False).last_hidden_state[0].float()

            # Inspect '=' positions inside generated region only
            xs_list, ys_list = [], []
            for pos in range(len(p_ids), len(full_ids)):
                if full_ids[pos] in self.vmap.eq_set:
                    is_pos = self.vmap.parse_strict_binary_expr(full_ids, pos, max_idx=self.args.max_idx) is not None
                    xs_list.append(h[pos].cpu())
                    ys_list.append(1.0 if is_pos else 0.0)
            if xs_list:
                trace_feats.append((src_type, torch.stack(xs_list, 0), torch.tensor(ys_list)))
            if (qi + 1) % max(1, len(harvest_items) // 5) == 0 or (qi + 1) == len(harvest_items):
                n_p = sum(int((y == 1).sum()) for _, _, y in trace_feats)
                n_n = sum(int((y == 0).sum()) for _, _, y in trace_feats)
                log(f"    harvested {qi + 1}/{len(harvest_items)} traces -> pos={n_p}, neg={n_n}")

        # Problem-level stratified 80/20 split by source_type -> zero problem/trace overlap between train & val!
        rng = random.Random(123)
        by_src: Dict[str, List[int]] = {}
        for idx_t, (src_type, _, _) in enumerate(trace_feats):
            by_src.setdefault(src_type, []).append(idx_t)
        val_idx: set[int] = set()
        for src_type in sorted(by_src):
            idxs = list(by_src[src_type])
            rng.shuffle(idxs)
            n_val_src = max(1, len(idxs) // 5) if len(idxs) >= 2 else 0
            val_idx.update(idxs[:n_val_src])
        if not val_idx and len(trace_feats) >= 2:
            val_idx.add(0)

        tr_X = [trace_feats[i][1] for i in range(len(trace_feats)) if i not in val_idx]
        tr_y = [trace_feats[i][2] for i in range(len(trace_feats)) if i not in val_idx]
        va_X = [trace_feats[i][1] for i in range(len(trace_feats)) if i in val_idx]
        va_y = [trace_feats[i][2] for i in range(len(trace_feats)) if i in val_idx]
        n_tr_traces = len(trace_feats) - len(val_idx)
        n_va_traces = len(val_idx)
        log(f"    problem-level stratified split: train_traces={n_tr_traces}, val_traces={n_va_traces}")

        # Add abundant Python stdlib '=' positions as extra training negatives (chunked <= 256 tokens; >=400 in full mode)
        code_tr = self._load_code_chunks(CODE_TRAIN_MODULES, max_chunks=32 if self.args.smoke else 240, chunk_len=256)
        code_negs = []
        with torch.no_grad():
            for seq in code_tr:
                ids_t = torch.tensor([seq], dtype=torch.long, device=self.device)
                m = torch.isin(ids_t[0], eq_t)
                if m.any():
                    h = self.llm.model(input_ids=ids_t, use_cache=False).last_hidden_state[0].float()
                    code_negs.append(h[m].cpu())
        n_code_neg = 0
        if code_negs:
            cn = torch.cat(code_negs, 0)
            n_code_neg = int(cn.shape[0])
            tr_X.append(cn)
            tr_y.append(torch.zeros(cn.shape[0]))
        log(f"    Python stdlib code '=' train negatives: {n_code_neg} (from {len(code_tr)} chunks)")

        # Add diverse <think>-wrapped synthetic positives + matched hard negatives (strictly disjoint train/val!)
        def extract_final_eq_hidden(texts: Sequence[str]) -> torch.Tensor:
            vecs = []
            with torch.no_grad():
                for s in texts:
                    ids_t = torch.tensor([self.tok.encode(s, add_special_tokens=False)], dtype=torch.long, device=self.device)
                    vecs.append(self.llm.model(input_ids=ids_t, use_cache=False).last_hidden_state[0, -1].float().cpu())
            return torch.stack(vecs, 0) if vecs else torch.zeros(0, self.d_llm)

        n_syn_tr = 64 if self.args.smoke else 192
        n_syn_va = 32 if self.args.smoke else 64
        syn_pos_tr_txt, syn_neg_tr_txt = make_synth_think_gate_examples(n_syn_tr, n_syn_tr, seed=555)
        syn_pos_va_txt, syn_neg_va_txt = make_synth_think_gate_examples(n_syn_va, n_syn_va, seed=666)

        X_syn_tr_pos = extract_final_eq_hidden(syn_pos_tr_txt)
        X_syn_tr_neg = extract_final_eq_hidden(syn_neg_tr_txt)
        tr_X.extend([X_syn_tr_pos, X_syn_tr_neg])
        tr_y.extend([torch.ones(X_syn_tr_pos.shape[0]), torch.zeros(X_syn_tr_neg.shape[0])])

        X_syn_va_pos = extract_final_eq_hidden(syn_pos_va_txt)
        X_syn_va_neg = extract_final_eq_hidden(syn_neg_va_txt)
        X_syn_va = torch.cat([X_syn_va_pos, X_syn_va_neg], 0).to(self.device)
        y_syn_va = torch.cat([torch.ones(X_syn_va_pos.shape[0]), torch.zeros(X_syn_va_neg.shape[0])], 0).to(self.device)

        X_tr = torch.cat(tr_X, 0).to(self.device)
        y_tr = torch.cat(tr_y, 0).to(self.device)
        X_va = torch.cat(va_X, 0).to(self.device) if va_X else torch.zeros(0, self.d_llm, device=self.device)
        y_va = torch.cat(va_y, 0).to(self.device) if va_y else torch.zeros(0, device=self.device)

        # 3. Train GateProbe (standardized + folded; cap pos_weight <= 1.0 to keep Python '=' FP near zero)
        mu = X_tr.mean(0)
        sd = X_tr.std(0) + 1e-4
        Xn = (X_tr - mu) / sd
        torch.manual_seed(0)
        lin = nn.Linear(self.d_llm, 1).to(self.device)
        nn.init.zeros_(lin.weight)
        nn.init.zeros_(lin.bias)
        n_pos_tr = float((y_tr == 1).sum().item())
        n_neg_tr = float((y_tr == 0).sum().item())
        pw = torch.tensor([min(1.0, max(0.5, n_neg_tr / max(1.0, n_pos_tr)))], device=self.device)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
        gate_steps = 400
        for _ in range(gate_steps):
            logit = lin(Xn).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, y_tr, pos_weight=pw) + 1e-3 * lin.weight.pow(2).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        probe = GateProbe(self.d_llm).to(self.device)
        with torch.no_grad():
            w = lin.weight / sd
            probe.lin.weight.copy_(w)
            probe.lin.bias.copy_(lin.bias - (w * mu).sum())
        probe.eval()
        self.probe = probe
        torch.save(probe.state_dict(), self.probe_path)

        # 4. Held-out '=' metrics at threshold=0.5, threshold=0.75, and best-F1 threshold
        def eval_probe_split(X_s: torch.Tensor, y_s: torch.Tensor, thr: float = 0.5) -> Dict[str, Any]:
            if X_s.shape[0] == 0:
                return {"n_val": 0, "threshold": thr}
            with torch.no_grad():
                pv = (probe(X_s).squeeze(-1) > thr).float()
                tp = int(((pv == 1) & (y_s == 1)).sum().item())
                fp = int(((pv == 1) & (y_s == 0)).sum().item())
                fn = int(((pv == 0) & (y_s == 1)).sum().item())
                tn = int(((pv == 0) & (y_s == 0)).sum().item())
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-9, prec + rec)
            return {
                "threshold": round(thr, 3),
                "n_val": int(X_s.shape[0]), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                "acc": round((tp + tn) / max(1, X_s.shape[0]), 4),
            }

        X_comb = torch.cat([X_va, X_syn_va], 0) if X_va.shape[0] > 0 else X_syn_va
        y_comb = torch.cat([y_va, y_syn_va], 0) if y_va.shape[0] > 0 else y_syn_va

        val_metrics = eval_probe_split(X_va, y_va, thr=0.5)
        synth_val_metrics = eval_probe_split(X_syn_va, y_syn_va, thr=0.5)
        comb_val_metrics = eval_probe_split(X_comb, y_comb, thr=0.5)

        val_metrics_t75 = eval_probe_split(X_va, y_va, thr=0.75)
        synth_val_metrics_t75 = eval_probe_split(X_syn_va, y_syn_va, thr=0.75)
        comb_val_metrics_t75 = eval_probe_split(X_comb, y_comb, thr=0.75)

        best_thr = 0.5
        best_f1 = -1.0
        for cand_thr in (0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
            m_c = eval_probe_split(X_comb, y_comb, thr=cand_thr)
            if m_c.get("f1", -1.0) > best_f1:
                best_f1 = m_c["f1"]
                best_thr = cand_thr
        val_metrics_best = eval_probe_split(X_va, y_va, thr=best_thr)
        comb_val_metrics_best = eval_probe_split(X_comb, y_comb, thr=best_thr)

        # 5. Held-out Python code & prose NLL damage (at threshold 0.5 and 0.75)
        code_ev = self._load_code_chunks(CODE_EVAL_MODULES, max_chunks=16 if self.args.smoke else 80, chunk_len=256)
        prose_ev = [self.tok.encode(s, add_special_tokens=False) for s in ENGLISH_PROBES]
        dmg_code_base = self._eval_nll_damage(code_ev, use_probe=False)
        dmg_code_gated = self._eval_nll_damage(code_ev, use_probe=True, threshold=0.5)
        dmg_code_gated_t75 = self._eval_nll_damage(code_ev, use_probe=True, threshold=0.75)
        dmg_prose_base = self._eval_nll_damage(prose_ev, use_probe=False)
        dmg_prose_gated = self._eval_nll_damage(prose_ev, use_probe=True, threshold=0.5)

        # 6. Quick Gated Eval 1 check (GateProbe active instead of forced ON) on id_3_6 and ood_24
        gated_e1_n = 8 if self.args.smoke else 16
        gated_e1: Dict[str, Any] = {"n_per_bucket": gated_e1_n, "threshold": 0.5, "buckets": {}}
        for bname, bi in (("id_3_6", 0), ("ood_24", 2)):
            rng_g = random.Random(1000 + 37 * bi)
            cases_g = [gen_expr_pair(rng_g, bname) for _ in range(gated_e1_n)]
            g_ok = g_pref = g_fires = g_missed = 0
            for a, op, b, ans in cases_g:
                prompt = f"Compute: {a} {op} {b} ="
                p_ids = self.tok.encode(prompt, add_special_tokens=False)
                ans_str = str(ans)
                g_gen, g_st = generate_with_inline_calc(
                    self.graft, p_ids, max_new_tokens=len(ans_str) + 4, eos_token_id=self.eos_id,
                    probe=self.probe, use_graft=True, force_open_at_eq=False, gate_threshold=0.5,
                )
                g_txt = self.tok.decode(g_gen, skip_special_tokens=True).strip()
                m_g = re.match(r"^(-?\d+)", g_txt)
                if m_g and int(m_g.group(1)) == ans:
                    g_ok += 1
                if g_txt.startswith(ans_str):
                    g_pref += 1
                g_fires += int(g_st["true_fires"])
                g_missed += int(g_st["missed_exprs"])
            gated_e1["buckets"][bname] = {
                "gated_exact": round(g_ok / max(1, gated_e1_n), 4),
                "gated_prefix_exact": round(g_pref / max(1, gated_e1_n), 4),
                "probe_fire_rate": round(g_fires / max(1, gated_e1_n), 4),
                "n_correct": g_ok,
                "true_fires": g_fires,
                "missed_exprs": g_missed,
                "n": gated_e1_n,
            }
            log(f"    gated_eval1 {bname:8s} | exact={g_ok}/{gated_e1_n} ({g_ok / max(1, gated_e1_n):.4f}) "
                f"| fires={g_fires}/{gated_e1_n} missed={g_missed}")

        res = {
            "n_params": self.d_llm + 1,
            "split_type": "problem_level_stratified_80_20",
            "n_harvest_traces": len(harvest_items),
            "train_harvest_traces": n_tr_traces,
            "val_harvest_traces": n_va_traces,
            "train_pos": int(n_pos_tr),
            "train_neg": int(n_neg_tr),
            "train_code_neg": int(n_code_neg),
            "heldout_think_eq": val_metrics,
            "heldout_synth_think_eq": synth_val_metrics,
            "heldout_combined_eq": comb_val_metrics,
            "heldout_think_eq_t75": val_metrics_t75,
            "heldout_synth_think_eq_t75": synth_val_metrics_t75,
            "heldout_combined_eq_t75": comb_val_metrics_t75,
            "best_f1_threshold": best_thr,
            "heldout_think_eq_best_f1": val_metrics_best,
            "heldout_combined_eq_best_f1": comb_val_metrics_best,
            "damage_python_base": dmg_code_base,
            "damage_python_gated": dmg_code_gated,
            "damage_python_gated_t75": dmg_code_gated_t75,
            "damage_python_nll_delta": round(dmg_code_gated["nll"] - dmg_code_base["nll"], 6),
            "damage_python_nll_delta_t75": round(dmg_code_gated_t75["nll"] - dmg_code_base["nll"], 6),
            "damage_python_eq_nll_delta": (
                round(dmg_code_gated["nll_at_eq"] - dmg_code_base["nll_at_eq"], 6)
                if (dmg_code_base["nll_at_eq"] is not None and dmg_code_gated["nll_at_eq"] is not None)
                else None
            ),
            "damage_python_eq_nll_delta_t75": (
                round(dmg_code_gated_t75["nll_at_eq"] - dmg_code_base["nll_at_eq"], 6)
                if (dmg_code_base["nll_at_eq"] is not None and dmg_code_gated_t75["nll_at_eq"] is not None)
                else None
            ),
            "damage_prose_base": dmg_prose_base,
            "damage_prose_gated": dmg_prose_gated,
            "damage_prose_nll_delta": round(dmg_prose_gated["nll"] - dmg_prose_base["nll"], 6),
            "gated_eval1_check": gated_e1,
            "seconds": round(time.time() - t0, 1),
        }
        log(f"  Step 4 done: think_F1@0.5={val_metrics.get('f1')} @0.75={val_metrics_t75.get('f1')} "
            f"comb_F1@0.5={comb_val_metrics.get('f1')} best_thr={best_thr} "
            f"py_delta@0.5={res['damage_python_nll_delta']} @0.75={res['damage_python_nll_delta_t75']} "
            f"prose_delta={res['damage_prose_nll_delta']}")
        self.store.set("step4_gate_probe", res)
        cleanup_cuda()

    # ------------------------------------------------------------------ Step 5
    def _append_trace_record(self, rec: Dict[str, Any]) -> None:
        self.traces_path.parent.mkdir(parents=True, exist_ok=True)
        with self.traces_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    @torch.no_grad()
    def _forced_close_continuation(
        self,
        p_ids: List[int],
        gen_ids: List[int],
        max_cont_tokens: int = 64,
    ) -> Tuple[Optional[int], str]:
        """Graceful </think> close fallback (64-token continuation after appending
        '\\n</think>\\nTherefore, the final answer is \\\\boxed{'). Applied identically to
        BOTH Base and Grafted traces that hit max_new_tokens without closing </think>."""
        assert self.graft is not None
        suffix = "\n</think>\nTherefore, the final answer is \\boxed{"
        suf_ids = self.tok.encode(suffix, add_special_tokens=False)
        cont_prompt = p_ids + gen_ids + suf_ids
        cont_ids, _ = generate_with_inline_calc(
            self.graft, cont_prompt, max_new_tokens=max_cont_tokens,
            eos_token_id=self.eos_id, use_graft=False
        )
        cont_txt = self.tok.decode(cont_ids, skip_special_tokens=True)
        m = re.match(r"\s*\$?\s*(-?[\d,]+)", cont_txt)
        if m:
            try:
                return int(m.group(1).replace(",", "")), cont_txt
            except ValueError:
                pass
        return extract_final_int(suffix + cont_txt), cont_txt

    def step5_eval_big_word_problems(self) -> None:
        assert self.graft is not None and self.probe is not None
        log("=== STEP 5: Eval 2 — Big-Number (9-14 digit) Word Problems inside <think> ===")
        if self.store.done("step5_eval2_big_wp"):
            log("  step5_eval2_big_wp already done; skipping.")
            return

        # If neither Step 5 nor Step 6 has finished yet, reset trace log to avoid duplicate records on retry
        if not self.store.done("step6_eval3_gsm8k") and self.traces_path.exists():
            self.traces_path.unlink()

        t0 = time.time()
        skip_base = bool(self.args.skip_base_eval2)
        if skip_base:
            log("  --skip-base-eval2 enabled: skipping Base arm, running Grafted arm only.")
        problems = make_word_problems(self.args.eval2_n, problem_set=self.args.problem_set)
        base_ok = graft_ok = 0
        base_forced_ok = graft_forced_ok = 0
        agg_gate = {k: 0 for k in ("total_eq_seen", "valid_expr_eq", "probe_fires", "true_fires",
                                   "false_probe_fires", "missed_exprs", "organ_tokens_emitted")}
        base_traces: List[Dict[str, Any]] = []
        graft_traces: List[Dict[str, Any]] = []
        base_tags_list: List[Dict[str, bool]] = []
        graft_tags_list: List[Dict[str, bool]] = []
        base_correct_list: List[bool] = []
        graft_correct_list: List[bool] = []
        base_gen_lens: List[int] = []
        graft_gen_lens: List[int] = []
        comparisons: List[Dict[str, Any]] = []

        for idx, prob in enumerate(problems):
            q_text = prob["question"]
            p_str = format_r1_prompt(
                self.tok,
                q_text,
                direct_eq_prompt=self.args.direct_eq_prompt,
                calculator_prompt=self.args.calculator_prompt,
            )
            p_ids = self.tok.encode(p_str, add_special_tokens=False)
            gt = prob["answer"]

            if skip_base:
                b_ids: List[int] = []
                b_txt = ""
                b_pred: Optional[int] = None
                b_correct = False
                b_forced_pred: Optional[int] = None
                b_forced_correct = False
                b_tags: Dict[str, bool] = {k: False for k in FAILURE_TAG_KEYS}
            else:
                b_ids, _ = generate_with_inline_calc(
                    self.graft, p_ids, max_new_tokens=self.args.max_new_tokens,
                    eos_token_id=self.eos_id, use_graft=False
                )
                b_txt = self.tok.decode(b_ids, skip_special_tokens=True)
                b_pred = extract_final_int(b_txt)
                b_correct = (b_pred == gt)
                base_ok += int(b_correct)
                b_tags = diagnose_failure(q_text, b_txt, b_pred, len(b_ids), self.args.max_new_tokens)
                if b_tags["hit_token_limit"] and b_tags["truncated_think"]:
                    b_forced_pred, _ = self._forced_close_continuation(p_ids, b_ids, max_cont_tokens=64)
                else:
                    b_forced_pred = b_pred
                b_forced_correct = (b_forced_pred == gt)
                base_forced_ok += int(b_forced_correct)
                base_traces.append({"text": b_txt, "gen_ids": b_ids})
                base_tags_list.append(b_tags)
                base_correct_list.append(b_correct)
                base_gen_lens.append(len(b_ids))

            g_ids, g_stats = generate_with_inline_calc(
                self.graft, p_ids, max_new_tokens=self.args.max_new_tokens,
                eos_token_id=self.eos_id, probe=self.probe, use_graft=True,
                trust_vec=self.trust_vec, trust_layer=self.trust_layer,
                handoff_cue_ids=self.handoff_cue_ids,
            )
            g_txt = self.tok.decode(g_ids, skip_special_tokens=True)
            g_pred = extract_final_int(g_txt)
            g_correct = (g_pred == gt)
            graft_ok += int(g_correct)
            g_tags = diagnose_failure(q_text, g_txt, g_pred, len(g_ids), self.args.max_new_tokens)
            if g_tags["hit_token_limit"] and g_tags["truncated_think"]:
                g_forced_pred, _ = self._forced_close_continuation(p_ids, g_ids, max_cont_tokens=64)
            else:
                g_forced_pred = g_pred
            g_forced_correct = (g_forced_pred == gt)
            graft_forced_ok += int(g_forced_correct)
            graft_traces.append({"text": g_txt, "gen_ids": g_ids, "graft_events": g_stats["graft_events"]})
            graft_tags_list.append(g_tags)
            graft_correct_list.append(g_correct)
            graft_gen_lens.append(len(g_ids))
            for k in agg_gate:
                agg_gate[k] += int(g_stats[k])

            self._append_trace_record({
                "eval": "eval2_big_wp",
                "idx": idx,
                "question": q_text,
                "gold": gt,
                "base_skipped": skip_base,
                "base_pred": b_pred,
                "graft_pred": g_pred,
                "base_correct": b_correct,
                "graft_correct": g_correct,
                "base_forced_pred": b_forced_pred,
                "graft_forced_pred": g_forced_pred,
                "base_forced_correct": b_forced_correct,
                "graft_forced_correct": g_forced_correct,
                "base_n_gen_tokens": len(b_ids),
                "graft_n_gen_tokens": len(g_ids),
                "base_failure_tags": b_tags,
                "graft_failure_tags": g_tags,
                "base_text": b_txt,
                "graft_text": g_txt,
                "graft_events": g_stats["graft_events"],
            })

            # Save up to 5 side-by-side trace comparisons (prioritize cases where Grafted fixes Base!)
            g_stats_clean = {k: g_stats[k] for k in agg_gate}
            if len(comparisons) < 5 and (g_correct and not b_correct):
                comparisons.append({
                    "id": prob["id"], "question": q_text, "ground_truth": gt,
                    "base_skipped": skip_base,
                    "base_pred": b_pred, "base_correct": b_correct, "base_trace": b_txt[:2400],
                    "graft_pred": g_pred, "graft_correct": g_correct, "graft_trace": g_txt[:2400],
                    "gate_stats": g_stats_clean,
                })
            elif idx >= len(problems) - (5 - len(comparisons)) and len(comparisons) < 5:
                comparisons.append({
                    "id": prob["id"], "question": q_text, "ground_truth": gt,
                    "base_skipped": skip_base,
                    "base_pred": b_pred, "base_correct": b_correct, "base_trace": b_txt[:2400],
                    "graft_pred": g_pred, "graft_correct": g_correct, "graft_trace": g_txt[:2400],
                    "gate_stats": g_stats_clean,
                })

            if (idx + 1) % max(1, len(problems) // 8) == 0 or (idx + 1) == len(problems):
                base_s = "SKIPPED" if skip_base else f"{base_ok}/{idx + 1} (fc={base_forced_ok})"
                log(f"    big_wp {idx + 1}/{len(problems)} | base={base_s} "
                    f"| graft={graft_ok}/{idx + 1} (fc={graft_forced_ok}) "
                    f"| true_fires={agg_gate['true_fires']} false_fires={agg_gate['false_probe_fires']}")

        if skip_base:
            base_grade = None
            base_fb: Dict[str, Any] = {"skipped": True}
        else:
            base_grade = grade_thinking(base_traces, self.tok, self.args.max_new_tokens, is_grafted=False)
            base_fb = aggregate_failure_breakdown(base_tags_list, base_correct_list, base_gen_lens)
        graft_grade = grade_thinking(graft_traces, self.tok, self.args.max_new_tokens, is_grafted=True)
        graft_fb = aggregate_failure_breakdown(graft_tags_list, graft_correct_list, graft_gen_lens)

        n = len(problems)
        res = {
            "n": n,
            "base_skipped": skip_base,
            "direct_eq_prompt": self.args.direct_eq_prompt,
            "calculator_prompt": self.args.calculator_prompt,
            "trust_steer": self.args.trust_steer,
            "trust_alpha": self.args.trust_alpha if self.args.trust_steer else None,
            "trust_layer": self.trust_layer,
            "trust_info": self.trust_info,
            "handoff_cue": self.handoff_cue_str,
            "base_final_acc": None if skip_base else round(base_ok / max(1, n), 4),
            "graft_final_acc": round(graft_ok / max(1, n), 4),
            "base_correct": None if skip_base else base_ok,
            "graft_correct": graft_ok,
            "base_forced_close_acc": None if skip_base else round(base_forced_ok / max(1, n), 4),
            "graft_forced_close_acc": round(graft_forced_ok / max(1, n), 4),
            "base_forced_close_correct": None if skip_base else base_forced_ok,
            "graft_forced_close_correct": graft_forced_ok,
            "base_thinking_grade": base_grade,
            "graft_thinking_grade": graft_grade,
            "base_failure_breakdown": base_fb,
            "graft_failure_breakdown": graft_fb,
            "gate_stats": agg_gate,
            "traces_file": str(self.traces_path),
            "side_by_side_comparisons": comparisons,
            "seconds": round(time.time() - t0, 1),
        }
        base_sum_s = (
            "SKIPPED"
            if skip_base
            else f"{res['base_final_acc']:.4f} (fc={res['base_forced_close_acc']:.4f})"
        )
        log(f"  Step 5 done: base={base_sum_s} "
            f"vs graft={res['graft_final_acc']:.4f} (fc={res['graft_forced_close_acc']:.4f}) "
            f"| base_fb={base_fb} | graft_fb={graft_fb}")
        self.store.set("step5_eval2_big_wp", res)
        cleanup_cuda()

    # ------------------------------------------------------------------ Step 6
    def step6_eval_gsm8k(self) -> None:
        assert self.graft is not None and self.probe is not None
        log("=== STEP 6: Eval 3 — Standard GSM8K No-Regression Check ===")
        if self.store.done("step6_eval3_gsm8k"):
            log("  step6_eval3_gsm8k already done; skipping.")
            return

        t0 = time.time()
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        n = min(self.args.eval3_n, len(ds))
        base_ok = graft_ok = 0
        base_forced_ok = graft_forced_ok = 0
        agg_gate = {k: 0 for k in ("total_eq_seen", "valid_expr_eq", "probe_fires", "true_fires",
                                   "false_probe_fires", "missed_exprs", "organ_tokens_emitted")}
        base_traces: List[Dict[str, Any]] = []
        graft_traces: List[Dict[str, Any]] = []
        base_tags_list: List[Dict[str, bool]] = []
        graft_tags_list: List[Dict[str, bool]] = []
        base_correct_list: List[bool] = []
        graft_correct_list: List[bool] = []
        base_gen_lens: List[int] = []
        graft_gen_lens: List[int] = []

        for i in range(n):
            q = ds[i]["question"]
            gt = parse_gsm8k_answer(ds[i]["answer"])
            p_str = format_r1_prompt(self.tok, q)
            p_ids = self.tok.encode(p_str, add_special_tokens=False)

            b_ids, _ = generate_with_inline_calc(
                self.graft, p_ids, max_new_tokens=self.args.max_new_tokens,
                eos_token_id=self.eos_id, use_graft=False
            )
            b_txt = self.tok.decode(b_ids, skip_special_tokens=True)
            b_pred = extract_final_int(b_txt)
            b_correct = bool(gt is not None and b_pred == gt)
            base_ok += int(b_correct)
            b_tags = diagnose_failure(q, b_txt, b_pred, len(b_ids), self.args.max_new_tokens)
            if b_tags["hit_token_limit"] and b_tags["truncated_think"]:
                b_forced_pred, _ = self._forced_close_continuation(p_ids, b_ids, max_cont_tokens=64)
            else:
                b_forced_pred = b_pred
            b_forced_correct = bool(gt is not None and b_forced_pred == gt)
            base_forced_ok += int(b_forced_correct)
            base_traces.append({"text": b_txt, "gen_ids": b_ids})
            base_tags_list.append(b_tags)
            base_correct_list.append(b_correct)
            base_gen_lens.append(len(b_ids))

            g_ids, g_stats = generate_with_inline_calc(
                self.graft, p_ids, max_new_tokens=self.args.max_new_tokens,
                eos_token_id=self.eos_id, probe=self.probe, use_graft=True,
                trust_vec=self.trust_vec, trust_layer=self.trust_layer,
                handoff_cue_ids=self.handoff_cue_ids,
            )
            g_txt = self.tok.decode(g_ids, skip_special_tokens=True)
            g_pred = extract_final_int(g_txt)
            g_correct = bool(gt is not None and g_pred == gt)
            graft_ok += int(g_correct)
            g_tags = diagnose_failure(q, g_txt, g_pred, len(g_ids), self.args.max_new_tokens)
            if g_tags["hit_token_limit"] and g_tags["truncated_think"]:
                g_forced_pred, _ = self._forced_close_continuation(p_ids, g_ids, max_cont_tokens=64)
            else:
                g_forced_pred = g_pred
            g_forced_correct = bool(gt is not None and g_forced_pred == gt)
            graft_forced_ok += int(g_forced_correct)
            graft_traces.append({"text": g_txt, "gen_ids": g_ids, "graft_events": g_stats["graft_events"]})
            graft_tags_list.append(g_tags)
            graft_correct_list.append(g_correct)
            graft_gen_lens.append(len(g_ids))
            for k in agg_gate:
                agg_gate[k] += int(g_stats[k])

            self._append_trace_record({
                "eval": "eval3_gsm8k",
                "idx": i,
                "question": q,
                "gold": gt,
                "base_pred": b_pred,
                "graft_pred": g_pred,
                "base_correct": b_correct,
                "graft_correct": g_correct,
                "base_forced_pred": b_forced_pred,
                "graft_forced_pred": g_forced_pred,
                "base_forced_correct": b_forced_correct,
                "graft_forced_correct": g_forced_correct,
                "base_n_gen_tokens": len(b_ids),
                "graft_n_gen_tokens": len(g_ids),
                "base_failure_tags": b_tags,
                "graft_failure_tags": g_tags,
                "base_text": b_txt,
                "graft_text": g_txt,
                "graft_events": g_stats["graft_events"],
            })

            if (i + 1) % max(1, n // 8) == 0 or (i + 1) == n:
                log(f"    gsm8k {i + 1}/{n} | base={base_ok}/{i + 1} (fc={base_forced_ok}) "
                    f"| graft={graft_ok}/{i + 1} (fc={graft_forced_ok}) "
                    f"| true_fires={agg_gate['true_fires']} false_fires={agg_gate['false_probe_fires']}")

        base_grade = grade_thinking(base_traces, self.tok, self.args.max_new_tokens, is_grafted=False)
        graft_grade = grade_thinking(graft_traces, self.tok, self.args.max_new_tokens, is_grafted=True)
        base_fb = aggregate_failure_breakdown(base_tags_list, base_correct_list, base_gen_lens)
        graft_fb = aggregate_failure_breakdown(graft_tags_list, graft_correct_list, graft_gen_lens)

        res = {
            "n": n,
            "base_acc": round(base_ok / max(1, n), 4),
            "graft_acc": round(graft_ok / max(1, n), 4),
            "base_correct": base_ok,
            "graft_correct": graft_ok,
            "acc_delta": round((graft_ok - base_ok) / max(1, n), 4),
            "base_forced_close_acc": round(base_forced_ok / max(1, n), 4),
            "graft_forced_close_acc": round(graft_forced_ok / max(1, n), 4),
            "base_forced_close_correct": base_forced_ok,
            "graft_forced_close_correct": graft_forced_ok,
            "base_thinking_grade": base_grade,
            "graft_thinking_grade": graft_grade,
            "base_failure_breakdown": base_fb,
            "graft_failure_breakdown": graft_fb,
            "gate_stats": agg_gate,
            "traces_file": str(self.traces_path),
            "seconds": round(time.time() - t0, 1),
        }
        log(f"  Step 6 done: GSM8K base={res['base_acc']:.4f} (fc={res['base_forced_close_acc']:.4f}) "
            f"vs graft={res['graft_acc']:.4f} (fc={res['graft_forced_close_acc']:.4f}, delta={res['acc_delta']:+.4f}) "
            f"| base_fb={base_fb} | graft_fb={graft_fb}")
        self.store.set("step6_eval3_gsm8k", res)
        cleanup_cuda()

    @torch.no_grad()
    def run_base_only_calib(self) -> None:
        """Base-only calibration on 2step_8_10 (n=12) and default (n=8) waiting for </think> + EOS."""
        max_tok = int(self.args.max_new_tokens)
        bs = max(1, int(self.args.calib_bs))
        log(f"=== BASE-ONLY CALIBRATION (model={self.args.model}, max_new_tokens={max_tok}, bs={bs}) ===")
        t0_all = time.time()
        suites = [
            ("2step_8_10", 12),
            ("default", 8),
        ]
        summary: Dict[str, Any] = {
            "model": self.args.model,
            "max_new_tokens": max_tok,
            "calib_bs": bs,
            "direct_eq_prompt": self.args.direct_eq_prompt,
            "calculator_prompt": self.args.calculator_prompt,
            "suites": {},
        }

        old_padding_side = getattr(self.tok, "padding_side", "right")
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.eos_id

        try:
            for suite_name, n_prob in suites:
                log(f"--- Suite: {suite_name} (n={n_prob}, max_new_tokens={max_tok}, bs={bs}) ---")
                t0 = time.time()
                problems = make_word_problems(n_prob, problem_set=suite_name)
                records: List[Dict[str, Any]] = []
                n_ok = 0
                n_closed = 0
                n_hit_limit = 0
                gen_lens: List[int] = []

                for b_start in range(0, n_prob, bs):
                    batch = problems[b_start:b_start + bs]
                    prompts = [
                        format_r1_prompt(self.tok, p["question"], direct_eq_prompt=self.args.direct_eq_prompt)
                        for p in batch
                    ]
                    enc = self.tok(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    in_t = enc["input_ids"].to(self.device)
                    attn_t = enc["attention_mask"].to(self.device)
                    prompt_padded_len = in_t.shape[1]

                    gen_out = self.llm.generate(
                        input_ids=in_t,
                        attention_mask=attn_t,
                        max_new_tokens=max_tok,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        eos_token_id=self.eos_id,
                        pad_token_id=self.eos_id,
                        use_cache=True,
                    )

                    for bi, prob in enumerate(batch):
                        idx = b_start + bi
                        gt = prob["answer"]
                        raw_gen = gen_out[bi, prompt_padded_len:].tolist()
                        # Trim at first EOS token (inclusive) so n_gen_tokens reflects actual generated length
                        if self.eos_id in raw_gen:
                            eos_pos = raw_gen.index(self.eos_id)
                            gen_ids = raw_gen[:eos_pos + 1]
                        else:
                            gen_ids = raw_gen
                        txt = self.tok.decode(gen_ids, skip_special_tokens=True)
                        n_gen = len(gen_ids)
                        closed_think = "</think>" in txt
                        hit_limit = n_gen >= max_tok
                        pred = extract_final_int(txt)
                        correct = (pred == gt)

                        n_ok += int(correct)
                        n_closed += int(closed_think)
                        n_hit_limit += int(hit_limit)
                        gen_lens.append(n_gen)

                        rec = {
                            "id": prob["id"],
                            "template": prob["template"],
                            "n_steps": prob["n_steps"],
                            "n_gen_tokens": n_gen,
                            "closed_think": closed_think,
                            "hit_limit": hit_limit,
                            "correct": correct,
                            "pred": pred,
                            "gold": gt,
                            "tail": txt[-240:],
                        }
                        records.append(rec)
                        log(
                            f"  [{suite_name} {idx + 1:2d}/{n_prob}] "
                            f"n_gen={n_gen:4d} closed={str(closed_think):5s} "
                            f"correct={str(correct):5s} pred={pred} gold={gt}"
                        )

                sorted_lens = sorted(gen_lens)
                mid = len(sorted_lens) // 2
                median_gen = (
                    float(sorted_lens[mid])
                    if len(sorted_lens) % 2 == 1
                    else 0.5 * (sorted_lens[mid - 1] + sorted_lens[mid])
                )
                suite_res = {
                    "n": n_prob,
                    "n_correct": n_ok,
                    "base_acc": round(n_ok / max(1, n_prob), 4),
                    "n_closed_think": n_closed,
                    "closed_think_rate": round(n_closed / max(1, n_prob), 4),
                    "n_hit_limit": n_hit_limit,
                    "hit_limit_rate": round(n_hit_limit / max(1, n_prob), 4),
                    "mean_gen_tokens": round(sum(gen_lens) / max(1, n_prob), 2),
                    "median_gen_tokens": round(median_gen, 2),
                    "min_gen_tokens": min(gen_lens) if gen_lens else 0,
                    "max_gen_tokens": max(gen_lens) if gen_lens else 0,
                    "seconds": round(time.time() - t0, 1),
                    "problems": records,
                }
                summary["suites"][suite_name] = suite_res
                self.store.set(f"base_calib_{suite_name}", suite_res)
                log(
                    f"  Suite {suite_name} summary: base_acc={suite_res['base_acc']:.4f} ({n_ok}/{n_prob}) | "
                    f"closed_think={suite_res['closed_think_rate']:.4f} ({n_closed}/{n_prob}) | "
                    f"mean_tok={suite_res['mean_gen_tokens']} median_tok={suite_res['median_gen_tokens']} "
                    f"({suite_res['seconds']}s)"
                )
                cleanup_cuda()
        finally:
            self.tok.padding_side = old_padding_side

        summary["total_seconds"] = round(time.time() - t0_all, 1)
        self.store.set("base_only_calib", summary)
        log(f"=== BASE-ONLY CALIBRATION COMPLETE ({summary['total_seconds']}s) -> {self.store.path} ===")

    def run_all(self) -> None:
        if self.args.base_only_calib:
            self.run_base_only_calib()
            return
        self.step1_socket_calibration()
        self.step2_train_w_out()
        if self.args.skip_step3:
            log("=== STEP 3: Skipped (--skip-step3) ===")
        else:
            self.step3_eval_standalone_ood()
        self.step4_harvest_and_train_gate()
        self.step5_eval_big_word_problems()
        if self.args.skip_step6:
            log("=== STEP 6: Skipped (--skip-step6) ===")
        else:
            self.step6_eval_gsm8k()
        log("=== ALL STEPS COMPLETE ===")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt-dir", default=str(HERE / "runs" / "ckpts"))
    ap.add_argument("--max-idx", type=int, default=64)
    ap.add_argument("--w-out-steps", type=int, default=250)
    ap.add_argument("--w-out-bs", type=int, default=16)
    ap.add_argument("--w-out-lr", type=float, default=1e-2)
    ap.add_argument("--eval1-n", type=int, default=64)
    ap.add_argument("--eval2-n", type=int, default=80)
    ap.add_argument("--eval3-n", type=int, default=80)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--problem-set", choices=["default", "2step_8_10"], default="default")
    ap.add_argument("--base-only-calib", action="store_true")
    ap.add_argument("--calib-bs", type=int, default=4)
    ap.add_argument("--direct-eq-prompt", action="store_true")
    ap.add_argument("--calculator-prompt", action="store_true")
    ap.add_argument("--skip-step3", action="store_true")
    ap.add_argument("--skip-base-eval2", action="store_true")
    ap.add_argument("--skip-step6", action="store_true")
    ap.add_argument("--trust-steer", action="store_true")
    ap.add_argument("--trust-alpha", type=float, default=2.5)
    ap.add_argument("--trust-layer", type=int, default=None)
    ap.add_argument("--handoff-cue", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    if not args.tag:
        args.tag = default_tag_for_model(args.model)
    if args.base_only_calib:
        if args.out is None:
            args.out = str(HERE / "runs" / f"base_calib_{args.tag}.json")
    elif args.smoke:
        args.w_out_steps = min(args.w_out_steps, 150)
        args.eval1_n = min(args.eval1_n, 16)
        args.eval2_n = min(args.eval2_n, 8)
        args.eval3_n = min(args.eval3_n, 8)
        args.max_new_tokens = min(args.max_new_tokens, 512)
        if args.out is None:
            args.out = str(HERE / "runs" / f"think_calc_{args.tag}_smoke_results.json")
    else:
        if args.out is None:
            args.out = str(HERE / "runs" / f"think_calc_{args.tag}_results.json")
    return args


def main() -> None:
    args = parse_args()
    try:
        exp = Experiment(args)
        exp.run_all()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # prevent HuggingFace datasets teardown hang


if __name__ == "__main__":
    main()
