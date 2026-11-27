"""08-think-math-kernel: Frozen Thinking Model + 1,537-param (d_llm + 1) Hidden-State GateProbe
+ Safe <0.1 ms Inline AST Math Kernel.

Zero tool-calling syntax. Zero RL. Zero edits to base model weights. Zero eval()/exec().
Only trainable parameters:
  - GateProbe: nn.Linear(d_llm, 1, bias=True) -> d_llm + 1 params (1,537 on R1-1.5B),
    trained on harvested <think> '=' hidden states + Python stdlib negatives.

Steps (each saved atomically to --out; reruns skip completed steps):
  1. Eval 1 (Un-gated): Standalone mixed-operation OOD math (+ - * / ^ sqrt, decimals,
     12-48 digit ints, multi-term) + fair chat-template Base check
  2. Harvest real <think> traces & train (d_llm + 1)-param GateProbe (problem-level
     stratified 80/20 split) + Python/prose NLL damage + Gated Eval 1
  3. Eval 2: Hard multi-step big-number word problems (+ - * / with 6-12 digit ints &
     decimals, n=32 / 8 smoke) inside <think> (Base vs Inline Math Kernel)
  4. Eval 3: Standard GSM8K (n=32 / 8 smoke) no-regression & <think> speedup check
     (skippable via --skip-step6)
"""

from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import json
import os
import platform
import random
import re
import statistics
import sys
import time
import traceback
from collections import deque
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from math_kernel import extract_and_eval_expr  # noqa: E402

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
# GATE PROBE & TOKEN HELPERS
# =============================================================================

class GateProbe(nn.Module):
    """(d_llm + 1)-param logistic gate on frozen LLM last_hidden_state: g = sigmoid(w . h + b)."""

    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.lin(h.float()))


def collect_eq_token_ids(tok: Any) -> List[int]:
    """Collect single-token IDs whose decoded text ends with '=' (e.g. '=', ' =', '\\n=')."""
    out: List[int] = []
    for v in ("=", " =", "\n=", "\n =", "=\t", "= "):
        ids = tok.encode(v, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in out:
            out.append(ids[0])
    assert out, "Could not find any single-token '=' ID in tokenizer!"
    return out


# =============================================================================
# DATA GENERATORS (EVAL 1 MIXED OOD + EVAL 2 BIG-NUMBER WORD PROBLEMS)
# =============================================================================

def rand_int_digits(rng: random.Random, n: int) -> int:
    if n <= 1:
        return rng.randint(0, 9)
    return rng.randint(10 ** (n - 1), 10**n - 1)


def _fmt_dec(d: Decimal) -> str:
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def gen_eval1_case(rng: random.Random, bucket: str) -> Tuple[str, str]:
    """Generate `(expr_without_eq, exact_answer_str)` for Eval 1 mixed-operation OOD buckets."""
    if bucket == "id_mixed_ops":
        k = rng.randint(0, 3)
        if k == 0:
            a, b = rand_int_digits(rng, rng.randint(3, 6)), rand_int_digits(rng, rng.randint(3, 6))
            return f"{a} + {b}", str(a + b)
        if k == 1:
            a, b = rand_int_digits(rng, rng.randint(3, 6)), rand_int_digits(rng, rng.randint(3, 6))
            return f"{a} - {b}", str(a - b)
        if k == 2:
            a, b = rand_int_digits(rng, rng.randint(3, 5)), rand_int_digits(rng, rng.randint(2, 4))
            return f"{a} * {b}", str(a * b)
        d = rng.randint(2, 999)
        q = rand_int_digits(rng, rng.randint(3, 6))
        return f"{q * d} / {d}", str(q)

    if bucket == "ood_12_48_int":
        k = rng.randint(0, 2)
        L = rng.choice([12, 18, 24, 36, 48])
        if k == 0:
            a, b = rand_int_digits(rng, L), rand_int_digits(rng, L)
            return f"{a} + {b}", str(a + b)
        if k == 1:
            a, b = rand_int_digits(rng, L), rand_int_digits(rng, L)
            return f"{a} - {b}", str(a - b)
        Lm = rng.choice([12, 16, 20])
        a, b = rand_int_digits(rng, Lm), rand_int_digits(rng, Lm)
        return f"{a} * {b}", str(a * b)

    if bucket == "multi_term":
        k = rng.randint(0, 2)
        if k == 0:
            a, b, c = rand_int_digits(rng, 9), rand_int_digits(rng, 9), rand_int_digits(rng, 9)
            return f"{a:,} + {b:,} - {c:,}", str(a + b - c)
        if k == 1:
            a, b, c = rand_int_digits(rng, 5), rand_int_digits(rng, 4), rand_int_digits(rng, 6)
            return f"({a} + {b}) * {c}", str((a + b) * c)
        a, b, c, d = (
            rand_int_digits(rng, 6),
            rand_int_digits(rng, 5),
            rand_int_digits(rng, 4),
            rand_int_digits(rng, 6),
        )
        return f"{a} + {b} * {c} - {d}", str(a + b * c - d)

    if bucket == "decimals":
        k = rng.randint(0, 2)
        da = Decimal(rng.randint(100, 999999)) / Decimal(100)
        db = Decimal(rng.randint(100, 99999)) / Decimal(100)
        if k == 0:
            return f"{_fmt_dec(da)} + {_fmt_dec(db)}", _fmt_dec(da + db)
        if k == 1:
            return f"{_fmt_dec(da)} - {_fmt_dec(db)}", _fmt_dec(da - db)
        return f"{_fmt_dec(da)} * {_fmt_dec(db)}", _fmt_dec(da * db)

    if bucket == "pow_sqrt":
        k = rng.randint(0, 2)
        if k == 0:
            b = rng.randint(2, 35)
            e = rng.randint(2, 9)
            return f"{b}^{e}", str(b**e)
        if k == 1:
            r = rng.randint(12, 99999)
            return f"sqrt({r * r})", str(r)
        r = rng.randint(5, 999)
        b = rng.randint(2, 20)
        e = rng.randint(2, 5)
        return f"sqrt({r * r}) + {b}^{e}", str(r + b**e)

    if bucket == "negative":
        k = rng.randint(0, 1)
        if k == 0:
            L = rng.randint(10, 24)
            a = rand_int_digits(rng, L)
            b = a + rand_int_digits(rng, rng.randint(6, L))
            return f"{a} - {b}", str(a - b)
        a = rand_int_digits(rng, 6)
        b = rand_int_digits(rng, 4)
        return f"-{a} + {b}", str(-a + b)

    raise ValueError(f"Unknown Eval 1 bucket: {bucket}")


def make_hard_mixed_word_problems(n: int, seed: int = 202608) -> List[Dict[str, Any]]:
    """Deterministic benchmark of hard multi-step word problems with 6-12 digit ints & decimals
    using +, -, *, /. Every problem has an exact integer or exact 2-decimal-place answer."""
    rng = random.Random(seed)
    problems: List[Dict[str, Any]] = []
    cities = ["Northbridge", "Valoria", "Kestrel", "Solhaven", "Alderbrook", "Maris", "Oakhaven", "Vesper"]
    orgs = ["Apex Energy", "Helios Logistics", "Meridian Rail", "Orion Systems", "Vertex Bio", "Cobalt Mining"]
    items = ["microchips", "solar cells", "battery packs", "fiber cables", "sensor modules", "turbine blades"]

    for i in range(n):
        tpl = i % 5
        if tpl == 0:
            # 2-step multiplication + subtraction (6-7 digit * 3-4 digit - 9-10 digit)
            org = rng.choice(orgs)
            units = rand_int_digits(rng, rng.randint(6, 7))
            price = rand_int_digits(rng, rng.randint(3, 4))
            gross = units * price
            cost = rng.randint(gross // 5, (gross * 3) // 4)
            ans = Decimal(gross - cost)
            q = (
                f"{org} sold {units} units of equipment at {price} dollars per unit. "
                f"Total operating and manufacturing expenses were {cost} dollars. "
                f"What was the company's net profit in dollars?"
            )
            ops = "*, -"
        elif tpl == 1:
            # 3-step big-number multi-term addition & subtraction (9-12 digit)
            city = rng.choice(cities)
            a = rand_int_digits(rng, rng.randint(9, 12))
            b = rand_int_digits(rng, rng.randint(9, 12))
            c = rand_int_digits(rng, rng.randint(9, 11))
            d = rng.randint(10**8, (a + b + c) // 2)
            ans = Decimal(a + b + c - d)
            q = (
                f"A regional grid around {city} generated {a} kilowatt-hours in Q1, "
                f"{b} kilowatt-hours in Q2, and {c} kilowatt-hours in Q3. "
                f"If {d} kilowatt-hours were lost during transmission, "
                f"how many usable kilowatt-hours remained?"
            )
            ops = "+, +, -"
        elif tpl == 2:
            # 2-step exact division + addition (9-11 digit dividend)
            item = rng.choice(items)
            crates = rng.randint(125, 999)
            per_crate = rand_int_digits(rng, rng.randint(6, 8))
            total_items = crates * per_crate
            bonus = rand_int_digits(rng, rng.randint(6, 8))
            ans = Decimal(per_crate + bonus)
            q = (
                f"A warehouse distributed {total_items} {item} equally among {crates} shipping containers. "
                f"Each container also received {bonus} spare {item}. "
                f"How many {item} were in each container in total?"
            )
            ops = "/, +"
        elif tpl == 3:
            # 2-step exact decimal multiplication + addition (large quantities * decimal rate)
            org = rng.choice(orgs)
            qty = rand_int_digits(rng, rng.randint(6, 8))
            rate_cents = rng.randint(125, 9875)  # $1.25 to $98.75
            rate = Decimal(rate_cents) / Decimal(100)
            fee_cents = rng.randint(100005, 9999995)
            fee = Decimal(fee_cents) / Decimal(100)
            ans = Decimal(qty) * rate + fee
            q = (
                f"{org} purchased {qty} liters of specialty fuel at {_fmt_dec(rate)} dollars per liter, "
                f"plus a flat delivery surcharge of {_fmt_dec(fee)} dollars. "
                f"What was the total cost in dollars? (Give the exact decimal answer.)"
            )
            ops = "*, + (decimal)"
        else:
            # 3-step big inventory: a + b * c - d
            item = rng.choice(items)
            a = rand_int_digits(rng, rng.randint(8, 10))
            b = rand_int_digits(rng, rng.randint(5, 6))
            c = rng.randint(112, 987)
            d = rng.randint(10**7, (a + b * c) // 2)
            ans = Decimal(a + b * c - d)
            q = (
                f"A distributor began the quarter with {a} {item} in stock. "
                f"It received {b} pallets containing {c} {item} each, "
                f"and then fulfilled orders totaling {d} {item}. "
                f"How many {item} remain in stock?"
            )
            ops = "+, *, -"

        problems.append({
            "id": i,
            "template": tpl,
            "ops": ops,
            "question": q,
            "answer": _fmt_dec(ans),
        })
    return problems


def make_direct_arith_think_prompts(n: int, seed: int = 33333) -> List[str]:
    """Concise multi-op arithmetic prompts that quickly produce explicit '=' equations inside <think>."""
    rng = random.Random(seed)
    prompts = []
    for i in range(n):
        k = i % 5
        if k == 0:
            a, b, c = rand_int_digits(rng, 4), rand_int_digits(rng, 4), rand_int_digits(rng, 3)
            prompts.append(f"Compute {a} + {b}, then subtract {c} from the result.")
        elif k == 1:
            a, b, c = rand_int_digits(rng, 3), rand_int_digits(rng, 3), rand_int_digits(rng, 4)
            prompts.append(f"Compute {a} * {b} + {c}. Show each arithmetic step.")
        elif k == 2:
            d = rng.randint(3, 49)
            q = rng.randint(100, 9999)
            prompts.append(f"Divide {q * d} by {d}, then add {rng.randint(10, 999)}.")
        elif k == 3:
            da = Decimal(rng.randint(115, 9995)) / Decimal(100)
            db = Decimal(rng.randint(115, 995)) / Decimal(100)
            prompts.append(f"Compute {_fmt_dec(da)} * {_fmt_dec(db)} and show the exact result.")
        else:
            r = rng.randint(11, 99)
            b = rng.randint(2, 12)
            prompts.append(f"Compute sqrt({r * r}) + {b}^3. Show each step.")
    return prompts


def make_hard_negative_think_prompts(n: int, seed: int = 999) -> List[str]:
    """Prompts that encourage <think> traces with algebra, equations with variables, and symbolic definitions."""
    rng = random.Random(seed)
    prompts = []
    for i in range(n):
        k = i % 4
        if k == 0:
            a, b = rng.randint(2, 19), rng.randint(5, 80)
            prompts.append(f"Solve for x: {a}x + {b} = {a * rng.randint(2, 30) + b}.")
        elif k == 1:
            a, b = rng.randint(2, 12), rng.randint(2, 12)
            prompts.append(f"Expand and simplify the algebraic expression ({a}x + {b})(x - 3).")
        elif k == 2:
            prompts.append(
                f"Write a short Python function that computes the sum of squares up to n={rng.randint(5, 25)}."
            )
        else:
            a, b = rng.randint(2, 9), rng.randint(10, 50)
            prompts.append(f"If 2x + y = {b} and x - y = {a}, find x and y.")
    return prompts


def make_synth_think_gate_examples(n_pos: int, n_neg: int, seed: int) -> Tuple[List[str], List[str]]:
    """Build diverse <think>-wrapped positive evaluable numeric expressions ending in '='
    and matched hard-negative examples (algebra, assignments, comparisons, unknown funcs, bare constants)."""
    rng = random.Random(seed)
    pos_prefixes = [
        "<think>\nLet's compute ",
        "<think>\nFirst, we calculate ",
        "<think>\nStep 1: we need to find ",
        "<think>\nNext, evaluating gives ",
        "<think>\nSo the intermediate result is ",
        "<think>\nStep 2: total = ",
        "<think>\nNow, computing ",
        "<think>\nTherefore, ",
    ]
    neg_prefixes = [
        "<think>\nLet's set up ",
        "<think>\nFirst, we define ",
        "<think>\nStep 1: we have ",
        "<think>\nSo the equation is ",
        "<think>\nLet's check the formula: ",
        "<think>\nStep 2: ",
        "<think>\nIn Python: ",
        "<think>\nNow, simplifying gives ",
    ]

    pos_texts: List[str] = []
    buckets = ("id_mixed_ops", "ood_12_48_int", "multi_term", "decimals", "pow_sqrt", "negative")
    for i in range(n_pos):
        bname = buckets[i % len(buckets)]
        expr, _ = gen_eval1_case(rng, bname)
        pref = pos_prefixes[i % len(pos_prefixes)]
        if i % 3 == 0:
            p1, p2 = rand_int_digits(rng, 3), rand_int_digits(rng, 3)
            pref = f"<think>\nWe start with {p1} and {p2}. First, {p1} + {p2} = {p1 + p2}.\nNext, "
        sp = " " if rng.random() < 0.85 else ""
        pos_texts.append(f"{pref}{expr}{sp}=")

    neg_texts: List[str] = []
    vars_list = ["x", "y", "n", "total", "balance", "diff", "result", "area", "profit"]
    for i in range(n_neg):
        pref = neg_prefixes[i % len(neg_prefixes)]
        k = i % 7
        a = rand_int_digits(rng, rng.randint(1, 5))
        b = rand_int_digits(rng, rng.randint(1, 5))
        v = rng.choice(vars_list)
        if k == 0:
            neg_texts.append(f"{pref}{v} =")
        elif k == 1:
            neg_texts.append(f"{pref}{rng.randint(2, 15)}{v} + {b} =")
        elif k == 2:
            neg_texts.append(f"{pref}{v} + {b} =")
        elif k == 3:
            neg_texts.append(f"{pref}if {v} ==")
        elif k == 4:
            neg_texts.append(f"{pref}{a} =")
        elif k == 5:
            neg_texts.append(f"{pref}{v} * {a} - {b} =")
        else:
            neg_texts.append(f"{pref}def compute(a={a}, {v}=")

    return pos_texts, neg_texts


# =============================================================================
# ANSWER EXTRACTION, FAILURE DIAGNOSTICS & <think> GRADING
# =============================================================================

def _parse_decimal_str(s: str) -> Optional[Decimal]:
    cleaned = s.replace(",", "").replace("$", "").replace(" ", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not m:
        return None
    try:
        return Decimal(m.group(0))
    except InvalidOperation:
        return None


def extract_final_number(text: str) -> Optional[Decimal]:
    """Extract final numeric answer (int or decimal) from a thinking model response."""
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        val = _parse_decimal_str(boxed[-1])
        if val is not None:
            return val
    tail = text.split("</think>")[-1] if "</think>" in text else text
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", tail)
    nums = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if nums:
        try:
            return Decimal(nums[-1])
        except InvalidOperation:
            pass
    return None


def answers_match(pred: Optional[Decimal], gold_str: str, tol: Decimal = Decimal("1e-6")) -> bool:
    if pred is None:
        return False
    try:
        gold = Decimal(gold_str)
    except InvalidOperation:
        return False
    return abs(pred - gold) <= tol


def parse_gsm8k_answer(ans_str: str) -> Optional[str]:
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", ans_str)
    if not m:
        return None
    return m.group(1).replace(",", "")


RECHECK_PAT = re.compile(
    r"\b(wait|hold on|double[- ]check|let me check|recalculate|re-check|recheck|let's check|actually)\b",
    re.IGNORECASE,
)


def default_tag_for_model(model_name: str) -> str:
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


BIN_EQ_IN_THINK_PAT = re.compile(r"(?<![\w.,])(\d+(?:,\d+)*)\s*([+\-*/×÷])\s*(\d+(?:,\d+)*)\s*=(?!=)")
MULTI_TERM_SUM_PAT = re.compile(r"(?<![\w.])\d[\d,]*\s*[+\-*/×÷]\s*\d[\d,]*\s*[+\-*/×÷]\s*\d[\d,]*(?![\w.])")
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
    pred: Optional[Decimal],
    n_gen_tokens: int,
    max_new_tokens: int,
) -> Dict[str, bool]:
    """Compute per-problem 8-tag diagnostic breakdown for a thinking trace."""
    has_close = "</think>" in text
    think_part = text.split("</think>", 1)[0] if has_close else text
    if think_part.startswith("<think>"):
        think_part = think_part[len("<think>"):]

    truncated_think = not has_close
    hit_token_limit = int(n_gen_tokens) >= int(max_new_tokens)
    has_boxed = BOXED_NUM_PAT.search(text) is not None
    hit_limit_before_boxed = bool(hit_token_limit and not has_boxed)
    no_final_answer = bool(has_close and (pred is None))

    big_bin_count = 0
    small_bin_count = 0
    for m in BIN_EQ_IN_THINK_PAT.finditer(think_part):
        a_s = m.group(1).replace(",", "")
        b_s = m.group(3).replace(",", "")
        if len(a_s) >= 6 or len(b_s) >= 6:
            big_bin_count += 1
        elif len(a_s) <= 2 and len(b_s) <= 2:
            small_bin_count += 1

    wrote_big_binary_eq = big_bin_count >= 1
    column_by_column_only = (small_bin_count >= 3) and (big_bin_count == 0)
    wrote_multi_term_sum = MULTI_TERM_SUM_PAT.search(think_part) is not None

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
    n = len(tags_list)
    out: Dict[str, Any] = {
        "n": n,
        "n_wrong": sum(1 for c in correct_list if not c),
        "mean_gen_tokens": round(sum(gen_lens) / max(1, n), 2),
    }
    for k in FAILURE_TAG_KEYS:
        out[k] = sum(1 for t in tags_list if t.get(k))
    return out


_RHS_NUM_RE = re.compile(r"^\s*(-?\d+(?:,\d+)*(?:\.\d+)?)")


def audit_think_arithmetic(text: str) -> Dict[str, int]:
    """Audit all evaluable math expressions `<expr> = <num>` inside `<think>...</think>`
    using `extract_and_eval_expr`."""
    think_part = text.split("</think>")[0] if "</think>" in text else text
    n_tot = 0
    n_ok = 0
    for m in re.finditer(r"=", think_part):
        pos = m.start()
        prefix = think_part[max(0, pos - 256):pos + 1]
        parsed = extract_and_eval_expr(prefix)
        if parsed is None:
            continue
        _, expected_str, _ = parsed
        rhs_m = _RHS_NUM_RE.match(think_part[pos + 1:pos + 160])
        if not rhs_m:
            continue
        got_dec = _parse_decimal_str(rhs_m.group(1))
        n_tot += 1
        if answers_match(got_dec, expected_str, tol=Decimal("1e-3")):
            n_ok += 1
    return {"n_eq": n_tot, "n_correct": n_ok}


def grade_thinking(
    traces: Sequence[Dict[str, Any]],
    tokenizer: Any,
    max_new_tokens: int,
    is_kernel: bool = False,
) -> Dict[str, Any]:
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
        if is_kernel:
            out["kernel_recheck_rate"] = None
            out["n_kernel_events"] = 0
        return out

    think_lens: List[int] = []
    truncated = 0
    recheck_counts: List[int] = []
    tot_eq = 0
    ok_eq = 0
    ev_total = 0
    ev_rechecked = 0

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

        if is_kernel:
            events = tr.get("kernel_events") or []
            for e_idx, ev in enumerate(events):
                ev_total += 1
                end_i = int(ev["end_gen_idx"])
                next_cap = end_i + 80
                if e_idx + 1 < len(events) and "start_gen_idx" in events[e_idx + 1]:
                    next_cap = min(next_cap, int(events[e_idx + 1]["start_gen_idx"]))
                after_txt = tokenizer.decode(gen_ids[end_i:max(end_i, next_cap)], skip_special_tokens=True)[:160]
                if RECHECK_PAT.search(after_txt) is not None:
                    ev_rechecked += 1

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
    if is_kernel:
        res["kernel_recheck_rate"] = round(ev_rechecked / ev_total, 4) if ev_total > 0 else None
        res["n_kernel_events"] = int(ev_total)
    return res


DIRECT_EQ_INSTRUCTION = (
    "Write each arithmetic calculation directly on one line ending with = "
    "(e.g. a * b - c = ) without breaking numbers into digit-by-digit columns or chunks."
)


def format_r1_prompt(tokenizer: Any, question: str, direct_eq_prompt: bool = False) -> str:
    """Format a question for thinking models ending with '<think>\\n'."""
    prefix = f"{DIRECT_EQ_INSTRUCTION}\n" if direct_eq_prompt else ""
    user_msg = (
        f"{prefix}{question}\n"
        "Please reason step by step, and put your final numerical answer within \\boxed{}."
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
# INLINE GATED MATH KERNEL GENERATION ENGINE
# =============================================================================

@torch.no_grad()
def generate_with_inline_kernel(
    llm: nn.Module,
    tok: Any,
    eq_set: set[int],
    prompt_ids: List[int],
    max_new_tokens: int,
    eos_token_id: int,
    probe: Optional[GateProbe] = None,
    use_kernel: bool = True,
    force_open_at_eq: bool = False,
    gate_threshold: float = 0.5,
) -> Tuple[List[int], Dict[str, Any]]:
    """Greedy generation (`do_sample=False`) with optional gated inline AST math kernel.

    When `use_kernel=True`:
      - At every '=' token, evaluates `probe(h_last) > gate_threshold` (or `force_open_at_eq`)
        AND runs `extract_and_eval_expr(prefix_text)` in <0.1 ms.
      - When both fire, tokenizes the exact result string (+ trailing ' ') and streams those
        tokens directly into the decode loop (keeping the frozen model's KV cache 100% synced),
        then immediately hands control back to the frozen thinking model!
    """
    dev = next(llm.parameters()).device
    lm_head = llm.get_output_embeddings()

    full_ids = list(prompt_ids)
    cur_in = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
    past = None
    generated: List[int] = []
    pending_ids: deque[int] = deque()
    active_event: Optional[Dict[str, Any]] = None

    stats: Dict[str, Any] = {
        "total_eq_seen": 0,
        "valid_expr_eq": 0,
        "probe_fires": 0,
        "true_fires": 0,
        "false_probe_fires": 0,
        "missed_exprs": 0,
        "kernel_tokens_emitted": 0,
        "kernel_latencies_ms": [],
        "kernel_events": [],
    }

    for _step_i in range(max_new_tokens):
        out = llm.model(input_ids=cur_in, past_key_values=past, use_cache=True)
        past = out.past_key_values
        h_last = out.last_hidden_state[:, -1, :].float()  # [1, d_llm]

        # If we are currently streaming a kernel result, pop the next token directly
        if pending_ids:
            nxt = pending_ids.popleft()
            generated.append(nxt)
            full_ids.append(nxt)
            if not pending_ids and active_event is not None:
                active_event["end_gen_idx"] = len(generated)
                stats["kernel_events"].append(active_event)
                active_event = None
            cur_in = torch.tensor([[nxt]], dtype=torch.long, device=dev)
            continue

        # Check inline math kernel trigger at '='
        emitted_from_kernel = False
        if use_kernel and full_ids[-1] in eq_set:
            stats["total_eq_seen"] += 1
            prefix_txt = tok.decode(full_ids[max(0, len(full_ids) - 160):], skip_special_tokens=False)
            parsed = extract_and_eval_expr(prefix_txt)
            is_valid = parsed is not None
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

            if probe_open and parsed is not None:
                expr_str, result_str, lat_ms = parsed
                stats["kernel_latencies_ms"].append(lat_ms)
                emit_str = (result_str + " ") if prefix_txt.endswith(" ") else (" " + result_str + " ")
                emit_ids = tok.encode(emit_str, add_special_tokens=False)
                if emit_ids:
                    stats["kernel_tokens_emitted"] += len(emit_ids)
                    active_event = {
                        "expr": expr_str,
                        "result": result_str,
                        "latency_ms": round(lat_ms, 4),
                        "start_gen_idx": len(generated),
                        "end_gen_idx": len(generated) + len(emit_ids),
                    }
                    pending_ids.extend(emit_ids)
                    nxt = pending_ids.popleft()
                    generated.append(nxt)
                    full_ids.append(nxt)
                    if not pending_ids:
                        active_event["end_gen_idx"] = len(generated)
                        stats["kernel_events"].append(active_event)
                        active_event = None
                    cur_in = torch.tensor([[nxt]], dtype=torch.long, device=dev)
                    emitted_from_kernel = True

        if emitted_from_kernel:
            continue

        logits = lm_head(h_last.to(lm_head.weight.dtype)).float()
        nxt = int(logits[0].argmax().item())
        generated.append(nxt)
        full_ids.append(nxt)
        if nxt == eos_token_id:
            break
        cur_in = torch.tensor([[nxt]], dtype=torch.long, device=dev)

    if active_event is not None:
        active_event["end_gen_idx"] = len(generated)
        stats["kernel_events"].append(active_event)

    return generated, stats


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class Experiment:
    def __init__(self, args: argparse.Namespace):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = (
            torch.bfloat16
            if (self.device.type == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float32
        )
        self.ckpt_dir = Path(args.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        mode_tag = "smoke" if args.smoke else "full"
        self.probe_path = self.ckpt_dir / f"gate_probe_{args.tag}_{mode_tag}.pt"

        out_p = Path(args.out)
        if out_p.name.endswith("_results.json"):
            self.traces_path = out_p.with_name(out_p.name[:-len("_results.json")] + "_traces.jsonl")
        else:
            suf = "_smoke" if args.smoke else ""
            self.traces_path = out_p.with_name(f"math_kernel_{args.tag}{suf}_traces.jsonl")
        if args.fresh and self.traces_path.exists():
            self.traces_path.unlink()

        config = {
            "model": args.model,
            "tag": args.tag,
            "smoke": args.smoke,
            "eval1_n": args.eval1_n,
            "eval2_n": args.eval2_n,
            "eval3_n": args.eval3_n,
            "max_new_tokens": args.max_new_tokens,
            "direct_eq_prompt": args.direct_eq_prompt,
            "skip_step6": args.skip_step6,
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
        self.eq_ids = collect_eq_token_ids(self.tok)
        self.eq_set = set(self.eq_ids)

        self.store.data["meta"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "gpu": torch.cuda.get_device_name(0) if self.device.type == "cuda" else None,
            "d_llm": self.d_llm,
            "eq_ids": self.eq_ids,
        }
        self.store.save()
        self.probe: Optional[GateProbe] = None

    # ------------------------------------------------------------------ Step 1
    def step1_eval1_ungated(self) -> None:
        log("=== STEP 1: Eval 1 — Standalone Mixed-Operation OOD Math (Un-gated Kernel + Fair Chat Base) ===")
        if self.store.done("step1_eval1_ood"):
            log("  step1_eval1_ood already done; skipping.")
            return

        t0 = time.time()
        buckets = ["id_mixed_ops", "ood_12_48_int", "multi_term", "decimals", "pow_sqrt", "negative"]
        n_per = self.args.eval1_n
        all_lats: List[float] = []
        res: Dict[str, Any] = {
            "eval_mode": "ungated_inline_math_kernel",
            "n_per_bucket": n_per,
            "buckets": {},
        }

        for bi, bname in enumerate(buckets):
            rng = random.Random(1000 + 37 * bi)
            cases = [gen_eval1_case(rng, bname) for _ in range(n_per)]
            exact_ok = prefix_ok = 0
            sample_failures: List[Dict[str, Any]] = []

            for expr, ans_str in cases:
                prompt = f"Compute: {expr} ="
                p_ids = self.tok.encode(prompt, add_special_tokens=False)
                max_tok = len(ans_str) + 6

                g_gen, g_st = generate_with_inline_kernel(
                    self.llm, self.tok, self.eq_set, p_ids,
                    max_new_tokens=max_tok, eos_token_id=self.eos_id,
                    use_kernel=True, force_open_at_eq=True,
                )
                all_lats.extend(g_st["kernel_latencies_ms"])
                g_txt = self.tok.decode(g_gen, skip_special_tokens=True).strip()
                m_g = re.match(r"^(-?\d+(?:\.\d+)?)", g_txt)
                ok_exact = bool(m_g and answers_match(_parse_decimal_str(m_g.group(1)), ans_str))
                ok_pref = g_txt.startswith(ans_str)
                if ok_exact:
                    exact_ok += 1
                elif len(sample_failures) < 3:
                    sample_failures.append({"expr": expr, "gold": ans_str, "got": g_txt[:80]})
                if ok_pref:
                    prefix_ok += 1

            b_res: Dict[str, Any] = {
                "kernel_ungated_exact": round(exact_ok / max(1, n_per), 4),
                "kernel_ungated_prefix_exact": round(prefix_ok / max(1, n_per), 4),
                "n_exact": exact_ok,
                "n": n_per,
            }
            if sample_failures:
                b_res["sample_failures"] = sample_failures
            res["buckets"][bname] = b_res
            log(
                f"    {bname:16s} | kernel_ungated={b_res['kernel_ungated_exact']:.4f} "
                f"(prefix={b_res['kernel_ungated_prefix_exact']:.4f}, {exact_ok}/{n_per})"
            )

        # Fair chat-template Base check on OOD / multi-term / decimals
        fair_buckets = [("ood_12_48_int", 1), ("multi_term", 2), ("decimals", 3)]
        n_fair = min(n_per, 8 if self.args.smoke else 16)
        fair_max_tok = 512
        fair_res: Dict[str, Any] = {"n_per_bucket": n_fair, "max_new_tokens": fair_max_tok, "buckets": {}}
        log(f"  Running fair chat-template Base check (n={n_fair}, max_new_tokens={fair_max_tok})...")
        for bname, bi in fair_buckets:
            rng = random.Random(1000 + 37 * bi)
            cases = [gen_eval1_case(rng, bname) for _ in range(n_fair)]
            base_ok = n_trunc = 0
            for expr, ans_str in cases:
                p_str = format_r1_prompt(self.tok, f"Compute {expr}.")
                p_ids = self.tok.encode(p_str, add_special_tokens=False)
                b_gen, _ = generate_with_inline_kernel(
                    self.llm, self.tok, self.eq_set, p_ids,
                    max_new_tokens=fair_max_tok, eos_token_id=self.eos_id, use_kernel=False,
                )
                b_txt = self.tok.decode(b_gen, skip_special_tokens=True)
                if "</think>" not in b_txt:
                    n_trunc += 1
                if answers_match(extract_final_number(b_txt), ans_str):
                    base_ok += 1
            fair_res["buckets"][bname] = {
                "base_chat_exact": round(base_ok / max(1, n_fair), 4),
                "n_correct": base_ok,
                "n_truncated_think": n_trunc,
                "n": n_fair,
            }
            log(
                f"    base_chat_fair {bname:14s} | exact={base_ok}/{n_fair} "
                f"({base_ok / max(1, n_fair):.4f}) | truncated_think={n_trunc}/{n_fair}"
            )

        res["base_chat_fair"] = fair_res
        res["median_kernel_latency_ms"] = round(statistics.median(all_lats), 4) if all_lats else None
        res["seconds"] = round(time.time() - t0, 1)
        log(f"  Step 1 done: median_kernel_latency_ms={res['median_kernel_latency_ms']} ms")
        self.store.set("step1_eval1_ood", res)
        cleanup_cuda()

    # ------------------------------------------------------------------ Step 2
    def _load_code_chunks(self, modules: Sequence[str], max_chunks: int, chunk_len: int = 256) -> List[List[int]]:
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
        """Evaluate next-token NLL overall and at '=' positions, plus probe & joint-kernel false fires."""
        assert self.probe is not None
        lm_head = self.llm.get_output_embeddings()
        eq_t = torch.tensor(self.eq_ids, dtype=torch.long, device=self.device)
        tot = n_tok = eq_tot = eq_n = 0.0
        gate_sum = 0.0
        n_probe_fires = 0
        n_kernel_fires = 0

        for ids_list in seqs:
            if len(ids_list) < 2:
                continue
            ids = torch.tensor([ids_list], dtype=torch.long, device=self.device)
            h = self.llm.model(input_ids=ids, use_cache=False).last_hidden_state.float()
            logits = lm_head(h[:, :-1, :].to(lm_head.weight.dtype)).float()[0]
            ce = F.cross_entropy(logits, ids[0, 1:], reduction="none")
            eq_mask = torch.isin(ids[0, :-1], eq_t)

            if use_probe and eq_mask.any():
                eq_positions = eq_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                g_vals = self.probe(h[0, eq_positions]).squeeze(-1)
                gate_sum += float(g_vals.sum().item())
                for idx_k, pos in enumerate(eq_positions):
                    if float(g_vals[idx_k].item()) > threshold:
                        n_probe_fires += 1
                        prefix_txt = self.tok.decode(ids_list[max(0, pos - 160):pos + 1], skip_special_tokens=False)
                        parsed = extract_and_eval_expr(prefix_txt)
                        if parsed is not None:
                            n_kernel_fires += 1
                            # If kernel actually fired on code/prose and would force a mismatched token,
                            # apply a conservative NLL penalty (+10.0 nats) at that position
                            _, result_str, _ = parsed
                            emit_str = (result_str + " ") if prefix_txt.endswith(" ") else (" " + result_str + " ")
                            emit_ids = self.tok.encode(emit_str, add_special_tokens=False)
                            if emit_ids and emit_ids[0] != ids_list[pos + 1]:
                                ce[pos] = ce[pos] + 10.0

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
            "n_probe_fires_at_eq": int(n_probe_fires) if use_probe else None,
            "n_kernel_fires_at_eq": int(n_kernel_fires) if use_probe else None,
            "threshold": threshold if use_probe else None,
        }

    def step2_harvest_and_train_gate(self) -> None:
        log(f"=== STEP 2: Harvest <think> Traces & Train {self.d_llm + 1:,}-param GateProbe ===")
        if self.store.done("step2_gate_probe") and self.probe_path.exists():
            self.probe = GateProbe(self.d_llm).to(self.device)
            self.probe.load_state_dict(torch.load(self.probe_path, map_location=self.device))
            self.probe.eval()
            log(f"  Loaded cached GateProbe from {self.probe_path}")
            return

        t0 = time.time()
        n_dir = 6 if self.args.smoke else 32
        n_gsm = 6 if self.args.smoke else 36
        n_big = 4 if self.args.smoke else 24
        n_neg = 4 if self.args.smoke else 28

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

        for p in make_hard_mixed_word_problems(n_big, seed=77777):
            harvest_items.append(("big_wp", p["question"]))
        for q in make_hard_negative_think_prompts(n_neg, seed=88888):
            harvest_items.append(("hard_neg", q))

        trace_feats: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        eq_t = torch.tensor(self.eq_ids, dtype=torch.long, device=self.device)
        max_h_tok = 320 if self.args.smoke else 448

        for qi, (src_type, q) in enumerate(harvest_items):
            p_str = format_r1_prompt(self.tok, q)
            p_ids = self.tok.encode(p_str, add_special_tokens=False)
            gen_ids, _ = generate_with_inline_kernel(
                self.llm, self.tok, self.eq_set, p_ids,
                max_new_tokens=max_h_tok, eos_token_id=self.eos_id, use_kernel=False,
            )
            full_ids = p_ids + gen_ids
            ids_t = torch.tensor([full_ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                h = self.llm.model(input_ids=ids_t, use_cache=False).last_hidden_state[0].float()

            xs_list, ys_list = [], []
            for pos in range(len(p_ids), len(full_ids)):
                if full_ids[pos] in self.eq_set:
                    prefix_txt = self.tok.decode(full_ids[max(0, pos - 160):pos + 1], skip_special_tokens=False)
                    is_pos = extract_and_eval_expr(prefix_txt) is not None
                    xs_list.append(h[pos].cpu())
                    ys_list.append(1.0 if is_pos else 0.0)
            if xs_list:
                trace_feats.append((src_type, torch.stack(xs_list, 0), torch.tensor(ys_list)))
            if (qi + 1) % max(1, len(harvest_items) // 5) == 0 or (qi + 1) == len(harvest_items):
                n_p = sum(int((y == 1).sum()) for _, _, y in trace_feats)
                n_n = sum(int((y == 0).sum()) for _, _, y in trace_feats)
                log(f"    harvested {qi + 1}/{len(harvest_items)} traces -> pos={n_p}, neg={n_n}")

        # Problem-level stratified 80/20 split by source_type -> zero trace overlap
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

        # Python stdlib '=' training negatives
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

        # Synthetic <think>-wrapped positives + matched hard negatives (disjoint train/val)
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

        # Train GateProbe (standardized + folded; cap pos_weight <= 1.0)
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
        for _ in range(400):
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

        # Held-out '=' metrics at threshold=0.5, 0.75, and best-F1
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

        # Held-out Python code & prose NLL damage
        code_ev = self._load_code_chunks(CODE_EVAL_MODULES, max_chunks=16 if self.args.smoke else 80, chunk_len=256)
        prose_ev = [self.tok.encode(s, add_special_tokens=False) for s in ENGLISH_PROBES]
        dmg_code_base = self._eval_nll_damage(code_ev, use_probe=False)
        dmg_code_gated = self._eval_nll_damage(code_ev, use_probe=True, threshold=0.5)
        dmg_code_gated_t75 = self._eval_nll_damage(code_ev, use_probe=True, threshold=0.75)
        dmg_prose_base = self._eval_nll_damage(prose_ev, use_probe=False)
        dmg_prose_gated = self._eval_nll_damage(prose_ev, use_probe=True, threshold=0.5)

        # Gated Eval 1 check across all 6 mixed-operation OOD buckets (GateProbe active!)
        gated_e1_n = 8 if self.args.smoke else 16
        gated_e1: Dict[str, Any] = {"n_per_bucket": gated_e1_n, "threshold": 0.5, "buckets": {}}
        buckets = ["id_mixed_ops", "ood_12_48_int", "multi_term", "decimals", "pow_sqrt", "negative"]
        for bi, bname in enumerate(buckets):
            rng_g = random.Random(1000 + 37 * bi)
            cases_g = [gen_eval1_case(rng_g, bname) for _ in range(gated_e1_n)]
            g_ok = g_pref = g_fires = g_missed = 0
            for expr, ans_str in cases_g:
                prompt = f"Compute: {expr} ="
                p_ids = self.tok.encode(prompt, add_special_tokens=False)
                g_gen, g_st = generate_with_inline_kernel(
                    self.llm, self.tok, self.eq_set, p_ids,
                    max_new_tokens=len(ans_str) + 6, eos_token_id=self.eos_id,
                    probe=self.probe, use_kernel=True, force_open_at_eq=False, gate_threshold=0.5,
                )
                g_txt = self.tok.decode(g_gen, skip_special_tokens=True).strip()
                m_g = re.match(r"^(-?\d+(?:\.\d+)?)", g_txt)
                if m_g and answers_match(_parse_decimal_str(m_g.group(1)), ans_str):
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
            log(
                f"    gated_eval1 {bname:14s} | exact={g_ok}/{gated_e1_n} ({g_ok / max(1, gated_e1_n):.4f}) "
                f"| fires={g_fires}/{gated_e1_n} missed={g_missed}"
            )

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
            "heldout_combined_eq_t75": comb_val_metrics_t75,
            "best_f1_threshold": best_thr,
            "heldout_think_eq_best_f1": val_metrics_best,
            "heldout_combined_eq_best_f1": comb_val_metrics_best,
            "damage_python_base": dmg_code_base,
            "damage_python_gated": dmg_code_gated,
            "damage_python_gated_t75": dmg_code_gated_t75,
            "damage_python_nll_delta": round(dmg_code_gated["nll"] - dmg_code_base["nll"], 6),
            "damage_python_nll_delta_t75": round(dmg_code_gated_t75["nll"] - dmg_code_base["nll"], 6),
            "damage_prose_base": dmg_prose_base,
            "damage_prose_gated": dmg_prose_gated,
            "damage_prose_nll_delta": round(dmg_prose_gated["nll"] - dmg_prose_base["nll"], 6),
            "gated_eval1_check": gated_e1,
            "seconds": round(time.time() - t0, 1),
        }
        log(
            f"  Step 2 done: think_F1@0.5={val_metrics.get('f1')} @0.75={val_metrics_t75.get('f1')} "
            f"comb_F1@0.5={comb_val_metrics.get('f1')} best_thr={best_thr} "
            f"py_delta@0.5={res['damage_python_nll_delta']} prose_delta={res['damage_prose_nll_delta']}"
        )
        self.store.set("step2_gate_probe", res)
        cleanup_cuda()

    # ------------------------------------------------------------------ Helpers
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
    ) -> Tuple[Optional[Decimal], str]:
        suffix = "\n</think>\nTherefore, the final answer is \\boxed{"
        suf_ids = self.tok.encode(suffix, add_special_tokens=False)
        cont_prompt = p_ids + gen_ids + suf_ids
        cont_ids, _ = generate_with_inline_kernel(
            self.llm, self.tok, self.eq_set, cont_prompt,
            max_new_tokens=max_cont_tokens, eos_token_id=self.eos_id, use_kernel=False,
        )
        cont_txt = self.tok.decode(cont_ids, skip_special_tokens=True)
        m = re.match(r"\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", cont_txt)
        if m:
            val = _parse_decimal_str(m.group(1))
            if val is not None:
                return val, cont_txt
        return extract_final_number(suffix + cont_txt), cont_txt

    # ------------------------------------------------------------------ Step 3
    def step3_eval2_big_wp(self) -> None:
        assert self.probe is not None
        log("=== STEP 3: Eval 2 — Hard Multi-Step Big-Number Word Problems (+ - * / & decimals) ===")
        if self.store.done("step3_eval2_big_wp"):
            log("  step3_eval2_big_wp already done; skipping.")
            return

        if not self.store.done("step4_eval3_gsm8k") and self.traces_path.exists():
            self.traces_path.unlink()

        t0 = time.time()
        problems = make_hard_mixed_word_problems(self.args.eval2_n, seed=202608)
        base_ok = kern_ok = 0
        base_fc_ok = kern_fc_ok = 0
        agg_gate = {
            k: 0 for k in (
                "total_eq_seen", "valid_expr_eq", "probe_fires", "true_fires",
                "false_probe_fires", "missed_exprs", "kernel_tokens_emitted",
            )
        }
        all_lats: List[float] = []
        base_traces: List[Dict[str, Any]] = []
        kern_traces: List[Dict[str, Any]] = []
        base_tags_list: List[Dict[str, bool]] = []
        kern_tags_list: List[Dict[str, bool]] = []
        base_correct_list: List[bool] = []
        kern_correct_list: List[bool] = []
        base_gen_lens: List[int] = []
        kern_gen_lens: List[int] = []
        comparisons: List[Dict[str, Any]] = []

        for idx, prob in enumerate(problems):
            q_text = prob["question"]
            p_str = format_r1_prompt(self.tok, q_text, direct_eq_prompt=self.args.direct_eq_prompt)
            p_ids = self.tok.encode(p_str, add_special_tokens=False)
            gt = prob["answer"]

            b_ids, _ = generate_with_inline_kernel(
                self.llm, self.tok, self.eq_set, p_ids,
                max_new_tokens=self.args.max_new_tokens, eos_token_id=self.eos_id, use_kernel=False,
            )
            b_txt = self.tok.decode(b_ids, skip_special_tokens=True)
            b_pred = extract_final_number(b_txt)
            b_correct = answers_match(b_pred, gt)
            base_ok += int(b_correct)
            b_tags = diagnose_failure(q_text, b_txt, b_pred, len(b_ids), self.args.max_new_tokens)
            if b_tags["hit_token_limit"] and b_tags["truncated_think"]:
                b_fc_pred, _ = self._forced_close_continuation(p_ids, b_ids, max_cont_tokens=64)
            else:
                b_fc_pred = b_pred
            b_fc_correct = answers_match(b_fc_pred, gt)
            base_fc_ok += int(b_fc_correct)
            base_traces.append({"text": b_txt, "gen_ids": b_ids})
            base_tags_list.append(b_tags)
            base_correct_list.append(b_correct)
            base_gen_lens.append(len(b_ids))

            k_ids, k_stats = generate_with_inline_kernel(
                self.llm, self.tok, self.eq_set, p_ids,
                max_new_tokens=self.args.max_new_tokens, eos_token_id=self.eos_id,
                probe=self.probe, use_kernel=True,
            )
            k_txt = self.tok.decode(k_ids, skip_special_tokens=True)
            k_pred = extract_final_number(k_txt)
            k_correct = answers_match(k_pred, gt)
            kern_ok += int(k_correct)
            k_tags = diagnose_failure(q_text, k_txt, k_pred, len(k_ids), self.args.max_new_tokens)
            if k_tags["hit_token_limit"] and k_tags["truncated_think"]:
                k_fc_pred, _ = self._forced_close_continuation(p_ids, k_ids, max_cont_tokens=64)
            else:
                k_fc_pred = k_pred
            k_fc_correct = answers_match(k_fc_pred, gt)
            kern_fc_ok += int(k_fc_correct)
            kern_traces.append({"text": k_txt, "gen_ids": k_ids, "kernel_events": k_stats["kernel_events"]})
            kern_tags_list.append(k_tags)
            kern_correct_list.append(k_correct)
            kern_gen_lens.append(len(k_ids))
            all_lats.extend(k_stats["kernel_latencies_ms"])
            for k in agg_gate:
                agg_gate[k] += int(k_stats[k])

            self._append_trace_record({
                "eval": "eval2_big_wp",
                "idx": idx,
                "ops": prob["ops"],
                "question": q_text,
                "gold": gt,
                "base_pred": str(b_pred) if b_pred is not None else None,
                "kernel_pred": str(k_pred) if k_pred is not None else None,
                "base_correct": b_correct,
                "kernel_correct": k_correct,
                "base_forced_pred": str(b_fc_pred) if b_fc_pred is not None else None,
                "kernel_forced_pred": str(k_fc_pred) if k_fc_pred is not None else None,
                "base_forced_correct": b_fc_correct,
                "kernel_forced_correct": k_fc_correct,
                "base_n_gen_tokens": len(b_ids),
                "kernel_n_gen_tokens": len(k_ids),
                "base_failure_tags": b_tags,
                "kernel_failure_tags": k_tags,
                "base_text": b_txt,
                "kernel_text": k_txt,
                "kernel_events": k_stats["kernel_events"],
            })

            k_stats_clean = {k: k_stats[k] for k in agg_gate}
            if len(comparisons) < 5 and (k_correct and not b_correct):
                comparisons.append({
                    "id": prob["id"], "ops": prob["ops"], "question": q_text, "ground_truth": gt,
                    "base_pred": str(b_pred) if b_pred is not None else None, "base_correct": b_correct,
                    "base_trace": b_txt[:2400],
                    "kernel_pred": str(k_pred) if k_pred is not None else None, "kernel_correct": k_correct,
                    "kernel_trace": k_txt[:2400],
                    "gate_stats": k_stats_clean,
                })
            elif idx >= len(problems) - (5 - len(comparisons)) and len(comparisons) < 5:
                comparisons.append({
                    "id": prob["id"], "ops": prob["ops"], "question": q_text, "ground_truth": gt,
                    "base_pred": str(b_pred) if b_pred is not None else None, "base_correct": b_correct,
                    "base_trace": b_txt[:2400],
                    "kernel_pred": str(k_pred) if k_pred is not None else None, "kernel_correct": k_correct,
                    "kernel_trace": k_txt[:2400],
                    "gate_stats": k_stats_clean,
                })

            if (idx + 1) % max(1, len(problems) // 8) == 0 or (idx + 1) == len(problems):
                log(
                    f"    big_wp {idx + 1}/{len(problems)} | base={base_ok}/{idx + 1} (fc={base_fc_ok}) "
                    f"| kernel={kern_ok}/{idx + 1} (fc={kern_fc_ok}) "
                    f"| true_fires={agg_gate['true_fires']} false_fires={agg_gate['false_probe_fires']}"
                )

        base_grade = grade_thinking(base_traces, self.tok, self.args.max_new_tokens, is_kernel=False)
        kern_grade = grade_thinking(kern_traces, self.tok, self.args.max_new_tokens, is_kernel=True)
        base_fb = aggregate_failure_breakdown(base_tags_list, base_correct_list, base_gen_lens)
        kern_fb = aggregate_failure_breakdown(kern_tags_list, kern_correct_list, kern_gen_lens)

        n = len(problems)
        res = {
            "n": n,
            "direct_eq_prompt": self.args.direct_eq_prompt,
            "base_final_acc": round(base_ok / max(1, n), 4),
            "kernel_final_acc": round(kern_ok / max(1, n), 4),
            "base_correct": base_ok,
            "kernel_correct": kern_ok,
            "base_forced_close_acc": round(base_fc_ok / max(1, n), 4),
            "kernel_forced_close_acc": round(kern_fc_ok / max(1, n), 4),
            "base_forced_close_correct": base_fc_ok,
            "kernel_forced_close_correct": kern_fc_ok,
            "median_kernel_latency_ms": round(statistics.median(all_lats), 4) if all_lats else None,
            "base_thinking_grade": base_grade,
            "kernel_thinking_grade": kern_grade,
            "base_failure_breakdown": base_fb,
            "kernel_failure_breakdown": kern_fb,
            "gate_stats": agg_gate,
            "traces_file": str(self.traces_path),
            "side_by_side_comparisons": comparisons,
            "seconds": round(time.time() - t0, 1),
        }
        log(
            f"  Step 3 done: base={res['base_final_acc']:.4f} (fc={res['base_forced_close_acc']:.4f}) "
            f"vs kernel={res['kernel_final_acc']:.4f} (fc={res['kernel_forced_close_acc']:.4f}) "
            f"| med_lat={res['median_kernel_latency_ms']}ms | base_fb={base_fb} | kern_fb={kern_fb}"
        )
        self.store.set("step3_eval2_big_wp", res)
        cleanup_cuda()

    # ------------------------------------------------------------------ Step 4
    def step4_eval3_gsm8k(self) -> None:
        assert self.probe is not None
        log("=== STEP 4: Eval 3 — Standard GSM8K No-Regression & <think> Speedup Check ===")
        if self.store.done("step4_eval3_gsm8k"):
            log("  step4_eval3_gsm8k already done; skipping.")
            return

        t0 = time.time()
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        n = min(self.args.eval3_n, len(ds))
        base_ok = kern_ok = 0
        base_fc_ok = kern_fc_ok = 0
        agg_gate = {
            k: 0 for k in (
                "total_eq_seen", "valid_expr_eq", "probe_fires", "true_fires",
                "false_probe_fires", "missed_exprs", "kernel_tokens_emitted",
            )
        }
        all_lats: List[float] = []
        base_traces: List[Dict[str, Any]] = []
        kern_traces: List[Dict[str, Any]] = []
        base_tags_list: List[Dict[str, bool]] = []
        kern_tags_list: List[Dict[str, bool]] = []
        base_correct_list: List[bool] = []
        kern_correct_list: List[bool] = []
        base_gen_lens: List[int] = []
        kern_gen_lens: List[int] = []

        for i in range(n):
            q = ds[i]["question"]
            gt = parse_gsm8k_answer(ds[i]["answer"])
            p_str = format_r1_prompt(self.tok, q)
            p_ids = self.tok.encode(p_str, add_special_tokens=False)

            b_ids, _ = generate_with_inline_kernel(
                self.llm, self.tok, self.eq_set, p_ids,
                max_new_tokens=self.args.max_new_tokens, eos_token_id=self.eos_id, use_kernel=False,
            )
            b_txt = self.tok.decode(b_ids, skip_special_tokens=True)
            b_pred = extract_final_number(b_txt)
            b_correct = bool(gt is not None and answers_match(b_pred, gt))
            base_ok += int(b_correct)
            b_tags = diagnose_failure(q, b_txt, b_pred, len(b_ids), self.args.max_new_tokens)
            if b_tags["hit_token_limit"] and b_tags["truncated_think"]:
                b_fc_pred, _ = self._forced_close_continuation(p_ids, b_ids, max_cont_tokens=64)
            else:
                b_fc_pred = b_pred
            b_fc_correct = bool(gt is not None and answers_match(b_fc_pred, gt))
            base_fc_ok += int(b_fc_correct)
            base_traces.append({"text": b_txt, "gen_ids": b_ids})
            base_tags_list.append(b_tags)
            base_correct_list.append(b_correct)
            base_gen_lens.append(len(b_ids))

            k_ids, k_stats = generate_with_inline_kernel(
                self.llm, self.tok, self.eq_set, p_ids,
                max_new_tokens=self.args.max_new_tokens, eos_token_id=self.eos_id,
                probe=self.probe, use_kernel=True,
            )
            k_txt = self.tok.decode(k_ids, skip_special_tokens=True)
            k_pred = extract_final_number(k_txt)
            k_correct = bool(gt is not None and answers_match(k_pred, gt))
            kern_ok += int(k_correct)
            k_tags = diagnose_failure(q, k_txt, k_pred, len(k_ids), self.args.max_new_tokens)
            if k_tags["hit_token_limit"] and k_tags["truncated_think"]:
                k_fc_pred, _ = self._forced_close_continuation(p_ids, k_ids, max_cont_tokens=64)
            else:
                k_fc_pred = k_pred
            k_fc_correct = bool(gt is not None and answers_match(k_fc_pred, gt))
            kern_fc_ok += int(k_fc_correct)
            kern_traces.append({"text": k_txt, "gen_ids": k_ids, "kernel_events": k_stats["kernel_events"]})
            kern_tags_list.append(k_tags)
            kern_correct_list.append(k_correct)
            kern_gen_lens.append(len(k_ids))
            all_lats.extend(k_stats["kernel_latencies_ms"])
            for k in agg_gate:
                agg_gate[k] += int(k_stats[k])

            self._append_trace_record({
                "eval": "eval3_gsm8k",
                "idx": i,
                "question": q,
                "gold": gt,
                "base_pred": str(b_pred) if b_pred is not None else None,
                "kernel_pred": str(k_pred) if k_pred is not None else None,
                "base_correct": b_correct,
                "kernel_correct": k_correct,
                "base_forced_pred": str(b_fc_pred) if b_fc_pred is not None else None,
                "kernel_forced_pred": str(k_fc_pred) if k_fc_pred is not None else None,
                "base_forced_correct": b_fc_correct,
                "kernel_forced_correct": k_fc_correct,
                "base_n_gen_tokens": len(b_ids),
                "kernel_n_gen_tokens": len(k_ids),
                "base_failure_tags": b_tags,
                "kernel_failure_tags": k_tags,
                "base_text": b_txt,
                "kernel_text": k_txt,
                "kernel_events": k_stats["kernel_events"],
            })

            if (i + 1) % max(1, n // 8) == 0 or (i + 1) == n:
                log(
                    f"    gsm8k {i + 1}/{n} | base={base_ok}/{i + 1} (fc={base_fc_ok}) "
                    f"| kernel={kern_ok}/{i + 1} (fc={kern_fc_ok}) "
                    f"| true_fires={agg_gate['true_fires']} false_fires={agg_gate['false_probe_fires']}"
                )

        base_grade = grade_thinking(base_traces, self.tok, self.args.max_new_tokens, is_kernel=False)
        kern_grade = grade_thinking(kern_traces, self.tok, self.args.max_new_tokens, is_kernel=True)
        base_fb = aggregate_failure_breakdown(base_tags_list, base_correct_list, base_gen_lens)
        kern_fb = aggregate_failure_breakdown(kern_tags_list, kern_correct_list, kern_gen_lens)
        speedup = (
            round(base_grade["mean_think_tokens"] / kern_grade["mean_think_tokens"], 3)
            if kern_grade["mean_think_tokens"] > 0
            else None
        )

        res = {
            "n": n,
            "base_acc": round(base_ok / max(1, n), 4),
            "kernel_acc": round(kern_ok / max(1, n), 4),
            "base_correct": base_ok,
            "kernel_correct": kern_ok,
            "acc_delta": round((kern_ok - base_ok) / max(1, n), 4),
            "base_forced_close_acc": round(base_fc_ok / max(1, n), 4),
            "kernel_forced_close_acc": round(kern_fc_ok / max(1, n), 4),
            "base_forced_close_correct": base_fc_ok,
            "kernel_forced_close_correct": kern_fc_ok,
            "think_token_speedup_ratio": speedup,
            "median_kernel_latency_ms": round(statistics.median(all_lats), 4) if all_lats else None,
            "base_thinking_grade": base_grade,
            "kernel_thinking_grade": kern_grade,
            "base_failure_breakdown": base_fb,
            "kernel_failure_breakdown": kern_fb,
            "gate_stats": agg_gate,
            "traces_file": str(self.traces_path),
            "seconds": round(time.time() - t0, 1),
        }
        log(
            f"  Step 4 done: GSM8K base={res['base_acc']:.4f} (fc={res['base_forced_close_acc']:.4f}) "
            f"vs kernel={res['kernel_acc']:.4f} (fc={res['kernel_forced_close_acc']:.4f}, delta={res['acc_delta']:+.4f}) "
            f"| speedup={speedup}x | med_lat={res['median_kernel_latency_ms']}ms"
        )
        self.store.set("step4_eval3_gsm8k", res)
        cleanup_cuda()

    def run_all(self) -> None:
        self.step1_eval1_ungated()
        self.step2_harvest_and_train_gate()
        self.step3_eval2_big_wp()
        if self.args.skip_step6:
            log("=== STEP 4 (GSM8K): Skipped (--skip-step6) ===")
        else:
            self.step4_eval3_gsm8k()
        log("=== ALL STEPS COMPLETE ===")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt-dir", default=str(HERE / "runs" / "ckpts"))
    ap.add_argument("--eval1-n", type=int, default=32)
    ap.add_argument("--eval2-n", type=int, default=32)
    ap.add_argument("--eval3-n", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--direct-eq-prompt", action="store_true")
    ap.add_argument("--skip-step6", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    if not args.tag:
        args.tag = default_tag_for_model(args.model)
    if args.smoke:
        args.eval1_n = min(args.eval1_n, 8)
        args.eval2_n = min(args.eval2_n, 8)
        args.eval3_n = min(args.eval3_n, 8)
        args.max_new_tokens = min(args.max_new_tokens, 512)
        if args.out is None:
            args.out = str(HERE / "runs" / f"math_kernel_{args.tag}_smoke_results.json")
    else:
        if args.out is None:
            args.out = str(HERE / "runs" / f"math_kernel_{args.tag}_results.json")
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
