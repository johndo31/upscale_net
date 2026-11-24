"""04d — Natural-notation addition and grafts that make Qwen's own layers take part.

Follow-up to real_graft_exp.py (imports its SparseOrgan, socket calibration, bracket circuit and
bracket task). Two criticisms of 04b are addressed:

  1. The addition prompt was built for the circuit (LSB-first digit pairs).
  2. Qwen's layers did nothing: the circuit read embeddings and wrote just before the output layer.

PART F  addition in normal notation  ("Add: 4821+3957=", answer "08778")
  * Both operands have the SAME number of digits n (leading zeros allowed, e.g. "0042+0917=").
    The answer always has n+1 digits; its first digit is the carry-out (0 or 1). So the
    "first answer digit" metric is in practice a 0/1 decision scored 10-way, and the base model is
    penalised by the leading-zero format (a lenient metric that parses the integer is also reported).
  * Written circuit (no training of the circuit; only the write-out matrix W_out is trained):
    2 layers, 3 heads, ReLU MLP, causal attention with relative-position bias, no LayerNorm.
    It computes every answer digit: at "=" it outputs the carry-out, at the k-th generated digit it
    outputs (a_k + b_k + carry_into_column_k) mod 10, and an "answer region" channel that gates the
    write-out on "=" and on digits after "=". Generation is real greedy decoding over the full vocab.
  * ADDED POSITIONAL INPUT (not learned, not read from Qwen): for every token, fixed code counts the
    index of the token within its run of digit tokens (1, 2, 3, ...; 0 for non-digits; it resets on
    every non-digit). This is the "abacus"-style input (McLeish et al. 2024). It is given to the
    circuit as a one-hot vector AND a thermometer vector over 1..max_idx. For answer digits the same
    counter is the answer-position index ("=" has index 0). Everything else (digit identity, "+",
    "=", other) goes through the closed-form ridge read-in from Qwen's token embeddings plus clamp.
    Column alignment through the index only works because both operands have the same length.
  * Arms: base, LoRA trained on the first answer digit, LoRA trained on the full answer (teacher
    forcing), trained circuit (same architecture + same index input, random init, trained with
    W_out), written circuit (only W_out trained, teacher-forced on all answer positions).
    The trained-circuit arm gets its write-out gate from fixed code (is "=" or a digit after "="),
    i.e. it is given the gate for free; the written arm computes the gate itself.

PART G1  Qwen uses the result (bracket circuit from 04b, prompt as gen_real_parens)
  The circuit's write-out is added to the residual stream at the OUTPUT of decoder layer k (k layers
  done), gated by the read-in "=" channel, through a forward hook. Only W_out (2 -> d_llm) is
  trained, with the loss at the final output, gradients flowing through the frozen upper layers.
    plain    : Yes/No at "=" (k default = depth/3).
    composed : after "=" the prompt continues with an instruction that needs the bracket result plus
               world knowledge; scored by the logits of the first tokens of the two candidate
               answers at the last position. Templates are filtered by an oracle check (base model
               told the result in words), split into train and held-out with disjoint facts.
               Baselines: base, LoRA on train templates (same steps), oracle text. k is swept over
               ~1/6, 1/3, 2/3 of depth. Text loss with the hook active is reported on ENGLISH_PROBES
               and on 30+ lines that contain "=" (overall and after the first "=").

PART G2  Qwen reads the input
  Brackets written as words ("open close ..."; second form "left right ..."). The circuit read-in is
  a ridge regression on Qwen's hidden states at layer j (default depth/6), fitted on tokens whose
  symbol class is known from the surface form (no task labels), on form A only; tested on A and B.

All results are merged into --out after every finished cell; reruns skip finished cells, failed
cells are stored as {"error": ...} and retried on the next run.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from real_graft_exp import (  # noqa: E402  (also puts 03-transformer-writer on sys.path)
    ENGLISH_PROBES,
    SparseOrgan,
    build_real_parens_organ,
    calibrate_port_socket,
    gen_real_parens,
)

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:7.0f}s] {msg}", flush=True)


# =============================================================================
# PART F CIRCUIT: natural-notation addition with an abacus index input
# =============================================================================

# Read-in channels (from Qwen embeddings via ridge + clamp)
CH_DIG = 0          # 0..9 one-hot digit
CH_PLUS = 10
CH_EQ = 11
CH_OTHER = 12
CH_CONST = 13
N_READ = 14
# Scratch channels written by the circuit
CH_SP = 14          # a "+" occurred strictly before this token
CH_SE = 15          # a "=" occurred strictly before this token
CH_AV = 16          # value fetched from same-index digits (b_k: a_k ; answer k: (a_k+b_k)/2 ; "=": 0)
CH_G = 17           # at b_k: column k generates a carry (a_k+b_k >= 10)
CH_K = 18           # at b_k: column k kills a carry   (a_k+b_k <= 8)
CH_C = 19           # carry into the column this answer position computes
CH_OUT = 20         # 20..29 one-hot output digit
CH_GATE = 30        # answer region: "=" or a digit after "="
CH_IDX = 31         # index input: one-hot 1..M, then thermometer 1..M
N_OUT = 11          # 10 digit logits + gate


def add_bus_dim(max_idx: int) -> int:
    return CH_IDX + 2 * max_idx


def build_natural_add_organ(max_idx: int, device: torch.device) -> SparseOrgan:
    """Written 2-layer, 3-head circuit for MSB-first addition with equal-length operands.

    Token stream: [other...] a_1..a_n "+" b_1..b_n "=" s_0 s_1 ... s_n   (column 1 = most significant)
    Index input: counter of the token within its digit run (a_k, b_k and answer digit s_{k-1} all
    have index k; "=" and every non-digit have index 0).

    Layer 0
      H0  seen-"+"  : const query, key = "+" channel (score 40)    -> CH_SP
      H1  seen-"="  : const query, key = "=" channel (score 40)    -> CH_SE
      H2  same-index fetch: score 60 on past digits with the same index, 30 on "other" tokens (a
          sink holding value 0), 0 elsewhere. Value = digit value.  -> CH_AV
          at b_k it finds a_k; at answer index k it finds a_k and b_k (mean); at "=" only the sink.
      MLP gated on "is b digit" (digit AND seen "+" AND NOT seen "="): S = b_k + a_k,
          G = [S >= 10] -> CH_G, K = [S <= 8] -> CH_K  (propagate = neither).
    Layer 1
      H0  carry lookup: score = B*(G+K)_j - B*[index_j <= index_query] + alpha*distance, alpha>0.
          Among non-propagate columns j that are LOWER than the query's column (j > k), the bias
          prefers the farthest token, i.e. the smallest j = the nearest lower non-propagate column.
          Value = G. If no such column exists, attention lands on position 0 (value 0).  -> CH_C
      MLP gated on answer region: X = 2*AV + C, ten-way one-hot of X mod 10 by triangular bumps
          -> CH_OUT..+9; gate neurons -> CH_GATE.
    """
    M = max_idx
    d_head = M + 1
    s = math.sqrt(d_head)  # SparseOrgan divides scores by sqrt(d_head)
    organ = SparseOrgan(d_model=add_bus_dim(M), d_mlp=64, n_out=N_OUT, n_layers=2, n_heads=3, d_head=d_head).to(device)
    one = lambda m: CH_IDX + m - 1          # noqa: E731
    thermo = lambda m: CH_IDX + M + m - 1   # noqa: E731
    with torch.no_grad():
        # ---- L0.H0 / L0.H1: seen "+" / seen "=" ----
        for h, (src, dst) in enumerate(((CH_PLUS, CH_SP), (CH_EQ, CH_SE))):
            organ.Q[0, h, CH_CONST, 0] = 40.0 * s
            organ.K[0, h, src, 0] = 1.0
            organ.V[0, h, src, 0] = 1.0
            organ.O[0, h, 0, dst] = 1.0
        # ---- L0.H2: same-index fetch with an "other"-token sink ----
        beta = 60.0
        for m in range(1, M + 1):
            organ.Q[0, 2, one(m), m - 1] = beta * s
            organ.K[0, 2, one(m), m - 1] = 1.0
        organ.Q[0, 2, CH_CONST, M] = 0.5 * beta * s
        organ.K[0, 2, CH_OTHER, M] = 1.0
        for d in range(1, 10):
            organ.V[0, 2, CH_DIG + d, 0] = float(d)
        organ.O[0, 2, 0, CH_AV] = 1.0
        # ---- L0.MLP: generate / kill at b digits ----
        def sum_neuron(n: int, sign: float, bias: float) -> None:
            for d in range(10):
                organ.W1[0, CH_DIG + d, n] = sign * float(d) + 50.0   # value of b_k + is_digit gate
            organ.W1[0, CH_AV, n] = sign                               # a_k
            organ.W1[0, CH_SP, n] = 50.0
            organ.W1[0, CH_SE, n] = -50.0
            organ.b1[0, n] = bias - 100.0
        sum_neuron(0, +1.0, -9.25); organ.W2[0, 0, CH_G] = +2.0
        sum_neuron(1, +1.0, -9.75); organ.W2[0, 1, CH_G] = -2.0
        sum_neuron(2, -1.0, +8.75); organ.W2[0, 2, CH_K] = +2.0
        sum_neuron(3, -1.0, +8.25); organ.W2[0, 3, CH_K] = -2.0
        # ---- L1.H0: carry lookup (positive distance bias = prefer the farthest eligible column) ----
        alpha = 20.0
        big = alpha * 4096.0 + 200.0   # dominates alpha*distance for sequences up to 4096 tokens
        organ.alibi[1, 0] = alpha
        organ.Q[1, 0, CH_CONST, 0] = big * s
        organ.K[1, 0, CH_G, 0] = 1.0
        organ.K[1, 0, CH_K, 0] = 1.0
        for m in range(1, M + 1):
            organ.Q[1, 0, thermo(m), m] = -big * s   # query thermometer: m <= own index
            organ.K[1, 0, one(m), m] = 1.0           # key one-hot: own index == m
        organ.V[1, 0, CH_G, 0] = 1.0
        organ.O[1, 0, 0, CH_C] = 1.0
        # ---- L1.MLP: (2*AV + C) mod 10 at answer positions, plus the gate ----
        n = 0
        for t_val in range(20):
            for off, coeff in ((-0.5, 2.0), (0.0, -4.0), (0.5, 2.0)):
                organ.W1[1, CH_AV, n] = 2.0
                organ.W1[1, CH_C, n] = 1.0
                # gate: 2*is_eq + seen_eq + is_digit must equal 2
                organ.W1[1, CH_EQ, n] = 100.0
                organ.W1[1, CH_SE, n] = 50.0
                for d in range(10):
                    organ.W1[1, CH_DIG + d, n] = 50.0
                organ.b1[1, n] = -(float(t_val) + off) - 100.0
                organ.W2[1, n, CH_OUT + (t_val % 10)] = coeff
                n += 1
        # gate neuron 60: answer digit = relu(seen_eq + is_digit - 1); neuron 61: relu(is_eq)
        organ.W1[1, CH_SE, 60] = 1.0
        for d in range(10):
            organ.W1[1, CH_DIG + d, 60] = 1.0
        organ.b1[1, 60] = -1.0
        organ.W2[1, 60, CH_GATE] = 1.0
        organ.W1[1, CH_EQ, 61] = 1.0
        organ.W2[1, 61, CH_GATE] = 1.0
        # ---- readout ----
        for d in range(10):
            organ.OUT_W[CH_OUT + d, d] = 6.0
        organ.OUT_W[CH_GATE, 10] = 1.0
    return organ


def digit_counters(is_digit: torch.Tensor) -> torch.Tensor:
    """is_digit: bool [B, T] -> index of each token within its digit run (1-based; 0 for non-digits)."""
    d = is_digit.long()
    cs = torch.cumsum(d, dim=1)
    reset = torch.where(is_digit, torch.zeros_like(cs), cs)
    last = torch.cummax(reset, dim=1).values
    return (cs - last) * d


def index_features(counters: torch.Tensor, max_idx: int) -> torch.Tensor:
    """Fixed, non-learned abacus input: one-hot and thermometer of the digit index (clamped to max_idx)."""
    c = counters.clamp(max=max_idx).unsqueeze(-1)
    m = torch.arange(1, max_idx + 1, device=counters.device)
    return torch.cat([(c == m).float(), (m <= c).float()], dim=-1)


def gen_add_normal(rng: random.Random, n: int, cascade: bool = False) -> Tuple[str, List[int]]:
    """'Add: 4821+3957=' with both operands exactly n digits (leading zeros allowed).
    Returns (prompt, answer digits): n+1 digits MSB first, the first one is the carry-out (0/1).
    cascade=True: every column except the lowest sums to 9 and the lowest column generates a
    carry or not (50/50), so the carry has to travel through all n columns."""
    if cascade:
        a = [rng.randint(0, 9) for _ in range(n - 1)]
        b = [9 - x for x in a]
        if rng.random() < 0.5:
            a0 = rng.randint(1, 9); b0 = rng.randint(10 - a0, 9)
        else:
            a0 = rng.randint(0, 8); b0 = rng.randint(0, 8 - a0)
        a.append(a0); b.append(b0)
    else:
        a = [rng.randint(0, 9) for _ in range(n)]
        b = [rng.randint(0, 9) for _ in range(n)]
    sa, sb = "".join(map(str, a)), "".join(map(str, b))
    ans = str(int(sa) + int(sb)).zfill(n + 1)
    assert len(ans) == n + 1
    return f"Add: {sa}+{sb}=", [int(c) for c in ans]


def _symbolic_bus(rows: List[List[str]], max_idx: int) -> torch.Tensor:
    """Exact read-in for the self-test: tokens are '0'..'9', '+', '=', or anything else (= other)."""
    B, T = len(rows), len(rows[0])
    x = torch.zeros(B, T, add_bus_dim(max_idx))
    isd = torch.zeros(B, T, dtype=torch.bool)
    for i, row in enumerate(rows):
        for t, tok in enumerate(row):
            if tok in "0123456789" and len(tok) == 1:
                x[i, t, CH_DIG + int(tok)] = 1.0; isd[i, t] = True
            elif tok == "+":
                x[i, t, CH_PLUS] = 1.0
            elif tok == "=":
                x[i, t, CH_EQ] = 1.0
            else:
                x[i, t, CH_OTHER] = 1.0
    x[:, :, CH_CONST] = 1.0
    x[:, :, CH_IDX:] = index_features(digit_counters(isd), max_idx)
    return x


def verify_natural_add_cpu(max_idx: int, n_cases: int = 128, seed: int = 0) -> Dict[str, str]:
    """Mandatory self-test: teacher-forced all-digit exactness, gate exactness and a free-running
    autoregressive generation with the circuit alone. Raises on any error."""
    torch.manual_seed(seed)
    organ = build_natural_add_organ(max_idx, torch.device("cpu"))
    organ.eval()
    rng = random.Random(seed)
    out = {}
    prefix = ["Add", ":", " "]
    for name, n, casc in [("n=6", 6, False), ("n=32", 32, False), ("n=64", 64, False),
                          ("carry n=64", 64, True), ("n=128", 128, False), ("carry n=128", 128, True)]:
        rows, answers = [], []
        for _ in range(n_cases):
            p, ans = gen_add_normal(rng, n, casc)
            body = p[len("Add: "):]
            rows.append(prefix + list(body) + [str(d) for d in ans])
            answers.append(ans)
        x = _symbolic_bus(rows, max_idx)
        with torch.no_grad():
            o = organ.forward_states(x)
        P = len(prefix) + 2 * n + 1          # index of "="
        pred = o[:, P:P + n + 1, :10].argmax(-1)
        tgt = torch.tensor(answers)
        exact = int((pred == tgt).all(dim=1).sum())
        dig = float((pred == tgt).float().mean())
        gate = o[:, :, 10].clamp(0, 1)
        gate_ok = bool((gate[:, :P] == 0).all() and (gate[:, P:] == 1).all())
        # margin of the winning logit (written logits are 6 vs 0)
        top2 = o[:, P:P + n + 1, :10].topk(2, dim=-1).values
        margin = float((top2[..., 0] - top2[..., 1]).min())
        # free-running generation on the first 8 cases
        gen_ok = 0
        for i in range(min(8, n_cases)):
            seq = rows[i][:P + 1]
            for _ in range(n + 1):
                with torch.no_grad():
                    lo = organ.forward_states(_symbolic_bus([seq], max_idx))[0, -1, :10]
                seq = seq + [str(int(lo.argmax()))]
            gen_ok += int([int(c) for c in seq[P + 1:]] == answers[i])
        n_gen = min(8, n_cases)
        line = (f"{name:12s} exact={exact}/{n_cases} per_digit={dig:.4f} gate_exact={gate_ok} "
                f"min_margin={margin:.3f} autoregressive={gen_ok}/{n_gen}")
        log(f"  [CPU self-test natural_add | nnz={organ.nnz()}] {line}")
        out[name] = line
        assert exact == n_cases and gate_ok and gen_ok == n_gen, f"natural_add self-test failed: {line}"
    out["nnz"] = str(organ.nnz())
    return out


def verify_parens_cpu(n_cases: int = 64) -> None:
    organ = build_real_parens_organ(torch.device("cpu"))
    rng = random.Random(1)
    for L in (6, 32, 64, 128):
        ok = 0
        for _ in range(n_cases):
            p, y = gen_real_parens(rng, L)
            toks = ["P0", "P1", "P2"] + [c for c in p.split(": ")[1] if c in "()="]
            x = torch.zeros(1, len(toks), 16)
            x[0, :, 4] = 1.0
            for i, ch in enumerate(toks):
                x[0, i, {"(": 0, ")": 1, "=": 2}.get(ch, 3)] = 1.0
            with torch.no_grad():
                ok += int(int(organ.forward_states(x)[0, -1].argmax()) == y)
        log(f"  [CPU self-test parens] L={L:3d} acc={ok}/{n_cases}")
        assert ok == n_cases


# =============================================================================
# SHARED LLM HELPERS
# =============================================================================

def dual_rail_clamp(z: torch.Tensor, dz: float = 0.3) -> torch.Tensor:
    """Same clamp as RealGraftedLLM._project_in: <=0.3 -> 0, >=0.85 -> 1."""
    return (F.relu(z - dz) / (0.85 - dz)).clamp(0.0, 1.0)


def decoder_layers(llm: nn.Module):
    return llm.model.layers


def add_residual_hook(llm: nn.Module, k: int, delta: torch.Tensor):
    """Add delta [B,T,d] to the residual stream at the output of decoder layer k (k layers done, 1..L)."""
    layer = decoder_layers(llm)[k - 1]

    def hook(_mod, _inp, out):
        if isinstance(out, tuple):
            return (out[0] + delta.to(out[0].dtype),) + tuple(out[1:])
        return out + delta.to(out.dtype)

    return layer.register_forward_hook(hook)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def first_tok(tokenizer, text: str) -> int:
    return tokenizer.encode(text, add_special_tokens=False)[0]


def text_loss(logits_fn, tokenizer, texts: Sequence[str], device, eq_ids: Sequence[int]) -> Dict[str, float]:
    """Mean NLL per predicted token, overall and for targets after the first '=' token."""
    eq = set(eq_ids)
    tot, n_tok, tot_a, n_a = 0.0, 0, 0.0, 0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            logits = logits_fn(ids).float()
        nll = F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="none")
        tot += float(nll.sum()); n_tok += int(nll.numel())
        lst = ids[0].tolist()
        first_eq = next((i for i, t in enumerate(lst) if t in eq), None)
        if first_eq is not None:
            after = nll[first_eq:]          # targets at positions first_eq+1 ..
            tot_a += float(after.sum()); n_a += int(after.numel())
    return {"nll": round(tot / max(1, n_tok), 5), "nll_after_eq": round(tot_a / max(1, n_a), 5) if n_a else None,
            "n_tokens": n_tok, "n_tokens_after_eq": n_a}


def eq_text_lines() -> List[str]:
    """30+ ordinary lines that contain '=' (maths and code)."""
    lines = [
        "x = 3 + 4",
        "total = price * quantity",
        "for i in range(10): s = s + i",
        "if a == b: c = 0",
        "E = mc^2 is the most famous equation in physics.",
        "speed = distance / time",
        "The area of a circle is A = pi * r^2.",
        "let count = items.length;",
        "const name = \"world\";",
        "y = 2 * x + 1",
        "result = sum(values) / len(values)",
        "In Python, a = [1, 2, 3] creates a list.",
        "We set the learning rate to lr = 0.001 for all runs.",
        "f(x) = x^2 - 4x + 4 has a double root at x = 2.",
        "int main() { int n = 5; return n; }",
        "df = pd.read_csv(\"data.csv\")",
        "The force is F = m * a, where m is mass.",
        "color = 'blue' if sky else 'grey'",
        "Hence the probability is p = 1/6 for a fair die.",
        "model = Model(hidden=256, layers=4)",
    ]
    rng = random.Random(4242)
    for _ in range(16):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        op = rng.choice(["+", "-", "*"])
        val = a + b if op == "+" else (a - b if op == "-" else a * b)
        lines.append(f"{a} {op} {b} = {val}")
    return lines


class Results:
    """Results JSON merged after every cell; finished cells are skipped on rerun."""

    def __init__(self, path: str, config: dict):
        self.path = Path(path)
        self.data = {"config": config, "meta": {}, "results": {}}
        if self.path.exists():
            try:
                old = json.loads(self.path.read_text())
                self.data["meta"] = old.get("meta", {})
                self.data["results"] = old.get("results", {})
                log(f"Resuming from {self.path} ({sum(len(v) for v in self.data['results'].values())} cells)")
            except Exception as e:  # pragma: no cover
                log(f"Could not read {self.path}: {e}; starting fresh")
        self.data["config"] = config

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["elapsed_s_this_run"] = round(time.time() - T0, 1)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        os.replace(tmp, self.path)

    def done(self, part: str, key: str) -> bool:
        r = self.data["results"].get(part, {}).get(key)
        return isinstance(r, dict) and "error" not in r

    def get(self, part: str, key: str):
        return self.data["results"].get(part, {}).get(key)

    def run(self, part: str, key: str, fn: Callable[[], dict]) -> Optional[dict]:
        if self.done(part, key):
            log(f"skip {part}/{key} (done)")
            return self.get(part, key)
        log(f"--- cell {part}/{key} ---")
        t = time.time()
        try:
            r = fn()
            r["cell_seconds"] = round(time.time() - t, 1)
        except Exception as e:
            r = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:],
                 "cell_seconds": round(time.time() - t, 1)}
            log(f"ERROR in {part}/{key}: {r['error']}")
        cleanup_cuda()
        self.data["results"].setdefault(part, {})[key] = r
        self.save()
        log(f"done {part}/{key} in {time.time() - t:.0f}s: {short(r)}")
        return r


def short(r: dict, limit: int = 600) -> str:
    def fmt(v):
        if isinstance(v, float):
            return round(v, 4)
        if isinstance(v, dict):
            return {k: fmt(x) for k, x in v.items() if k != "traceback"}
        return v
    s = json.dumps({k: fmt(v) for k, v in r.items() if k != "traceback"})
    return s if len(s) <= limit else s[:limit] + "..."


class StepTimer:
    def __init__(self, name: str, total: int):
        self.name, self.total, self.t = name, total, time.time()

    def tick(self, step: int, loss: float) -> None:
        s = step + 1
        if s in (1, 5, 20) or s % max(1, self.total // 10) == 0 or s == self.total:
            log(f"    {self.name} step {s}/{self.total} loss={loss:.4f} ({(time.time() - self.t) / s:.3f} s/step)")


def lora_session(llm: nn.Module, r: int = 16):
    """Attach LoRA; returns (params, detach_fn). detach_fn always restores the plain model."""
    from writer_loop import attach_lora, unwrap_and_reset_lora
    params = attach_lora(llm, r=r, alpha=32.0)
    llm.gradient_checkpointing_enable()
    llm.enable_input_require_grads()

    def detach():
        try:
            llm.gradient_checkpointing_disable()
        except Exception:
            pass
        try:
            llm.disable_input_require_grads()
        except Exception:
            pass
        unwrap_and_reset_lora(llm, merge=False)
        llm.eval()
        cleanup_cuda()

    return params, detach


# =============================================================================
# PART F HARNESS
# =============================================================================

class NaturalAddGraft(nn.Module):
    """Frozen LLM + read-in (ridge on embeddings, clamp) + abacus index input + circuit + W_out at
    the last hidden state of every gated position."""

    def __init__(self, base_llm, organ: SparseOrgan, cal_in, max_idx: int, digit_ids: List[int],
                 eq_id: int, gate_mode: str):
        super().__init__()
        self.base_llm = base_llm
        self.organ = organ
        self.max_idx = max_idx
        self.gate_mode = gate_mode
        self.register_buffer("W_in", cal_in[0].float().clone(), persistent=False)
        self.register_buffer("b_in", cal_in[1].float().clone(), persistent=False)
        self.register_buffer("digit_ids", torch.tensor(digit_ids, dtype=torch.long), persistent=False)
        self.eq_id = eq_id
        d_llm = base_llm.config.hidden_size
        self.W_out = nn.Linear(10, d_llm, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.W_out.weight)

    def bus(self, ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h0 = self.base_llm.get_input_embeddings()(ids).float()
        z = dual_rail_clamp(h0 @ self.W_in.T + self.b_in)
        isd = torch.isin(ids, self.digit_ids.to(ids.device))
        idx = index_features(digit_counters(isd), self.max_idx)
        B, T = ids.shape
        scratch = torch.zeros(B, T, CH_IDX - N_READ, device=ids.device)
        return torch.cat([z, scratch, idx], dim=-1)

    def code_gate(self, ids: torch.Tensor) -> torch.Tensor:
        is_eq = ids == self.eq_id
        seen = (torch.cumsum(is_eq.long(), 1) - is_eq.long()) > 0
        isd = torch.isin(ids, self.digit_ids.to(ids.device))
        return (is_eq | (seen & isd)).float()

    def delta(self, ids: torch.Tensor) -> torch.Tensor:
        o = self.organ.forward_states(self.bus(ids))
        if self.gate_mode == "circuit":
            gate = o[..., 10].clamp(0.0, 1.0)
        else:
            gate = self.code_gate(ids)
        return gate.unsqueeze(-1) * self.W_out(torch.tanh(o[..., :10] / 10.0) * 10.0)

    def logits_from(self, ids: torch.Tensor, pos_from: int) -> torch.Tensor:
        """Teacher-forced logits for positions pos_from..T-1: [B, T-pos_from, V]."""
        with torch.no_grad():
            h = self.base_llm.model(input_ids=ids, use_cache=False).last_hidden_state[:, pos_from:].float()
        h = h + self.delta(ids)[:, pos_from:]
        lm = self.base_llm.get_output_embeddings()
        return lm(h.to(lm.weight.dtype)).float()

    def step_fn(self, full, new, past):
        out = self.base_llm.model(input_ids=new, past_key_values=past, use_cache=True)
        h = out.last_hidden_state[:, -1].float() + self.delta(full)[:, -1]
        lm = self.base_llm.get_output_embeddings()
        return lm(h.to(lm.weight.dtype)).float(), out.past_key_values

    def all_logits(self, ids: torch.Tensor) -> torch.Tensor:
        return self.logits_from(ids, 0)


def plain_step_fn(llm):
    def fn(full, new, past):
        out = llm(input_ids=new, past_key_values=past, use_cache=True)
        return out.logits[:, -1].float(), out.past_key_values
    return fn


@torch.no_grad()
def greedy_generate(step_fn, prompt: torch.Tensor, n_new: int) -> Tuple[torch.Tensor, torch.Tensor]:
    full, new, past, outs, first = prompt, prompt, None, [], None
    for i in range(n_new):
        logits, past = step_fn(full, new, past)
        if i == 0:
            first = logits
        nxt = logits.argmax(-1)
        outs.append(nxt)
        full = torch.cat([full, nxt[:, None]], 1)
        new = nxt[:, None]
    return torch.stack(outs, 1), first


class PartF:
    def __init__(self, ctx):
        self.ctx = ctx
        a = ctx.args
        tok, dev = ctx.tokenizer, ctx.device
        self.digit_ids = [first_tok(tok, str(d)) for d in range(10)]
        self.plus_id = first_tok(tok, "+")
        self.eq_id = first_tok(tok, "=")
        self.prefix_ids = tok.encode("Add: ", add_special_tokens=False)
        # tokenization check: every digit, '+' and '=' must be its own token
        p, ans = gen_add_normal(random.Random(0), 6)
        ids = tok.encode(p + "".join(map(str, ans)), add_special_tokens=False)
        expect = self.prefix_ids + [self.digit_ids[int(c)] if c.isdigit() else (self.plus_id if c == "+" else self.eq_id)
                                    for c in (p[len("Add: "):] + "".join(map(str, ans)))]
        pieces = [tok.decode([i]) for i in ids]
        log(f"F tokenization: {pieces}")
        if ids != expect:
            raise RuntimeError(f"Prompt is not tokenized one symbol per token: {pieces}")
        ctx.res.data["meta"]["f_tokenization"] = pieces
        self.cal = calibrate_port_socket(ctx.llm, N_READ, [[d] for d in self.digit_ids] + [[self.plus_id], [self.eq_id], self.prefix_ids],
                                         other_ch=CH_OTHER, const_ch=CH_CONST, device=dev)
        self.check_readin()
        # shared test sets
        n_test = a.f_test_n
        self.splits = {}
        specs = [("id", [4, 5, 6, 7, 8], False)] + [(f"L{n}", [n], False) for n in a.f_lens] + [(f"carry{a.f_carry_len}", [a.f_carry_len], True)]
        for si, (name, lens, casc) in enumerate(specs):
            rng = random.Random(777 + 31 * si)
            groups: Dict[int, Tuple[list, list]] = {}
            for i in range(n_test):
                n = lens[i % len(lens)]
                prompt, ans = gen_add_normal(rng, n, casc)
                groups.setdefault(n, ([], []))
                groups[n][0].append(tok.encode(prompt, add_special_tokens=False))
                groups[n][1].append(ans)
            self.splits[name] = [(n, torch.tensor(x, dtype=torch.long, device=dev), torch.tensor(y, dtype=torch.long, device=dev))
                                 for n, (x, y) in sorted(groups.items())]
        self.digit_t = torch.tensor(self.digit_ids, device=dev)

    def check_readin(self) -> None:
        """Log whether the clamped read-in gives exact one-hot classes on a real prompt."""
        ctx = self.ctx
        p, ans = gen_add_normal(random.Random(1), 8)
        ids = torch.tensor([ctx.tokenizer.encode(p + "".join(map(str, ans)), add_special_tokens=False)], device=ctx.device)
        with torch.no_grad():
            z = dual_rail_clamp(ctx.llm.get_input_embeddings()(ids).float() @ self.cal[0].T + self.cal[1])[0]
        exp = torch.zeros_like(z)
        for t, i in enumerate(ids[0].tolist()):
            if i in self.digit_ids:
                exp[t, self.digit_ids.index(i)] = 1
            elif i == self.plus_id:
                exp[t, CH_PLUS] = 1
            elif i == self.eq_id:
                exp[t, CH_EQ] = 1
            else:
                exp[t, CH_OTHER] = 1
        exp[:, CH_CONST] = 1
        err = float((z - exp).abs().max())
        log(f"F read-in max abs error vs exact one-hot on a real prompt: {err:.4g}")
        self.ctx.res.data["meta"]["f_readin_max_err"] = err
        # background leakage over a vocabulary sample
        V = ctx.llm.get_input_embeddings().weight.shape[0]
        sample = torch.tensor(random.Random(5).sample(range(min(V, 150000)), 4096), device=ctx.device)
        with torch.no_grad():
            zb = dual_rail_clamp(ctx.llm.get_input_embeddings()(sample).float() @ self.cal[0].T + self.cal[1])
        special = set(self.digit_ids + [self.plus_id, self.eq_id])
        mask = torch.tensor([int(t) not in special for t in sample.tolist()], device=ctx.device)
        leak = float((zb[mask][:, :CH_OTHER] > 0).any(-1).float().mean())
        self.ctx.res.data["meta"]["f_readin_vocab_leak_rate"] = leak
        log(f"F read-in: fraction of 4096 random non-symbol tokens firing a digit/+/= channel: {leak:.4f}")

    # ---------------- evaluation ----------------
    def evaluate(self, step_fn, nll_fn=None) -> dict:
        res = {}
        bs = self.ctx.args.eval_bs
        tok = self.ctx.tokenizer
        for name, groups in self.splits.items():
            first_ok = exact = lenient = dig_ok = dig_tot = total = 0
            for n, X, Y in groups:
                for s in range(0, X.shape[0], bs):
                    x, y = X[s:s + bs], Y[s:s + bs]
                    gen, first = greedy_generate(step_fn, x, n + 1)
                    first_ok += int((first.index_select(-1, self.digit_t).argmax(-1) == y[:, 0]).sum())
                    tgt = self.digit_t[y]
                    exact += int((gen == tgt).all(1).sum())
                    dig_ok += int((gen == tgt).sum()); dig_tot += tgt.numel()
                    for row, yy in zip(gen.tolist(), y.tolist()):
                        txt = ""
                        for t in row:
                            piece = tok.decode([t]).strip()
                            if piece.isdigit():
                                txt += piece
                            else:
                                break
                        lenient += int(txt != "" and int(txt) == int("".join(map(str, yy))))
                    total += x.shape[0]
            res[name] = {"first_digit_acc": round(first_ok / total, 4), "full_exact": round(exact / total, 4),
                         "full_exact_lenient": round(lenient / total, 4), "per_digit": round(dig_ok / max(1, dig_tot), 4), "n": total}
            log(f"    eval {name}: {res[name]}")
        if nll_fn is not None:
            res["english_nll"] = text_loss(nll_fn, tok, ENGLISH_PROBES, self.ctx.device, [self.eq_id])["nll"]
        return res

    def batch(self, rng: random.Random, bs: int):
        n = rng.choice(self.ctx.args.f_train_lens)
        xs, ys = [], []
        for _ in range(bs):
            p, ans = gen_add_normal(rng, n)
            xs.append(self.ctx.tokenizer.encode(p, add_special_tokens=False) + [self.digit_ids[d] for d in ans[:-1]])
            ys.append(ans)
        dev = self.ctx.device
        ids = torch.tensor(xs, dtype=torch.long, device=dev)
        P = ids.shape[1] - n - 1            # position of '='
        return ids, P, self.digit_t[torch.tensor(ys, device=dev)]

    # ---------------- arms ----------------
    def arm_base(self) -> dict:
        llm = self.ctx.llm
        r = self.evaluate(plain_step_fn(llm), nll_fn=lambda ids: llm(input_ids=ids, use_cache=False).logits)
        return r

    def arm_graft(self, written: bool, seed: int) -> dict:
        a, ctx = self.ctx.args, self.ctx
        torch.manual_seed(seed)
        rng = random.Random(seed)
        organ = build_natural_add_organ(a.max_idx, ctx.device)
        if not written:
            organ.init_random_for_sgd(seed=seed)
        for p in organ.parameters():
            p.requires_grad_(not written)
        graft = NaturalAddGraft(ctx.llm, organ, self.cal, a.max_idx, self.digit_ids, self.eq_id,
                                gate_mode="circuit" if written else "code").to(ctx.device)
        steps = a.written_steps if written else a.circuit_steps
        lr = a.written_lr if written else a.circuit_lr
        params = [p for p in graft.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-3 if written else 0.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=lr * 0.05)
        nll0 = text_loss(graft.all_logits, ctx.tokenizer, ENGLISH_PROBES, ctx.device, [self.eq_id])["nll"]
        timer = StepTimer("written" if written else "trained_circuit", steps)
        for step in range(steps):
            ids, P, tgt = self.batch(rng, a.batch_size)
            logits = graft.logits_from(ids, P)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            timer.tick(step, loss.item())
        graft.eval()
        r = self.evaluate(graft.step_fn, nll_fn=graft.all_logits)
        r["english_nll_step0"] = nll0
        r["n_trainable_params"] = sum(p.numel() for p in params)
        r["organ_nnz"] = organ.nnz()
        r["final_train_loss"] = round(loss.item(), 5) if steps else None
        return r

    def arm_lora(self, mode: str, seed: int) -> dict:
        a, ctx = self.ctx.args, self.ctx
        llm = ctx.llm
        torch.manual_seed(seed)
        rng = random.Random(seed)
        params, detach = lora_session(llm)
        try:
            opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.lora_steps), eta_min=2e-5)
            amp = torch.bfloat16 if (ctx.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
            micro = min(8, a.batch_size)
            accum = max(1, a.batch_size // micro)
            timer = StepTimer(f"lora_{mode}", a.lora_steps)
            llm.train()
            for step in range(a.lora_steps):
                opt.zero_grad(set_to_none=True)
                tot = 0.0
                for _ in range(accum):
                    ids, P, tgt = self.batch(rng, micro)
                    with torch.autocast(device_type=ctx.device.type, dtype=amp, enabled=ctx.device.type == "cuda"):
                        if mode == "first":
                            ids = ids[:, :P + 1]
                            lg = llm(input_ids=ids, use_cache=False).logits[:, -1].float()
                            loss = F.cross_entropy(lg, tgt[:, 0])
                        else:
                            lg = llm(input_ids=ids, use_cache=False).logits[:, P:].float()
                            loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1))
                    (loss / accum).backward()
                    tot += loss.item() / accum
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step()
                timer.tick(step, tot)
            llm.gradient_checkpointing_disable()
            llm.eval()
            r = self.evaluate(plain_step_fn(llm), nll_fn=lambda ids: llm(input_ids=ids, use_cache=False).logits)
            r["n_trainable_params"] = sum(p.numel() for p in params)
            r["final_train_loss"] = round(tot, 5)
            return r
        finally:
            detach()

    def run(self) -> None:
        a, res = self.ctx.args, self.ctx.res
        arms = a.f_arms
        if "base" in arms:
            res.run("f", "base", self.arm_base)
        for arm, fn in (("written_circuit", lambda s: self.arm_graft(True, s)),
                        ("trained_circuit", lambda s: self.arm_graft(False, s)),
                        ("lora_first", lambda s: self.arm_lora("first", s)),
                        ("lora_full", lambda s: self.arm_lora("full", s))):
            if arm in arms:
                for seed in a.seeds:
                    res.run("f", f"{arm}/s{seed}", lambda: fn(seed))


# =============================================================================
# PART G1: write-out into the residual stream at layer k
# =============================================================================

# Each fact appears in exactly one template, so held-out templates never share a fact with training.
FACT_PAIRS = [
    (("the capital of France", "Paris"), ("the capital of Japan", "Tokyo")),
    (("the capital of Italy", "Rome"), ("the capital of Egypt", "Cairo")),
    (("the capital of Germany", "Berlin"), ("the capital of Spain", "Madrid")),
    (("the capital of Russia", "Moscow"), ("the capital of Greece", "Athens")),
    (("the capital of China", "Beijing"), ("the capital of the United Kingdom", "London")),
    (("the capital of Canada", "Ottawa"), ("the capital of Kenya", "Nairobi")),
    (("the capital of Portugal", "Lisbon"), ("the capital of Austria", "Vienna")),
    (("the capital of Norway", "Oslo"), ("the capital of Ireland", "Dublin")),
    (("the opposite of hot", "cold"), ("the opposite of up", "down")),
    (("the opposite of big", "small"), ("the opposite of fast", "slow")),
    (("the opposite of day", "night"), ("the opposite of happy", "sad")),
    (("the opposite of early", "late"), ("the opposite of rich", "poor")),
    (("the opposite of wet", "dry"), ("the opposite of win", "lose")),
    (("the opposite of true", "false"), ("the opposite of north", "south")),
    (("the colour of the sky on a clear day", "blue"), ("the colour of grass", "green")),
    (("the colour of snow", "white"), ("the colour of coal", "black")),
    (("the colour of a ripe banana", "yellow"), ("the colour of blood", "red")),
    (("the colour of chocolate", "brown"), ("the colour of a carrot", "orange")),
    (("the number word after three", "four"), ("the number word after seven", "eight")),
    (("the number word after one", "two"), ("the number word after nine", "ten")),
    (("the number word after five", "six"), ("the number word after eleven", "twelve")),
    (("the animal that barks", "dog"), ("the animal that says moo", "cow")),
    (("the largest land animal", "elephant"), ("the animal that meows", "cat")),
    (("the largest planet in the solar system", "Jupiter"), ("the planet we live on", "Earth")),
    (("the name for frozen water", "ice"), ("the gas humans need to breathe", "oxygen")),
    (("the main language of Brazil", "Portuguese"), ("the main language of Mexico", "Spanish")),
    (("the month after January", "February"), ("the day after Monday", "Tuesday")),
    (("the currency of Japan", "yen"), ("the author of Hamlet", "Shakespeare")),
    (("the month before December", "November"), ("the season after winter", "spring")),
    (("the organ that pumps blood", "heart"), ("the metal with the symbol Fe", "iron")),
]

PHRASINGS = [
    "If the brackets are balanced, name {qa}; otherwise name {qb}.",
    "If the answer is Yes, give {qa}. If the answer is No, give {qb}.",
    "Balanced means: say {qa}. Not balanced means: say {qb}.",
    "Reply with {qa} when the brackets are balanced, and with {qb} when they are not.",
    "Task: for a balanced sequence write {qa}; for an unbalanced sequence write {qb}. One word only.",
]


# Second template family: translating the yes/no result (knowledge of the language word is needed).
LANG_PAIRS = [
    (("the German word for yes", "Ja"), ("the German word for no", "Nein")),
    (("the Spanish word for yes", "Sí"), ("the Spanish word for no", "No")),
    (("the Italian word for yes", "Sì"), ("the Italian word for no", "No")),
    (("the Portuguese word for yes", "Sim"), ("the Portuguese word for no", "Não")),
    (("the Russian word for yes", "Да"), ("the Russian word for no", "Нет")),
    (("the Polish word for yes", "Tak"), ("the Polish word for no", "Nie")),
    (("the Dutch word for yes", "Ja"), ("the Dutch word for no", "Nee")),
    (("the Swedish word for yes", "Ja"), ("the Swedish word for no", "Nej")),
]

# Third family, NO world knowledge needed (only routes the bit to an answer word). Reported separately:
# it shows the bit reaches the answer but not that it is composed with stored knowledge.
ECHO_PAIRS = [
    (("the word balanced", "balanced"), ("the word unbalanced", "unbalanced")),
    (("the word correct", "correct"), ("the word incorrect", "incorrect")),
    (("the word valid", "valid"), ("the word invalid", "invalid")),
    (("the word good", "good"), ("the word bad", "bad")),
    (("the word No", "No"), ("the word Yes", "Yes")),
    (("the word false", "false"), ("the word true", "true")),
]

# Facts used ONLY inside few-shot demonstrations (never in any template, train or held-out).
# Demonstrations state the result in words; they contain no bracket strings and no '='.
DEMO_FACTS = [
    ("the capital of Peru", "Lima", "the capital of Cuba", "Havana", 1),
    ("the opposite of tall", "short", "the opposite of loud", "quiet", 0),
    ("the number word after two", "three", "the number word after six", "seven", 0),
    ("the French word for yes", "Oui", "the French word for no", "Non", 1),
    ("the capital of Poland", "Warsaw", "the capital of Turkey", "Ankara", 0),
    ("the opposite of hard", "soft", "the opposite of thick", "thin", 1),
    ("the Finnish word for yes", "Kyllä", "the Finnish word for no", "Ei", 1),
    ("the capital of Chile", "Santiago", "the capital of Thailand", "Bangkok", 0),
]
G1_SYSTEM = "You are a helpful assistant. Answer with one word."
# Measured locally on Qwen2.5-1.5B-Instruct (6 templates, 4+4 cases, CPU): plain 0.54, arrow 0.46
# mean oracle accuracy, i.e. chance. The other formats were not measured locally.
G1_FORMATS = ["plain", "arrow", "fewshot", "fewshot8", "chat", "chat_fewshot", "chat_fewshot8", "chat_result_first"]


def g1_instr(qa: str, qb: str) -> str:
    return f"If the answer is Yes, name {qa}. If the answer is No, name {qb}."


def build_composed(fmt: str, tok, t: dict, seq: str, result: Optional[int]) -> str:
    """Composed prompt. result=None: the query ('= ?', the bit must come from the circuit);
    result=0/1: oracle text ('= No' / '= Yes', the bit stated in words). Every format ends with
    'Answer:' so the candidates are the first tokens of ' <word>'."""
    qa, qb = t["qa"], t["qb"]
    res = "?" if result is None else ("Yes" if result else "No")
    q = f"Balanced brackets: {seq} = {res}"
    if fmt == "plain":
        return f"{q} {t['instr']} Answer:"
    if fmt == "arrow":
        return f"{q}\nYes -> {qa}. No -> {qb}.\nAnswer:"
    if fmt in ("fewshot", "fewshot8"):
        demos = DEMO_FACTS[:4] if fmt == "fewshot" else DEMO_FACTS
        demo = "".join(f"Balanced brackets: {'Yes' if dy else 'No'}\n{g1_instr(dqa, dqb)}\nAnswer: {da if dy else db}\n\n"
                       for dqa, da, dqb, db, dy in demos)
        return demo + f"{q}\n{g1_instr(qa, qb)}\nAnswer:"
    if fmt in ("chat", "chat_fewshot", "chat_fewshot8"):
        msgs = [{"role": "system", "content": G1_SYSTEM}]
        if fmt != "chat":
            for dqa, da, dqb, db, dy in (DEMO_FACTS[:4] if fmt == "chat_fewshot" else DEMO_FACTS):
                msgs += [{"role": "user", "content": f"Balanced brackets: {'Yes' if dy else 'No'}\n{g1_instr(dqa, dqb)}"},
                         {"role": "assistant", "content": f"Answer: {da if dy else db}"}]
        msgs.append({"role": "user", "content": f"{q}\n{g1_instr(qa, qb)}"})
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "Answer:"
    if fmt == "chat_result_first":
        # instruction first, the bracket line (with the bit) last before the answer
        msgs = [{"role": "system", "content": G1_SYSTEM},
                {"role": "user", "content": f"{g1_instr(qa, qb)}\n{q}"}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "Answer:"
    raise ValueError(f"unknown G1 format {fmt}")


def g1_templates(tok) -> List[dict]:
    temps = []
    for fam, pairs in (("fact", FACT_PAIRS), ("lang", LANG_PAIRS), ("echo", ECHO_PAIRS)):
        for (qa, aa), (qb, ab) in pairs:
            i = len(temps)
            temps.append({"id": i, "family": fam, "qa": qa, "qb": qb, "yes": aa, "no": ab,
                          "instr": PHRASINGS[i % len(PHRASINGS)].format(qa=qa, qb=qb),
                          "cand_yes": first_tok(tok, " " + aa), "cand_no": first_tok(tok, " " + ab)})
    return temps


def g1_split(temps: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Fixed train/held-out split by template (before any filtering), stratified by family.
    Each fact appears in exactly one template, so held-out facts never occur in training."""
    train, held = [], []
    for fam in sorted({t["family"] for t in temps}):
        ts = [t for t in temps if t["family"] == fam]
        order = list(range(len(ts)))
        random.Random(2024).shuffle(order)
        n_held = max(1, round(len(ts) * 0.4))
        held += [ts[i] for i in sorted(order[:n_held])]
        train += [ts[i] for i in sorted(order[n_held:])]
    return train, held


def bracket_seq(rng: random.Random, L: int) -> Tuple[str, int]:
    p, y = gen_real_parens(rng, L)
    return p[len("Balanced brackets: "):-2], y        # "( ( ) )", label


class MidLayerGraft(nn.Module):
    """Bracket circuit read from embeddings; write-out added at the output of decoder layer k."""

    def __init__(self, llm, organ, cal_in, k: int, n_sym: int = 4, gate_ch: int = 2):
        super().__init__()
        self.llm, self.organ, self.k, self.n_sym, self.gate_ch = llm, organ, k, n_sym, gate_ch
        for p in organ.parameters():
            p.requires_grad_(False)
        self.register_buffer("W_in", cal_in[0].float().clone(), persistent=False)
        self.register_buffer("b_in", cal_in[1].float().clone(), persistent=False)
        self.W_out = nn.Linear(organ.n_out, llm.config.hidden_size, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.W_out.weight)

    def delta(self, ids):
        with torch.no_grad():
            z = self.llm.get_input_embeddings()(ids).float() @ self.W_in.T + self.b_in
            z = torch.cat([dual_rail_clamp(z[..., :self.n_sym]), z[..., self.n_sym:]], -1)
            org = self.organ.forward_states(z)
            gate = z[..., self.gate_ch:self.gate_ch + 1].clamp(0.0, 1.0)
        return gate * self.W_out(torch.tanh(org / 10.0) * 10.0)

    def last_logits(self, ids):
        h = add_residual_hook(self.llm, self.k, self.delta(ids))
        try:
            hs = self.llm.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
        finally:
            h.remove()
        lm = self.llm.get_output_embeddings()
        return lm(hs).float()

    def all_logits(self, ids):
        h = add_residual_hook(self.llm, self.k, self.delta(ids))
        try:
            return self.llm(input_ids=ids, use_cache=False).logits.float()
        finally:
            h.remove()


class LastStateGraft(MidLayerGraft):
    """Reference: the 04b wiring (write-out added to the final hidden state, gated on '=').
    In the composed prompt the answer is read many tokens after '=', where the gate is 0, so this
    cannot change the answer by construction."""

    def __init__(self, llm, organ, cal_in):
        super().__init__(llm, organ, cal_in, k=llm.config.num_hidden_layers)

    def last_logits(self, ids):
        with torch.no_grad():
            hs = self.llm.model(input_ids=ids, use_cache=False).last_hidden_state.float()
        h = (hs + self.delta(ids))[:, -1]
        lm = self.llm.get_output_embeddings()
        return lm(h.to(lm.weight.dtype)).float()


def cand_accuracy(fn, groups, bs: int) -> float:
    """groups: list of (ids [N,T], cands [2] (no, yes), y [N] [, family])."""
    ok = tot = 0
    with torch.no_grad():
        for ids, cands, y, *_ in groups:
            for s in range(0, ids.shape[0], bs):
                lg = fn(ids[s:s + bs]).index_select(-1, cands)
                ok += int((lg.argmax(-1) == y[s:s + bs]).sum()); tot += int(y[s:s + bs].numel())
    return round(ok / max(1, tot), 4)


class PartG:
    def __init__(self, ctx):
        self.ctx = ctx
        tok = ctx.tokenizer
        self.tok_yes, self.tok_no = first_tok(tok, "Yes"), first_tok(tok, "No")
        self.eq_ids = sorted({first_tok(tok, "="), first_tok(tok, " =")})
        self.lp = sorted({first_tok(tok, "("), first_tok(tok, " (")})
        self.rp = sorted({first_tok(tok, ")"), first_tok(tok, " )")})
        self.cal_par = calibrate_port_socket(ctx.llm, 16, [self.lp, self.rp, self.eq_ids], other_ch=3, const_ch=4, device=ctx.device)
        L = ctx.n_layers
        a = ctx.args
        self.k_default = a.g1_k if a.g1_k > 0 else max(1, round(L / 3))
        self.k_sweep = [int(x) for x in a.g1_ks.split(",")] if a.g1_ks else sorted({max(1, round(L / 6)), max(1, round(L / 3)), max(1, round(2 * L / 3))})
        self.lens = [("id", [4, 6, 8])] + [(f"L{n}", [n]) for n in a.g_lens]
        ctx.res.data["meta"]["g1_k_default"] = self.k_default
        ctx.res.data["meta"]["g1_k_sweep"] = self.k_sweep
        ctx.res.data["meta"]["n_layers"] = L

    def enc(self, text: str) -> List[int]:
        return self.ctx.tokenizer.encode(text, add_special_tokens=False)

    def tens(self, x):
        return torch.tensor(x, dtype=torch.long, device=self.ctx.device)

    # ---------- plain ----------
    def plain_sets(self):
        n = self.ctx.args.g_test_n
        out = {}
        for si, (name, lens) in enumerate(self.lens):
            rng = random.Random(9100 + si)
            g: Dict[int, Tuple[list, list]] = {}
            for i in range(n):
                p, y = gen_real_parens(rng, lens[i % len(lens)])
                ids = self.enc(p)
                g.setdefault(len(ids), ([], []))[0].append(ids)
                g[len(ids)][1].append(y)
            cands = self.tens([self.tok_no, self.tok_yes])
            out[name] = [(self.tens(x), cands, self.tens(y)) for x, y in g.values()]
        return out

    def train_mid(self, graft: MidLayerGraft, batch_fn, steps: int, seed: int, name: str) -> float:
        a = self.ctx.args
        rng = random.Random(seed)
        torch.manual_seed(seed)
        opt = torch.optim.AdamW([graft.W_out.weight], lr=a.g_lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=a.g_lr * 0.05)
        timer = StepTimer(name, steps)
        loss = torch.tensor(float("nan"))
        for step in range(steps):
            ids, tgt = batch_fn(rng)
            loss = F.cross_entropy(graft.last_logits(ids), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([graft.W_out.weight], 1.0)
            opt.step(); sched.step()
            timer.tick(step, loss.item())
        return round(loss.item(), 5)

    def plain_batch(self, rng):
        L = rng.choice([4, 6, 8])
        xs, ys = [], []
        for _ in range(self.ctx.args.batch_size):
            p, y = gen_real_parens(rng, L)
            xs.append(self.enc(p)); ys.append(self.tok_yes if y else self.tok_no)
        return self.tens(xs), self.tens(ys)

    def text_losses(self, logits_fn) -> dict:
        ctx = self.ctx
        return {"probes": text_loss(logits_fn, ctx.tokenizer, ENGLISH_PROBES, ctx.device, self.eq_ids),
                "eq_lines": text_loss(logits_fn, ctx.tokenizer, eq_text_lines(), ctx.device, self.eq_ids)}

    # ---------- composed ----------
    def enc_checked(self, text: str, n_brackets: int) -> List[int]:
        """Encode and assert the hard constraint: the only tokens containing '(' / ')' are the
        query's own bracket tokens, and the only token containing '=' is the query's '='."""
        ids = self.enc(text)
        tok = self.ctx.tokenizer
        br = [i for i, t in enumerate(ids) if any(c in tok.decode([t]) for c in "()")]
        eq = [i for i, t in enumerate(ids) if "=" in tok.decode([t])]
        ok = (len(br) == n_brackets and all(ids[i] in self.lp or ids[i] in self.rp for i in br)
              and len(eq) == 1 and ids[eq[0]] in self.eq_ids and (not br or eq[0] == br[-1] + 1))
        if not ok:
            raise AssertionError(f"G1 prompt violates the bracket/'=' constraint: {[tok.decode([t]) for t in ids]}")
        return ids

    def oracle_table(self, fmt: str, temps: List[dict]) -> dict:
        """Base model only, result told in words: accuracy per template (n balanced + n unbalanced)."""
        a = self.ctx.args
        llm, tok = self.ctx.llm, self.ctx.tokenizer
        out = {}
        for t in temps:
            if t["cand_yes"] == t["cand_no"]:
                out[str(t["id"])] = {"acc": 0.0, "why": "same first token"}
                continue
            rng = random.Random(31337 + t["id"])
            ok = tot = 0
            per = {}
            for y_want in (1, 0):
                by_len: Dict[int, list] = {}
                got = 0
                while got < a.g1_check_n:
                    L = rng.choice([4, 6, 8])
                    seq, y = bracket_seq(rng, L)
                    if y != y_want:
                        continue
                    got += 1
                    ids = self.enc_checked(build_composed(fmt, tok, t, seq, y), len(seq.split()))
                    by_len.setdefault(len(ids), []).append(ids)
                c = 0
                for xs in by_len.values():
                    for s0 in range(0, len(xs), a.eval_bs):
                        with torch.no_grad():
                            lg = llm(input_ids=self.tens(xs[s0:s0 + a.eval_bs]), use_cache=False).logits[:, -1].float()
                        c += int(((lg[:, t["cand_yes"]] > lg[:, t["cand_no"]]).long() == y_want).sum())
                per["balanced" if y_want else "unbalanced"] = round(c / got, 4)
                ok += c; tot += got
            out[str(t["id"])] = {"acc": round(ok / tot, 4), **per}
        return out

    def g1check(self):
        """Measure oracle accuracy for every format and template; select the format by mean
        oracle accuracy on TRAIN templates; keep templates that pass (>= threshold) in it."""
        a, res, tok = self.ctx.args, self.ctx.res, self.ctx.tokenizer
        fams = {f.strip() for f in a.g1_families.split(",") if f.strip()}
        temps = [t for t in g1_templates(tok) if t["family"] in fams]
        train_all, held_all = g1_split(temps)
        train_ids = {t["id"] for t in train_all}
        res.data["meta"]["g1_template_list"] = [{k: t[k] for k in ("id", "family", "qa", "yes", "qb", "no")}
                                                | {"split": "train" if t["id"] in train_ids else "held_out"} for t in temps]
        fmts = [f.strip() for f in a.g1_formats.split(",") if f.strip()]
        table = {}
        for fmt in fmts:
            r = res.run("g1check", f"oracle_{fmt}", lambda fmt=fmt: {"format": fmt, "n_per_class": a.g1_check_n,
                                                                     "per_template": self.oracle_table(fmt, temps)})
            if r and "error" not in r:
                table[fmt] = r["per_template"]
        if not table:
            raise RuntimeError("no G1 format could be evaluated")
        def mean_train(fmt):
            v = [table[fmt][str(i)]["acc"] for i in train_ids]
            return sum(v) / len(v)
        summary = {f: {"mean_train": round(mean_train(f), 4),
                       "mean_all": round(sum(x["acc"] for x in table[f].values()) / len(table[f]), 4),
                       "n_pass_train": sum(table[f][str(t["id"])]["acc"] >= a.g1_pass for t in train_all),
                       "n_pass_held": sum(table[f][str(t["id"])]["acc"] >= a.g1_pass for t in held_all)} for f in table}
        fmt = a.g1_format if a.g1_format else max(table, key=mean_train)
        train = [t for t in train_all if table[fmt][str(t["id"])]["acc"] >= a.g1_pass]
        held = [t for t in held_all if table[fmt][str(t["id"])]["acc"] >= a.g1_pass]
        if a.smoke and (len(train) < 1 or len(held) < 1):
            log("SMOKE: oracle filter bypassed so the composed code path runs (results meaningless)")
            train, held = train_all, held_all
        sel = {"format": fmt, "selected_by": "--g1-format" if a.g1_format else "max mean oracle accuracy on train templates",
               "pass_threshold": a.g1_pass, "per_format": summary,
               "train_kept": [t["id"] for t in train], "held_kept": [t["id"] for t in held],
               "n_train": len(train), "n_held": len(held),
               "kept_by_family": {f: {"train": sum(t["family"] == f for t in train), "held": sum(t["family"] == f for t in held)}
                                  for f in sorted(fams)},
               "example_query": build_composed(fmt, tok, temps[0], "( ( ) )", None)}
        res.data["meta"]["g1_selection"] = sel
        res.save()
        log(f"G1 format summary: {summary}")
        log(f"G1 selected format '{fmt}': train templates kept {len(train)}/{len(train_all)}, held-out kept {len(held)}/{len(held_all)}, "
            f"by family {sel['kept_by_family']}")
        if len(held) < 8:
            log(f"WARNING: only {len(held)} held-out templates survive the oracle filter (wanted >= 8)")
        return fmt, train, held

    def composed_sets(self, fmt, temps, oracle: bool):
        n = self.ctx.args.g_test_n
        tok = self.ctx.tokenizer
        out = {}
        for si, (name, lens) in enumerate(self.lens):
            rng = random.Random(7300 + si)
            g: Dict[Tuple[int, int], Tuple[list, list]] = {}
            for i in range(n):
                t = temps[i % len(temps)]
                L = lens[(i // len(temps)) % len(lens)]
                seq, y = bracket_seq(rng, L)
                ids = self.enc_checked(build_composed(fmt, tok, t, seq, y if oracle else None), len(seq.split()))
                g.setdefault((t["id"], len(ids)), ([], []))
                g[(t["id"], len(ids))][0].append(ids); g[(t["id"], len(ids))][1].append(y)
            tmap = {t["id"]: t for t in temps}
            out[name] = [(self.tens(x), self.tens([tmap[tid]["cand_no"], tmap[tid]["cand_yes"]]), self.tens(y), tmap[tid]["family"])
                         for (tid, _), (x, y) in g.items()]
        return out

    @staticmethod
    def by_family(fn, sets, bs) -> dict:
        """Held-out accuracy per template family (fact / lang need world knowledge; echo does not)."""
        fams = sorted({g[3] for v in sets.values() for g in v})
        return {f: {k: cand_accuracy(fn, [g for g in v if g[3] == f], bs) for k, v in sets.items()} for f in fams}

    def composed_batch_fn(self, fmt, train):
        tok = self.ctx.tokenizer
        def fn(rng):
            t = rng.choice(train)
            L = rng.choice([4, 6, 8])
            xs, ys = [], []
            for _ in range(self.ctx.args.batch_size):
                seq, y = bracket_seq(rng, L)
                xs.append(self.enc_checked(build_composed(fmt, tok, t, seq, None), len(seq.split())))
                ys.append(t["cand_yes"] if y else t["cand_no"])
            return self.tens(xs), self.tens(ys)
        return fn

    def lora_composed(self, fmt, train, held_sets, train_id_set, seed):
        a, ctx = self.ctx.args, self.ctx
        llm = ctx.llm
        torch.manual_seed(seed)
        rng = random.Random(seed)
        params, detach = lora_session(llm)
        try:
            opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.g1_steps), eta_min=2e-5)
            amp = torch.bfloat16 if (ctx.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
            micro = min(8, a.batch_size)
            accum = max(1, a.batch_size // micro)
            bfn = self.composed_batch_fn(fmt, train)
            timer = StepTimer("lora_composed", a.g1_steps)
            llm.train()
            for step in range(a.g1_steps):
                opt.zero_grad(set_to_none=True)
                tot = 0.0
                ids, tgt = bfn(rng)
                for m in range(accum):
                    x, y = ids[m * micro:(m + 1) * micro], tgt[m * micro:(m + 1) * micro]
                    with torch.autocast(device_type=ctx.device.type, dtype=amp, enabled=ctx.device.type == "cuda"):
                        loss = F.cross_entropy(llm(input_ids=x, use_cache=False).logits[:, -1].float(), y)
                    (loss / accum).backward(); tot += loss.item() / accum
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step()
                timer.tick(step, tot)
            llm.gradient_checkpointing_disable()
            llm.eval()
            fn = lambda x: llm(input_ids=x, use_cache=False).logits[:, -1].float()  # noqa: E731
            r = {"held_out": {k: cand_accuracy(fn, v, a.eval_bs) for k, v in held_sets.items()},
                 "held_out_by_family": self.by_family(fn, held_sets, a.eval_bs),
                 "train_templates_id": cand_accuracy(fn, train_id_set, a.eval_bs),
                 "text_loss": self.text_losses(lambda ids: llm(input_ids=ids, use_cache=False).logits),
                 "n_trainable_params": sum(p.numel() for p in params), "final_train_loss": round(tot, 5)}
            return r
        finally:
            detach()

    def run_g1(self) -> None:
        a, ctx, res = self.ctx.args, self.ctx, self.ctx.res
        llm = ctx.llm
        last = lambda x: llm(input_ids=x, use_cache=False).logits[:, -1].float()  # noqa: E731
        res.run("g1", "text_loss_base", lambda: self.text_losses(lambda ids: llm(input_ids=ids, use_cache=False).logits))
        psets = self.plain_sets()
        res.run("g1", "plain_base", lambda: {k: cand_accuracy(last, v, a.eval_bs) for k, v in psets.items()})
        for seed in a.g_seeds:
            def plain_cell(seed=seed):
                g = MidLayerGraft(llm, build_real_parens_organ(ctx.device), self.cal_par, self.k_default).to(ctx.device)
                fl = self.train_mid(g, self.plain_batch, a.g1_steps, seed, f"g1_plain_k{self.k_default}")
                return {"k": self.k_default, "acc": {k: cand_accuracy(g.last_logits, v, a.eval_bs) for k, v in psets.items()},
                        "text_loss": self.text_losses(g.all_logits), "final_train_loss": fl}
            res.run("g1", f"plain_graft_k{self.k_default}/s{seed}", plain_cell)
        del psets
        self.run_composed()

    def run_composed(self) -> None:
        a, ctx, res = self.ctx.args, self.ctx, self.ctx.res
        llm = ctx.llm
        last = lambda x: llm(input_ids=x, use_cache=False).logits[:, -1].float()  # noqa: E731
        fmt, train, held = self.g1check()
        if len(train) < 1 or len(held) < a.g1_min_held:
            raise RuntimeError(f"too few templates after the oracle filter (train={len(train)}, held={len(held)}, "
                               f"minimum held-out {a.g1_min_held})")
        tag = f"{fmt}"
        held_sets = self.composed_sets(fmt, held, oracle=False)
        train_id = self.composed_sets(fmt, train, oracle=False)["id"]
        counts = {"format": fmt, "n_train_templates": len(train), "n_held_templates": len(held)}
        res.run("g1", f"composed_base/{tag}", lambda: {**counts, "held_out": {k: cand_accuracy(last, v, a.eval_bs) for k, v in held_sets.items()},
                                                       "held_out_by_family": self.by_family(last, held_sets, a.eval_bs),
                                                       "train_templates_id": cand_accuracy(last, train_id, a.eval_bs)})
        oracle_sets = self.composed_sets(fmt, held, oracle=True)
        res.run("g1", f"composed_oracle_text/{tag}", lambda: {**counts, "held_out": {k: cand_accuracy(last, v, a.eval_bs) for k, v in oracle_sets.items()},
                                                              "held_out_by_family": self.by_family(last, oracle_sets, a.eval_bs)})
        del oracle_sets
        for seed in a.g_seeds:
            def last_cell(seed=seed):
                g = LastStateGraft(llm, build_real_parens_organ(ctx.device), self.cal_par).to(ctx.device)
                fl = self.train_mid(g, self.composed_batch_fn(fmt, train), a.g1_steps, seed, "g1_composed_laststate")
                return {**counts, "wiring": "last hidden state (04b), gated on '='",
                        "held_out": {s: cand_accuracy(g.last_logits, v, a.eval_bs) for s, v in held_sets.items()},
                        "train_templates_id": cand_accuracy(g.last_logits, train_id, a.eval_bs),
                        "w_out_norm": round(g.W_out.weight.norm().item(), 5), "final_train_loss": fl}
            res.run("g1", f"composed_graft_laststate/{tag}/s{seed}", last_cell)
        for k in self.k_sweep:
            for seed in a.g_seeds:
                def comp_cell(k=k, seed=seed):
                    g = MidLayerGraft(llm, build_real_parens_organ(ctx.device), self.cal_par, k).to(ctx.device)
                    fl = self.train_mid(g, self.composed_batch_fn(fmt, train), a.g1_steps, seed, f"g1_composed_k{k}")
                    return {**counts, "k": k, "held_out": {s: cand_accuracy(g.last_logits, v, a.eval_bs) for s, v in held_sets.items()},
                            "held_out_by_family": self.by_family(g.last_logits, held_sets, a.eval_bs),
                            "train_templates_id": cand_accuracy(g.last_logits, train_id, a.eval_bs),
                            "text_loss": self.text_losses(g.all_logits), "final_train_loss": fl}
                res.run("g1", f"composed_graft_k{k}/{tag}/s{seed}", comp_cell)
        for seed in a.g_seeds:
            res.run("g1", f"composed_lora/{tag}/s{seed}", lambda seed=seed: {**counts, **self.lora_composed(fmt, train, held_sets, train_id, seed)})

    # =========================================================================
    # PART G2: read-in from hidden states at layer j
    # =========================================================================
    FORMS = {"A_open_close": ("open", "close"), "B_left_right": ("left", "right")}

    def word_prompt(self, seq: str, form: str) -> str:
        o, c = self.FORMS[form]
        return "Balanced brackets: " + " ".join(o if s == "(" else c for s in seq.split()) + " ="

    def fit_hidden_readin(self, j: int):
        ctx, a = self.ctx, self.ctx.args
        tok = ctx.tokenizer
        o_id, c_id = first_tok(tok, " open"), first_tok(tok, " close")
        rng = random.Random(555)
        texts = []
        for _ in range(a.g2_fit_n):
            seq, _ = bracket_seq(rng, rng.choice([4, 6, 8, 10, 12, 16]))
            texts.append(self.word_prompt(seq, "A_open_close"))
        texts += ENGLISH_PROBES + eq_text_lines()
        feats = {0: [], 1: [], 2: [], 3: []}
        eq = set(self.eq_ids)
        for txt in texts:
            ids = self.tens([self.enc(txt)])
            with torch.no_grad():
                h = ctx.llm.model(input_ids=ids, use_cache=False, output_hidden_states=True).hidden_states[j][0].float()
            for t, i in enumerate(ids[0].tolist()):
                ch = 0 if i == o_id else 1 if i == c_id else 2 if i in eq else 3
                feats[ch].append(h[t])
        cap = 4000
        X, Y = [], []
        counts = {}
        g = torch.Generator().manual_seed(0)
        for ch, rows in feats.items():
            R = torch.stack(rows)
            if R.shape[0] > cap:
                R = R[torch.randperm(R.shape[0], generator=g)[:cap].to(R.device)]
            counts[ch] = int(R.shape[0])
            reps = max(1, 256 // R.shape[0])      # upweight rare classes ('=' has few rows)
            R = R.repeat(reps, 1)
            X.append(R)
            y = torch.zeros(R.shape[0], 16, device=R.device); y[:, ch] = 1.0; y[:, 4] = 1.0
            Y.append(y)
        X, Y = torch.cat(X), torch.cat(Y)
        mu, sd = X.mean(0), X.std(0) + 1e-3
        Xs = torch.cat([(X - mu) / sd, torch.ones(X.shape[0], 1, device=X.device)], 1)
        A = Xs.T @ Xs
        A.diagonal().add_(a.g2_ridge * X.shape[0])
        sol = torch.linalg.solve(A, Xs.T @ Y)
        Wn, c = sol[:-1], sol[-1]
        W_in = (Wn / sd[:, None]).T.contiguous()
        b_in = (c - (mu / sd) @ Wn).contiguous()
        return W_in, b_in, counts

    def readin_diag(self, W_in, b_in, j, form: str, n: int = 32) -> dict:
        ctx = self.ctx
        o, c = self.FORMS[form]
        oid, cid = first_tok(ctx.tokenizer, " " + o), first_tok(ctx.tokenizer, " " + c)
        rng = random.Random(808)
        exact = tot = 0
        per = {"open": [0, 0], "close": [0, 0], "eq": [0, 0], "other": [0, 0]}
        for _ in range(n):
            seq, _ = bracket_seq(rng, rng.choice([8, 32]))
            ids = self.tens([self.enc(self.word_prompt(seq, form))])
            with torch.no_grad():
                h = ctx.llm.model(input_ids=ids, use_cache=False, output_hidden_states=True).hidden_states[j][0].float()
            z = dual_rail_clamp((h @ W_in.T + b_in)[:, :4])
            for t, i in enumerate(ids[0].tolist()):
                ch = 0 if i == oid else 1 if i == cid else 2 if i in self.eq_ids else 3
                e = torch.zeros(4, device=z.device); e[ch] = 1
                ok = bool((z[t] == e).all())
                name = ["open", "close", "eq", "other"][ch]
                per[name][0] += int(ok); per[name][1] += 1
                exact += int(ok); tot += 1
        return {"exact_onehot_rate": round(exact / tot, 4), **{k: round(v[0] / max(1, v[1]), 4) for k, v in per.items()}}

    def run_g2(self) -> None:
        a, ctx, res = self.ctx.args, self.ctx, self.ctx.res
        llm = ctx.llm
        L = ctx.n_layers
        j = a.g2_j if a.g2_j > 0 else max(1, round(L / 6))
        for form, (o, c) in self.FORMS.items():
            for w in (o, c):
                ids = self.enc(" " + w)
                if len(ids) != 1:
                    raise RuntimeError(f"G2 word ' {w}' is {len(ids)} tokens")
        W_in, b_in, counts = self.fit_hidden_readin(j)
        res.run("g2", f"readin_j{j}", lambda: {"j": j, "fit_rows_per_class": counts,
                                              **{f: self.readin_diag(W_in, b_in, j, f) for f in self.FORMS}})
        n = a.g_test_n
        sets = {}
        cands = self.tens([self.tok_no, self.tok_yes])
        for form in self.FORMS:
            for si, (name, lens) in enumerate(self.lens):
                rng = random.Random(6100 + si)
                g: Dict[int, Tuple[list, list]] = {}
                for i in range(n):
                    seq, y = bracket_seq(rng, lens[i % len(lens)])
                    ids = self.enc(self.word_prompt(seq, form))
                    g.setdefault(len(ids), ([], []))[0].append(ids); g[len(ids)][1].append(y)
                sets[f"{form}/{name}"] = [(self.tens(x), cands, self.tens(y)) for x, y in g.values()]
        last = lambda x: llm(input_ids=x, use_cache=False).logits[:, -1].float()  # noqa: E731
        res.run("g2", "base", lambda: {k: cand_accuracy(last, v, a.eval_bs) for k, v in sets.items()})
        organ = build_real_parens_organ(ctx.device)
        for p in organ.parameters():
            p.requires_grad_(False)

        class HiddenGraft(nn.Module):
            def __init__(s):
                super().__init__()
                s.W_out = nn.Linear(2, llm.config.hidden_size, bias=False, dtype=torch.float32)
                nn.init.zeros_(s.W_out.weight)

            def last_logits(s, ids):
                with torch.no_grad():
                    out = llm.model(input_ids=ids, use_cache=False, output_hidden_states=True)
                    z = out.hidden_states[j].float() @ W_in.T + b_in
                    z = torch.cat([dual_rail_clamp(z[..., :4]), z[..., 4:]], -1)
                    org = organ.forward_states(z)[:, -1]
                    gate = z[:, -1, 2:3].clamp(0.0, 1.0)
                    hl = out.last_hidden_state[:, -1].float()
                lm = llm.get_output_embeddings()
                return lm((hl + gate * s.W_out(torch.tanh(org / 10.0) * 10.0)).to(lm.weight.dtype)).float()

        def batch(rng):
            L_ = rng.choice([4, 6, 8])
            xs, ys = [], []
            for _ in range(a.batch_size):
                seq, y = bracket_seq(rng, L_)
                xs.append(self.enc(self.word_prompt(seq, "A_open_close"))); ys.append(self.tok_yes if y else self.tok_no)
            return self.tens(xs), self.tens(ys)

        for seed in a.g_seeds:
            def cell(seed=seed):
                g = HiddenGraft().to(ctx.device)
                rng = random.Random(seed); torch.manual_seed(seed)
                opt = torch.optim.AdamW([g.W_out.weight], lr=a.g_lr, weight_decay=1e-3)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.g2_steps), eta_min=a.g_lr * 0.05)
                timer = StepTimer(f"g2_j{j}", a.g2_steps)
                fl = float("nan")
                for step in range(a.g2_steps):
                    ids, tgt = batch(rng)
                    loss = F.cross_entropy(g.last_logits(ids), tgt)
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
                    fl = loss.item(); timer.tick(step, fl)
                return {"j": j, "acc": {k: cand_accuracy(g.last_logits, v, a.eval_bs) for k, v in sets.items()},
                        "final_train_loss": round(fl, 5)}
            res.run("g2", f"graft_j{j}/s{seed}", cell)


# =============================================================================
# MAIN
# =============================================================================

class Ctx:
    pass


def parse_ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parts", default="f,g1,g2", help="any of f,g1check,g1,g2 (g1 runs g1check itself)")
    ap.add_argument("--smoke", action="store_true", help="tiny everything; end to end in minutes")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seeds", default="0,1,2", help="seeds for Part F trained arms")
    ap.add_argument("--f-arms", default="base,written_circuit,trained_circuit,lora_first,lora_full")
    ap.add_argument("--g-seeds", default="0,1,2", help="seeds for Part G trained arms")
    ap.add_argument("--max-idx", type=int, default=160, help="largest digit index the abacus input encodes")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-bs", type=int, default=64)
    ap.add_argument("--lora-steps", type=int, default=1000)
    ap.add_argument("--circuit-steps", type=int, default=3000, help="trained-circuit arm steps")
    ap.add_argument("--circuit-lr", type=float, default=3e-3)
    ap.add_argument("--written-steps", type=int, default=500, help="W_out steps for the written circuit")
    ap.add_argument("--written-lr", type=float, default=1e-2)
    ap.add_argument("--f-train-lens", default="4,5,6,7,8")
    ap.add_argument("--f-lens", default="16,32,64")
    ap.add_argument("--f-carry-len", type=int, default=64)
    ap.add_argument("--f-test-n", type=int, default=512)
    ap.add_argument("--g1-steps", type=int, default=500, help="W_out steps (and LoRA steps) in G1")
    ap.add_argument("--g1-k", type=int, default=-1, help="layer for plain G1 (default depth/3)")
    ap.add_argument("--g1-ks", default="", help="comma list for the composed sweep (default ~1/6,1/3,2/3)")
    ap.add_argument("--g1-check-n", type=int, default=16, help="oracle cases per truth value per template")
    ap.add_argument("--g1-formats", default=",".join(G1_FORMATS), help="composed prompt formats measured by g1check")
    ap.add_argument("--g1-format", default="", help="force a format instead of selecting by train oracle accuracy")
    ap.add_argument("--g1-pass", type=float, default=0.9, help="oracle accuracy a template needs to be kept")
    ap.add_argument("--g1-min-held", type=int, default=4)
    ap.add_argument("--g1-families", default="fact,lang,echo", help="template families (echo needs no world knowledge)")
    ap.add_argument("--g-lr", type=float, default=1e-2)
    ap.add_argument("--g-lens", default="16,32,64,128")
    ap.add_argument("--g-test-n", type=int, default=512)
    ap.add_argument("--g2-steps", type=int, default=300)
    ap.add_argument("--g2-j", type=int, default=-1, help="read-in layer (default depth/6)")
    ap.add_argument("--g2-fit-n", type=int, default=256)
    ap.add_argument("--g2-ridge", type=float, default=1e-3)
    ap.add_argument("--selftest-n", type=int, default=128)
    ap.add_argument("--out", default="runs/natural_graft_results.json")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.g_seeds = "0", "0"
        args.batch_size, args.eval_bs = 4, 4
        args.lora_steps, args.circuit_steps, args.written_steps = 2, 3, 3
        args.f_lens, args.f_carry_len, args.f_test_n = "10", 10, 4
        args.g1_steps, args.g2_steps, args.g1_check_n = 2, 2, 1
        args.g_lens, args.g_test_n, args.g2_fit_n = "16", 4, 8
        args.selftest_n = 16
    args.seeds, args.g_seeds = parse_ints(args.seeds), parse_ints(args.g_seeds)
    args.f_train_lens, args.f_lens, args.g_lens = parse_ints(args.f_train_lens), parse_ints(args.f_lens), parse_ints(args.g_lens)
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    log(f"natural_graft_exp | parts={parts} smoke={args.smoke} model={args.model}")

    log("CPU self-test of the written circuits (must be 100%)...")
    st = verify_natural_add_cpu(args.max_idx, args.selftest_n)
    verify_parens_cpu(32)
    if args.verify_only:
        log("SELF-TEST PASSED")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    ctx = Ctx()
    ctx.args = args
    ctx.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    dtype = torch.bfloat16 if (ctx.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    cfg = {k: v for k, v in vars(args).items()}
    ctx.res = Results(args.out, cfg)
    ctx.res.data["meta"]["selftest_natural_add"] = st
    ctx.tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        llm = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    llm = llm.to(ctx.device).eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    ctx.llm = llm
    ctx.n_layers = llm.config.num_hidden_layers
    ctx.res.data["meta"].update({"device": str(ctx.device), "dtype": str(dtype),
                                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    log(f"loaded {args.model} on {ctx.device} ({dtype}), layers={ctx.n_layers}")
    for part in parts:   # a setup error from an earlier run is retried, so drop the stale record
        ctx.res.data["results"].get(part, {}).pop("setup", None)

    if "f" in parts:
        try:
            PartF(ctx).run()
        except Exception as e:
            ctx.res.data["results"].setdefault("f", {})["setup"] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
            ctx.res.save(); log(f"ERROR in part f setup: {e}")
        cleanup_cuda()
    g = None
    if "g1check" in parts or "g1" in parts or "g2" in parts:
        try:
            g = PartG(ctx)
        except Exception as e:
            ctx.res.data["results"].setdefault("g1", {})["setup"] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
            ctx.res.save(); log(f"ERROR in part g setup: {e}")
    for part, fn in (("g1check", "g1check"), ("g1", "run_g1"), ("g2", "run_g2")):
        if part == "g1check" and "g1" in parts:
            continue   # run_g1 runs the check itself (finished check cells are skipped)
        if part in parts and g is not None:
            try:
                getattr(g, fn)()
            except Exception as e:
                ctx.res.data["results"].setdefault(part, {})["setup"] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
                ctx.res.save(); log(f"ERROR in part {part}: {e}")
            cleanup_cuda()
    ctx.res.save()
    log(f"ALL DONE -> {args.out}")


if __name__ == "__main__":
    main()
