"""Compiled 2-layer causal Transformer organ for variable-length integer '+' and '-'.

Self-contained module implementing:
  - SparseOrgan: standard 2-layer causal Transformer (no LayerNorm, additive ALiBi, ReLU MLP).
  - build_varlen_calc_organ(max_idx=64, device="cpu"): weight-written (zero-training) 2-layer
    SparseOrgan supporting:
      * Variable-length operands (n_a != n_b, 1..max_idx digits, no leading zeros required)
      * Addition ('+') and Subtraction ('-') with arbitrary carry/borrow chains
      * a == b -> "0", a < b -> "-" followed by (b - a)
      * Optional spaces around '+', '-', '=' (treated as CH_OTHER)
      * No leading zeros & autonomous structural gate shutoff once units digit (p=1) is emitted
  - Active-expression window parser & bus builder (shared between standalone CPU tests and
    the frozen LLM graft).
  - run_organ_standalone(expr_str, organ=None, max_idx=64): pure-neural autoregressive runner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# SPARSE CAUSAL TRANSFORMER ORGAN
# =============================================================================

class SparseOrgan(nn.Module):
    """Standard causal Transformer organ (no LayerNorm, additive ALiBi, ReLU MLP)
    parameterized by (n_layers, n_heads, d_model, d_head, d_mlp, n_out)."""

    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        n_out: int,
        n_layers: int = 2,
        n_heads: int = 1,
        d_head: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.n_out = n_out
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head

        self.Q = nn.Parameter(torch.zeros(n_layers, n_heads, d_model, d_head))
        self.K = nn.Parameter(torch.zeros(n_layers, n_heads, d_model, d_head))
        self.V = nn.Parameter(torch.zeros(n_layers, n_heads, d_model, d_head))
        self.O = nn.Parameter(torch.zeros(n_layers, n_heads, d_head, d_model))
        self.alibi = nn.Parameter(torch.zeros(n_layers, n_heads))
        self.W1 = nn.Parameter(torch.zeros(n_layers, d_model, d_mlp))
        self.b1 = nn.Parameter(torch.zeros(n_layers, d_mlp))
        self.W2 = nn.Parameter(torch.zeros(n_layers, d_mlp, d_model))
        self.OUT_W = nn.Parameter(torch.zeros(d_model, n_out))
        self.OUT_b = nn.Parameter(torch.zeros(n_out))

    def nnz(self) -> int:
        return int(sum((p.detach().abs() > 1e-8).sum().item() for p in self.parameters()))

    def forward_states(self, x_in: torch.Tensor) -> torch.Tensor:
        """x_in: [B, T, d_model] -> logits [B, T, n_out]."""
        _, T, _ = x_in.shape
        device = x_in.device
        x = x_in
        pos = torch.arange(T, device=device)
        dist = (pos[:, None] - pos[None, :]).to(x.dtype)
        # Strictly causal (j < i), with self-attention at pos 0 so softmax is well-defined
        mask = pos[None, :] < pos[:, None]
        mask[0, 0] = True
        attn_mask = torch.where(mask, torch.zeros_like(dist), torch.full_like(dist, float("-inf")))
        scale = 1.0 / math.sqrt(self.d_head)

        for l in range(self.n_layers):
            ad = torch.zeros_like(x)
            for h in range(self.n_heads):
                q = x @ self.Q[l, h]
                k = x @ self.K[l, h]
                v = x @ self.V[l, h]
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                scores = scores + self.alibi[l, h] * dist.unsqueeze(0) + attn_mask.unsqueeze(0)
                probs = F.softmax(scores, dim=-1)
                ad = ad + torch.matmul(probs, v) @ self.O[l, h]
            x = x + ad
            x = x + F.relu(x @ self.W1[l] + self.b1[l]) @ self.W2[l]
        return x @ self.OUT_W + self.OUT_b


# =============================================================================
# BUS LAYOUT FOR VARIABLE-LENGTH '+' / '-' ORGAN
# =============================================================================

# Read-in channels (from LLM embeddings via ridge + dual-rail clamp, or exact one-hot)
CH_DIG = 0          # 0..9: one-hot digit '0'..'9'
CH_PLUS = 10        # '+' / ' +'
CH_MINUS = 11       # '-' / ' -'
CH_EQ = 12          # '=' / ' ='
CH_OTHER = 13       # spaces, prefix, sink
CH_CONST = 14       # constant 1.0
N_READ = 15

# Structural / abacus inputs (non-learned, scoped to active expression window)
CH_SEG_A = 15       # 1.0 on digits of operand a
CH_SEG_B = 16       # 1.0 on digits of operand b

# Scratch channels written by Layer 0 and Layer 1
CH_SEEN_PLUS = 17
CH_SEEN_MINUS = 18
CH_SEEN_EQ = 19
CH_VAL_A = 20       # fetched a_p (0..9)
CH_VAL_B = 21       # fetched b_p (0..9)
CH_G_ADD = 22       # at b_p: is_b & seen_plus & (a_p + b_p >= 10)
CH_K_ADD = 23       # at b_p: is_b & seen_plus & (a_p + b_p <= 8)
CH_K_ADD_A = 24     # at a_p: is_a & (a_p <= 8)
CH_G_SUB = 25       # at b_p: is_b & seen_minus & (a_p - b_p <= -1)
CH_K_SUB = 26       # at b_p: is_b & seen_minus & (a_p - b_p >= +1)
CH_K_SUB_A = 27     # at a_p: is_a & (a_p >= 1)
CH_C_ADD = 28       # incoming carry for a + b at place p
CH_BOR_AB = 29      # incoming borrow for a - b at place p
CH_BOR_BA = 30      # incoming borrow for b - a at place p
CH_IS_NEG = 31      # 1.0 iff op == '-' and a < b
CH_OUT = 32         # 32..42: 10 output digits ('0'..'9') + 1 minus sign ('-' at CH_OUT+10)
CH_GATE = 43        # structural answer gate (1.0 while target_p >= 1, 0.0 once p=1 emitted)
CH_BASE_VEC = 44    # vector channels start here:
#   [CH_BASE_VEC          .. +M)    : one-hot place p in 1..M (INPUT)
#   [CH_BASE_VEC + M      .. +2M)   : thermometer [m <= p] for m in 1..M (INPUT)
#   [CH_BASE_VEC + 2M     .. +3M)   : thermo(n_b) copied onto '=' & answer by L0.H5 (SCRATCH)
#   [CH_BASE_VEC + 3M     .. +4M)   : is_a * one(p) computed by L0.MLP (SCRATCH)

N_SYM_OUT = 11      # 0..9 digits, 10 = '-'
N_OUT = 12          # 0..10 output symbols + 11 = structural gate


def calc_internal_places(max_idx: int) -> int:
    """Internal place count M = max_idx + 1 so max_idx-digit addition with carry-out fits."""
    return max_idx + 1


def calc_bus_dim(max_idx: int) -> int:
    M = calc_internal_places(max_idx)
    return CH_BASE_VEC + 4 * M


def build_varlen_calc_organ(max_idx: int = 64, device: torch.device | str = "cpu") -> SparseOrgan:
    """Compile a 2-layer, 6-head SparseOrgan for variable-length integer '+' and '-'.

    Supports operand lengths 1..max_idx (with internal place range 1..max_idx+1 for carry-out),
    unequal operand lengths (n_a != n_b), arbitrary carry/borrow chains, a == b ("0"),
    a < b ("-" followed by b - a), optional spaces around '+', '-', '=', no leading zeros,
    and autonomous structural gate shutoff once the units digit (p=1) is emitted.
    """
    M = calc_internal_places(max_idx)
    d_model = calc_bus_dim(max_idx)
    d_head = 2 * M + 3
    d_mlp = max(256, 16 + M)
    s = math.sqrt(d_head)

    organ = SparseOrgan(
        d_model=d_model,
        d_mlp=d_mlp,
        n_out=N_OUT,
        n_layers=2,
        n_heads=6,
        d_head=d_head,
    ).to(device)
    organ.max_idx = max_idx  # type: ignore[attr-defined]
    organ.M = M              # type: ignore[attr-defined]

    one = lambda m: CH_BASE_VEC + (m - 1)                # noqa: E731
    thermo = lambda m: CH_BASE_VEC + M + (m - 1)         # noqa: E731
    thermo_nb = lambda m: CH_BASE_VEC + 2 * M + (m - 1)  # noqa: E731
    a_one = lambda m: CH_BASE_VEC + 3 * M + (m - 1)      # noqa: E731

    with torch.no_grad():
        # =====================================================================
        # LAYER 0 ATTENTION (6 heads)
        # =====================================================================
        # H0, H1, H2: seen '+', seen '-', seen '=' (strictly causal)
        for h, (src, dst) in enumerate((
            (CH_PLUS, CH_SEEN_PLUS),
            (CH_MINUS, CH_SEEN_MINUS),
            (CH_EQ, CH_SEEN_EQ),
        )):
            organ.Q[0, h, CH_CONST, 0] = 40.0 * s
            organ.K[0, h, src, 0] = 1.0
            organ.V[0, h, src, 0] = 1.0
            organ.O[0, h, 0, dst] = 1.0

        # H3: fetch a_p (same place p AND CH_SEG_A), fallback to CH_OTHER sink (val 0)
        # H4: fetch b_p (same place p AND CH_SEG_B), fallback to CH_OTHER sink (val 0)
        beta = 60.0
        for h, seg_ch, dst_ch in ((3, CH_SEG_A, CH_VAL_A), (4, CH_SEG_B, CH_VAL_B)):
            for m in range(1, M + 1):
                organ.Q[0, h, one(m), m - 1] = beta * s
                organ.K[0, h, one(m), m - 1] = 1.0
            organ.Q[0, h, CH_CONST, M] = beta * s
            organ.K[0, h, seg_ch, M] = 1.0
            organ.Q[0, h, CH_CONST, M + 1] = 1.5 * beta * s
            organ.K[0, h, CH_OTHER, M + 1] = 1.0
            for d in range(1, 10):
                organ.V[0, h, CH_DIG + d, 0] = float(d)
            organ.O[0, h, 0, dst_ch] = 1.0

        # H5: copy thermo(n_b) from earliest CH_SEG_B token (b_{n_b}) -> CH_THERMO_NB
        alpha_nb = 20.0
        big_nb = alpha_nb * 512.0 + 400.0
        organ.alibi[0, 5] = alpha_nb
        organ.Q[0, 5, CH_CONST, 0] = big_nb * s
        organ.K[0, 5, CH_SEG_B, 0] = 1.0
        for m in range(1, M + 1):
            organ.V[0, 5, thermo(m), m - 1] = 1.0
            organ.O[0, 5, m - 1, thermo_nb(m)] = 1.0

        # =====================================================================
        # LAYER 0 MLP: G/K flags at b_p, K_a flags at a_p, and a_one(m)
        # =====================================================================
        def set_b_pair(n_lo: int, n_hi: int, op_ch: int, a_sign: float, b_sign: float,
                       t_lo: float, t_hi: float, dst_ch: int) -> None:
            """Compute 2*relu(X - t_lo) - 2*relu(X - t_hi) gated on (CH_SEG_B & op_ch)."""
            for n_idx, thr, w2_val in ((n_lo, t_lo, +2.0), (n_hi, t_hi, -2.0)):
                for d in range(1, 10):
                    organ.W1[0, CH_DIG + d, n_idx] = b_sign * float(d)
                organ.W1[0, CH_VAL_A, n_idx] = a_sign
                organ.W1[0, CH_SEG_B, n_idx] = 50.0
                organ.W1[0, op_ch, n_idx] = 50.0
                organ.b1[0, n_idx] = -thr - 100.0
                organ.W2[0, n_idx, dst_ch] = w2_val

        # G_add: a_p + b_p >= 10  <=>  (a+b) > 9.5
        set_b_pair(0, 1, CH_SEEN_PLUS, +1.0, +1.0, 9.25, 9.75, CH_G_ADD)
        # K_add: a_p + b_p <= 8   <=> -(a+b) > -8.5
        set_b_pair(2, 3, CH_SEEN_PLUS, -1.0, -1.0, -8.75, -8.25, CH_K_ADD)
        # G_sub: a_p - b_p <= -1  <=> (b-a) > 0.5
        set_b_pair(4, 5, CH_SEEN_MINUS, -1.0, +1.0, 0.25, 0.75, CH_G_SUB)
        # K_sub: a_p - b_p >= +1  <=> (a-b) > 0.5
        set_b_pair(6, 7, CH_SEEN_MINUS, +1.0, -1.0, 0.25, 0.75, CH_K_SUB)

        # At a_p (for p > n_b where b_p = 0):
        # K_add_a: is_a & (a_p <= 8)  <=> -a_p > -8.5
        for n_idx, thr, w2_val in ((8, -8.75, +2.0), (9, -8.25, -2.0)):
            for d in range(1, 10):
                organ.W1[0, CH_DIG + d, n_idx] = -float(d)
            organ.W1[0, CH_SEG_A, n_idx] = 50.0
            organ.b1[0, n_idx] = -thr - 50.0
            organ.W2[0, n_idx, CH_K_ADD_A] = w2_val

        # K_sub_a: is_a & (a_p >= 1)  <=> a_p > 0.5
        for n_idx, thr, w2_val in ((10, 0.25, +2.0), (11, 0.75, -2.0)):
            for d in range(1, 10):
                organ.W1[0, CH_DIG + d, n_idx] = +float(d)
            organ.W1[0, CH_SEG_A, n_idx] = 50.0
            organ.b1[0, n_idx] = -thr - 50.0
            organ.W2[0, n_idx, CH_K_SUB_A] = w2_val

        # a_one(m) = relu(one(m) + CH_SEG_A - 1) for m = 1..M
        for m in range(1, M + 1):
            n_idx = 12 + (m - 1)
            organ.W1[0, one(m), n_idx] = 1.0
            organ.W1[0, CH_SEG_A, n_idx] = 1.0
            organ.b1[0, n_idx] = -1.0
            organ.W2[0, n_idx, a_one(m)] = 1.0

        # =====================================================================
        # LAYER 1 ATTENTION: Carry / Borrow Lookahead & Sign Comparison
        # =====================================================================
        alpha_p = 20.0
        big = alpha_p * float(M) * 3.0 + 1000.0  # >> alpha_p * M

        def wire_lookahead(h: int, elig_chs: Sequence[int], require_lt_p: bool) -> None:
            # Slot 0: eligibility (+big)
            organ.Q[1, h, CH_CONST, 0] = big * s
            for ch in elig_chs:
                organ.K[1, h, ch, 0] = 1.0
            # Slot 1: place-value tie-breaker (+alpha_p * j, prefers largest j)
            organ.Q[1, h, CH_CONST, 1] = alpha_p * s
            for m in range(1, M + 1):
                organ.K[1, h, one(m), 1] = float(m)
            # Slot 2: sink CH_OTHER (+1.5*big if require_lt_p else +0.5*big)
            organ.Q[1, h, CH_CONST, 2] = (1.5 if require_lt_p else 0.5) * big * s
            organ.K[1, h, CH_OTHER, 2] = 1.0
            # Slots 3..3+M-1: +big * [j < p] = +big * (thermo(p)[j] - one(p)[j])
            if require_lt_p:
                for m in range(1, M + 1):
                    organ.Q[1, h, thermo(m), 3 + (m - 1)] = +big * s
                    organ.Q[1, h, one(m), 3 + (m - 1)] = -big * s
                    organ.K[1, h, one(m), 3 + (m - 1)] = 1.0
            # Slots 3+M..3+2M-1: -2*big * [is_a(j) AND j <= n_b]
            for m in range(1, M + 1):
                organ.Q[1, h, thermo_nb(m), 3 + M + (m - 1)] = -2.0 * big * s
                organ.K[1, h, a_one(m), 3 + M + (m - 1)] = 1.0

        # L1.H0: Addition carry lookahead -> CH_C_ADD
        wire_lookahead(0, (CH_G_ADD, CH_K_ADD, CH_K_ADD_A), require_lt_p=True)
        organ.V[1, 0, CH_G_ADD, 0] = 1.0
        organ.O[1, 0, 0, CH_C_ADD] = 1.0

        # L1.H1: Subtraction borrow lookahead -> CH_BOR_AB (slot 0) and CH_BOR_BA (slot 1)
        wire_lookahead(1, (CH_G_SUB, CH_K_SUB, CH_K_SUB_A), require_lt_p=True)
        organ.V[1, 1, CH_G_SUB, 0] = 1.0
        organ.O[1, 1, 0, CH_BOR_AB] = 1.0
        organ.V[1, 1, CH_K_SUB, 1] = 1.0
        organ.V[1, 1, CH_K_SUB_A, 1] = 1.0
        organ.O[1, 1, 1, CH_BOR_BA] = 1.0

        # L1.H2: Global sign comparison (a < b) -> CH_IS_NEG (no j < p mask)
        wire_lookahead(2, (CH_G_SUB, CH_K_SUB, CH_K_SUB_A), require_lt_p=False)
        organ.V[1, 2, CH_G_SUB, 0] = 1.0
        organ.O[1, 2, 0, CH_IS_NEG] = 1.0

        # =====================================================================
        # LAYER 1 MLP: Exact 11-way output ('0'..'9', '-') + structural CH_GATE
        # =====================================================================
        n_ptr = 0

        # Branch 1 (60 neurons): Addition (a_p + b_p + c_add) mod 10
        # Active iff (CH_EQ + CH_SEEN_EQ) == 1 AND thermo(1) == 1 AND CH_SEEN_PLUS == 1
        for t_val in range(20):
            for off, coeff in ((-0.5, +2.0), (0.0, -4.0), (+0.5, +2.0)):
                organ.W1[1, CH_VAL_A, n_ptr] = +1.0
                organ.W1[1, CH_VAL_B, n_ptr] = +1.0
                organ.W1[1, CH_C_ADD, n_ptr] = +1.0
                organ.W1[1, CH_EQ, n_ptr] = 50.0
                organ.W1[1, CH_SEEN_EQ, n_ptr] = 50.0
                organ.W1[1, thermo(1), n_ptr] = 50.0
                organ.W1[1, CH_SEEN_PLUS, n_ptr] = 50.0
                organ.b1[1, n_ptr] = -(float(t_val) + off) - 150.0
                organ.W2[1, n_ptr, CH_OUT + (t_val % 10)] = coeff
                n_ptr += 1

        # Branch 2 (60 neurons): Non-negative subtraction (a_p - b_p - bor_ab + 10) mod 10
        # Active iff (CH_EQ + CH_SEEN_EQ) == 1 AND thermo(1) == 1 AND CH_SEEN_MINUS == 1 AND CH_IS_NEG == 0
        for t_val in range(20):
            for off, coeff in ((-0.5, +2.0), (0.0, -4.0), (+0.5, +2.0)):
                organ.W1[1, CH_VAL_A, n_ptr] = +1.0
                organ.W1[1, CH_VAL_B, n_ptr] = -1.0
                organ.W1[1, CH_BOR_AB, n_ptr] = -1.0
                organ.W1[1, CH_EQ, n_ptr] = 50.0
                organ.W1[1, CH_SEEN_EQ, n_ptr] = 50.0
                organ.W1[1, thermo(1), n_ptr] = 50.0
                organ.W1[1, CH_SEEN_MINUS, n_ptr] = 50.0
                organ.W1[1, CH_IS_NEG, n_ptr] = -50.0
                organ.b1[1, n_ptr] = 10.0 - (float(t_val) + off) - 150.0
                organ.W2[1, n_ptr, CH_OUT + (t_val % 10)] = coeff
                n_ptr += 1

        # Branch 3 (60 neurons): Negative subtraction (b_p - a_p - bor_ba + 10) mod 10
        # Active iff CH_SEEN_EQ == 1 (after '-') AND thermo(1) == 1 AND CH_SEEN_MINUS == 1 AND CH_IS_NEG == 1
        for t_val in range(20):
            for off, coeff in ((-0.5, +2.0), (0.0, -4.0), (+0.5, +2.0)):
                organ.W1[1, CH_VAL_B, n_ptr] = +1.0
                organ.W1[1, CH_VAL_A, n_ptr] = -1.0
                organ.W1[1, CH_BOR_BA, n_ptr] = -1.0
                organ.W1[1, CH_SEEN_EQ, n_ptr] = 50.0
                organ.W1[1, thermo(1), n_ptr] = 50.0
                organ.W1[1, CH_SEEN_MINUS, n_ptr] = 50.0
                organ.W1[1, CH_IS_NEG, n_ptr] = 50.0
                organ.b1[1, n_ptr] = 10.0 - (float(t_val) + off) - 200.0
                organ.W2[1, n_ptr, CH_OUT + (t_val % 10)] = coeff
                n_ptr += 1

        # Branch 4 (1 neuron): Emit '-' at '=' iff CH_EQ & CH_SEEN_MINUS & CH_IS_NEG & thermo(1)
        organ.W1[1, CH_EQ, n_ptr] = 1.0
        organ.W1[1, CH_SEEN_MINUS, n_ptr] = 1.0
        organ.W1[1, CH_IS_NEG, n_ptr] = 1.0
        organ.W1[1, thermo(1), n_ptr] = 1.0
        organ.b1[1, n_ptr] = -3.0
        organ.W2[1, n_ptr, CH_OUT + 10] = 1.0
        n_ptr += 1

        # Structural gate (1 neuron): CH_GATE = relu((CH_EQ + CH_SEEN_EQ) + thermo(1) - 1)
        organ.W1[1, CH_EQ, n_ptr] = 1.0
        organ.W1[1, CH_SEEN_EQ, n_ptr] = 1.0
        organ.W1[1, thermo(1), n_ptr] = 1.0
        organ.b1[1, n_ptr] = -1.0
        organ.W2[1, n_ptr, CH_GATE] = 1.0
        n_ptr += 1

        # Readout: crisp margin 6.0 on 11 symbol channels, 1.0 on structural gate
        for c in range(N_SYM_OUT):
            organ.OUT_W[CH_OUT + c, c] = 6.0
        organ.OUT_W[CH_GATE, 11] = 1.0

    return organ


# =============================================================================
# ACTIVE EXPRESSION PARSER & BUS BUILDERS
# =============================================================================

@dataclass
class ActiveExprSpan:
    """Span of a binary integer expression `a +/- b =` inside a token sequence."""
    start_idx: int          # index of first digit of a
    a_positions: List[int]  # indices of digits of a (MSB..LSB)
    op_idx: int             # index of '+' or '-'
    op: str                 # '+' or '-'
    b_positions: List[int]  # indices of digits of b (MSB..LSB)
    eq_idx: int             # index of '='


def parse_active_expr_symbols(
    sym_types: Sequence[str],
    eq_pos: int,
    max_idx: int = 64,
) -> Optional[ActiveExprSpan]:
    """Parse backwards from `eq_pos` (which must have sym_types[eq_pos] == '=') to check if
    an active binary integer expression `a +/- b =` immediately precedes `eq_pos`.

    sym_types elements are in {'0'..'9', '+', '-', '=', 'SPACE', 'COMMA', 'OTHER'}.
    Spaces ('SPACE') are allowed around '+', '-', '='.
    Thousands-separator commas ('COMMA') are allowed inside integer operands when flanked by
    digits with exact 3-digit groups (e.g. `325,263,952`); commas are excluded from
    `a_positions`/`b_positions` so right-aligned digit place values `p = 1..n` ignore commas.
    Rejects chained ops (e.g. `1 + 2 + 3 =`), decimals (`.`), malformed commas (`,123`, `3,5`,
    `123,`), variables (`x + 5 =`), multiplication/division (`*`, `/`), or operands exceeding
    `max_idx` digits.
    """
    if eq_pos < 0 or eq_pos >= len(sym_types) or sym_types[eq_pos] != "=":
        return None

    def is_dig(s: str) -> bool:
        return len(s) == 1 and s.isdigit()

    def parse_int_rtl(idx: int) -> Tuple[List[int], int]:
        """Parse an integer operand (with optional digit-flanked commas, including mis-grouped
        commas like `6,286,917,586,52`) right-to-left.
        Returns (digit_positions_msb_to_lsb, next_idx_left_of_operand). Empty list on failure.
        """
        pos_list: List[int] = []
        while idx >= 0:
            s = sym_types[idx]
            if is_dig(s):
                pos_list.append(idx)
                idx -= 1
            elif s == "COMMA":
                # Valid digit-flanked comma: preceded by a digit (idx-1) and followed by a digit
                # (pos_list non-empty; consecutive ',,' rejected since sym_types[idx-1] must be digit)
                if pos_list and idx - 1 >= 0 and is_dig(sym_types[idx - 1]):
                    idx -= 1
                else:
                    return [], idx
            else:
                break
        if not pos_list or len(pos_list) > max_idx:
            return [], idx
        pos_list.reverse()
        return pos_list, idx

    i = eq_pos - 1
    while i >= 0 and sym_types[i] == "SPACE":
        i -= 1
    # Parse operand b (digits right-to-left)
    b_pos, i = parse_int_rtl(i)
    if not b_pos:
        return None

    while i >= 0 and sym_types[i] == "SPACE":
        i -= 1
    if i < 0 or sym_types[i] not in ("+", "-"):
        return None
    op_idx = i
    op = sym_types[i]
    i -= 1

    while i >= 0 and sym_types[i] == "SPACE":
        i -= 1
    # Parse operand a (digits right-to-left)
    a_pos, i = parse_int_rtl(i)
    if not a_pos:
        return None

    # Reject a stray comma immediately touching the left of operand `a`
    if i >= 0 and sym_types[i] == "COMMA":
        return None

    # Keep `raw_start_idx` before stripping leading zeros so boundary checks inspect the token
    # before the first '0' (e.g. still rejecting `1.005 + 3 =` or `x005 + 3 =`).
    # Strip leading '0' positions from `a_pos` and `b_pos` while keeping at least the final digit
    # (so "000" -> "0", "005" -> "5", "00199034677" -> "199034677"). Stripped leading-zero
    # positions receive seg=0, place=0 and are ignored cleanly by the organ.
    raw_start_idx = a_pos[0]
    while len(a_pos) > 1 and sym_types[a_pos[0]] == "0":
        a_pos.pop(0)
    while len(b_pos) > 1 and sym_types[b_pos[0]] == "0":
        b_pos.pop(0)

    return ActiveExprSpan(
        start_idx=raw_start_idx,
        a_positions=a_pos,
        op_idx=op_idx,
        op=op,
        b_positions=b_pos,
        eq_idx=eq_pos,
    )


def place_features(places: torch.Tensor, M: int) -> torch.Tensor:
    """places: LongTensor [B, T] in 0..M -> [B, T, 2*M] (one-hot 1..M, thermometer 1..M)."""
    c = places.clamp(0, M).unsqueeze(-1)
    m = torch.arange(1, M + 1, device=places.device)
    return torch.cat([(c == m).float(), (m <= c).float()], dim=-1)


def build_window_structural_features(
    T: int,
    a_local: Sequence[int],
    b_local: Sequence[int],
    eq_local: int,
    answer_target_places: Sequence[int],
    max_idx: int,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (seg_a [T], seg_b [T], places [T]) for a single active expression window.

    `answer_target_places` gives the target place `p` for positions `eq_local, eq_local+1, ...`
    (length = T - eq_local).
    """
    M = calc_internal_places(max_idx)
    seg_a = torch.zeros(T, dtype=torch.float32, device=device)
    seg_b = torch.zeros(T, dtype=torch.float32, device=device)
    places = torch.zeros(T, dtype=torch.long, device=device)

    n_a = len(a_local)
    for k, pos in enumerate(a_local):
        seg_a[pos] = 1.0
        places[pos] = n_a - k  # MSB=n_a .. LSB=1

    n_b = len(b_local)
    for k, pos in enumerate(b_local):
        seg_b[pos] = 1.0
        places[pos] = n_b - k  # MSB=n_b .. LSB=1

    for offset, p_val in enumerate(answer_target_places):
        idx = eq_local + offset
        if 0 <= idx < T:
            places[idx] = min(max(0, int(p_val)), M)

    return seg_a, seg_b, places


def assemble_bus(
    z_read: torch.Tensor,
    seg_a: torch.Tensor,
    seg_b: torch.Tensor,
    places: torch.Tensor,
    max_idx: int,
) -> torch.Tensor:
    """Assemble full [B, T, d_model] organ input bus from read-in [B, T, N_READ] and structural inputs."""
    B, T, _ = z_read.shape
    M = calc_internal_places(max_idx)
    d_model = calc_bus_dim(max_idx)
    x = torch.zeros(B, T, d_model, dtype=torch.float32, device=z_read.device)
    x[:, :, :N_READ] = z_read
    x[:, :, CH_SEG_A] = seg_a
    x[:, :, CH_SEG_B] = seg_b
    x[:, :, CH_BASE_VEC:CH_BASE_VEC + 2 * M] = place_features(places, M)
    return x


@torch.no_grad()
def determine_p_top_neural(
    organ: SparseOrgan,
    z_expr: torch.Tensor,
    a_local: Sequence[int],
    b_local: Sequence[int],
    eq_local: int,
    op: str,
    max_idx: int,
) -> Tuple[int, bool]:
    """Pure-neural determination of (p_top, is_neg) using ONLY `organ.forward_states` (zero Python math!).

    - For '+': queries organ at `eq_local` with `p = max(n_a, n_b) + 1`. If organ predicts digit > 0
      (i.e. carry-out '1'), p_top = max_n + 1; else p_top = max_n (or 1 if max_n == 0).
    - For '-': first queries `eq_local` with `p = 1` to check whether organ emits '-' (is_neg).
      Then queries all candidate places `p = max_n..1` in a single batched `organ.forward_states`
      call (with a trailing '-' token appended if `is_neg`, so Branch 3 computes `b - a`) and
      picks the highest `p` whose predicted digit is non-zero (or `1` if all zero, i.e. `a == b`).
    """
    dev = z_expr.device
    T_expr = z_expr.shape[1]
    n_a, n_b = len(a_local), len(b_local)
    max_n = max(n_a, n_b)
    M = calc_internal_places(max_idx)

    if op == "+":
        cand_p = min(max_n + 1, M)
        seg_a, seg_b, places = build_window_structural_features(
            T_expr, a_local, b_local, eq_local, [cand_p], max_idx, device=dev
        )
        x = assemble_bus(z_expr, seg_a.unsqueeze(0), seg_b.unsqueeze(0), places.unsqueeze(0), max_idx)
        out = organ.forward_states(x)[0, eq_local, :N_SYM_OUT]
        top_dig = int(out.argmax().item())
        p_top = cand_p if top_dig > 0 else max(1, max_n)
        return p_top, False

    # Subtraction ('-'): first check if organ emits '-' at '='
    seg_a, seg_b, places = build_window_structural_features(
        T_expr, a_local, b_local, eq_local, [1], max_idx, device=dev
    )
    x0 = assemble_bus(z_expr, seg_a.unsqueeze(0), seg_b.unsqueeze(0), places.unsqueeze(0), max_idx)
    sym0 = int(organ.forward_states(x0)[0, eq_local, :N_SYM_OUT].argmax().item())
    is_neg = (sym0 == 10)

    # Batched query across p = 1..max_n using organ.forward_states
    if is_neg:
        # Append a synthetic '-' read-in token after '=' so Branch 3 computes digits of (b - a)
        minus_tok = torch.zeros(1, 1, N_READ, dtype=torch.float32, device=dev)
        minus_tok[0, 0, CH_MINUS] = 1.0
        minus_tok[0, 0, CH_CONST] = 1.0
        z_q = torch.cat([z_expr, minus_tok], dim=1)  # [1, T_expr + 1, N_READ]
        query_pos = eq_local + 1
    else:
        z_q = z_expr
        query_pos = eq_local

    Tq = z_q.shape[1]
    z_batch = z_q.expand(max_n, Tq, N_READ)
    seg_a_b = torch.zeros(max_n, Tq, dtype=torch.float32, device=dev)
    seg_b_b = torch.zeros(max_n, Tq, dtype=torch.float32, device=dev)
    places_b = torch.zeros(max_n, Tq, dtype=torch.long, device=dev)

    for k, pos in enumerate(a_local):
        seg_a_b[:, pos] = 1.0
        places_b[:, pos] = n_a - k
    for k, pos in enumerate(b_local):
        seg_b_b[:, pos] = 1.0
        places_b[:, pos] = n_b - k
    for b_idx in range(max_n):
        p_val = b_idx + 1  # 1..max_n
        places_b[b_idx, query_pos] = p_val
        if is_neg:
            places_b[b_idx, eq_local] = 1

    xb = assemble_bus(z_batch, seg_a_b, seg_b_b, places_b, max_idx)
    logits = organ.forward_states(xb)[:, query_pos, :10]  # [max_n, 10]
    preds = logits.argmax(dim=-1).tolist()  # index b_idx corresponds to place p = b_idx + 1

    p_top = 1
    for p_val in range(max_n, 0, -1):
        if preds[p_val - 1] != 0:
            p_top = p_val
            break
    return p_top, is_neg


# =============================================================================
# STANDALONE PURE-NEURAL EXECUTION HELPER (CPU/GPU)
# =============================================================================

SYM_TO_CHAR = [str(d) for d in range(10)] + ["-"]


def _chars_to_readin(chars: Sequence[str], device: torch.device | str = "cpu") -> torch.Tensor:
    """Exact one-hot read-in [1, T, N_READ] for standalone CPU/GPU testing."""
    T = len(chars)
    z = torch.zeros(1, T, N_READ, dtype=torch.float32, device=device)
    for t, ch in enumerate(chars):
        if len(ch) == 1 and ch.isdigit():
            z[0, t, CH_DIG + int(ch)] = 1.0
        elif ch == "+":
            z[0, t, CH_PLUS] = 1.0
        elif ch == "-":
            z[0, t, CH_MINUS] = 1.0
        elif ch == "=":
            z[0, t, CH_EQ] = 1.0
        else:
            z[0, t, CH_OTHER] = 1.0
    z[0, :, CH_CONST] = 1.0
    return z


@torch.no_grad()
def run_organ_standalone(
    expr_str: str,
    organ: Optional[SparseOrgan] = None,
    max_idx: int = 64,
    return_details: bool = False,
):
    """Run the compiled 2-layer SparseOrgan autoregressively on `expr_str` (e.g.
    "Prefix 12 + 34 = 46. Now 48219384721 - 39571284905 =").

    Slices the active binary expression window ending at the final '=', finds (p_top, is_neg)
    purely via `organ.forward_states`, and generates answer characters autoregressively until
    the organ's own structural gate `CH_GATE` shuts off (<= 0.5).
    """
    if organ is None:
        organ = build_varlen_calc_organ(max_idx=max_idx, device="cpu")
    dev = organ.Q.device
    M_idx = getattr(organ, "max_idx", max_idx)

    chars = list(expr_str)
    eq_pos = len(chars) - 1
    while eq_pos >= 0 and chars[eq_pos] == " ":
        eq_pos -= 1
    if eq_pos < 0 or chars[eq_pos] != "=":
        raise ValueError(f"Expression must end with '=': {expr_str!r}")

    sym_types = [
        c if (c.isdigit() or c in "+-=") else ("SPACE" if c in (" ", "\t", "\n", "\r") else ("COMMA" if c == "," else "OTHER"))
        for c in chars[:eq_pos + 1]
    ]
    span = parse_active_expr_symbols(sym_types, eq_pos, max_idx=M_idx)
    if span is None:
        raise ValueError(f"Could not parse active binary integer expression before '=': {expr_str!r}")

    # Slice active window and prepend a guaranteed CH_OTHER sink at position 0
    win_chars = ["<SINK>"] + chars[span.start_idx:span.eq_idx + 1]
    offset = 1 - span.start_idx
    a_local = [p + offset for p in span.a_positions]
    b_local = [p + offset for p in span.b_positions]
    eq_local = span.eq_idx + offset

    z_expr = _chars_to_readin(win_chars, device=dev)
    p_top, is_neg = determine_p_top_neural(organ, z_expr, a_local, b_local, eq_local, span.op, M_idx)

    # Autoregressive decoding driven purely by organ outputs + organ structural gate
    seq_chars = list(win_chars)
    target_places = [p_top]
    emitted: List[str] = []
    gates: List[float] = []
    min_margin = float("inf")

    for _ in range(M_idx + 4):
        z_cur = _chars_to_readin(seq_chars, device=dev)
        seg_a, seg_b, places = build_window_structural_features(
            len(seq_chars), a_local, b_local, eq_local, target_places, M_idx, device=dev
        )
        x = assemble_bus(z_cur, seg_a.unsqueeze(0), seg_b.unsqueeze(0), places.unsqueeze(0), M_idx)
        out = organ.forward_states(x)[0, -1]
        gate_val = float(out[11].clamp(0.0, 1.0).item())
        gates.append(gate_val)
        if gate_val <= 0.5:
            break
        sym_logits = out[:N_SYM_OUT]
        top2 = sym_logits.topk(2).values
        min_margin = min(min_margin, float((top2[0] - top2[1]).item()))
        sym = int(sym_logits.argmax().item())
        ch = SYM_TO_CHAR[sym]
        emitted.append(ch)
        seq_chars.append(ch)
        if ch == "-":
            # Emitted minus sign at '='; next token still targets p_top
            target_places.append(p_top)
        else:
            # Emitted digit at current target_p; next token targets target_p - 1 (0 -> gate shuts off)
            target_places.append(target_places[-1] - 1)

    ans_str = "".join(emitted)
    if return_details:
        return {
            "answer": ans_str,
            "p_top": p_top,
            "is_neg": is_neg,
            "gates": gates,
            "min_margin": min_margin,
        }
    return ans_str
