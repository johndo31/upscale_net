"""Minimal standard PyTorch 2-layer Transformer + sparse weight parser/serializer.

Architecture (no LayerNorm, standard scaled dot-product softmax attention + ReLU MLP):
  Vocab V = 6:
    0, 1, 2, 3 : data symbols
    4          : BOS (always at position 0)
    5          : EOS (always at final position L+1; output logits read here)

  Fixed transparent input embedding E (6 x 16):
    ch 0..3 : one-hot data token (0..3)
    ch 4    : is_BOS
    ch 5    : is_EOS
    ch 6    : constant 1.0 (bias channel)
    ch 7..15: 9 zero-initialized residual working channels

  2 layers (l = 0, 1), each with:
    - 2 strictly-causal attention heads (h = 0, 1), d_head = 4, per-head scalar ALiBi slope:
        scores[i, j] = (Q_i . K_j) / sqrt(d_head) + alibi * (i - j)
        mask: j < i for i > 0; j == 0 for i == 0 (so BOS has valid softmax)
        x = x + sum_h (softmax(scores_h) @ V_h @ O_h)
    - ReLU MLP (d_mlp = 8):
        x = x + relu(x @ W1 + b1) @ W2

  Readout at EOS position (last token):
    logits = x[:, -1, :] @ OUT_W + OUT_b   (shape [B, 4])
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 6
BOS_ID = 4
EOS_ID = 5

D_MODEL = 16
N_LAYERS = 2
N_HEADS = 2
D_HEAD = 4
D_MLP = 8
N_OUT = 4


def make_fixed_embedding(device=None, dtype=torch.float32) -> torch.Tensor:
    E = torch.zeros(VOCAB_SIZE, D_MODEL, device=device, dtype=dtype)
    for t in range(VOCAB_SIZE):
        E[t, t] = 1.0
    E[:, 6] = 1.0  # constant bias channel
    return E


class TinyTransformer(nn.Module):
    def __init__(self, learnable_embed: bool = False):
        super().__init__()
        E = make_fixed_embedding()
        if learnable_embed:
            self.embed = nn.Parameter(E)
        else:
            self.register_buffer("embed", E)

        self.Q = nn.Parameter(torch.zeros(N_LAYERS, N_HEADS, D_MODEL, D_HEAD))
        self.K = nn.Parameter(torch.zeros(N_LAYERS, N_HEADS, D_MODEL, D_HEAD))
        self.V = nn.Parameter(torch.zeros(N_LAYERS, N_HEADS, D_MODEL, D_HEAD))
        self.O = nn.Parameter(torch.zeros(N_LAYERS, N_HEADS, D_HEAD, D_MODEL))
        self.alibi = nn.Parameter(torch.zeros(N_LAYERS, N_HEADS))

        self.W1 = nn.Parameter(torch.zeros(N_LAYERS, D_MODEL, D_MLP))
        self.b1 = nn.Parameter(torch.zeros(N_LAYERS, D_MLP))
        self.W2 = nn.Parameter(torch.zeros(N_LAYERS, D_MLP, D_MODEL))

        self.OUT_W = nn.Parameter(torch.zeros(D_MODEL, N_OUT))
        self.OUT_b = nn.Parameter(torch.zeros(N_OUT))

    def init_random_for_sgd(self, seed: int = 0) -> None:
        g = torch.Generator(device=self.Q.device).manual_seed(seed)
        with torch.no_grad():
            for p in (self.Q, self.K, self.V, self.O, self.W1, self.W2, self.OUT_W):
                nn.init.normal_(p, mean=0.0, std=0.2, generator=g)
            nn.init.zeros_(self.b1)
            nn.init.zeros_(self.OUT_b)
            # Standard spread of initial ALiBi slopes (both recency and early-position bias)
            self.alibi.copy_(torch.tensor([[-1.0, 0.0], [-0.5, 0.5]], device=self.alibi.device))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, T] int64 -> logits at final position: [B, 4]."""
        B, T = tokens.shape
        device = tokens.device
        x = F.embedding(tokens, self.embed)  # [B, T, D_MODEL]

        pos = torch.arange(T, device=device)
        dist = (pos[:, None] - pos[None, :]).to(x.dtype)  # [T, T], i - j >= 0 for j <= i

        # Strictly causal mask: j < i for i > 0, and j == 0 for i == 0
        mask = pos[None, :] < pos[:, None]  # [T, T]
        mask[0, 0] = True
        attn_mask = torch.where(mask, torch.zeros_like(dist), torch.full_like(dist, float("-inf")))

        scale = 1.0 / math.sqrt(D_HEAD)

        for l in range(N_LAYERS):
            # Attention
            attn_delta = torch.zeros_like(x)
            for h in range(N_HEADS):
                q = x @ self.Q[l, h]  # [B, T, D_HEAD]
                k = x @ self.K[l, h]  # [B, T, D_HEAD]
                v = x @ self.V[l, h]  # [B, T, D_HEAD]
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                scores = scores + self.alibi[l, h] * dist.unsqueeze(0) + attn_mask.unsqueeze(0)
                probs = F.softmax(scores, dim=-1)
                head_out = torch.matmul(probs, v) @ self.O[l, h]
                attn_delta = attn_delta + head_out
            x = x + attn_delta

            # MLP
            h_mlp = F.relu(x @ self.W1[l] + self.b1[l])
            x = x + h_mlp @ self.W2[l]

        logits = x[:, -1, :] @ self.OUT_W + self.OUT_b
        return logits

    def count_nonzero(self) -> int:
        total = 0
        for p in (self.Q, self.K, self.V, self.O, self.alibi, self.W1, self.b1, self.W2, self.OUT_W, self.OUT_b):
            total += int((p.detach() != 0).sum().item())
        return total


# Regex patterns for sparse weight lines
_RE_ALIBI = re.compile(r"^L([01])\.H([01])\.alibi\s*=\s*([-+]?\d+(?:\.\d+)?)$")
_RE_HEAD = re.compile(r"^L([01])\.H([01])\.([QKVO])\[(\d+),(\d+)\]\s*=\s*([-+]?\d+(?:\.\d+)?)$")
_RE_MLP_W = re.compile(r"^L([01])\.M\.(W1|W2)\[(\d+),(\d+)\]\s*=\s*([-+]?\d+(?:\.\d+)?)$")
_RE_MLP_B = re.compile(r"^L([01])\.M\.b1\[(\d+)\]\s*=\s*([-+]?\d+(?:\.\d+)?)$")
_RE_OUT_W = re.compile(r"^OUT\.W\[(\d+),(\d+)\]\s*=\s*([-+]?\d+(?:\.\d+)?)$")
_RE_OUT_B = re.compile(r"^OUT\.b\[(\d+)\]\s*=\s*([-+]?\d+(?:\.\d+)?)$")


@dataclass
class ParseResult:
    model: Optional[TinyTransformer]
    n_entries: int
    error: Optional[str] = None


def parse_sparse_weights(
    text: str,
    max_abs_val: float = 1000.0,
    max_entries: int = 256,
    device=None,
) -> ParseResult:
    """Parse sparse weight lines into a TinyTransformer.

    Accepts optional markdown/reasoning before a `WEIGHTS:` block (if present),
    or raw assignment lines directly. Ignores blank lines and `#` comments inside
    the weights section; rejects invalid indices or out-of-range numbers.
    """
    if "WEIGHTS:" in text:
        body = text.split("WEIGHTS:", 1)[1]
    else:
        body = text

    # Strip fenced code markers if present
    lines = []
    for raw in body.splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or s.startswith("```"):
            continue
        if s.upper().startswith("END"):
            break
        # Strip inline comment
        if "#" in s:
            s = s.split("#", 1)[0].strip()
        if s:
            lines.append(s)

    if not lines:
        return ParseResult(None, 0, "no weight lines found")

    model = TinyTransformer(learnable_embed=False)
    n_entries = 0

    with torch.no_grad():
        for line in lines:
            if n_entries >= max_entries:
                return ParseResult(None, n_entries, f"exceeded max_entries={max_entries}")

            m = _RE_ALIBI.match(line)
            if m:
                l, h, val = int(m.group(1)), int(m.group(2)), float(m.group(3))
                if not math.isfinite(val) or abs(val) > max_abs_val:
                    return ParseResult(None, n_entries, f"value out of range: {line}")
                model.alibi[l, h] = val
                n_entries += 1
                continue

            m = _RE_HEAD.match(line)
            if m:
                l, h, kind = int(m.group(1)), int(m.group(2)), m.group(3)
                r, c, val = int(m.group(4)), int(m.group(5)), float(m.group(6))
                if not math.isfinite(val) or abs(val) > max_abs_val:
                    return ParseResult(None, n_entries, f"value out of range: {line}")
                tensor = getattr(model, kind)[l, h]
                if not (0 <= r < tensor.shape[0] and 0 <= c < tensor.shape[1]):
                    return ParseResult(None, n_entries, f"index out of bounds: {line}")
                tensor[r, c] = val
                n_entries += 1
                continue

            m = _RE_MLP_W.match(line)
            if m:
                l, kind = int(m.group(1)), m.group(2)
                r, c, val = int(m.group(3)), int(m.group(4)), float(m.group(5))
                if not math.isfinite(val) or abs(val) > max_abs_val:
                    return ParseResult(None, n_entries, f"value out of range: {line}")
                tensor = getattr(model, kind)[l]
                if not (0 <= r < tensor.shape[0] and 0 <= c < tensor.shape[1]):
                    return ParseResult(None, n_entries, f"index out of bounds: {line}")
                tensor[r, c] = val
                n_entries += 1
                continue

            m = _RE_MLP_B.match(line)
            if m:
                l, c, val = int(m.group(1)), int(m.group(2)), float(m.group(3))
                if not math.isfinite(val) or abs(val) > max_abs_val:
                    return ParseResult(None, n_entries, f"value out of range: {line}")
                if not (0 <= c < D_MLP):
                    return ParseResult(None, n_entries, f"index out of bounds: {line}")
                model.b1[l, c] = val
                n_entries += 1
                continue

            m = _RE_OUT_W.match(line)
            if m:
                r, c, val = int(m.group(1)), int(m.group(2)), float(m.group(3))
                if not math.isfinite(val) or abs(val) > max_abs_val:
                    return ParseResult(None, n_entries, f"value out of range: {line}")
                if not (0 <= r < D_MODEL and 0 <= c < N_OUT):
                    return ParseResult(None, n_entries, f"index out of bounds: {line}")
                model.OUT_W[r, c] = val
                n_entries += 1
                continue

            m = _RE_OUT_B.match(line)
            if m:
                c, val = int(m.group(1)), float(m.group(2))
                if not math.isfinite(val) or abs(val) > max_abs_val:
                    return ParseResult(None, n_entries, f"value out of range: {line}")
                if not (0 <= c < N_OUT):
                    return ParseResult(None, n_entries, f"index out of bounds: {line}")
                model.OUT_b[c] = val
                n_entries += 1
                continue

            return ParseResult(None, n_entries, f"unparseable line: {line}")

    if device is not None:
        model = model.to(device)
    model.eval()
    return ParseResult(model, n_entries, None)


def _fmt(v: float) -> str:
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return f"{v:.4g}"


def serialize_sparse_weights(model: TinyTransformer, eps: float = 1e-6) -> str:
    """Serialize non-zero weights of a TinyTransformer to canonical sparse text."""
    lines: List[str] = []
    with torch.no_grad():
        for l in range(N_LAYERS):
            for h in range(N_HEADS):
                a = float(model.alibi[l, h].item())
                if abs(a) > eps:
                    lines.append(f"L{l}.H{h}.alibi = {_fmt(a)}")
                for kind in ("Q", "K", "V", "O"):
                    t = getattr(model, kind)[l, h]
                    nz = torch.nonzero(t.abs() > eps, as_tuple=False)
                    for idx in nz.tolist():
                        lines.append(f"L{l}.H{h}.{kind}[{idx[0]},{idx[1]}] = {_fmt(float(t[idx[0], idx[1]].item()))}")
            for kind in ("W1", "W2"):
                t = getattr(model, kind)[l]
                nz = torch.nonzero(t.abs() > eps, as_tuple=False)
                for idx in nz.tolist():
                    lines.append(f"L{l}.M.{kind}[{idx[0]},{idx[1]}] = {_fmt(float(t[idx[0], idx[1]].item()))}")
            b1 = model.b1[l]
            for c in torch.nonzero(b1.abs() > eps, as_tuple=False).flatten().tolist():
                lines.append(f"L{l}.M.b1[{c}] = {_fmt(float(b1[c].item()))}")
        nz = torch.nonzero(model.OUT_W.abs() > eps, as_tuple=False)
        for idx in nz.tolist():
            lines.append(f"OUT.W[{idx[0]},{idx[1]}] = {_fmt(float(model.OUT_W[idx[0], idx[1]].item()))}")
        for c in torch.nonzero(model.OUT_b.abs() > eps, as_tuple=False).flatten().tolist():
            lines.append(f"OUT.b[{c}] = {_fmt(float(model.OUT_b[c].item()))}")
    return "\n".join(lines)
