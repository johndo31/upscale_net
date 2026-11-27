"""Safe, <0.1 ms, AST-based deterministic inline math kernel (zero eval/exec).

Extracts and evaluates the active numeric expression immediately preceding '='
inside a thinking model's decode stream (e.g. `<think>`).

Supported:
  - Arbitrary-precision integers (including loose thousands commas like `6,286,917,586,52`)
  - Exact decimals via `fractions.Fraction` / `decimal.Decimal`
  - Multi-term `+`, `-`, `*`, `×`, `·`, `⋅`, `/`, `÷`, `^`, `**`, unary `+`/`-`
  - Simple LaTeX operators (`\\times`, `\\div`, `\\cdot`, `\\sqrt{...}`) and Unicode `−`, `√(...)`
  - Nested parentheses `(...)` and `sqrt(...)`
  - Clean formatting: exact integer when whole, exact trimmed decimal when terminating,
    rounded trimmed decimal otherwise

Rejected (returns None):
  - Algebraic variables / identifiers (`x + 5 =`, `2x + 3 =`, `sin(1) + 2 =`)
  - Assignments (`total =`, `x =`) and comparisons (`==`, `!=`, `<=`, `>=`, `+=`, `:=`)
  - Bare constants with no computation (`42 =`, `-5 =`, `(7) =`)
  - Malformed numbers/commas (`,123 + 4 =`, `123, + 4 =`, `3.5.2 + 1 =`)
  - Unsafe / DoS expressions (huge exponents like `9**9**9`, >80-digit inputs, >120-digit outputs)
  - Division by zero, `sqrt` of negative numbers, complex results
"""

from __future__ import annotations

import ast
import math
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from typing import List, Optional, Tuple, Union

# =============================================================================
# SAFETY LIMITS
# =============================================================================

MAX_LOOKBACK_CHARS = 512
MAX_INPUT_DIGITS = 80
MAX_OUTPUT_DIGITS = 120
MAX_AST_NODES = 128
MAX_AST_DEPTH = 32
MAX_EXPONENT_ABS = 1000
MAX_TERMINATING_DEC_PLACES = 16
NON_TERMINATING_DEC_PLACES = 6

# Characters immediately before '=' that indicate comparison or compound assignment
_FORBIDDEN_BEFORE_EQ = frozenset("!<>=+-*/:%&|^~")

# Whitelisted function names
_ALLOWED_FUNCS = frozenset({"sqrt"})

# Token specification for left-to-right scanning of the lookback window
_TOKEN_RE = re.compile(
    r"(?P<DOUBLE_NL>[ \t\r]*\n[ \t\r]*\n[ \t\r\n]*)"
    r"|(?P<WS>[ \t\r]+|\n)"
    r"|(?P<OP>\*\*|[+\-*/^])"
    r"|(?P<LPAREN>\()"
    r"|(?P<RPAREN>\))"
    r"|(?P<NUM>\d+(?:,\d+)*(?:\.\d+)?)"
    r"|(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<OTHER>.)"
)

# Simple LaTeX operator replacements (applied only to the bounded lookback window)
_LATEX_TIMES_RE = re.compile(r"\\times(?![A-Za-z])")
_LATEX_CDOT_RE = re.compile(r"\\cdot(?![A-Za-z])")
_LATEX_DIV_RE = re.compile(r"\\div(?![A-Za-z])")
_LATEX_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
_UNICODE_SQRT_NUM_RE = re.compile(r"√\s*(\d+(?:,\d+)*(?:\.\d+)?)")
_UNICODE_SQRT_PAREN_RE = re.compile(r"√\s*\(")

# Strict characters allowed immediately before an expression starts (no whitespace)
_DELIMITER_BEFORE_EXPR = frozenset(":;=$€£¥\"'`«»—–•·")


@dataclass(frozen=True)
class _Tok:
    kind: str
    text: str
    start: int
    end: int


def _normalize_window(window: str) -> str:
    """Normalize Unicode math operators and simple LaTeX ops within the lookback window."""
    s = (
        window.replace("×", "*")
        .replace("·", "*")
        .replace("⋅", "*")
        .replace("÷", "/")
        .replace("−", "-")
    )
    if "\\" in s:
        s = _LATEX_TIMES_RE.sub("*", s)
        s = _LATEX_CDOT_RE.sub("*", s)
        s = _LATEX_DIV_RE.sub("/", s)
        # Convert non-nested \sqrt{...} -> sqrt(...)
        for _ in range(4):
            s_new = _LATEX_SQRT_RE.sub(r"sqrt(\1)", s)
            if s_new == s:
                break
            s = s_new
    if "√" in s:
        s = _UNICODE_SQRT_PAREN_RE.sub("sqrt(", s)
        s = _UNICODE_SQRT_NUM_RE.sub(r"sqrt(\1)", s)
    return s


def _tokenize(window: str) -> List[_Tok]:
    return [
        _Tok(m.lastgroup or "OTHER", m.group(0), m.start(), m.end())
        for m in _TOKEN_RE.finditer(window)
    ]


def _prev_non_ws(tokens: List[_Tok], idx: int) -> Tuple[int, bool, bool]:
    """Find previous non-WS token before `idx`.
    Returns `(prev_idx, has_double_nl, directly_adjacent)`.
    `prev_idx == -1` if none exists.
    """
    has_dnl = False
    j = idx - 1
    while j >= 0:
        k = tokens[j].kind
        if k == "DOUBLE_NL":
            has_dnl = True
            j -= 1
            continue
        if k == "WS":
            j -= 1
            continue
        adj = (j == idx - 1) and (tokens[j].end == tokens[idx].start)
        return j, has_dnl, adj
    return -1, has_dnl, False


def extract_candidate_expr(prefix_text: str) -> Optional[str]:
    """Extract the active numeric expression immediately preceding the trailing '='.
    Returns None if no valid numeric expression ends at '='.
    """
    if not prefix_text:
        return None
    s = prefix_text.rstrip(" \t")
    if not s.endswith("="):
        return None
    # Reject comparisons / compound assignments (==, !=, <=, >=, +=, -=, *=, /=, :=, etc.)
    if len(s) >= 2 and s[-2] in _FORBIDDEN_BEFORE_EQ:
        return None

    lhs = s[:-1]
    if not lhs or not lhs.strip():
        return None

    window = _normalize_window(lhs[-MAX_LOOKBACK_CHARS:])
    tokens = _tokenize(window)
    if not tokens:
        return None

    # Find last non-WS token before '='
    last_idx = len(tokens) - 1
    while last_idx >= 0 and tokens[last_idx].kind in ("WS", "DOUBLE_NL"):
        last_idx -= 1
    if last_idx < 0:
        return None

    last_tok = tokens[last_idx]
    if last_tok.kind not in ("NUM", "RPAREN"):
        return None

    # Walk backwards from `last_idx` using a strict 2-state machine:
    #   state == "BEFORE_ATOM": we just accepted NUM, LPAREN (matched), or FUNC.
    #                           Going left, we can accept OP, LPAREN (if paren_depth > 0),
    #                           or FUNC attached to LPAREN; else stop (if paren_depth == 0).
    #   state == "BEFORE_OP":   we just accepted OP or RPAREN.
    #                           Going left, we NEED a value (NUM, RPAREN), or (for unary +/-)
    #                           an OP/LPAREN/boundary.
    paren_depth = 1 if last_tok.kind == "RPAREN" else 0
    state = "BEFORE_OP" if last_tok.kind == "RPAREN" else "BEFORE_ATOM"
    cur_idx = last_idx
    start_idx = last_idx

    while True:
        prev_idx, has_dnl, adj = _prev_non_ws(tokens, cur_idx)
        cur_tok = tokens[cur_idx]

        if prev_idx < 0 or has_dnl:
            if state == "BEFORE_ATOM":
                if paren_depth == 0:
                    start_idx = cur_idx
                    break
                return None
            # state == "BEFORE_OP": only valid at start/paragraph-break if cur_tok is unary +/- attached to right
            if (
                cur_tok.kind == "OP"
                and cur_tok.text in ("+", "-")
                and paren_depth == 0
                and cur_idx + 1 < len(tokens)
                and tokens[cur_idx + 1].kind in ("NUM", "LPAREN", "IDENT")
                and cur_tok.end == tokens[cur_idx + 1].start
            ):
                start_idx = cur_idx
                break
            return None

        prev_tok = tokens[prev_idx]

        if state == "BEFORE_ATOM":
            # Direct attachment checks (no whitespace between prev_tok and cur_tok)
            if adj:
                if prev_tok.kind == "IDENT":
                    if cur_tok.kind == "LPAREN" and prev_tok.text in _ALLOWED_FUNCS:
                        cur_idx = prev_idx
                        start_idx = cur_idx
                        state = "BEFORE_ATOM"
                        continue
                    # Attached unknown identifier (e.g. `sin(1)`, `f(2)`, `x1`) -> reject!
                    return None
                if prev_tok.kind == "OTHER":
                    if prev_tok.text in _DELIMITER_BEFORE_EXPR and paren_depth == 0:
                        start_idx = cur_idx
                        break
                    # Attached '.', ',', '%', '[', ']', etc. -> malformed number or indexing
                    return None
                if prev_tok.kind == "NUM":
                    # Implicit multiplication like `2(3+4)` -> reject to avoid ambiguity
                    return None
                if prev_tok.kind == "RPAREN":
                    # `(1+2)(3+4)` -> reject
                    return None

            # General (whitespace-separated or attached OP/LPAREN) transitions
            if prev_tok.kind == "OP":
                cur_idx = prev_idx
                start_idx = cur_idx
                state = "BEFORE_OP"
                continue

            if prev_tok.kind == "LPAREN":
                if paren_depth > 0:
                    paren_depth -= 1
                    cur_idx = prev_idx
                    start_idx = cur_idx
                    state = "BEFORE_ATOM"
                    continue
                # Unmatched '(' directly before expression (e.g. "((1 + 2) * 3 =") -> reject
                return None

            # prev_tok is IDENT/NUM/RPAREN/OTHER separated by whitespace -> prose boundary!
            if paren_depth == 0:
                start_idx = cur_idx
                break
            return None

        else:  # state == "BEFORE_OP"
            if prev_tok.kind == "NUM":
                cur_idx = prev_idx
                start_idx = cur_idx
                state = "BEFORE_ATOM"
                continue

            if prev_tok.kind == "RPAREN":
                paren_depth += 1
                cur_idx = prev_idx
                start_idx = cur_idx
                state = "BEFORE_OP"
                continue

            # If cur_tok is unary +/- (e.g. `-5`, `(-5)`, `2 * -3`)
            if cur_tok.kind == "OP" and cur_tok.text in ("+", "-"):
                if prev_tok.kind == "LPAREN":
                    if paren_depth > 0:
                        paren_depth -= 1
                        cur_idx = prev_idx
                        start_idx = cur_idx
                        state = "BEFORE_ATOM"
                        continue
                    # Prose "( -5 + 3 =" -> stop at unary '-'
                    if (
                        cur_idx + 1 < len(tokens)
                        and cur_tok.end == tokens[cur_idx + 1].start
                    ):
                        start_idx = cur_idx
                        break
                    return None
                if prev_tok.kind == "OP":
                    # e.g. `2 * -3` or `2 ^ -1`
                    cur_idx = prev_idx
                    start_idx = cur_idx
                    state = "BEFORE_OP"
                    continue
                # Unary +/- right after prose/delimiter (e.g. `Compute -5 + 3 =`, `= -5 + 3 =`)
                # Must be directly attached to the operand on the right, and NOT attached to prev_tok
                # (unless prev_tok is a delimiter like `:` or `=`).
                is_attached_right = (
                    cur_idx + 1 < len(tokens)
                    and tokens[cur_idx + 1].kind in ("NUM", "LPAREN", "IDENT")
                    and cur_tok.end == tokens[cur_idx + 1].start
                )
                prev_is_boundary = (not adj) or (
                    prev_tok.kind == "OTHER" and prev_tok.text in _DELIMITER_BEFORE_EXPR
                )
                if is_attached_right and prev_is_boundary and paren_depth == 0:
                    start_idx = cur_idx
                    break

            # Anything else before an operator (e.g. `x + 5`, `total - 5`, `arr[0] + 1`) -> REJECT!
            return None

    if paren_depth != 0:
        return None

    raw_expr = window[tokens[start_idx].start:tokens[last_idx].end].strip()
    return raw_expr if raw_expr else None


# =============================================================================
# SAFE AST EVALUATOR (EXACT FRACTION / DECIMAL ARITHMETIC)
# =============================================================================

_NumericVal = Union[Fraction, Decimal]


def _check_input_digits(raw_num: str) -> bool:
    digits_only = raw_num.replace(",", "").replace(".", "")
    return len(digits_only) <= MAX_INPUT_DIGITS


def _prepare_ast_source(raw_expr: str) -> Optional[str]:
    """Strip loose thousands commas, collapse newlines to spaces, convert `^` to `**`."""
    # Validate every NUM token's digit count and strip internal commas
    tokens = _tokenize(raw_expr)
    parts: List[str] = []
    for tok in tokens:
        if tok.kind == "NUM":
            if not _check_input_digits(tok.text):
                return None
            parts.append(tok.text.replace(",", ""))
        elif tok.kind in ("WS", "DOUBLE_NL"):
            parts.append(" ")
        elif tok.kind == "OP" and tok.text == "^":
            parts.append("**")
        else:
            parts.append(tok.text)
    return "".join(parts).strip()


def _to_decimal(val: _NumericVal) -> Decimal:
    if isinstance(val, Decimal):
        return val
    with localcontext() as ctx:
        ctx.prec = 64
        return Decimal(val.numerator) / Decimal(val.denominator)


def _safe_pow(base: _NumericVal, exp: _NumericVal) -> Optional[_NumericVal]:
    """Evaluate `base ** exp` with strict anti-DoS bounds."""
    # Exact square root shortcut for `x ** 0.5` or `x ** (1/2)`
    if isinstance(exp, Fraction) and exp == Fraction(1, 2):
        return _safe_sqrt(base)

    if isinstance(base, Fraction) and isinstance(exp, Fraction) and exp.denominator == 1:
        e_int = exp.numerator
        if abs(e_int) > MAX_EXPONENT_ABS:
            if base in (Fraction(0), Fraction(1), Fraction(-1)):
                if base == 0 and e_int < 0:
                    return None
                return base ** e_int
            return None
        if base == 0:
            if e_int < 0:
                return None
            return Fraction(1) if e_int == 0 else Fraction(0)
        # Estimate output digit count for numerator and denominator
        abs_e = abs(e_int)
        num_abs = abs(base.numerator)
        den_abs = base.denominator
        if num_abs > 1 and abs_e * math.log10(num_abs) > MAX_OUTPUT_DIGITS:
            return None
        if den_abs > 1 and abs_e * math.log10(den_abs) > MAX_OUTPUT_DIGITS:
            return None
        return base ** e_int

    # General non-integer / Decimal exponent
    d_base = _to_decimal(base)
    d_exp = _to_decimal(exp)
    if abs(d_exp) > 100:
        return None
    if d_base < 0:
        # Only integer exponents allowed for negative bases (already handled for Fraction)
        if d_exp == d_exp.to_integral_value():
            e_int = int(d_exp)
            if abs(e_int) > MAX_EXPONENT_ABS:
                return None
            with localcontext() as ctx:
                ctx.prec = 64
                res = d_base ** e_int
            if not res.is_finite():
                return None
            return res
        return None
    if d_base == 0:
        if d_exp < 0:
            return None
        return Decimal(1) if d_exp == 0 else Decimal(0)
    # Estimate magnitude
    try:
        log10_est = float(d_exp) * math.log10(float(d_base))
    except (ValueError, OverflowError):
        return None
    if abs(log10_est) > MAX_OUTPUT_DIGITS:
        return None
    with localcontext() as ctx:
        ctx.prec = 64
        try:
            res = d_base ** d_exp
        except (InvalidOperation, OverflowError):
            return None
    if not res.is_finite():
        return None
    return res


def _safe_sqrt(val: _NumericVal) -> Optional[_NumericVal]:
    """Evaluate `sqrt(val)` exactly when possible, otherwise high-precision Decimal."""
    if isinstance(val, Fraction):
        if val < 0:
            return None
        if val == 0:
            return Fraction(0)
        sn = math.isqrt(val.numerator)
        sd = math.isqrt(val.denominator)
        if sn * sn == val.numerator and sd * sd == val.denominator:
            return Fraction(sn, sd)
        with localcontext() as ctx:
            ctx.prec = 64
            d = Decimal(val.numerator) / Decimal(val.denominator)
            return d.sqrt()
    else:
        if val < 0:
            return None
        if val == 0:
            return Decimal(0)
        with localcontext() as ctx:
            ctx.prec = 64
            return val.sqrt()


def _eval_ast_node(
    node: ast.AST,
    src: str,
    counter: List[int],
    depth: int,
) -> Optional[_NumericVal]:
    """Strictly whitelisted recursive AST evaluator."""
    counter[0] += 1
    if counter[0] > MAX_AST_NODES or depth > MAX_AST_DEPTH:
        return None

    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body, src, counter, depth + 1)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return None
        if isinstance(node.value, int):
            return Fraction(node.value)
        # Extract exact decimal literal text from `src` to avoid float rounding artifacts
        if node.col_offset is not None and node.end_col_offset is not None:
            lit = src[node.col_offset:node.end_col_offset].strip()
            try:
                return Fraction(Decimal(lit))
            except (InvalidOperation, ValueError):
                return None
        return Fraction(str(node.value))

    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand, src, counter, depth + 1)
        if operand is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        return None

    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left, src, counter, depth + 1)
        right = _eval_ast_node(node.right, src, counter, depth + 1)
        if left is None or right is None:
            return None

        if isinstance(node.op, ast.Pow):
            return _safe_pow(left, right)

        if isinstance(left, Fraction) and isinstance(right, Fraction):
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                # Guard against huge product
                if (
                    abs(left.numerator) > 1
                    and abs(right.numerator) > 1
                    and (left.numerator.bit_length() + right.numerator.bit_length()) > MAX_OUTPUT_DIGITS * 4
                ):
                    return None
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    return None
                return left / right
            return None

        # At least one operand is Decimal (e.g. from irrational sqrt)
        dl = _to_decimal(left)
        dr = _to_decimal(right)
        with localcontext() as ctx:
            ctx.prec = 64
            try:
                if isinstance(node.op, ast.Add):
                    res = dl + dr
                elif isinstance(node.op, ast.Sub):
                    res = dl - dr
                elif isinstance(node.op, ast.Mult):
                    res = dl * dr
                elif isinstance(node.op, ast.Div):
                    if dr == 0:
                        return None
                    res = dl / dr
                else:
                    return None
            except (InvalidOperation, OverflowError, ZeroDivisionError):
                return None
        return res if res.is_finite() else None

    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "sqrt"
            and len(node.args) == 1
            and len(node.keywords) == 0
        ):
            arg = _eval_ast_node(node.args[0], src, counter, depth + 1)
            if arg is None:
                return None
            return _safe_sqrt(arg)
        return None

    return None


def _has_computation(tree: ast.AST) -> bool:
    """Require at least one binary operator or whitelisted function call (reject bare constants)."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.Call)):
            return True
    return False


def _is_terminating_denom(q: int) -> Tuple[bool, int]:
    """Check if denominator `q` has only prime factors 2 and 5.
    Returns `(is_terminating, decimal_places)`.
    """
    n2 = 0
    while q % 2 == 0:
        q //= 2
        n2 += 1
    n5 = 0
    while q % 5 == 0:
        q //= 5
        n5 += 1
    return (q == 1), max(n2, n5)


def _format_decimal_clean(d: Decimal, max_places: Optional[int] = None) -> Optional[str]:
    """Format Decimal cleanly without scientific notation or trailing fractional zeros."""
    if not d.is_finite():
        return None
    if max_places is not None:
        q = Decimal(10) ** (-max_places)
        with localcontext() as ctx:
            ctx.prec = max(64, MAX_OUTPUT_DIGITS + 16)
            d = d.quantize(q)
    # Normalize -0 -> 0
    if d == 0:
        return "0"
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    if len(s.lstrip("-").replace(".", "")) > MAX_OUTPUT_DIGITS:
        return None
    return s


def _format_result(val: _NumericVal) -> Optional[str]:
    """Format exact Fraction or Decimal cleanly."""
    if isinstance(val, Fraction):
        if val.denominator == 1:
            s = str(val.numerator)
            if len(s.lstrip("-")) > MAX_OUTPUT_DIGITS:
                return None
            return s
        is_term, places = _is_terminating_denom(val.denominator)
        if is_term and places <= MAX_TERMINATING_DEC_PLACES:
            int_digits = len(str(abs(val.numerator) // val.denominator))
            with localcontext() as ctx:
                ctx.prec = max(64, int_digits + places + 8)
                d = Decimal(val.numerator) / Decimal(val.denominator)
            return _format_decimal_clean(d, max_places=None)
        # Non-terminating rational (e.g. 1/3 -> 0.333333)
        int_digits = len(str(abs(val.numerator) // val.denominator))
        if int_digits > MAX_OUTPUT_DIGITS:
            return None
        with localcontext() as ctx:
            ctx.prec = max(64, int_digits + NON_TERMINATING_DEC_PLACES + 12)
            d = Decimal(val.numerator) / Decimal(val.denominator)
        return _format_decimal_clean(d, max_places=NON_TERMINATING_DEC_PLACES)
    else:
        # Decimal (from irrational sqrt or non-integer pow)
        if not val.is_finite():
            return None
        # If extremely close to an integer (e.g. sqrt(2)*sqrt(2) = 1.9999999999999999...), snap cleanly
        nearest = val.to_integral_value()
        if abs(val - nearest) < Decimal("1e-24"):
            s = format(nearest, "f").split(".")[0]
            if s == "-0":
                s = "0"
            return s if len(s.lstrip("-")) <= MAX_OUTPUT_DIGITS else None
        return _format_decimal_clean(val, max_places=NON_TERMINATING_DEC_PLACES)


def eval_expr_str(raw_expr: str) -> Optional[str]:
    """Safely parse and evaluate a standalone math expression string (without trailing '=').
    Returns exact/clean formatted result string, or None if invalid/unsafe.
    """
    src = _prepare_ast_source(raw_expr)
    if not src:
        return None
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError:
        return None
    if not _has_computation(tree):
        return None
    val = _eval_ast_node(tree, src, counter=[0], depth=0)
    if val is None:
        return None
    return _format_result(val)


def extract_and_eval_expr(prefix_text: str) -> Optional[Tuple[str, str, float]]:
    """Extract and evaluate the active numeric expression ending at '=' in `prefix_text`.

    Returns:
      `(expr_str, result_str, latency_ms)` if a valid, safe numeric expression is found,
      otherwise `None`.
    """
    t0 = time.perf_counter()
    expr_str = extract_candidate_expr(prefix_text)
    if expr_str is None:
        return None
    result_str = eval_expr_str(expr_str)
    if result_str is None:
        return None
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return expr_str, result_str, latency_ms
