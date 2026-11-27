"""Comprehensive local CPU unit test suite (220+ cases) for `math_kernel.py`.

Verifies 100% pass across:
  - Basic & multi-term arithmetic (+, -, *, /, ^, **, parentheses, unary minus)
  - Unicode & LaTeX operators (×, ·, ⋅, ÷, −, √, \\times, \\div, \\cdot, \\sqrt{})
  - Exact Decimal / Fraction precision (0.1 + 0.2 = 0.3, 3.5 * 1.25 = 4.375, etc.)
  - Loose thousands commas (e.g. 6,286,917,586,52 + 3,137,930,031 =)
  - 12-to-48 digit OOD integer arithmetic & carry/borrow cascades
  - Exact & irrational sqrt(...)
  - Realistic <think> prose contexts
  - Strict rejections: algebra, assignments, comparisons, code injection, malformed syntax, DoS limits
  - Median / p95 / max latency reporting in milliseconds
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path
from typing import List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from math_kernel import extract_and_eval_expr  # noqa: E402


def build_positive_cases() -> List[Tuple[str, str]]:
    cases: List[Tuple[str, str]] = [
        # --- Basic operations ---
        ("2 + 3 =", "5"),
        ("2+3=", "5"),
        ("10 - 4 =", "6"),
        ("4 - 10 =", "-6"),
        ("7 * 8 =", "56"),
        ("56 / 8 =", "7"),
        ("100 / 4 =", "25"),
        ("0 + 0 =", "0"),
        ("0 * 999999 =", "0"),
        ("123456 - 123456 =", "0"),
        ("2 + 3 = ", "5"),  # trailing space after '='
        ("  15 + 25  =", "40"),
        # --- Operator precedence & multi-term ---
        ("2 + 3 * 4 =", "14"),
        ("(2 + 3) * 4 =", "20"),
        ("10 - 3 - 2 =", "5"),
        ("10 - (3 - 2) =", "9"),
        ("20 / 4 / 2 =", "2.5"),
        ("20 / (4 / 2) =", "10"),
        ("1 + 2 + 3 + 4 + 5 =", "15"),
        ("100 - 20 + 30 - 50 + 10 =", "70"),
        ("((2 + 3) * (4 + 5)) / 9 =", "5"),
        ("2 * (3 + 4 * (5 - 2)) =", "30"),
        ("10 + 20 * 3 - 40 / 8 =", "65"),
        ("325,263,952 + 905,359,078 - 110,685,943 =", str(325263952 + 905359078 - 110685943)),
        ("1,000 * 2,000 + 3,000 - 500 =", "2002500"),
        # --- Loose thousands commas & newlines ---
        ("6,286,917,586,52 + 3,137,930,031 =", str(628691758652 + 3137930031)),
        ("6,286,917,586,52\n+ 3,137,930,031 =", str(628691758652 + 3137930031)),
        ("12,34,56 + 1,2,3 =", str(123456 + 123)),
        ("1,000,000 - 1 =", "999999"),
        ("9,999,999,999 + 1 =", "10000000000"),
        # --- Unicode & LaTeX operators ---
        ("6 × 7 =", "42"),
        ("6 · 7 =", "42"),
        ("6 ⋅ 7 =", "42"),
        ("84 ÷ 7 =", "12"),
        ("10 − 3 =", "7"),
        ("2 × 3 + 12 ÷ 4 =", "9"),
        ("12 \\times 15 =", "180"),
        ("144 \\div 12 =", "12"),
        ("7 \\cdot 8 =", "56"),
        ("\\sqrt{144} + 5 =", "17"),
        ("\\sqrt{81} \\times \\sqrt{16} =", "36"),
        ("√144 + 1 =", "13"),
        ("√(9 + 16) =", "5"),
        # --- Exponentiation (^ and **) ---
        ("2^10 =", "1024"),
        ("2**10 =", "1024"),
        ("3^4 =", "81"),
        ("2^3^2 =", "512"),  # right-associative
        ("(2^3)^2 =", "64"),
        ("5^0 =", "1"),
        ("0^5 =", "0"),
        ("0^0 =", "1"),
        ("1^999 =", "1"),
        ("(-1)^999 =", "-1"),
        ("(-1)^1000 =", "1"),
        ("-2^2 =", "-4"),
        ("(-2)^2 =", "4"),
        ("(-3)^3 =", "-27"),
        ("2^-3 =", "0.125"),
        ("10^-4 =", "0.0001"),
        ("16^0.5 =", "4"),
        ("2.25^0.5 =", "1.5"),
        ("2^20 + 3^10 =", str(2**20 + 3**10)),
        # --- sqrt(...) exact & irrational ---
        ("sqrt(0) =", "0"),
        ("sqrt(1) =", "1"),
        ("sqrt(4) =", "2"),
        ("sqrt(144) =", "12"),
        ("sqrt(1000000000000) =", "1000000"),
        ("sqrt(2.25) =", "1.5"),
        ("sqrt(0.0625) =", "0.25"),
        ("sqrt(9 + 16) =", "5"),
        ("sqrt(144) + 5^2 =", "37"),
        ("sqrt(sqrt(81)) =", "3"),
        ("sqrt(2) =", "1.414214"),
        ("sqrt(3) =", "1.732051"),
        ("sqrt(2) * sqrt(2) =", "2"),
        ("sqrt(2) + sqrt(8) =", "4.242641"),
        ("2 * sqrt(25) - 3 =", "7"),
        # --- Exact decimals & fractions ---
        ("0.1 + 0.2 =", "0.3"),
        ("0.3 - 0.1 =", "0.2"),
        ("3.5 + 1.25 =", "4.75"),
        ("3.5 * 1.25 =", "4.375"),
        ("10.00 - 2.50 =", "7.5"),
        ("0.25 * 4 =", "1"),
        ("1.5 / 0.5 =", "3"),
        ("7 / 2 =", "3.5"),
        ("1 / 8 =", "0.125"),
        ("1 / 64 =", "0.015625"),
        ("1 / 1024 =", "0.0009765625"),
        ("(1 / 3) * 3 =", "1"),
        ("(1 / 7) * 7 =", "1"),
        ("1 / 3 =", "0.333333"),
        ("2 / 3 =", "0.666667"),
        ("100.50 + 200.75 - 50.25 =", "251"),
        ("12345.67 * 100 =", "1234567"),
        ("0.0001 + 0.0002 =", "0.0003"),
        ("-3.5 + 1.25 =", "-2.25"),
        ("-3.5 * -2 =", "7"),
        # --- Unary minus & negative results ---
        ("-5 + 3 =", "-2"),
        ("+5 + 3 =", "8"),
        ("(-5) + 3 =", "-2"),
        ("2 * -3 =", "-6"),
        ("10 + -4 =", "6"),
        ("10 - -4 =", "14"),
        ("(-5) * (-4) =", "20"),
        ("-(3 + 4) * 2 =", "-14"),
        ("Compute -5 + 3 =", "-2"),
        ("Step 1: -12.5 + 4.5 =", "-8"),
        ("total = -100 + 25 =", "-75"),
        # --- Realistic <think> prose contexts ---
        ("<think>\nLet's compute 48219384721 + 39571284905 =", str(48219384721 + 39571284905)),
        ("<think>\nFirst, we calculate 325,263,952 + 905,359,078 =", str(325263952 + 905359078)),
        ("<think>\nStep 1: total = 100 + 250 =", "350"),
        ("<think>\nStep 2: 350 - 75 =", "275"),
        ("<think>\nIn 2024, 150 + 250 =", "400"),
        ("<think>\nWe have 5 apples and 10 + 20 =", "30"),
        ("<think>\nFirst, 10 + 20 = 30.\nNext, 30 * 5 =", "150"),
        ("<think>\nSo the area is 12.5 * 8.4 =", "105"),
        ("<think>\nThe hypotenuse is sqrt(3^2 + 4^2) =", "5"),
        ("<think>\nNote (so 12 + 18 =", "30"),
        ("<think>\nStep 1 (2 + 3) * 4 =", "20"),
        ("<think>\nCost: $120 + 80 =", "200"),
        ("<think>\nThus, 1000 / 8 =", "125"),
        ("<think>\nChecking: 123456789 * 987654321 =", str(123456789 * 987654321)),
        # --- Big 12-to-48 digit integers & carry/borrow cascades ---
        ("999999999999 + 1 =", "1000000000000"),
        ("1000000000000 - 1 =", "999999999999"),
        (
            "123456789012345678901234 + 987654321098765432109876 =",
            str(123456789012345678901234 + 987654321098765432109876),
        ),
        (
            "123456789012345678901234567890123456789012345678 + 1 =",
            str(123456789012345678901234567890123456789012345678 + 1),
        ),
        (
            "999999999999999999999999999999999999999999999999 + 1 =",
            str(10**48),
        ),
        (
            "100000000000000000000000000000000000000000000000 - 1 =",
            str(10**47 - 1),
        ),
        (
            "48219 - 100000000000000000000000 =",
            str(48219 - 10**23),
        ),
        (
            "12345678901234567890 * 98765432109876543210 =",
            str(12345678901234567890 * 98765432109876543210),
        ),
        (
            "999999999999999999999999 / 3 =",
            str(999999999999999999999999 // 3),
        ),
    ]
    return cases


def build_negative_cases() -> List[str]:
    """All of these MUST return None."""
    return [
        # --- Empty / no trailing '=' ---
        "",
        "   ",
        "=",
        "   =",
        "2 + 3",
        "2 + 3 = 5",
        "2 + 3 =\n",
        # --- Comparisons & compound assignments ---
        "5 ==",
        "x ==",
        "1 + 2 ==",
        "5 !=",
        "5 <=",
        "5 >=",
        "x +=",
        "5 +=",
        "5 -=",
        "5 *=",
        "5 /=",
        "x :=",
        "5 %=",
        "5 ^=",
        "if a + b ==",
        "while count <=",
        # --- Prose / variable assignments ---
        "x =",
        "total =",
        "result =",
        "balance =",
        "<think>\nSo total =",
        "<think>\nLet x =",
        "def foo(a=1, b=",
        "kwargs = dict(x=",
        # --- Bare constants (no computation) ---
        "42 =",
        "0 =",
        "-5 =",
        "3.14 =",
        "(7) =",
        "Step 1 =",
        "total = 42 =",
        "1,000,000 =",
        # --- Algebraic expressions ---
        "x + 5 =",
        "5 + x =",
        "2x + 3 =",
        "2*x + 3 =",
        "x * 10 + 20 =",
        "10 + 20 * x =",
        "a + b =",
        "total - 5 =",
        "x-5 =",
        "2 * (x + 1) =",
        "sqrt(x) =",
        "sqrt(x + 1) =",
        "a1 + 2 =",
        "x_1 + 5 =",
        "y / 2 + 3 =",
        "n^2 + 1 =",
        # --- Unknown functions & code injection ---
        "sin(0) + 1 =",
        "cos(0) =",
        "abs(-5) =",
        "pow(2, 3) =",
        "max(1, 2) =",
        "f(16) =",
        "__import__('os').system('ls') =",
        "eval('1+1') =",
        "exec('x=1') =",
        "open('/etc/passwd') =",
        "().__class__.__base__ =",
        "[1, 2] + [3] =",
        "arr[0] + 1 =",
        "d['a'] + 1 =",
        "True + 1 =",
        "False + 2 =",
        "None + 1 =",
        "10 % 3 =",
        "50% + 10 =",
        "10 // 3 =",
        # --- Malformed numbers, commas, operators, parens ---
        ",123 + 4 =",
        "123, + 4 =",
        "1,,2 + 3 =",
        "3.5.2 + 1 =",
        "3. + 1 =",
        ".5 + 1 =",
        "1 + =",
        "1 * =",
        "1 + * 2 =",
        "1 + 2) =",
        "((1 + 2) * 3 =",
        "2(3 + 4) =",
        "(1 + 2)(3 + 4) =",
        "sqrt() =",
        "sqrt(1, 2) =",
        "1 + \n\n 2 =",
        # --- Math errors & DoS safety limits ---
        "1 / 0 =",
        "5 / (3 - 3) =",
        "0^-1 =",
        "0^-5 =",
        "sqrt(-1) =",
        "sqrt(-144) =",
        "(-4)^0.5 =",
        "9**9**9 =",
        "9^9^9 =",
        "2**10000 =",
        "2^10000 =",
        "10**500 =",
        "9999999999**50 =",
        "10^-500 =",
        ("9" * 90) + " + 1 =",
        ("9" * 65) + " * " + ("9" * 65) + " =",
    ]


def build_randomized_cases(n: int = 60, seed: int = 202608) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    cases: List[Tuple[str, str]] = []
    for i in range(n):
        kind = i % 6
        if kind == 0:
            # Big 12-36 digit addition / subtraction
            L = rng.randint(12, 36)
            a = rng.randint(10 ** (L - 1), 10**L - 1)
            b = rng.randint(10 ** (L - 1), 10**L - 1)
            op = rng.choice(["+", "-"])
            ans = a + b if op == "+" else a - b
            use_comma = rng.random() < 0.5
            a_s = f"{a:,}" if use_comma else str(a)
            b_s = f"{b:,}" if use_comma else str(b)
            cases.append((f"<think>\nNext, {a_s} {op} {b_s} =", str(ans)))
        elif kind == 1:
            # 3-term mixed + - *
            a = rng.randint(100, 99999)
            b = rng.randint(10, 9999)
            c = rng.randint(100, 99999)
            ans = a * b - c
            cases.append((f"Step 2: {a:,} * {b:,} - {c:,} =", str(ans)))
        elif kind == 2:
            # Exact division + addition
            d = rng.randint(2, 999)
            q = rng.randint(10**5, 10**9)
            c = rng.randint(10**4, 10**8)
            ans = q + c
            cases.append((f"So ({q * d} / {d}) + {c} =", str(ans)))
        elif kind == 3:
            # Exact decimal add/sub/mul
            a_c = rng.randint(10, 99999)
            b_c = rng.randint(10, 9999)
            # exact 2-decimal numbers: a_c/100 and b_c/100
            from decimal import Decimal
            da = Decimal(a_c) / Decimal(100)
            db = Decimal(b_c) / Decimal(100)
            op = rng.choice(["+", "-", "*"])
            res = da + db if op == "+" else (da - db if op == "-" else da * db)
            res_s = format(res, "f")
            if "." in res_s:
                res_s = res_s.rstrip("0").rstrip(".")
            cases.append((f"Price: {format(da, 'f')} {op} {format(db, 'f')} =", res_s))
        elif kind == 4:
            # Exact sqrt + pow
            r = rng.randint(2, 9999)
            b = rng.randint(2, 20)
            e = rng.randint(2, 5)
            ans = r + b**e
            cases.append((f"Compute sqrt({r * r}) + {b}^{e} =", str(ans)))
        else:
            # Parenthesized 4-term
            a = rng.randint(10, 999)
            b = rng.randint(10, 999)
            c = rng.randint(2, 99)
            d = rng.randint(10, 999)
            ans = (a + b) * c - d
            cases.append((f"({a} + {b}) × {c} − {d} =", str(ans)))
    return cases


def main() -> None:
    pos_cases = build_positive_cases() + build_randomized_cases(60)
    neg_cases = build_negative_cases()
    total = len(pos_cases) + len(neg_cases)
    failures: List[str] = []
    latencies_ms: List[float] = []

    for inp, expected in pos_cases:
        out = extract_and_eval_expr(inp)
        if out is None:
            failures.append(f"POS FAIL (got None): {inp!r} expected {expected!r}")
        else:
            expr_s, res_s, lat_ms = out
            latencies_ms.append(lat_ms)
            if res_s != expected:
                failures.append(
                    f"POS FAIL: {inp!r} -> expr={expr_s!r} got={res_s!r} expected={expected!r}"
                )

    for inp in neg_cases:
        out = extract_and_eval_expr(inp)
        if out is not None:
            failures.append(f"NEG FAIL (expected None): {inp!r} -> got {out!r}")

    passed = total - len(failures)
    med_ms = statistics.median(latencies_ms) if latencies_ms else 0.0
    p95_ms = sorted(latencies_ms)[int(0.95 * len(latencies_ms))] if latencies_ms else 0.0
    max_ms = max(latencies_ms) if latencies_ms else 0.0

    print(
        f"test_math_kernel: {passed}/{total} passed "
        f"(pos={len(pos_cases)}, neg={len(neg_cases)})"
    )
    print(
        f"latency_ms: median={med_ms:.4f} ms | p95={p95_ms:.4f} ms | max={max_ms:.4f} ms"
    )

    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures[:30]:
            print("  " + f)
        sys.exit(1)

    assert med_ms < 1.0, f"Median latency unexpectedly high: {med_ms:.4f} ms"
    print("ALL TESTS PASSED!")


if __name__ == "__main__":
    main()
