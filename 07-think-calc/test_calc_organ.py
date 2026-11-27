"""Comprehensive CPU test suite for the compiled 2-layer variable-length '+' / '-' SparseOrgan.

Tests 1,000+ cases across:
  - 1..64 digit addition and subtraction
  - Unequal operand lengths (e.g. 48-digit +/- 3-digit, 3-digit +/- 48-digit, 64-digit +/- 1-digit)
  - Adversarial carry chains (999...9 + 1, 999... + 999...)
  - Adversarial borrow chains (1000...0 - 1, 1000...0 - 48219, 1000...0 - 999...9)
  - Negative results (a < b, including unequal lengths and borrow chains)
  - Zero results (a == b, 0 + 0, 0 - 0, 0 - x, x - 0)
  - Optional spaces around '+', '-', '='
  - Multi-equation prefix slicing (ensuring earlier equations never leak into the active window)
  - Structural gate exactness (gate == 1.0 on every emitted token, gate == 0.0 right after p=1)
"""

from __future__ import annotations

import random
import time
import torch

from calc_organ import build_varlen_calc_organ, run_organ_standalone


def rand_int_digits(rng: random.Random, n: int) -> int:
    if n == 1:
        return rng.randint(0, 9)
    lo = 10 ** (n - 1)
    hi = 10 ** n - 1
    return rng.randint(lo, hi)


def check_case(organ, expr: str, expected: int, max_idx: int = 64) -> float:
    det = run_organ_standalone(expr, organ=organ, max_idx=max_idx, return_details=True)
    got = det["answer"]
    exp_str = str(expected)
    gates = det["gates"]
    gate_ok = (len(gates) == len(exp_str) + 1 and all(g > 0.99 for g in gates[:-1]) and gates[-1] < 0.01)
    assert got == exp_str and gate_ok, (
        f"FAILED on {expr!r}: expected={exp_str!r}, got={got!r}, gates={gates}, det={det}"
    )
    return det["min_margin"]


def main() -> None:
    t0 = time.time()
    max_idx = 64
    organ = build_varlen_calc_organ(max_idx=max_idx, device="cpu")
    organ.eval()
    nnz = organ.nnz()
    print(f"Built 2-layer varlen calc SparseOrgan: max_idx={max_idx}, d_model={organ.d_model}, "
          f"d_head={organ.d_head}, d_mlp={organ.d_mlp}, n_heads={organ.n_heads}, nnz={nnz}")

    rng = random.Random(2026)
    total = 0
    min_margin = float("inf")

    # 1. Deterministic edge cases (zeros, single digits, carry/borrow extremes, spaces, prefixes)
    edge_cases = [
        ("0+0=", 0),
        ("0-0=", 0),
        ("0 + 0 =", 0),
        ("0 - 0 =", 0),
        ("0 - 5 =", -5),
        ("5 - 0 =", 5),
        ("0 + 999999 =", 999999),
        ("999999 + 0 =", 999999),
        ("9 + 1 =", 10),
        ("10 - 1 =", 9),
        ("1 - 10 =", -9),
        ("42 - 42 =", 0),
        ("100000000000 - 48219 =", 100000000000 - 48219),
        ("48219 - 100000000000 =", 48219 - 100000000000),
        ("100000000000 - 99999999999 =", 1),
        ("99999999999 - 100000000000 =", -1),
        (f"{'9' * 64} + 1 =", 10 ** 64),
        (f"1 + {'9' * 64} =", 10 ** 64),
        (f"{'9' * 64} + {'9' * 64} =", 2 * (10 ** 64 - 1)),
        (f"1{'0' * 63} - 1 =", 10 ** 63 - 1),
        (f"1 - 1{'0' * 63} =", -(10 ** 63 - 1)),
        (f"1{'0' * 63} - {'9' * 63} =", 1),
        (f"{'9' * 63} - 1{'0' * 63} =", -1),
        (f"{'9' * 64} - {'9' * 64} =", 0),
        # Multi-equation prefix slicing
        ("First 123 + 456 = 579. Next 48219384721 + 39571284905 =", 48219384721 + 39571284905),
        ("Step 1: 999 - 1000 = -1; Step 2: 100000000000 - 48219 =", 100000000000 - 48219),
        ("Wait, x = 5, y = 10, so 1000000 - 999999 =", 1),
        ("  12345   +   67890   =", 12345 + 67890),
        ("  67890   -   12345   =", 67890 - 12345),
        ("  12345   -   67890   =", 12345 - 67890),
        # Thousands-separator commas inside integers
        ("325,263,952 + 905,359,078 =", 325263952 + 905359078),
        ("1,000,000 - 1 =", 999999),
        ("39 - 4,821 =", 39 - 4821),
        ("1,000,000,000,000 - 48,219 =", 1000000000000 - 48219),
        ("999,999,999 + 1 =", 1000000000),
        ("Step 1: 1,234 + 5,678 = 6,912. Step 2: 325,263,952 - 905,359,078 =", 325263952 - 905359078),
        # Mis-grouped commas & multiline expressions observed in real <think> traces
        ("6,286,917,586,52 + 3,137,930,031 =", 628691758652 + 3137930031),
        ("31,652,159,587,559 + 7,961,942,11 =", 31652159587559 + 796194211),
        ("970,901,515,336\n- 637,910,139,374 =", 970901515336 - 637910139374),
        ("123,456,789\n+\n987,654,321\n=", 123456789 + 987654321),
        # Leading-zero operands (stripped cleanly so n_a/n_b/p_top aren't inflated)
        ("00199034677 + 09724415826 =", 199034677 + 9724415826),
        ("000 + 000 =", 0),
        ("005 + 003 =", 8),
        ("00199034677 - 09724415826 =", 199034677 - 9724415826),
    ]
    for expr, exp in edge_cases:
        min_margin = min(min_margin, check_case(organ, expr, exp, max_idx))
        total += 1
    print(f"  [PASS] edge_cases: {len(edge_cases)}/{len(edge_cases)}")

    # 2. Randomized buckets totaling > 1,000 cases
    buckets = [
        ("short_1_6_add", 150, "add", (1, 6), (1, 6)),
        ("short_1_6_sub", 150, "sub", (1, 6), (1, 6)),
        ("med_7_24_add", 150, "add", (7, 24), (7, 24)),
        ("med_7_24_sub", 150, "sub", (7, 24), (7, 24)),
        ("long_25_64_add", 100, "add", (25, 64), (25, 64)),
        ("long_25_64_sub", 100, "sub", (25, 64), (25, 64)),
        ("unequal_long_short", 120, "both", (30, 64), (1, 5)),
        ("unequal_short_long", 120, "both", (1, 5), (30, 64)),
        ("equal_a_eq_b", 50, "eq", (1, 64), (1, 64)),
        ("carry_borrow_chains", 80, "chain", (8, 64), (8, 64)),
    ]

    for bname, count, mode, (la_lo, la_hi), (lb_lo, lb_hi) in buckets:
        b_margin = float("inf")
        for i in range(count):
            sp1 = " " if rng.random() < 0.6 else ""
            sp2 = " " if rng.random() < 0.6 else ""
            sp3 = " " if rng.random() < 0.6 else ""
            prefix = ""
            if rng.random() < 0.25:
                p_a, p_b = rng.randint(1, 9999), rng.randint(1, 9999)
                prefix = f"Earlier {p_a} + {p_b} = {p_a + p_b}. Now "

            if mode == "add":
                a = rand_int_digits(rng, rng.randint(la_lo, la_hi))
                b = rand_int_digits(rng, rng.randint(lb_lo, lb_hi))
                op, exp = "+", a + b
            elif mode == "sub":
                a = rand_int_digits(rng, rng.randint(la_lo, la_hi))
                b = rand_int_digits(rng, rng.randint(lb_lo, lb_hi))
                op, exp = "-", a - b
            elif mode == "both":
                a = rand_int_digits(rng, rng.randint(la_lo, la_hi))
                b = rand_int_digits(rng, rng.randint(lb_lo, lb_hi))
                if rng.random() < 0.5:
                    op, exp = "+", a + b
                else:
                    op, exp = "-", a - b
            elif mode == "eq":
                a = rand_int_digits(rng, rng.randint(la_lo, la_hi))
                b = a
                op, exp = "-", 0
            else:  # chain
                L = rng.randint(la_lo, la_hi)
                kind = i % 4
                if kind == 0:
                    # 9-sum carry cascade
                    a_digs = [rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(L - 2)]
                    b_digs = [9 - d for d in a_digs]
                    if b_digs[0] == 0:
                        b_digs[0] = 1; a_digs[0] = 8
                    a0 = rng.randint(1, 9); b0 = rng.randint(10 - a0, 9)
                    a = int("".join(map(str, a_digs + [a0])))
                    b = int("".join(map(str, b_digs + [b0])))
                    op, exp = "+", a + b
                elif kind == 1:
                    # 10^k - small borrow chain
                    a = 10 ** (L - 1)
                    b = rng.randint(1, 10 ** min(6, L - 1) - 1)
                    op, exp = "-", a - b
                elif kind == 2:
                    # small - 10^k negative borrow chain
                    a = rng.randint(1, 10 ** min(6, L - 1) - 1)
                    b = 10 ** (L - 1)
                    op, exp = "-", a - b
                else:
                    # Near-equal cancellation (many leading zeros cancelled!)
                    base = rand_int_digits(rng, L)
                    delta = rng.randint(1, 999)
                    if rng.random() < 0.5:
                        a, b = base, max(0, base - delta)
                    else:
                        a, b = max(0, base - delta), base
                    op, exp = "-", a - b

            expr = f"{prefix}{a}{sp1}{op}{sp2}{b}{sp3}="
            m = check_case(organ, expr, exp, max_idx)
            b_margin = min(b_margin, m)
            min_margin = min(min_margin, m)
            total += 1
        print(f"  [PASS] {bname:22s}: {count}/{count} (min_margin={b_margin:.3f})")

    elapsed = time.time() - t0
    print(f"\nALL {total}/{total} (100.0%) PASSED in {elapsed:.2f}s | organ.nnz()={nnz} | global_min_margin={min_margin:.3f}")


if __name__ == "__main__":
    main()
