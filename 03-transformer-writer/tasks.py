"""Variable-length algorithmic sequence task suite for Transformer Writer.

Every task operates on sequences of data symbols x = [x_1, ..., x_L] with x_i in {0, 1, 2, 3}.
Sequences are fed to TinyTransformer wrapped as [BOS(4), x_1, ..., x_L, EOS(5)].
The target prediction is read from the EOS position (logits[:, :n_classes]).

Splits:
  - "example" (4 tasks) : worked examples in prompt
  - "train"   (22 tasks): verified on L in [2..8] during self-training
  - "heldout" (10 tasks): never trained on; tests zero-shot weight compilation

Every task also has a balanced generator (positive + hard-negative examples) that works at
arbitrary length L, enabling exact OOD length-generalization testing at L = 16, 32, 64.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch

from target_tf import BOS_ID, EOS_ID, TinyTransformer


@dataclass(frozen=True)
class Task:
    name: str
    family: str
    tier: int
    n_classes: int
    alphabet: Tuple[int, ...]
    split: str  # "example" | "train" | "heldout"
    spec: str
    fn: Callable[[List[int]], int]
    gen: Callable[[random.Random, int], Tuple[List[int], int]]


# ---------- Balanced sequence generators (with hard negatives) ----------

def _gen_uniform(alphabet: Tuple[int, ...], fn: Callable[[List[int]], int]):
    def g(rng: random.Random, length: int) -> Tuple[List[int], int]:
        seq = [rng.choice(alphabet) for _ in range(length)]
        return seq, fn(seq)
    return g


def _gen_pos_eq(pos_idx: int, tok: int, alphabet: Tuple[int, ...], fn: Callable[[List[int]], int]):
    def g(rng: random.Random, length: int) -> Tuple[List[int], int]:
        seq = [rng.choice(alphabet) for _ in range(length)]
        idx = pos_idx if pos_idx >= 0 else length + pos_idx
        if rng.random() < 0.5:
            seq[idx] = tok
        else:
            others = [a for a in alphabet if a != tok]
            seq[idx] = rng.choice(others)
        return seq, fn(seq)
    return g


def _gen_has_token(tok: int, alphabet: Tuple[int, ...], fn: Callable[[List[int]], int], want_neg_prob: float = 0.5):
    others = [a for a in alphabet if a != tok]
    def g(rng: random.Random, length: int) -> Tuple[List[int], int]:
        if rng.random() < want_neg_prob:
            seq = [rng.choice(others) for _ in range(length)]
        else:
            # Include sparse-occurrence hard positives (only 1 or 2 copies even at L=64)
            seq = [rng.choice(others) for _ in range(length)]
            k = 1 if rng.random() < 0.6 else rng.randint(1, max(1, length // 2))
            for idx in rng.sample(range(length), k=min(k, length)):
                seq[idx] = tok
        return seq, fn(seq)
    return g


def _gen_has_both(t1: int, t2: int, alphabet: Tuple[int, ...], fn: Callable[[List[int]], int]):
    others = [a for a in alphabet if a not in (t1, t2)]
    def g(rng: random.Random, length: int) -> Tuple[List[int], int]:
        mode = rng.choice(["both", "both", "only1", "only2"])
        if mode == "both":
            seq = [rng.choice(others) for _ in range(length)]
            i1, i2 = rng.sample(range(length), 2)
            seq[i1] = t1
            seq[i2] = t2
            if rng.random() < 0.4:
                for i in range(length):
                    if i not in (i1, i2) and rng.random() < 0.3:
                        seq[i] = rng.choice(alphabet)
        elif mode == "only1":
            pool = others + [t1]
            seq = [rng.choice(pool) for _ in range(length)]
            seq[rng.randrange(length)] = t1
        else:
            pool = others + [t2]
            seq = [rng.choice(pool) for _ in range(length)]
            seq[rng.randrange(length)] = t2
        return seq, fn(seq)
    return g


def _gen_has_either(t1: int, t2: int, alphabet: Tuple[int, ...], fn: Callable[[List[int]], int]):
    others = [a for a in alphabet if a not in (t1, t2)]
    def g(rng: random.Random, length: int) -> Tuple[List[int], int]:
        if rng.random() < 0.5:
            seq = [rng.choice(others) for _ in range(length)]
        else:
            seq = [rng.choice(others) for _ in range(length)]
            seq[rng.randrange(length)] = rng.choice([t1, t2])
            if rng.random() < 0.4:
                seq[rng.randrange(length)] = rng.choice([t1, t2])
        return seq, fn(seq)
    return g


def _gen_all_tok(tok: int, alphabet: Tuple[int, ...], fn: Callable[[List[int]], int]):
    others = [a for a in alphabet if a != tok]
    def g(rng: random.Random, length: int) -> Tuple[List[int], int]:
        if rng.random() < 0.5:
            seq = [tok] * length
        else:
            # Hard negatives: only 1 or 2 dissenting tokens!
            seq = [tok] * length
            k = 1 if rng.random() < 0.7 else rng.randint(1, min(4, length))
            for idx in rng.sample(range(length), k):
                seq[idx] = rng.choice(others)
        return seq, fn(seq)
    return g


def _gen_all_same(alphabet: Tuple[int, ...], fn: Callable[[List[int]], int]):
    def g(rng: random.Random, length: int) -> Tuple[List[int], int]:
        t = rng.choice(alphabet)
        if rng.random() < 0.5:
            seq = [t] * length
        else:
            seq = [t] * length
            others = [a for a in alphabet if a != t]
            k = 1 if rng.random() < 0.7 else rng.randint(1, min(4, length))
            for idx in rng.sample(range(length), k):
                seq[idx] = rng.choice(others)
        return seq, fn(seq)
    return g


def _gen_majority(pos_tok: int, neg_tok: int, alphabet: Tuple[int, ...], fn: Callable[[List[int]], int]):
    others = [a for a in alphabet if a not in (pos_tok, neg_tok)]
    def g(rng: random.Random, length: int) -> Tuple[List[int], int]:
        # Include close-margin examples (diff = -1, 0, +1, +2) so circuits must actually count!
        if rng.random() < 0.6 and length >= 2:
            # Pick close counts of pos_tok and neg_tok
            n_rest = 0 if not others else rng.randint(0, max(0, length // 3))
            rem = length - n_rest
            if rng.random() < 0.5:
                # Positive: n_pos > n_neg
                n_pos = rem // 2 + 1
                if rng.random() < 0.5 and n_pos < rem:
                    n_pos = rng.randint(n_pos, rem)
                n_neg = rem - n_pos
            else:
                # Non-positive: n_pos <= n_neg (include exact tie!)
                n_pos = rem // 2
                if rng.random() < 0.5 and n_pos > 0:
                    n_pos = rng.randint(0, n_pos)
                n_neg = rem - n_pos
            seq = [pos_tok] * n_pos + [neg_tok] * n_neg + [rng.choice(others) for _ in range(n_rest)]
            rng.shuffle(seq)
        else:
            seq = [rng.choice(alphabet) for _ in range(length)]
        return seq, fn(seq)
    return g


def _gen_extremum(is_max: bool, alphabet: Tuple[int, ...], fn: Callable[[List[int]], int]):
    def g(rng: random.Random, length: int) -> Tuple[List[int], int]:
        # Sample target extremum uniformly over alphabet so all 4 classes are balanced even at L=64!
        target = rng.choice(alphabet)
        allowed = [a for a in alphabet if (a <= target if is_max else a >= target)]
        seq = [rng.choice(allowed) for _ in range(length)]
        seq[rng.randrange(length)] = target
        return seq, fn(seq)
    return g


def _gen_bigram(a: int, b: int, alphabet: Tuple[int, ...], fn: Callable[[List[int]], int]):
    def g(rng: random.Random, length: int) -> Tuple[List[int], int]:
        if rng.random() < 0.5:
            # Positive: build negative then inject one (a, b) pair
            seq = _make_no_bigram(rng, length, a, b, alphabet)
            pos = rng.randrange(length - 1)
            seq[pos] = a
            seq[pos + 1] = b
        else:
            # Hard negative: avoid (a, b), while allowing a and b separately (and even (b, a)!)
            seq = _make_no_bigram(rng, length, a, b, alphabet)
        return seq, fn(seq)
    return g


def _make_no_bigram(rng: random.Random, length: int, a: int, b: int, alphabet: Tuple[int, ...]) -> List[int]:
    seq: List[int] = []
    for _ in range(length):
        if seq and seq[-1] == a:
            choices = [x for x in alphabet if x != b]
        else:
            choices = list(alphabet)
        seq.append(rng.choice(choices))
    return seq


def _gen_two_same_adj(alphabet: Tuple[int, ...], fn: Callable[[List[int]], int]):
    def g(rng: random.Random, length: int) -> Tuple[List[int], int]:
        # Build strictly non-repeating adjacent sequence first
        seq = [rng.choice(alphabet)]
        for _ in range(length - 1):
            seq.append(rng.choice([x for x in alphabet if x != seq[-1]]))
        if rng.random() < 0.5:
            # Inject a single adjacent duplicate (hard positive!)
            pos = rng.randrange(length - 1)
            seq[pos + 1] = seq[pos]
        return seq, fn(seq)
    return g


def _gen_sorted(asc: bool, alphabet: Tuple[int, ...], fn: Callable[[List[int]], int]):
    def g(rng: random.Random, length: int) -> Tuple[List[int], int]:
        seq = sorted([rng.choice(alphabet) for _ in range(length)], reverse=not asc)
        if rng.random() < 0.5:
            # Make a hard negative by introducing a single adjacent inversion (if possible)
            diff_positions = [i for i in range(length - 1) if seq[i] != seq[i + 1]]
            if diff_positions:
                i = rng.choice(diff_positions)
                seq[i], seq[i + 1] = seq[i + 1], seq[i]
            else:
                # All equal: perturb one element to create an inversion
                i = rng.randrange(length - 1)
                if asc:
                    seq[i] = min(3, seq[i] + 1)
                    seq[i + 1] = max(0, seq[i] - 1)
                else:
                    seq[i] = max(0, seq[i] - 1)
                    seq[i + 1] = min(3, seq[i] + 1)
        return seq, fn(seq)
    return g


def _gen_palindrome_ends(alphabet: Tuple[int, ...], fn: Callable[[List[int]], int]):
    def g(rng: random.Random, length: int) -> Tuple[List[int], int]:
        seq = [rng.choice(alphabet) for _ in range(length)]
        if rng.random() < 0.5:
            seq[-1] = seq[0]
        else:
            seq[-1] = rng.choice([x for x in alphabet if x != seq[0]])
        return seq, fn(seq)
    return g


def _gen_dyck1(fn: Callable[[List[int]], int]):
    """0 = '(', 1 = ')'. Balanced at even lengths; at odd lengths always invalid."""
    def make_valid(rng: random.Random, n_pairs: int) -> List[int]:
        # Random valid Dyck-1 path of length 2*n_pairs via Catalan-style random walk
        seq: List[int] = []
        opened = 0
        closed = 0
        for _ in range(2 * n_pairs):
            rem_open = n_pairs - opened
            rem_close = n_pairs - closed
            if opened == closed:
                seq.append(0)
                opened += 1
            elif rem_open == 0:
                seq.append(1)
                closed += 1
            else:
                # Weight by remaining budget for varied nesting depths
                if rng.random() < rem_open / (rem_open + rem_close):
                    seq.append(0)
                    opened += 1
                else:
                    seq.append(1)
                    closed += 1
        return seq

    def g(rng: random.Random, length: int) -> Tuple[List[int], int]:
        # Ensure even length for Dyck-1 positives (if odd, round down or allow negative)
        L = length if length % 2 == 0 else length + 1
        n_pairs = L // 2
        if rng.random() < 0.5:
            seq = make_valid(rng, n_pairs)
        else:
            kind = rng.choice(["neg_prefix_equal_count", "neg_prefix_equal_count", "unbalanced_count"])
            seq = make_valid(rng, n_pairs)
            if kind == "neg_prefix_equal_count":
                # Swap a '(' with a later ')' or reverse to keep equal 0/1 counts while dipping < 0!
                # Find a balanced prefix split or just swap index 0 ('(') with a ')'
                close_idxs = [i for i, v in enumerate(seq) if v == 1]
                j = rng.choice(close_idxs)
                seq[0], seq[j] = seq[j], seq[0]
                if fn(seq) == 1:
                    seq[0], seq[-1] = 1, 0
            else:
                # Flip one bit so total count is off by +-2
                idx = rng.randrange(L)
                seq[idx] = 1 - seq[idx]
        return seq, fn(seq)
    return g


def _gen_induction(alphabet: Tuple[int, ...], fn: Callable[[List[int]], int]):
    """Generate [x_1, ..., x_L] where query q = x_L appears at least once in x_1..x_{L-2}
    (with x_{L-1} != q so the follower is strictly before the query).
    Includes distractors (earlier occurrences of q with DIFFERENT followers!) so the
    transformer must match the MOST RECENT occurrence of q.
    """
    def g(rng: random.Random, length: int) -> Tuple[List[int], int]:
        L = max(3, length)
        seq = [rng.choice(alphabet) for _ in range(L)]
        q = seq[-1]
        # Avoid trivial self-adjacent q at L-2 so follower is a distinct earlier token
        if seq[-2] == q:
            seq[-2] = rng.choice([a for a in alphabet if a != q])
        if q not in seq[:-2]:
            pos = rng.randrange(L - 2)
            seq[pos] = q
        # 50% of the time with L >= 5, inject a distractor earlier occurrence of q
        if L >= 5 and rng.random() < 0.5:
            p1 = rng.randrange(0, L - 4)
            p2 = rng.randrange(p1 + 2, L - 2)
            seq[p1] = q
            seq[p2] = q
            f1 = rng.choice(alphabet)
            f2 = rng.choice([a for a in alphabet if a != f1])
            seq[p1 + 1] = f1
            seq[p2 + 1] = f2
            for k in range(p2 + 1, L - 1):
                if seq[k] == q:
                    seq[k] = rng.choice([a for a in alphabet if a != q])
        return seq, fn(seq)
    return g


# ---------- Ground-truth functions ----------

def _fn_dyck1(seq: List[int]) -> int:
    bal = 0
    for x in seq:
        bal += 1 if x == 0 else -1
        if bal < 0:
            return 0
    return 1 if bal == 0 else 0


def _fn_induction(seq: List[int]) -> int:
    q = seq[-1]
    for i in range(len(seq) - 2, -1, -1):
        if seq[i] == q:
            return seq[i + 1]
    return 0


A4 = (0, 1, 2, 3)
A2 = (0, 1)


def build_tasks() -> List[Task]:
    tasks: List[Task] = [
        # ---- Worked examples (4) ----
        Task(
            name="last_tok", family="position", tier=1, n_classes=4, alphabet=A4, split="example",
            spec="Output the last token x_L of the sequence.",
            fn=lambda s: s[-1],
            gen=_gen_uniform(A4, lambda s: s[-1]),
        ),
        Task(
            name="has_0", family="presence", tier=1, n_classes=2, alphabet=A4, split="example",
            spec="Output 1 if token 0 appears anywhere in the sequence, else 0.",
            fn=lambda s: int(0 in s),
            gen=_gen_has_token(0, A4, lambda s: int(0 in s)),
        ),
        Task(
            name="majority_1", family="aggregate", tier=1, n_classes=2, alphabet=A2, split="example",
            spec="Given a binary sequence over {0,1}, output 1 if there are strictly more 1s than 0s, else 0.",
            fn=lambda s: int(s.count(1) > s.count(0)),
            gen=_gen_majority(1, 0, A2, lambda s: int(s.count(1) > s.count(0))),
        ),
        Task(
            name="bigram_0_1", family="bigram", tier=2, n_classes=2, alphabet=A4, split="example",
            spec="Output 1 if the adjacent pair (0, 1) occurs anywhere in the sequence, else 0.",
            fn=lambda s: int(any(s[i] == 0 and s[i + 1] == 1 for i in range(len(s) - 1))),
            gen=_gen_bigram(0, 1, A4, lambda s: int(any(s[i] == 0 and s[i + 1] == 1 for i in range(len(s) - 1)))),
        ),

        # ---- Train tasks (22) ----
        Task(
            name="first_tok", family="position", tier=1, n_classes=4, alphabet=A4, split="train",
            spec="Output the first token x_1 of the sequence.",
            fn=lambda s: s[0],
            gen=_gen_uniform(A4, lambda s: s[0]),
        ),
        Task(
            name="first_is_0", family="position", tier=1, n_classes=2, alphabet=A4, split="train",
            spec="Output 1 if the first token x_1 equals 0, else 0.",
            fn=lambda s: int(s[0] == 0),
            gen=_gen_pos_eq(0, 0, A4, lambda s: int(s[0] == 0)),
        ),
        Task(
            name="first_is_1", family="position", tier=1, n_classes=2, alphabet=A4, split="train",
            spec="Output 1 if the first token x_1 equals 1, else 0.",
            fn=lambda s: int(s[0] == 1),
            gen=_gen_pos_eq(0, 1, A4, lambda s: int(s[0] == 1)),
        ),
        Task(
            name="last_is_0", family="position", tier=1, n_classes=2, alphabet=A4, split="train",
            spec="Output 1 if the last token x_L equals 0, else 0.",
            fn=lambda s: int(s[-1] == 0),
            gen=_gen_pos_eq(-1, 0, A4, lambda s: int(s[-1] == 0)),
        ),
        Task(
            name="last_is_2", family="position", tier=1, n_classes=2, alphabet=A4, split="train",
            spec="Output 1 if the last token x_L equals 2, else 0.",
            fn=lambda s: int(s[-1] == 2),
            gen=_gen_pos_eq(-1, 2, A4, lambda s: int(s[-1] == 2)),
        ),
        Task(
            name="has_1", family="presence", tier=1, n_classes=2, alphabet=A4, split="train",
            spec="Output 1 if token 1 appears anywhere in the sequence, else 0.",
            fn=lambda s: int(1 in s),
            gen=_gen_has_token(1, A4, lambda s: int(1 in s)),
        ),
        Task(
            name="has_2", family="presence", tier=1, n_classes=2, alphabet=A4, split="train",
            spec="Output 1 if token 2 appears anywhere in the sequence, else 0.",
            fn=lambda s: int(2 in s),
            gen=_gen_has_token(2, A4, lambda s: int(2 in s)),
        ),
        Task(
            name="no_0", family="presence", tier=1, n_classes=2, alphabet=A4, split="train",
            spec="Output 1 if token 0 never appears in the sequence, else 0.",
            fn=lambda s: int(0 not in s),
            gen=_gen_has_token(0, A4, lambda s: int(0 not in s)),
        ),
        Task(
            name="has_0_and_1", family="presence", tier=1, n_classes=2, alphabet=A4, split="train",
            spec="Output 1 if both token 0 and token 1 appear in the sequence, else 0.",
            fn=lambda s: int((0 in s) and (1 in s)),
            gen=_gen_has_both(0, 1, A4, lambda s: int((0 in s) and (1 in s))),
        ),
        Task(
            name="has_1_or_2", family="presence", tier=1, n_classes=2, alphabet=A4, split="train",
            spec="Output 1 if token 1 or token 2 appears in the sequence, else 0.",
            fn=lambda s: int((1 in s) or (2 in s)),
            gen=_gen_has_either(1, 2, A4, lambda s: int((1 in s) or (2 in s))),
        ),
        Task(
            name="all_0", family="presence", tier=1, n_classes=2, alphabet=A4, split="train",
            spec="Output 1 if every token in the sequence equals 0, else 0.",
            fn=lambda s: int(all(x == 0 for x in s)),
            gen=_gen_all_tok(0, A4, lambda s: int(all(x == 0 for x in s))),
        ),
        Task(
            name="all_1", family="presence", tier=1, n_classes=2, alphabet=A4, split="train",
            spec="Output 1 if every token in the sequence equals 1, else 0.",
            fn=lambda s: int(all(x == 1 for x in s)),
            gen=_gen_all_tok(1, A4, lambda s: int(all(x == 1 for x in s))),
        ),
        Task(
            name="max_tok", family="aggregate", tier=1, n_classes=4, alphabet=A4, split="train",
            spec="Output the maximum token value max(x_1..x_L) in the sequence.",
            fn=lambda s: max(s),
            gen=_gen_extremum(True, A4, lambda s: max(s)),
        ),
        Task(
            name="majority_0", family="aggregate", tier=1, n_classes=2, alphabet=A2, split="train",
            spec="Given a binary sequence over {0,1}, output 1 if there are strictly more 0s than 1s, else 0.",
            fn=lambda s: int(s.count(0) > s.count(1)),
            gen=_gen_majority(0, 1, A2, lambda s: int(s.count(0) > s.count(1))),
        ),
        Task(
            name="more_2_than_1", family="aggregate", tier=1, n_classes=2, alphabet=A4, split="train",
            spec="Output 1 if token 2 occurs strictly more times than token 1, else 0.",
            fn=lambda s: int(s.count(2) > s.count(1)),
            gen=_gen_majority(2, 1, A4, lambda s: int(s.count(2) > s.count(1))),
        ),
        Task(
            name="bigram_1_2", family="bigram", tier=2, n_classes=2, alphabet=A4, split="train",
            spec="Output 1 if the adjacent pair (1, 2) occurs anywhere in the sequence, else 0.",
            fn=lambda s: int(any(s[i] == 1 and s[i + 1] == 2 for i in range(len(s) - 1))),
            gen=_gen_bigram(1, 2, A4, lambda s: int(any(s[i] == 1 and s[i + 1] == 2 for i in range(len(s) - 1)))),
        ),
        Task(
            name="bigram_2_0", family="bigram", tier=2, n_classes=2, alphabet=A4, split="train",
            spec="Output 1 if the adjacent pair (2, 0) occurs anywhere in the sequence, else 0.",
            fn=lambda s: int(any(s[i] == 2 and s[i + 1] == 0 for i in range(len(s) - 1))),
            gen=_gen_bigram(2, 0, A4, lambda s: int(any(s[i] == 2 and s[i + 1] == 0 for i in range(len(s) - 1)))),
        ),
        Task(
            name="bigram_1_1", family="bigram", tier=2, n_classes=2, alphabet=A4, split="train",
            spec="Output 1 if the adjacent pair (1, 1) occurs anywhere in the sequence, else 0.",
            fn=lambda s: int(any(s[i] == 1 and s[i + 1] == 1 for i in range(len(s) - 1))),
            gen=_gen_bigram(1, 1, A4, lambda s: int(any(s[i] == 1 and s[i + 1] == 1 for i in range(len(s) - 1)))),
        ),
        Task(
            name="two_same_adj", family="order", tier=2, n_classes=2, alphabet=A4, split="train",
            spec="Output 1 if any two adjacent tokens are equal (x_i == x_{i-1}), else 0.",
            fn=lambda s: int(any(s[i] == s[i + 1] for i in range(len(s) - 1))),
            gen=_gen_two_same_adj(A4, lambda s: int(any(s[i] == s[i + 1] for i in range(len(s) - 1)))),
        ),
        Task(
            name="sorted_asc", family="order", tier=2, n_classes=2, alphabet=A4, split="train",
            spec="Output 1 if the sequence is non-decreasing (x_1 <= x_2 <= ... <= x_L), else 0.",
            fn=lambda s: int(all(s[i] <= s[i + 1] for i in range(len(s) - 1))),
            gen=_gen_sorted(True, A4, lambda s: int(all(s[i] <= s[i + 1] for i in range(len(s) - 1)))),
        ),
        Task(
            name="dyck1", family="dyck", tier=2, n_classes=2, alphabet=A2, split="train",
            spec="Balanced parentheses over {0='(', 1=')'}: output 1 if prefix count(0) >= count(1) everywhere and count(0) == count(1) at the end, else 0.",
            fn=_fn_dyck1,
            gen=_gen_dyck1(_fn_dyck1),
        ),
        Task(
            name="induction", family="induction", tier=2, n_classes=4, alphabet=A4, split="train",
            spec="In-context induction: output the token immediately following the most recent earlier occurrence of the final token x_L.",
            fn=_fn_induction,
            gen=_gen_induction(A4, _fn_induction),
        ),

        # ---- Held-out unseen tasks (10) ----
        Task(
            name="has_3", family="presence", tier=1, n_classes=2, alphabet=A4, split="heldout",
            spec="Output 1 if token 3 appears anywhere in the sequence, else 0.",
            fn=lambda s: int(3 in s),
            gen=_gen_has_token(3, A4, lambda s: int(3 in s)),
        ),
        Task(
            name="first_is_3", family="position", tier=1, n_classes=2, alphabet=A4, split="heldout",
            spec="Output 1 if the first token x_1 equals 3, else 0.",
            fn=lambda s: int(s[0] == 3),
            gen=_gen_pos_eq(0, 3, A4, lambda s: int(s[0] == 3)),
        ),
        Task(
            name="min_tok", family="aggregate", tier=1, n_classes=4, alphabet=A4, split="heldout",
            spec="Output the minimum token value min(x_1..x_L) in the sequence.",
            fn=lambda s: min(s),
            gen=_gen_extremum(False, A4, lambda s: min(s)),
        ),
        Task(
            name="has_2_and_3", family="presence", tier=1, n_classes=2, alphabet=A4, split="heldout",
            spec="Output 1 if both token 2 and token 3 appear in the sequence, else 0.",
            fn=lambda s: int((2 in s) and (3 in s)),
            gen=_gen_has_both(2, 3, A4, lambda s: int((2 in s) and (3 in s))),
        ),
        Task(
            name="second_last", family="position", tier=2, n_classes=4, alphabet=A4, split="heldout",
            spec="Output the second-to-last token x_{L-1} of the sequence.",
            fn=lambda s: s[-2],
            gen=_gen_uniform(A4, lambda s: s[-2]),
        ),
        Task(
            name="palindrome_ends", family="composition", tier=1, n_classes=2, alphabet=A4, split="heldout",
            spec="Output 1 if the first token equals the last token (x_1 == x_L), else 0.",
            fn=lambda s: int(s[0] == s[-1]),
            gen=_gen_palindrome_ends(A4, lambda s: int(s[0] == s[-1])),
        ),
        Task(
            name="all_same", family="composition", tier=1, n_classes=2, alphabet=A4, split="heldout",
            spec="Output 1 if all tokens in the sequence are identical, else 0.",
            fn=lambda s: int(len(set(s)) == 1),
            gen=_gen_all_same(A4, lambda s: int(len(set(s)) == 1)),
        ),
        Task(
            name="bigram_3_1", family="bigram", tier=2, n_classes=2, alphabet=A4, split="heldout",
            spec="Output 1 if the adjacent pair (3, 1) occurs anywhere in the sequence, else 0.",
            fn=lambda s: int(any(s[i] == 3 and s[i + 1] == 1 for i in range(len(s) - 1))),
            gen=_gen_bigram(3, 1, A4, lambda s: int(any(s[i] == 3 and s[i + 1] == 1 for i in range(len(s) - 1)))),
        ),
        Task(
            name="sorted_desc", family="order", tier=2, n_classes=2, alphabet=A4, split="heldout",
            spec="Output 1 if the sequence is non-increasing (x_1 >= x_2 >= ... >= x_L), else 0.",
            fn=lambda s: int(all(s[i] >= s[i + 1] for i in range(len(s) - 1))),
            gen=_gen_sorted(False, A4, lambda s: int(all(s[i] >= s[i + 1] for i in range(len(s) - 1)))),
        ),
        Task(
            name="no_same_adj", family="order", tier=2, n_classes=2, alphabet=A4, split="heldout",
            spec="Output 1 if no two adjacent tokens are equal (x_i != x_{i-1} for all i), else 0.",
            fn=lambda s: int(all(s[i] != s[i + 1] for i in range(len(s) - 1))),
            gen=_gen_two_same_adj(A4, lambda s: int(all(s[i] != s[i + 1] for i in range(len(s) - 1)))),
        ),
    ]
    return tasks


TASKS: List[Task] = build_tasks()
TASKS_BY_NAME: Dict[str, Task] = {t.name: t for t in TASKS}


def sample_batch(
    task: Task,
    lengths: Sequence[int],
    n_per_length: int,
    seed: int,
    device=None,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Return {L: (tokens [N, L+2], labels [N])} grouped by exact length for fast batched eval."""
    rng = random.Random(seed)
    by_len: Dict[int, Tuple[List[List[int]], List[int]]] = {}
    for L in lengths:
        for _ in range(n_per_length):
            seq, y = task.gen(rng, L)
            actual_L = len(seq)
            if actual_L not in by_len:
                by_len[actual_L] = ([], [])
            by_len[actual_L][0].append([BOS_ID] + seq + [EOS_ID])
            by_len[actual_L][1].append(y)
    out: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for L, (seqs, ys) in by_len.items():
        toks = torch.tensor(seqs, dtype=torch.long, device=device)
        labs = torch.tensor(ys, dtype=torch.long, device=device)
        out[L] = (toks, labs)
    return out


@torch.no_grad()
def evaluate_model(
    model: TinyTransformer,
    task: Task,
    batches: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> float:
    """Exact classification accuracy across all batches for `task`."""
    correct = 0
    total = 0
    for _, (toks, labs) in batches.items():
        logits = model(toks)[:, : task.n_classes]
        preds = logits.argmax(dim=-1)
        correct += int((preds == labs).sum().item())
        total += int(labs.numel())
    return correct / max(1, total)


def make_eval_suite(
    task: Task,
    seed_base: int = 1000,
    device=None,
) -> Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
    """Standard verification & OOD length-generalization splits for one task.
    - id_train_verif : L in 2..8  (used by the verifier during self-training)
    - id_test        : L in 2..8  (fresh seed, held-out examples)
    - ood_16         : L = 16
    - ood_32         : L = 32
    - ood_64         : L = 64
    """
    h = sum((i + 1) * ord(c) for i, c in enumerate(task.name))
    return {
        "verif": sample_batch(task, [3, 4, 5, 6, 7, 8], n_per_length=32, seed=seed_base + h, device=device),
        "id": sample_batch(task, [3, 4, 5, 6, 7, 8], n_per_length=64, seed=seed_base + 10000 + h, device=device),
        "L16": sample_batch(task, [16], n_per_length=256, seed=seed_base + 20000 + h, device=device),
        "L32": sample_batch(task, [32], n_per_length=256, seed=seed_base + 30000 + h, device=device),
        "L64": sample_batch(task, [64], n_per_length=256, seed=seed_base + 40000 + h, device=device),
    }
