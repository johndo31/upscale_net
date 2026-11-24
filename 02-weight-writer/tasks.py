"""Tasks, prompt format and the exact verifier for the weight-writing experiment.

A task is a Boolean function of n inputs, given as a truth table. The model must write the
weights of a fixed tiny network:  hidden = relu(W1 @ x + b1);  output = 1 if W2 @ hidden + b2 > 0.5.
The verifier parses the answer, runs the network on all 2^n inputs and compares.
A function is an integer f: bit i of f is the output for input row i (rows in order 00..0 to 11..1).
"""
import ast
import itertools
import re

import numpy as np

HIDDEN = {2: 2, 3: 3}          # hidden units per number of inputs


def inputs(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int64)


def tt_str(f, n):
    return "".join(str((f >> i) & 1) for i in range(2 ** n))


def p_class(f, n):
    """Canonical representative of f under permutations of the inputs."""
    X = inputs(n)
    idx = {tuple(r): i for i, r in enumerate(X.tolist())}
    best = None
    for perm in itertools.permutations(range(n)):
        g = 0
        for i, r in enumerate(X.tolist()):
            j = idx[tuple(r[p] for p in perm)]
            g |= ((f >> j) & 1) << i
        best = g if best is None else min(best, g)
    return best


# ---------------------------------------------------------------- networks
def run(net, n):
    """Truth table (as int) computed by a network."""
    W1, b1, W2, b2 = net
    h = np.maximum(0.0, inputs(n) @ W1.T + b1)
    y = h @ W2 + b2
    return int(sum(int(v) << i for i, v in enumerate(y > 0.5)))


def _num(x):
    x = float(x)
    return str(int(x)) if x.is_integer() else repr(round(x, 4))


def render(net):
    W1, b1, W2, b2 = net
    rows = ", ".join("[" + ", ".join(_num(v) for v in r) + "]" for r in W1)
    return (f"W1 = [{rows}]\nb1 = [{', '.join(_num(v) for v in b1)}]\n"
            f"W2 = [{', '.join(_num(v) for v in W2)}]\nb2 = {_num(b2)}")


def size(net):
    """Smaller is simpler: active hidden units, then total |weight|."""
    W1, b1, W2, b2 = net
    return (int((W2 != 0).sum()), float(np.abs(W1).sum() + np.abs(b1).sum() + np.abs(W2).sum() + abs(b2)))


_LINE = {k: re.compile(rf"^[ \t]*{k}[ \t]*=[ \t]*(.+?)[ \t]*$", re.M) for k in ("W1", "b1", "W2", "b2")}


def _numeric(v):
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    return isinstance(v, list) and all(_numeric(x) for x in v)


def first_block(text):
    return text.split("\n\n")[0].split("\nTask")[0].strip("\n")


def parse(text, n):
    """Model answer -> network, or None if it is malformed or has the wrong shape."""
    H = HIDDEN[n]
    block = first_block(text)
    vals = {}
    for k, rx in _LINE.items():
        m = rx.search(block)
        if not m or len(m.group(1)) > 400:
            return None
        try:
            v = ast.literal_eval(m.group(1))        # literals only: no code is executed
        except Exception:
            return None
        if not _numeric(v):
            return None
        vals[k] = v
    try:
        W1 = np.array(vals["W1"], dtype=float)
        b1 = np.array(vals["b1"], dtype=float)
        W2 = np.array(vals["W2"], dtype=float).reshape(-1)
        b2 = np.array(vals["b2"], dtype=float).reshape(-1)
    except Exception:
        return None
    if W1.shape != (H, n) or b1.shape != (H,) or W2.shape != (H,) or b2.shape != (1,):
        return None
    for a in (W1, b1, W2, b2):
        if not np.isfinite(a).all() or np.abs(a).max() > 1e6:
            return None
    return W1, b1, W2, float(b2[0])


# ---------------------------------------------------------------- prompt
HEADER = ("Write the weights of a small neural network that computes a truth table.\n"
          "hidden = relu(W1 @ x + b1), output = 1 if W2 @ hidden + b2 > 0.5 else 0.\n\n")


def task_text(f, n):
    rows = "\n".join(f"{''.join(map(str, r))} -> {(f >> i) & 1}" for i, r in enumerate(inputs(n).tolist()))
    return f"Task: {n} inputs, {HIDDEN[n]} hidden units.\n{rows}\n"


def _net(W1, b1, W2, b2):
    return np.array(W1, float), np.array(b1, float), np.array(W2, float), float(b2)


# the only human-written solutions in the experiment: AND, XOR, 3-input majority, 3-input NOR
FEWSHOT = [
    (2, 8, _net([[1, 1], [0, 0]], [-1, 0], [1, 0], 0)),
    (2, 6, _net([[1, 1], [1, 1]], [0, -1], [1, -2], 0)),
    (3, 232, _net([[1, 1, 1], [0, 0, 0], [0, 0, 0]], [-1, 0, 0], [1, 0, 0], 0)),
    (3, 1, _net([[1, 1, 1], [0, 0, 0], [0, 0, 0]], [0, 0, 0], [-1, 0, 0], 1)),
]
for _n, _f, _w in FEWSHOT:
    assert run(_w, _n) == _f, (_n, _f, run(_w, _n))
    assert parse(render(_w), _n) is not None
PREFIX = HEADER + "".join(task_text(f, n) + render(w) + "\n\n" for n, f, w in FEWSHOT)


def prompt(f, n):
    return PREFIX + task_text(f, n)
