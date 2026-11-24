"""Brute-force teacher: for every Boolean function of n inputs find a small-integer network
of the fixed shape, and the minimum number of active hidden units it needs (its level).
Exhaustive over weights in {-R..R}. Writes solutions.json."""
import itertools
import json
import sys
import time

import numpy as np

import tasks


def solve_all(n, R=2, log=print):
    H = tasks.HIDDEN[n]
    X = tasks.inputs(n); N = 2 ** n
    vals = list(range(-R, R + 1))
    units = {}                                      # activation pattern -> cheapest (cost, w, b)
    for w in itertools.product(vals, repeat=n):
        for b in vals:
            p = tuple(np.maximum(0, X @ np.array(w) + b).tolist())
            if not any(p):
                continue
            c = (sum(map(abs, w)) + abs(b), w, b)
            if p not in units or c < units[p]:
                units[p] = c
    pats = sorted(units, key=lambda p: units[p])
    P = np.array(pats, dtype=np.int32)              # K x N
    K = len(pats)
    bit = (1 << np.arange(N)).astype(np.int64)
    found = {}                                      # f -> (level, [pattern idx], [v], c)

    def note(codes, level, decode):
        u, first = np.unique(codes, return_index=True)
        for f, i in zip(u.tolist(), first.tolist()):
            if f not in found:
                found[f] = (level,) + decode(i)

    for c in vals:                                  # level 0: constant output
        f = (2 ** N - 1) if c > 0.5 else 0
        found.setdefault(f, (0, (), (), c))
    nz = [v for v in sorted(vals, key=abs) if v != 0]
    total = 2 ** N
    for level in range(1, H + 1):
        t = time.time()
        vcombos = sorted(itertools.product(nz, repeat=level), key=lambda v: sum(map(abs, v)))
        if level == 1:
            for (v,) in vcombos:
                for c in sorted(vals, key=abs):
                    codes = ((v * P + c) > 0.5) @ bit
                    note(codes, 1, lambda i, v=v, c=c: ((i,), (v,), c))
        else:
            # all index tuples for the first level-1 units, last unit broadcast
            head = np.array(list(itertools.combinations(range(K), level - 1)), dtype=np.int32)
            for v in vcombos:
                base = sum(v[j] * P[head[:, j]] for j in range(level - 1))          # M x N
                for k in range(K):
                    keep = head[:, -1] < k                                           # ordered tuples only
                    if not keep.any():
                        continue
                    S = base[keep] + v[-1] * P[k]
                    for c in sorted(vals, key=abs):
                        codes = ((S + c) > 0.5) @ bit
                        hk = head[keep]
                        note(codes, level, lambda i, v=v, c=c, hk=hk, k=k: (tuple(hk[i].tolist()) + (k,), v, c))
                if len(found) == total:
                    break
        log(f"n={n} level {level}: {len(found)}/{total} functions found, {K} unit patterns, {time.time()-t:.1f}s")
        if len(found) == total:
            break

    out = {}
    for f, (level, idx, v, c) in found.items():
        W1 = np.zeros((H, n)); b1 = np.zeros(H); W2 = np.zeros(H)
        for j, (i, vj) in enumerate(zip(idx, v)):
            _, w, b = units[pats[i]]
            W1[j] = w; b1[j] = b; W2[j] = vj
        net = (W1, b1, W2, float(c))
        assert tasks.run(net, n) == f
        out[f] = {"level": level, "answer": tasks.render(net)}
    return out


if __name__ == "__main__":
    res = {}
    for n in (2, 3):
        sol = solve_all(n)
        res[str(n)] = {str(f): s for f, s in sorted(sol.items())}
        lv = [s["level"] for s in sol.values()]
        print(f"n={n}: solved {len(sol)}/{2 ** (2 ** n)}; by level", {l: lv.count(l) for l in sorted(set(lv))})
    json.dump(res, open("solutions.json", "w"))
