#!/usr/bin/env python
"""Pick the init to continue after the pilot phase.

Rule (decided before the data came in): lowest held-out loss at --step wins; if the best
is within --tie of zero-init, zero wins (simplest, and the exactness story is about it).
The slope over the last 500 steps and the pilot benchmarks are printed for the record.
Prints a markdown table; the LAST line is "winner: <init>" (pilot.sh reads it).

    python pick.py --step 2000 --inits zero split guided
"""
import argparse
import glob
import json
from pathlib import Path


def heldout(name):
    out = {}
    for line in open(Path("runs") / name / "log.jsonl"):
        r = json.loads(line)
        if "step" in r and "heldout_loss" in r:
            out[r["step"]] = r["heldout_loss"]
    return out


def bench(name):
    """Mean of the headline metrics of a bench.sh run, or None."""
    accs = []
    for f in glob.glob(f"runs/bench/{name}/**/results_*.json", recursive=True):
        for task, m in json.load(open(f))["results"].items():
            for k in ("acc_norm,none", "acc,none"):
                if k in m:
                    accs.append(m[k]); break
    return sum(accs) / len(accs) if accs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--inits", nargs="+", default=["zero", "split", "guided"])
    ap.add_argument("--tie", type=float, default=0.01)
    a = ap.parse_args()

    rows = []
    for init in a.inits:
        h = heldout(f"heal-small-{init}")
        now, before = h.get(a.step), h.get(a.step - 500)
        b = bench(f"small-{init}-pilot")
        rows.append((init, now, before, b))

    print(f"| init | held-out @ {a.step} | held-out @ {a.step - 500} | slope | pilot bench mean |")
    print("|---|---|---|---|---|")
    for init, now, before, b in rows:
        slope = f"{now - before:+.4f}" if now is not None and before is not None else "n/a"
        print(f"| {init} | {now:.4f} | {before:.4f} | {slope} | {100 * b:.1f} |" if now is not None and before is not None and b is not None
              else f"| {init} | {now} | {before} | {slope} | {b} |")

    scored = [(now, init) for init, now, _, _ in rows if now is not None]
    best_loss, best = min(scored)
    zero = dict((init, now) for init, now, _, _ in rows).get("zero")
    if zero is not None and best != "zero" and zero - best_loss < a.tie:
        print(f"\n{best} beats zero by only {zero - best_loss:.4f} (< {a.tie}): tie goes to zero.")
        best = "zero"
    print(f"\nwinner: {best}")


if __name__ == "__main__":
    main()
