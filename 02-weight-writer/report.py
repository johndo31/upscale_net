"""Summarise runs/*/round*.json as markdown tables.  Usage: python report.py [runs_dir ...]
With several dirs, rows are labelled <dir name>/<arm>."""
import glob
import json
import os
import re
import sys

ORDER = {"base": 0, "teacher": 1, "pass": 2, "hindsight": 3, "unverified": 4}


def fmt(x):
    return "-" if x is None else f"{x:.3f}"


def main(*roots):
    roots = roots or ("runs",)
    rows = []
    for di, root in enumerate(roots):
        for p in glob.glob(os.path.join(root, "*", "round*.json")):
            arm = os.path.basename(os.path.dirname(p))
            r = int(re.search(r"round(\d+)\.json$", p).group(1))
            try:
                d = json.load(open(p))
            except Exception as e:
                print(f"skipping {p}: {e}", file=sys.stderr)
                continue
            label = f"{os.path.basename(os.path.normpath(root))}/{arm}" if len(roots) > 1 else arm
            rows.append((di, ORDER.get(arm, 9), label, r, d))
    rows.sort(key=lambda x: x[:4])
    rows = [x[1:] for x in rows]
    if not rows:
        print(f"no results under {', '.join(roots)}")
        return
    print("| arm | round | pool | fns | train solved@k | train pass | held greedy | held pass | held solved@k | held solved@64 | valid |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    levels = set()
    for _, arm, r, d in rows:
        if d.get("empty_pool"):
            print(f"| {arm} | {r} | 0 | 0 | empty pool, arm stopped |||||||")
            continue
        m, k = d["metrics"], d["k"]
        tr, ho = m.get("train", {}), m.get("heldout", {})
        nv = sum(x.get("n_tasks", 0) for x in (tr, ho))
        valid = sum(x.get("valid_rate", 0) * x.get("n_tasks", 0) for x in (tr, ho)) / nv if nv else None
        levels |= set(ho.get("by_level", {}))
        print(f"| {arm} | {r} | {d.get('pool_size', '-')} | {d.get('pool_functions', '-')} | {fmt(tr.get(f'solved@{k}'))} | "
              f"{fmt(tr.get('mean_pass'))} | {fmt(ho.get('greedy_pass'))} | {fmt(ho.get('mean_pass'))} | "
              f"{fmt(ho.get(f'solved@{k}'))} | {fmt(ho.get('solved@64'))} | {fmt(valid)} |")
    print(f"\n(k = {rows[0][3].get('k', '?')} samples per task; valid = valid-answer rate over all sampled answers)\n")
    levels = sorted(levels, key=int)
    print("Held-out solved@k by level (number of tasks in brackets)\n")
    print("| arm | round | " + " | ".join(f"level {l}" for l in levels) + " |")
    print("|---|---|" + "---|" * len(levels))
    for _, arm, r, d in rows:
        if d.get("empty_pool"):
            continue
        k = d["k"]
        bl = d["metrics"].get("heldout", {}).get("by_level", {})
        cells = []
        for l in levels:
            x = bl.get(l)
            cells.append(f"{fmt(x.get(f'solved@{k}'))} ({x['n_tasks']})" if x and x.get("n_tasks") else "-")
        print(f"| {arm} | {r} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(*(sys.argv[1:] or [os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")]))
