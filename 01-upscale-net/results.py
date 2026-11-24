#!/usr/bin/env python
"""Collect lm-eval results and healing logs into one markdown table.

    python results.py --mode small     # looks at runs/bench/small-* and runs/heal-small
"""
import argparse
import glob
import json
from pathlib import Path

TASKS = [("hellaswag", "acc_norm,none"), ("arc_easy", "acc_norm,none"), ("arc_challenge", "acc_norm,none"),
         ("piqa", "acc_norm,none"), ("winogrande", "acc,none"), ("gsm8k", "exact_match,flexible-extract"),
         ("wikitext", "word_perplexity,none")]


def load_bench(name):
    """Merge every results_*.json under runs/bench/<name>/ (one per bench.sh invocation)."""
    merged = {}
    for f in sorted(glob.glob(f"runs/bench/{name}/**/results_*.json", recursive=True)):
        merged.update(json.load(open(f)).get("results", {}))
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="small")
    ap.add_argument("--tag", default=None, help="run tag for healed/control (default: mode; e.g. small-self)")
    args = ap.parse_args()
    tag = args.tag or args.mode
    cols = [(f"{args.mode}-source", "source"), (f"{args.mode}-grown-step0", "grown, step 0"),
            (f"{tag}-control", "source + same training (control)"), (f"{tag}-healed", "grown + healed")]
    # reference models benchmarked with bench_refs.sh (runs/bench/ref-<org>__<model>)
    for d in sorted(glob.glob("runs/bench/ref-*")):
        if Path(d).is_dir():
            cols.append((Path(d).name, Path(d).name[4:].split("_", 1)[-1]))   # ref-Org_Model -> Model
    data = {label: load_bench(name) for name, label in cols}

    print(f"| task | " + " | ".join(l for _, l in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for task, metric in TASKS:
        row = []
        for _, label in cols:
            v = data[label].get(task, {}).get(metric)
            if v is None:
                row.append("–")
            elif metric.startswith("word_perplexity"):
                row.append(f"{v:.2f}")
            else:
                row.append(f"{100 * v:.1f}")
        print(f"| {task}{' (ppl ↓)' if task == 'wikitext' else ' (%)'} | " + " | ".join(row) + " |")

    for log in (Path(f"runs/heal-{tag}/log.jsonl"), Path(f"runs/control-{tag}/log.jsonl")):
        if not log.exists():
            continue
        recs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        start = next((r for r in recs if r.get("event") == "start"), None)
        steps = [r for r in recs if "step" in r]
        if start and steps:
            last = steps[-1]
            toks = last["step"] * start["tokens_per_step"]
            evals = [r for r in steps if "heldout_loss" in r]
            print(f"\n{log.parent.name}: {last['step']} steps, {toks/1e6:.0f}M tokens, {start['optimizer']} lr {start['lr']}, "
                  f"~{steps[-1].get('tok_s', 0)} tok/s. Held-out loss {start['heldout_loss']:.3f} -> "
                  f"{evals[-1]['heldout_loss']:.3f}" if evals else "")


if __name__ == "__main__":
    main()
