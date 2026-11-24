#!/bin/bash
# Run 04b Real-World BPE Neural Organ Grafting (dual-rail deadzone fix on add_carry_64
# + full 4-arm decimal_sorted_64, merging into runs/real_graft_results.json) with auto-stop.
set -uo pipefail
cd "$(dirname "$0")"
COLAB="${COLAB:-$HOME/.local/bin/colab}"
SESSION="${SESSION:-graft04real}"
ALLOCATED=0

cleanup() {
  if [ "$ALLOCATED" -eq 1 ]; then
    echo "== [cleanup] Downloading 04b real-BPE graft results and stopping '$SESSION' =="
    mkdir -p runs
    "$COLAB" download -s "$SESSION" /content/graft04/runs/real_graft_results.json runs/real_graft_results.json 2>&1 || true
    "$COLAB" stop -s "$SESSION" 2>&1 || true
    "$COLAB" usage 2>&1 || true
  fi
}
trap cleanup EXIT

for attempt in $(seq 1 30); do
  for GPU in L4 A100 T4; do
    echo "$(date '+%H:%M:%S') == Trying $GPU for '$SESSION' (attempt $attempt/30) =="
    if COLAB_CLI_ASSIGN_RETRIES=0 "$COLAB" new -s "$SESSION" --gpu "$GPU"; then
      ALLOCATED=1
      break 2
    fi
  done
  echo "$(date '+%H:%M:%S') == L4/A100/T4 unavailable; retrying in 45s =="
  sleep 45
done
[ "$ALLOCATED" -eq 1 ] || exit 1
sleep 4

echo 'import os; os.makedirs("/content/tw", exist_ok=True); os.makedirs("/content/graft04/runs", exist_ok=True)' | "$COLAB" exec -s "$SESSION" --timeout 60 || exit 1
for f in target_tf.py tasks.py verify_realizability.py writer_loop.py; do
  "$COLAB" upload -s "$SESSION" "../03-transformer-writer/$f" "/content/tw/$f" || exit 1
done
"$COLAB" upload -s "$SESSION" real_graft_exp.py /content/graft04/real_graft_exp.py || exit 1
if [ -f runs/real_graft_results.json ]; then
  "$COLAB" upload -s "$SESSION" runs/real_graft_results.json /content/graft04/runs/real_graft_results.json || exit 1
fi

cat <<'EOF' | "$COLAB" exec -s "$SESSION" --timeout 3600
import subprocess, sys

cmds = [
    [
        sys.executable, "-u", "/content/graft04/real_graft_exp.py",
        "--model", "Qwen/Qwen2.5-1.5B-Instruct",
        "--tasks", "add_carry_64",
        "--arms", "compiled_graft_deadzone",
        "--batch-size", "32",
        "--out", "/content/graft04/runs/real_graft_results.json",
    ],
    [
        sys.executable, "-u", "/content/graft04/real_graft_exp.py",
        "--model", "Qwen/Qwen2.5-1.5B-Instruct",
        "--tasks", "decimal_sorted_64",
        "--arms", "all",
        "--graft-steps", "800",
        "--lora-steps", "350",
        "--batch-size", "32",
        "--out", "/content/graft04/runs/real_graft_results.json",
    ],
]
for cmd in cmds:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    print("CMD_EXIT:", rc, flush=True)
    if rc != 0:
        raise RuntimeError(f"Command exited with {rc}")
EOF
