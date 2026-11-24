#!/bin/bash
# Fast L4 verification of the 2-line deadzone-clamped neural socket fix (Exp 04 Arm 6)
# across all 4 tasks (all_0, bigram_1_2, sorted_asc, dyck1) with guaranteed auto-stop.
set -uo pipefail
cd "$(dirname "$0")"
COLAB="${COLAB:-$HOME/.local/bin/colab}"
SESSION="${SESSION:-graft04dz}"
ALLOCATED=0

cleanup() {
  if [ "$ALLOCATED" -eq 1 ]; then
    echo "== [cleanup] Downloading deadzone results and stopping '$SESSION' =="
    mkdir -p runs
    "$COLAB" download -s "$SESSION" /content/graft04/runs/graft_deadzone.json runs/graft_deadzone.json 2>&1 || true
    "$COLAB" stop -s "$SESSION" 2>&1 || true
    "$COLAB" usage 2>&1 || true
  fi
}
trap cleanup EXIT

for GPU in L4 T4 A100; do
  echo "== Trying $GPU for '$SESSION' =="
  if "$COLAB" new -s "$SESSION" --gpu "$GPU"; then
    ALLOCATED=1
    break
  fi
done
[ "$ALLOCATED" -eq 1 ] || exit 1

echo 'import os; os.makedirs("/content/tw", exist_ok=True); os.makedirs("/content/graft04", exist_ok=True)' | "$COLAB" exec -s "$SESSION" --timeout 60 || exit 1
for f in target_tf.py tasks.py verify_realizability.py writer_loop.py; do
  "$COLAB" upload -s "$SESSION" "../03-transformer-writer/$f" "/content/tw/$f" || exit 1
done
"$COLAB" upload -s "$SESSION" graft_exp.py /content/graft04/graft_exp.py || exit 1

cat <<'EOF' | "$COLAB" exec -s "$SESSION" --timeout 1800
import subprocess, sys
p = subprocess.Popen(
    [
        sys.executable, "-u", "/content/graft04/graft_exp.py",
        "--model", "Qwen/Qwen2.5-1.5B-Instruct",
        "--tasks", "all_0,bigram_1_2,sorted_asc,dyck1",
        "--arms", "compiled_graft_deadzone",
        "--batch-size", "32",
        "--out", "/content/graft04/runs/graft_deadzone.json",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print("GRAFT04DZ_EXIT:", rc, flush=True)
if rc != 0:
    raise RuntimeError(f"graft_exp deadzone exited with {rc}")
EOF
