#!/bin/bash
# Run 04-neural-graft on a Colab L4 (or A100) with guaranteed auto-stop and JSON sync.
set -uo pipefail
cd "$(dirname "$0")"
COLAB="${COLAB:-$HOME/.local/bin/colab}"
SESSION="${SESSION:-graft04}"

cleanup() {
  echo "== [cleanup] Downloading 04-neural-graft results and stopping '$SESSION' =="
  mkdir -p runs
  "$COLAB" download -s "$SESSION" /content/graft04/runs/graft_results.json runs/graft_results.json 2>&1 || true
  "$COLAB" stop -s "$SESSION" 2>&1 || true
  "$COLAB" usage 2>&1 || true
}
trap cleanup EXIT

ALLOCATED=0
for GPU in L4 A100 T4; do
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
        "--graft-steps", "800",
        "--lora-steps", "300",
        "--batch-size", "32",
        "--out", "/content/graft04/runs/graft_results.json",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print("GRAFT04_EXIT:", rc, flush=True)
if rc != 0:
    raise RuntimeError(f"graft_exp exited with {rc}")
EOF
