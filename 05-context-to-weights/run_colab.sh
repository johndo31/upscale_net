#!/bin/bash
# Run 05-context-to-weights on a Colab L4 (or A100) with guaranteed auto-stop and JSON sync.
set -uo pipefail
cd "$(dirname "$0")"
COLAB="${COLAB:-$HOME/.local/bin/colab}"
SESSION="${SESSION:-c2w05}"

cleanup() {
  echo "== [cleanup] Downloading 05-context-to-weights results and stopping '$SESSION' =="
  mkdir -p runs
  "$COLAB" download -s "$SESSION" /content/c2w05/runs/c2w_results.json runs/c2w_results.json 2>&1 || true
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

echo 'import os; os.makedirs("/content/tw", exist_ok=True); os.makedirs("/content/c2w05", exist_ok=True)' | "$COLAB" exec -s "$SESSION" --timeout 60 || exit 1
for f in target_tf.py tasks.py verify_realizability.py writer_loop.py; do
  "$COLAB" upload -s "$SESSION" "../03-transformer-writer/$f" "/content/tw/$f" || exit 1
done
"$COLAB" upload -s "$SESSION" c2w_exp.py /content/c2w05/c2w_exp.py || exit 1

cat <<'EOF' | "$COLAB" exec -s "$SESSION" --timeout 1800
import subprocess, sys
p = subprocess.Popen(
    [
        sys.executable, "-u", "/content/c2w05/c2w_exp.py",
        "--model", "Qwen/Qwen2.5-1.5B-Instruct",
        "--steps-per-card", "30",
        "--lr", "3e-4",
        "--out", "/content/c2w05/runs/c2w_results.json",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print("C2W05_EXIT:", rc, flush=True)
if rc != 0:
    raise RuntimeError(f"c2w_exp exited with {rc}")
EOF
