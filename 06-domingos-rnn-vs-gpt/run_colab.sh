#!/bin/bash
# Run 06-domingos-rnn-vs-gpt on Colab (G4/A100/L4/T4) with live streaming and auto-stop.
set -uo pipefail
cd "$(dirname "$0")"
COLAB="${COLAB:-$HOME/.local/bin/colab}"
SESSION="${SESSION:-domingos06}"

cleanup() {
  echo "== [cleanup] Downloading 06-domingos-rnn-vs-gpt results and stopping '$SESSION' =="
  mkdir -p runs
  "$COLAB" download -s "$SESSION" /content/dom06/runs/domingos_results.json runs/domingos_results.json 2>&1 || true
  "$COLAB" stop -s "$SESSION" 2>&1 || true
  "$COLAB" usage 2>&1 || true
}
trap cleanup EXIT

ALLOCATED=0
for GPU in G4 A100 L4 T4; do
  echo "== Trying $GPU for '$SESSION' =="
  if "$COLAB" new -s "$SESSION" --gpu "$GPU"; then
    ALLOCATED=1
    break
  fi
done
[ "$ALLOCATED" -eq 1 ] || exit 1

echo 'import os; os.makedirs("/content/dom06/runs", exist_ok=True)' | "$COLAB" exec -s "$SESSION" --timeout 60 || exit 1
"$COLAB" upload -s "$SESSION" rnn_vs_gpt.py /content/dom06/rnn_vs_gpt.py || exit 1

cat <<'EOF' | "$COLAB" exec -s "$SESSION" --timeout 1800
import subprocess, sys
p = subprocess.Popen(
    [
        sys.executable, "-u", "/content/dom06/rnn_vs_gpt.py",
        "--steps", "4000",
        "--batch-size", "32",
        "--seq-len", "512",
        "--d-model", "384",
        "--n-layers", "6",
        "--n-heads", "6",
        "--lr", "6e-4",
        "--eval-every", "500",
        "--out", "/content/dom06/runs/domingos_results.json",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print("DOMINGOS06_EXIT:", rc, flush=True)
if rc != 0:
    raise RuntimeError(f"rnn_vs_gpt exited with {rc}")
EOF
