#!/bin/bash
# Run 06-domingos-rnn-vs-gpt 250M-token + RoPE airtight control on Colab A100
# (microgpt sinusoidal, microgpt_rope, domingos_lstm, domingos_rnn) with guaranteed auto-stop.
set -uo pipefail
cd "$(dirname "$0")"
COLAB="${COLAB:-$HOME/.local/bin/colab}"
SESSION="${SESSION:-domingos250m}"
ALLOCATED=0

cleanup() {
  if [ "$ALLOCATED" -eq 1 ]; then
    echo "== [cleanup] Downloading 250M Domingos results and stopping '$SESSION' =="
    mkdir -p runs
    "$COLAB" download -s "$SESSION" /content/dom06/runs/domingos_250m_results.json runs/domingos_250m_results.json 2>&1 || true
    "$COLAB" stop -s "$SESSION" 2>&1 || true
    "$COLAB" usage 2>&1 || true
  fi
}
trap cleanup EXIT

for attempt in $(seq 1 120); do
  for GPU in ${GPUS:-A100 L4}; do
    echo "$(date '+%H:%M:%S') == Trying $GPU for '$SESSION' (attempt $attempt/120) =="
    if COLAB_CLI_ASSIGN_RETRIES=0 "$COLAB" new -s "$SESSION" --gpu "$GPU"; then
      ALLOCATED=1
      break 2
    fi
  done
  echo "$(date '+%H:%M:%S') == A100/L4 unavailable; retrying in 60s =="
  sleep 60
done
[ "$ALLOCATED" -eq 1 ] || exit 1
sleep 8

for i in 1 2 3; do
  if echo 'import os; os.makedirs("/content/dom06/runs", exist_ok=True)' | "$COLAB" exec -s "$SESSION" --timeout 60; then
    break
  fi
  sleep 5
done
"$COLAB" upload -s "$SESSION" rnn_vs_gpt.py /content/dom06/rnn_vs_gpt.py || exit 1

cat <<'EOF' | "$COLAB" exec -s "$SESSION" --timeout 7200
import subprocess, sys
# 7630 steps * 64 batch * 512 seq_len = 250,019,840 tokens (~250M tokens)
p = subprocess.Popen(
    [
        sys.executable, "-u", "/content/dom06/rnn_vs_gpt.py",
        "--steps", "7630",
        "--batch-size", "64",
        "--seq-len", "512",
        "--d-model", "384",
        "--n-layers", "6",
        "--n-heads", "6",
        "--lr", "6e-4",
        "--eval-every", "954",
        "--val-seqs", "256",
        "--archs", "microgpt,microgpt_rope,domingos_lstm,domingos_rnn",
        "--out", "/content/dom06/runs/domingos_250m_results.json",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print("DOMINGOS250M_EXIT:", rc, flush=True)
if rc != 0:
    raise RuntimeError(f"rnn_vs_gpt 250m exited with {rc}")
EOF
