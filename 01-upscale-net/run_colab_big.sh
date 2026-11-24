#!/bin/bash
# Run 01-upscale-net BIG mode (DeepSeek-R1-Distill-Qwen-1.5B -> 8.22B Llama-3-8B shape)
# on a Colab G4 (96 GB Blackwell RTX PRO 6000) with 100M tokens + 20% OpenR1-Math-220k mix,
# 1.5B control, lm-eval + GSM8K, live-streaming Popen, and guaranteed auto-stop!
set -uo pipefail
cd "$(dirname "$0")"
COLAB="${COLAB:-$HOME/.local/bin/colab}"
SESSION="${SESSION:-ds8b}"
TOKENS="${TOKENS:-500000000}"  # 500M tokens (~7629 steps): tests reasoning survival + late healing crossover
LIMIT="${LIMIT:-200}"          # 200 examples/task for fast, crisp benchmark + GSM8K eval

sync_results() {
  echo 'import tarfile, os
with tarfile.open("/content/ds8b/results_big.tar.gz", "w:gz") as tf:
    for root, _, files in os.walk("/content/ds8b/runs"):
        for fn in files:
            if fn.endswith((".json", ".jsonl", ".md", ".log")):
                p = os.path.join(root, fn)
                tf.add(p, arcname=os.path.relpath(p, "/content/ds8b"))
print("packed", os.path.getsize("/content/ds8b/results_big.tar.gz"), "bytes")' | "$COLAB" exec -s "$SESSION" --timeout 120 2>&1 || true
  mkdir -p runs-big-colab
  "$COLAB" download -s "$SESSION" /content/ds8b/results_big.tar.gz runs-big-colab/results_big.tar.gz 2>&1 || true
  if [ -f runs-big-colab/results_big.tar.gz ]; then
    tar -xzf runs-big-colab/results_big.tar.gz -C runs-big-colab 2>&1 || true
  fi
}

cleanup() {
  if [ "${ALLOCATED:-0}" -eq 1 ]; then
    echo "== [cleanup] Downloading DeepSeek 8B logs/results and stopping '$SESSION' =="
    sync_results
    "$COLAB" stop -s "$SESSION" 2>&1 || true
    "$COLAB" usage 2>&1 || true
  fi
}
trap cleanup EXIT

# Needs 80GB+ VRAM -> patiently wait for a G4 (96 GB Blackwell) slot (0 units while waiting)
MAX_ATTEMPTS="${MAX_ATTEMPTS:-240}"
RETRY_WAIT="${RETRY_WAIT:-60}"
ALLOCATED=0
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "$(date '+%H:%M:%S') == Trying G4 (96 GB Blackwell) for '$SESSION' (attempt $attempt/$MAX_ATTEMPTS) =="
  if COLAB_CLI_ASSIGN_RETRIES=0 "$COLAB" new -s "$SESSION" --gpu G4; then
    ALLOCATED=1
    break
  fi
  echo "$(date '+%H:%M:%S') == G4 unavailable; retrying in ${RETRY_WAIT}s =="
  sleep "$RETRY_WAIT"
done
[ "$ALLOCATED" -eq 1 ] || exit 1

echo 'import os; os.makedirs("/content/ds8b", exist_ok=True)' | "$COLAB" exec -s "$SESSION" --timeout 60 || exit 1
for f in grow.py verify.py heal.py bench.sh results.py run_all.sh; do
  "$COLAB" upload -s "$SESSION" "$f" "/content/ds8b/$f" || exit 1
done

echo "== Installing lm-eval and running DeepSeek 1.5B -> 8.22B ($TOKENS tokens, LIMIT=$LIMIT) =="
cat <<EOF | "$COLAB" exec -s "$SESSION" --timeout 43200
import os, subprocess, sys
os.chdir("/content/ds8b")
subprocess.check_call(["chmod", "+x", "bench.sh", "run_all.sh"])
# Install lm-eval quietly if not already installed
rc_chk = subprocess.call([sys.executable, "-c", "import lm_eval"])
if rc_chk != 0:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lm-eval", "datasets", "accelerate", "safetensors"])
env = os.environ.copy()
env["LIMIT"] = "$LIMIT"
env["DEVICE"] = "cuda"
env["PY"] = os.path.dirname(sys.executable)
p = subprocess.Popen(
    ["bash", "/content/ds8b/run_all.sh", "big", "$TOKENS"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print("DS8B_EXIT:", rc, flush=True)
if rc != 0:
    raise RuntimeError(f"run_all.sh big exited with {rc}")
EOF
