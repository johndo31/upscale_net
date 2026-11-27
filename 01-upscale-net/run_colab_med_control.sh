#!/bin/bash
# Run ONLY the 1.5B control (control-med + med-control benchmarks) matching heal-med (ds4b)
# exact hyperparameters (SEQ=2048, BATCH=1, ACC=32, Adafactor lr=1e-3, 100M tokens = 1525 steps,
# FineWeb-Edu + 20% OpenR1-Math-220k, seed=0) so held-out batches are token-for-token identical.
set -uo pipefail
cd "$(dirname "$0")"
COLAB="${COLAB:-$HOME/.local/bin/colab}"
SESSION="${SESSION:-ds15ctrl}"
TOKENS="${TOKENS:-100000000}"
LIMIT="${LIMIT:-200}"
MAX_HOURS="${MAX_HOURS:-8}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-120}"
RETRY_WAIT="${RETRY_WAIT:-60}"
POLL_S="${POLL_S:-60}"
ALLOCATED=0
ATTACH="${ATTACH:-0}"
STOP_VM_ON_EXIT=0

sync_results() {
  python3 -c '
import subprocess, sys
code = """import tarfile, os
os.makedirs("/content/ds15ctrl/runs", exist_ok=True)
with tarfile.open("/content/ds15ctrl/results_ctrl.tar.gz", "w:gz") as tf:
    for root, _, files in os.walk("/content/ds15ctrl/runs"):
        for fn in files:
            if fn.endswith((".json", ".jsonl", ".md", ".log")):
                p = os.path.join(root, fn)
                tf.add(p, arcname=os.path.relpath(p, "/content/ds15ctrl"))
print("packed", os.path.getsize("/content/ds15ctrl/results_ctrl.tar.gz"), "bytes")
"""
try:
    r = subprocess.run([sys.argv[1], "exec", "-s", sys.argv[2], "--timeout", "90"],
                       input=code, text=True, capture_output=True, timeout=120)
    sys.stdout.write(r.stdout + r.stderr)
except Exception as e:
    print("sync_pack_err:", e)
' "$COLAB" "$SESSION" 2>&1 || true
  mkdir -p runs-med-colab
  "$COLAB" download -s "$SESSION" /content/ds15ctrl/results_ctrl.tar.gz runs-med-colab/results_ctrl.tar.gz 2>&1 || true
  if [ -f runs-med-colab/results_ctrl.tar.gz ]; then
    tar -xzf runs-med-colab/results_ctrl.tar.gz -C runs-med-colab 2>&1 || true
  fi
}

cleanup() {
  if [ "${ALLOCATED:-0}" -eq 1 ] && [ "${STOP_VM_ON_EXIT:-0}" -eq 1 ]; then
    echo "== [cleanup] Downloading control-med logs/results and stopping '$SESSION' =="
    sync_results
    "$COLAB" stop -s "$SESSION" 2>&1 || true
    "$COLAB" usage 2>&1 || true
  elif [ "${ALLOCATED:-0}" -eq 1 ]; then
    echo "== [cleanup] Local watcher interrupted; leaving detached '$SESSION' alive (reattach with ATTACH=1) =="
  fi
}
trap cleanup EXIT
trap 'echo "== local watcher interrupted (leaving remote VM alive) =="; STOP_VM_ON_EXIT=0; exit 130' INT TERM

if [ "$ATTACH" = "1" ]; then
  echo "$(date '+%H:%M:%S') == Attaching to existing '$SESSION' session (MAX_HOURS=$MAX_HOURS) =="
  ALLOCATED=1
else
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    for GPU in A100; do
      echo "$(date '+%H:%M:%S') == Trying $GPU for '$SESSION' (attempt $attempt/$MAX_ATTEMPTS) =="
      if COLAB_CLI_ASSIGN_RETRIES=0 "$COLAB" new -s "$SESSION" --gpu "$GPU"; then
        ALLOCATED=1
        break 2
      fi
    done
    echo "$(date '+%H:%M:%S') == A100 unavailable; retrying in ${RETRY_WAIT}s =="
    sleep "$RETRY_WAIT"
  done
  [ "$ALLOCATED" -eq 1 ] || exit 1
  sleep 5

  echo 'import os; os.makedirs("/content/ds15ctrl/runs", exist_ok=True)' | "$COLAB" exec -s "$SESSION" --timeout 60 || { STOP_VM_ON_EXIT=1; exit 1; }
  for f in heal.py bench.sh; do
    "$COLAB" upload -s "$SESSION" "$f" "/content/ds15ctrl/$f" || { STOP_VM_ON_EXIT=1; exit 1; }
  done

  echo "== Installing deps and launching detached control-med (1.5B, $TOKENS tokens = 1525 steps, LIMIT=$LIMIT) =="
  cat <<EOF | "$COLAB" exec -s "$SESSION" --timeout 600 || { STOP_VM_ON_EXIT=1; exit 1; }
import os, subprocess, sys
os.chdir("/content/ds15ctrl")
subprocess.check_call(["chmod", "+x", "bench.sh"])
if subprocess.call([sys.executable, "-c", "import lm_eval"]) != 0:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lm-eval", "datasets", "accelerate", "safetensors"])
for fn in ("/content/ds15ctrl/runs/DONE", "/content/ds15ctrl/runs/EXIT_CODE"):
    if os.path.exists(fn):
        os.remove(fn)
env = os.environ.copy()
env["LIMIT"] = "$LIMIT"
env["DEVICE"] = "cuda"
env["PY"] = os.path.dirname(sys.executable)
env["BENCH_REASONING"] = "1"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
cmd = (
    "set -euo pipefail; "
    "python3 heal.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --name control-med "
    "--device cuda --steps 1525 --seq-len 2048 --batch 1 --grad-accum 32 "
    "--optimizer adafactor --lr 1e-3 --warmup 100 --eval-every 100 --save-every 250 --resume "
    "--mix-dataset open-r1/OpenR1-Math-220k --mix-subset default --mix-ratio 0.2 "
    "2>&1 | tee -a runs/control-med.log; "
    "./bench.sh runs/control-med/model med-control $LIMIT cuda"
)
wrapper = f"bash -c {cmd!r} >> /content/ds15ctrl/runs/driver_ctrl.log 2>&1; echo \$? > /content/ds15ctrl/runs/EXIT_CODE; touch /content/ds15ctrl/runs/DONE"
p = subprocess.Popen(["bash", "-c", wrapper], cwd="/content/ds15ctrl", env=env, start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("LAUNCHED_CTRL_PID:", p.pid, flush=True)
EOF
fi

START=$(date +%s)
DEADLINE=$(( START + MAX_HOURS * 3600 ))
POLLS=0
FAILS=0
while true; do
  sleep "$POLL_S"
  POLLS=$(( POLLS + 1 ))
  OUT=$(python3 -c '
import subprocess, sys
code = """import os
log = "/content/ds15ctrl/runs/driver_ctrl.log"
if os.path.exists(log):
    lines = open(log, errors="replace").read().splitlines()
    print("\\n".join(l[:260] for l in lines[-4:]))
if os.path.exists("/content/ds15ctrl/runs/DONE"):
    rc = open("/content/ds15ctrl/runs/EXIT_CODE").read().strip() if os.path.exists("/content/ds15ctrl/runs/EXIT_CODE") else "?"
    print("CTRL_DONE_RC=" + rc)
else:
    print("CTRL_STATUS=RUNNING")
"""
try:
    r = subprocess.run([sys.argv[1], "exec", "-s", sys.argv[2], "--timeout", "90"],
                       input=code, text=True, capture_output=True, timeout=120)
    sys.stdout.write(r.stdout + r.stderr)
    sys.exit(r.returncode)
except subprocess.TimeoutExpired:
    print("LOCAL_POLL_TIMEOUT")
    sys.exit(124)
' "$COLAB" "$SESSION" 2>&1)
  RC=$?
  MIN=$(( ( $(date +%s) - START ) / 60 ))
  if [ "$RC" -ne 0 ]; then
    FAILS=$(( FAILS + 1 ))
    echo "$(date '+%H:%M:%S') [+${MIN}m] poll failed ($FAILS/5)"
    [ "$FAILS" -ge 5 ] && exit 4
    continue
  fi
  FAILS=0
  echo "$(date '+%H:%M:%S') [+${MIN}m] $OUT" | tail -n 5
  if echo "$OUT" | grep -q 'CTRL_DONE_RC='; then
    DRC=$(echo "$OUT" | grep -o 'CTRL_DONE_RC=[0-9]*' | tail -1 | cut -d= -f2)
    echo "== ds15ctrl finished with rc=$DRC =="
    STOP_VM_ON_EXIT=1
    [ "$DRC" = "0" ] && exit 0 || exit 5
  fi
  if [ $(( POLLS % 15 )) -eq 0 ]; then
    sync_results
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "== MAX_HOURS=$MAX_HOURS reached; syncing and stopping =="
    STOP_VM_ON_EXIT=1
    exit 2
  fi
done
