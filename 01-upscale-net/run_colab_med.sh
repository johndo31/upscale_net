#!/bin/bash
# Run 01-upscale-net MED mode (Option 1B: DeepSeek-R1-Distill-Qwen-1.5B -> 3.80B exact fp32)
# on a Colab A100 40GB (or G4) with 300M tokens + 20% OpenR1-Math-220k + 1.5B control + GSM8K.
# Uses detached remote execution + polling + periodic log/JSON sync + hard MAX_HOURS cap + auto-stop!
set -uo pipefail
cd "$(dirname "$0")"
COLAB="${COLAB:-$HOME/.local/bin/colab}"
SESSION="${SESSION:-ds4b}"
TOKENS="${TOKENS:-100000000}"   # 100M tokens (~1525 steps): ~8.5-9.5h total on A100 (~55-65 units)
LIMIT="${LIMIT:-200}"           # 200 examples/task for crisp lm-eval + GSM8K
MAX_HOURS="${MAX_HOURS:-11}"    # hard safety cap (~74 units max on A100)
MAX_ATTEMPTS="${MAX_ATTEMPTS:-240}"
RETRY_WAIT="${RETRY_WAIT:-60}"
POLL_S="${POLL_S:-60}"
ALLOCATED=0
ATTACH="${ATTACH:-0}"
STOP_VM_ON_EXIT=0

sync_results() {
  echo 'import tarfile, os
os.makedirs("/content/ds4b/runs", exist_ok=True)
with tarfile.open("/content/ds4b/results_med.tar.gz", "w:gz") as tf:
    for root, _, files in os.walk("/content/ds4b/runs"):
        for fn in files:
            if fn.endswith((".json", ".jsonl", ".md", ".log")):
                p = os.path.join(root, fn)
                tf.add(p, arcname=os.path.relpath(p, "/content/ds4b"))
print("packed", os.path.getsize("/content/ds4b/results_med.tar.gz"), "bytes")' | "$COLAB" exec -s "$SESSION" --timeout 120 2>&1 || true
  mkdir -p runs-med-colab
  "$COLAB" download -s "$SESSION" /content/ds4b/results_med.tar.gz runs-med-colab/results_med.tar.gz 2>&1 || true
  if [ -f runs-med-colab/results_med.tar.gz ]; then
    tar -xzf runs-med-colab/results_med.tar.gz -C runs-med-colab 2>&1 || true
  fi
}

cleanup() {
  if [ "${ALLOCATED:-0}" -eq 1 ] && [ "${STOP_VM_ON_EXIT:-0}" -eq 1 ]; then
    echo "== [cleanup] Downloading DeepSeek 3.8B logs/results and stopping '$SESSION' =="
    sync_results
    "$COLAB" stop -s "$SESSION" 2>&1 || true
    "$COLAB" usage 2>&1 || true
  elif [ "${ALLOCATED:-0}" -eq 1 ]; then
    echo "== [cleanup] Local watcher interrupted; leaving detached '$SESSION' running on Colab (reattach with ATTACH=1) =="
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

echo 'import os; os.makedirs("/content/ds4b/runs", exist_ok=True)' | "$COLAB" exec -s "$SESSION" --timeout 60 || { STOP_VM_ON_EXIT=1; exit 1; }
for f in grow.py verify.py heal.py bench.sh results.py run_all.sh; do
  "$COLAB" upload -s "$SESSION" "$f" "/content/ds4b/$f" || { STOP_VM_ON_EXIT=1; exit 1; }
done
# Reuse already-completed Step 2/3 verification + baseline benchmarks if cached locally (saves ~18 min of A100)
if [ -d runs-med-colab/runs/bench/med-source ] && [ -d runs-med-colab/runs/bench/med-grown-step0 ] && [ -f runs-med-colab/runs/verify-med.log ]; then
  tar -czf /tmp/ds4b_seed_cache.tar.gz -C runs-med-colab/runs verify-med.log bench/med-source bench/med-grown-step0 bench/med-source.log bench/med-grown-step0.log 2>/dev/null || true
  if [ -f /tmp/ds4b_seed_cache.tar.gz ]; then
    "$COLAB" upload -s "$SESSION" /tmp/ds4b_seed_cache.tar.gz /content/ds4b/seed_cache.tar.gz || true
    echo 'import tarfile; tarfile.open("/content/ds4b/seed_cache.tar.gz").extractall("/content/ds4b/runs"); print("seeded baseline bench + verify cache")' | "$COLAB" exec -s "$SESSION" --timeout 60 || true
    rm -f /tmp/ds4b_seed_cache.tar.gz
  fi
fi

echo "== Installing deps and launching detached DeepSeek 1.5B -> 3.80B ($TOKENS tokens, LIMIT=$LIMIT) =="
cat <<EOF | "$COLAB" exec -s "$SESSION" --timeout 600 || { STOP_VM_ON_EXIT=1; exit 1; }
import os, subprocess, sys
os.chdir("/content/ds4b")
subprocess.check_call(["chmod", "+x", "bench.sh", "run_all.sh"])
if subprocess.call([sys.executable, "-c", "import lm_eval"]) != 0:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lm-eval", "datasets", "accelerate", "safetensors"])
for fn in ("/content/ds4b/runs/DONE", "/content/ds4b/runs/EXIT_CODE"):
    if os.path.exists(fn):
        os.remove(fn)
env = os.environ.copy()
env["LIMIT"] = "$LIMIT"
env["DEVICE"] = "cuda"
env["PY"] = os.path.dirname(sys.executable)
wrapper = "bash /content/ds4b/run_all.sh med $TOKENS >> /content/ds4b/runs/driver.log 2>&1; echo \$? > /content/ds4b/runs/EXIT_CODE; touch /content/ds4b/runs/DONE"
p = subprocess.Popen(["bash", "-c", wrapper], cwd="/content/ds4b", env=env, start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("LAUNCHED_DS4B_PID:", p.pid, flush=True)
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
log = "/content/ds4b/runs/driver.log"
if os.path.exists(log):
    lines = open(log, errors="replace").read().splitlines()
    print("\\n".join(l[:260] for l in lines[-4:]))
if os.path.exists("/content/ds4b/runs/DONE"):
    rc = open("/content/ds4b/runs/EXIT_CODE").read().strip() if os.path.exists("/content/ds4b/runs/EXIT_CODE") else "?"
    print("DS4B_DONE_RC=" + rc)
else:
    print("DS4B_STATUS=RUNNING")
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
  if echo "$OUT" | grep -q 'DS4B_DONE_RC='; then
    DRC=$(echo "$OUT" | grep -o 'DS4B_DONE_RC=[0-9]*' | tail -1 | cut -d= -f2)
    echo "== ds4b finished with rc=$DRC =="
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
