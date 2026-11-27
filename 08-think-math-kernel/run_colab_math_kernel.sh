#!/bin/bash
# Run 08-think-math-kernel (think_math_kernel_exp.py) on Colab as a DETACHED remote process
# with polling, periodic JSON/log/traces sync, and clean auto-stop on completion or MAX_HOURS.
#
# Usage:
#   SMOKE=1 bash run_colab_math_kernel.sh         # quick smoke test -> runs/math_kernel_<tag>_smoke_results.json
#   bash run_colab_math_kernel.sh                 # full run -> runs/math_kernel_<tag>_results.json
#   MODEL=Qwen/Qwen3-4B bash run_colab_math_kernel.sh
#   ATTACH=1 bash run_colab_math_kernel.sh        # reattach watcher to running remote session
#
# Local INT/TERM leaves the detached remote VM alive (reattach with ATTACH=1).
# Normal completion or MAX_HOURS downloads final results, stops the VM, and prints colab usage.
set -uo pipefail
cd "$(dirname "$0")"

COLAB="${COLAB:-$HOME/.local/bin/colab}"
GPUS="${GPUS:-L4 A100 T4}"
SESSION="${SESSION:-mathkernel}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
TAG="${TAG:-}"
if [ -z "$TAG" ]; then
  case "$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')" in
    deepseek-ai/deepseek-r1-distill-qwen-1.5b) TAG="r1-1.5b" ;;
    qwen/qwen3-4b) TAG="qwen3-4b" ;;
    xiaomimimo/mimo-7b-rl) TAG="mimo-7b-rl" ;;
    *) TAG="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/-/g')" ;;
  esac
fi
MAX_HOURS="${MAX_HOURS:-4}"
SMOKE="${SMOKE:-0}"
ATTACH="${ATTACH:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
RETRY_WAIT="${RETRY_WAIT:-45}"
POLL_S="${POLL_S:-60}"
SYNC_EVERY=5   # polls between JSON/log/traces downloads (= 5 minutes)

if [ "$SMOKE" = "1" ]; then
  NAME="math_kernel_${TAG}_smoke"
  SMOKE_FLAG="--smoke"
else
  NAME="math_kernel_${TAG}"
  SMOKE_FLAG=""
fi

REMOTE="/content/mathkernel"
L_JSON="runs/${NAME}_results.json"
L_LOG="runs/${NAME}.log"
L_TRACES="runs/${NAME}_traces.jsonl"
R_JSON="$REMOTE/$L_JSON"
R_LOG="$REMOTE/$L_LOG"
R_TRACES="$REMOTE/$L_TRACES"
R_DONE="$REMOTE/runs/${NAME}.DONE"
ALLOCATED=0
STOP_VM_ON_EXIT=0

ts() { date '+%Y-%m-%d %H:%M:%S'; }

sync_results() {
  mkdir -p runs
  "$COLAB" download -s "$SESSION" "$R_JSON" "$L_JSON" 2>&1 | tail -1 || true
  "$COLAB" download -s "$SESSION" "$R_LOG" "$L_LOG" 2>&1 | tail -1 || true
  "$COLAB" download -s "$SESSION" "$R_TRACES" "$L_TRACES" 2>&1 | tail -1 || true
}

cleanup() {
  if [ "$ALLOCATED" -eq 1 ] && [ "$STOP_VM_ON_EXIT" -eq 1 ]; then
    echo "$(ts) == [cleanup] downloading $L_JSON, $L_LOG, and $L_TRACES, stopping '$SESSION' =="
    sync_results
    "$COLAB" stop -s "$SESSION" 2>&1 || true
    "$COLAB" usage 2>&1 || true
  elif [ "$ALLOCATED" -eq 1 ]; then
    echo "$(ts) == [cleanup] local watcher detached; leaving '$SESSION' alive (reattach with ATTACH=1) =="
  fi
}
trap cleanup EXIT
trap 'echo "$(ts) == local watcher interrupted (leaving remote VM alive) =="; STOP_VM_ON_EXIT=0; exit 130' INT TERM

if [ "$ATTACH" = "1" ]; then
  echo "$(ts) == Attaching to existing '$SESSION' session (MAX_HOURS=$MAX_HOURS) =="
  ALLOCATED=1
else
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    for GPU in $GPUS; do
      echo "$(ts) == Trying $GPU for '$SESSION' (attempt $attempt/$MAX_ATTEMPTS) =="
      if COLAB_CLI_ASSIGN_RETRIES=0 "$COLAB" new -s "$SESSION" --gpu "$GPU"; then
        ALLOCATED=1
        break 2
      fi
    done
    echo "$(ts) == $GPUS unavailable; retrying in ${RETRY_WAIT}s =="
    sleep "$RETRY_WAIT"
  done
  [ "$ALLOCATED" -eq 1 ] || exit 1
  sleep 4

  echo "import os; os.makedirs('$REMOTE/runs/ckpts', exist_ok=True)" | "$COLAB" exec -s "$SESSION" --timeout 60 || { STOP_VM_ON_EXIT=1; exit 1; }
  "$COLAB" upload -s "$SESSION" math_kernel.py "$REMOTE/math_kernel.py" || { STOP_VM_ON_EXIT=1; exit 1; }
  "$COLAB" upload -s "$SESSION" think_math_kernel_exp.py "$REMOTE/think_math_kernel_exp.py" || { STOP_VM_ON_EXIT=1; exit 1; }
  if [ -f "$L_JSON" ]; then
    echo "$(ts) == uploading existing $L_JSON for resume =="
    "$COLAB" upload -s "$SESSION" "$L_JSON" "$R_JSON" || { STOP_VM_ON_EXIT=1; exit 1; }
  fi

  echo "$(ts) == launching detached run (model=$MODEL tag=$TAG smoke=$SMOKE) =="
  cat <<EOF | "$COLAB" exec -s "$SESSION" --timeout 180 || { STOP_VM_ON_EXIT=1; exit 1; }
import os, shlex, subprocess, sys
if os.path.exists("$R_DONE"):
    os.remove("$R_DONE")
if subprocess.call([sys.executable, "-c", "import datasets, transformers"]) != 0:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets", "transformers", "accelerate"])
cmd = [sys.executable, "-u", "$REMOTE/think_math_kernel_exp.py", "--model", "$MODEL", "--tag", "$TAG", "--out", "$R_JSON"] + shlex.split("$SMOKE_FLAG $EXTRA_ARGS")
line = " ".join(shlex.quote(c) for c in cmd)
wrapper = f"cd $REMOTE && {line} >> $R_LOG 2>&1; echo \$? > $R_DONE"
subprocess.Popen(["bash", "-c", wrapper], start_new_session=True,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
print("LAUNCHED:", line, flush=True)
EOF
fi

START=$(date +%s)
DEADLINE=$(( START + MAX_HOURS * 3600 ))
POLLS=0
FAILS=0
echo "$(ts) == entering poll loop (POLL_S=$POLL_S) =="
while true; do
  sleep "$POLL_S"
  POLLS=$(( POLLS + 1 ))
  OUT=$(cat <<EOF | perl -e 'alarm 120; exec @ARGV' -- "$COLAB" exec -s "$SESSION" --timeout 90 2>&1
import os
log = "$R_LOG"
if os.path.exists(log):
    lines = open(log, errors="replace").read().splitlines()
    print("\n".join(l[:300] for l in lines[-5:]))
print("DONE_RC=" + open("$R_DONE").read().strip() if os.path.exists("$R_DONE") else "STATUS=RUNNING")
EOF
)
  RC=$?
  MIN=$(( ( $(date +%s) - START ) / 60 ))
  if [ "$RC" -ne 0 ]; then
    FAILS=$(( FAILS + 1 ))
    echo "$(ts) [+${MIN}m] poll failed ($FAILS/5): $(echo "$OUT" | tail -2)"
    if [ "$FAILS" -ge 5 ]; then
      echo "$(ts) == session unreachable; giving up =="
      STOP_VM_ON_EXIT=1
      exit 1
    fi
    continue
  fi
  FAILS=0
  echo "$(ts) [+${MIN}m] ----"
  echo "$OUT"
  if echo "$OUT" | grep -q 'DONE_RC=[0-9]'; then
    DRC=$(echo "$OUT" | grep -o 'DONE_RC=[0-9]*' | tail -1 | cut -d= -f2)
    echo "$(ts) == remote process finished with exit code $DRC =="
    STOP_VM_ON_EXIT=1
    [ "$DRC" = "0" ] && exit 0 || exit 3
  fi
  if [ $(( POLLS % SYNC_EVERY )) -eq 0 ]; then
    echo "$(ts) == periodic download of results and log =="
    sync_results
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "$(ts) == MAX_HOURS=$MAX_HOURS reached; stopping (finished steps saved; rerun resumes) =="
    STOP_VM_ON_EXIT=1
    exit 2
  fi
done
