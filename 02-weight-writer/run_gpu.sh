#!/bin/bash
# Run ON THE POD: the whole weight-writing experiment, unattended, into runs_gpu/.
#   mkdir -p runs_gpu && nohup bash run_gpu.sh >> runs_gpu/run_gpu.log 2>&1 &
# Stages run one after another; each writes its own log in runs_gpu/logs/. A failed stage is
# reported and the next one starts. Re-running is safe: finished stages are skipped (.done markers),
# and wsp.py itself skips round JSONs that exist and re-evaluates rounds whose adapter was saved.
# At the end: report tables, then (AUTO_STOP=1) wait GRACE seconds and stop this pod via the RunPod API.
set -u
cd "$(dirname "$0")" || exit 1
export PATH=/workspace/venv/bin:$PATH HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

# ------------------------------------------------------------------ settings
OUT=runs_gpu
MAIN=$OUT/main                         # pilot (base/round0.json) and the loop arms
DEVICE="${DEVICE:-cuda}"
COMMON="--device $DEVICE --seed ${SEED:-0} --rank ${RANK:-16} --lr ${LR:-2e-4} --k 16"
PILOT_HELDOUT_K=64
TEACHER_EPOCHS="10 40 150"
TEACHER_MAX_STEPS=5000                 # 150 epochs x 205 examples / 16 = 1922 steps, so the cap never binds
LOOP_TRAIN="--epochs 40 --pool-cap 4 --max-steps 2000"
HINDSIGHT_ROUNDS=10
PASS_ROUNDS=6
UNVERIFIED_ROUNDS=3
AUTO_STOP="${AUTO_STOP:-1}"
GRACE="${GRACE:-600}"
# ------------------------------------------------------------------

LOGS=$OUT/logs
mkdir -p "$LOGS"
FAILED=""
ts() { echo "$(date '+%Y-%m-%d %H:%M:%S') == $*"; }

# a wsp.py started by an earlier copy of this script may still be running: let it finish first
while pgrep -f "python wsp.py" >/dev/null; do ts "waiting for a running wsp.py"; sleep 60; done

stage() {   # stage NAME wsp-args...   -> logs/NAME.log, marker logs/NAME.done on success
  local name=$1; shift
  if [ -f "$LOGS/$name.done" ]; then ts "skip $name (done)"; return 0; fi
  ts "start $name: python wsp.py $*"
  local t0=$SECONDS
  python wsp.py "$@" >> "$LOGS/$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    touch "$LOGS/$name.done"
    ts "done $name in $(( (SECONDS - t0) / 60 )) min: $(grep -E '^\[|^\{' "$LOGS/$name.log" | tail -1)"
  else
    FAILED="$FAILED $name"
    ts "FAILED $name (exit $rc) after $(( (SECONDS - t0) / 60 )) min; last lines:"
    tail -5 "$LOGS/$name.log"
  fi
}

ts "run_gpu.sh start on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
stage bench bench --device "$DEVICE"
stage pilot pilot $COMMON --heldout-k $PILOT_HELDOUT_K --out-dir $MAIN
for E in $TEACHER_EPOCHS; do
  stage teacher_e$E teacher $COMMON --epochs $E --max-steps $TEACHER_MAX_STEPS --out-dir $OUT/teacher_e$E
done
stage hindsight loop --arm hindsight --rounds $HINDSIGHT_ROUNDS $COMMON $LOOP_TRAIN --out-dir $MAIN
stage pass loop --arm pass --rounds $PASS_ROUNDS $COMMON $LOOP_TRAIN --out-dir $MAIN
stage unverified loop --arm unverified --rounds $UNVERIFIED_ROUNDS $COMMON $LOOP_TRAIN --out-dir $MAIN

ts "report"
TEACHER_DIRS=""; for E in $TEACHER_EPOCHS; do TEACHER_DIRS="$TEACHER_DIRS $OUT/teacher_e$E"; done
python report.py $MAIN $TEACHER_DIRS > $OUT/report.md 2> $LOGS/report.log
cat $OUT/report.md
ts "all done; failed stages:${FAILED:- none}"

if [ "$AUTO_STOP" = 1 ]; then
  export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_POD_ID|RUNPOD_API_KEY)=' | xargs)
  ts "stopping pod ${RUNPOD_POD_ID:-?} in $GRACE s (kill this script to prevent it)"
  sleep "$GRACE"
  ts "stopping pod ${RUNPOD_POD_ID:-?}"
  curl -s -X POST "https://api.runpod.io/graphql" -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
    -d "{\"query\":\"mutation { podStop(input:{podId:\\\"${RUNPOD_POD_ID:-}\\\"}) { id desiredStatus } }\"}"
  echo
fi
