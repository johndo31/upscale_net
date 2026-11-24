#!/bin/bash
# Follow-up stages, started while run_gpu.sh is still running ON THE POD:
#   mkdir -p runs_gpu && nohup bash run_gpu2.sh >> runs_gpu/run_gpu2.log 2>&1 &
# Waits for run_gpu.sh to finish its last stage, cancels its auto-stop, runs the extra stages
# (more exploration, more rounds), writes report2.md, then stops the pod itself.
set -u
cd "$(dirname "$0")" || exit 1
export PATH=/workspace/venv/bin:$PATH HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
OUT=runs_gpu; MAIN=$OUT/main; LOGS=$OUT/logs
COMMON="--device cuda --seed 0 --rank 16 --lr 2e-4"
LOOP_TRAIN="--epochs 40 --pool-cap 4 --max-steps 2000"
GRACE="${GRACE:-600}"
FAILED=""
ts() { echo "$(date '+%Y-%m-%d %H:%M:%S') == $*"; }

ts "run_gpu2.sh waiting for run_gpu.sh to finish its stages"
until grep -q "all done" $OUT/run_gpu.log; do sleep 20; done
pkill -f "bash run_gpu.sh" && ts "cancelled the auto-stop of run_gpu.sh"
while pgrep -f "python wsp.py" >/dev/null; do sleep 10; done

stage() {
  local name=$1; shift
  if [ -f "$LOGS/$name.done" ]; then ts "skip $name (done)"; return 0; fi
  ts "start $name: python wsp.py $*"
  local t0=$SECONDS
  python wsp.py "$@" >> "$LOGS/$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    touch "$LOGS/$name.done"
    ts "done $name in $(( (SECONDS - t0) / 60 )) min: $(grep -E '^\[' "$LOGS/$name.log" | tail -1)"
  else
    FAILED="$FAILED $name"
    ts "FAILED $name (exit $rc) after $(( (SECONDS - t0) / 60 )) min; last lines:"; tail -5 "$LOGS/$name.log"
  fi
}

# 1. more exploration through temperature (same 16 samples per task)
stage explT_pilot     pilot $COMMON --k 16 --temperature 1.3 --heldout-k 64 --out-dir $OUT/explore_T13
stage explT_hindsight loop --arm hindsight --rounds 10 $COMMON --k 16 --temperature 1.3 $LOOP_TRAIN --out-dir $OUT/explore_T13
# 2. more exploration through more samples (64 per task, same temperature)
stage explK_pilot     pilot $COMMON --k 64 --temperature 0.8 --heldout-k 64 --out-dir $OUT/explore_k64
stage explK_hindsight loop --arm hindsight --rounds 10 $COMMON --k 64 --temperature 0.8 $LOOP_TRAIN --out-dir $OUT/explore_k64
# 3. the main hindsight loop continued from round 10 to round 30
stage hindsight_long  loop --arm hindsight --rounds 30 $COMMON --k 16 $LOOP_TRAIN --out-dir $MAIN

ts "report"
python report.py $MAIN $OUT/explore_T13 $OUT/explore_k64 $OUT/teacher_e10 $OUT/teacher_e40 $OUT/teacher_e150 > $OUT/report2.md 2> $LOGS/report2.log
ts "ALL DONE 2; failed stages:${FAILED:- none}"

export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_POD_ID|RUNPOD_API_KEY)=' | xargs)
ts "stopping pod ${RUNPOD_POD_ID:-?} in $GRACE s (kill this script to prevent it)"
sleep "$GRACE"
ts "stopping pod ${RUNPOD_POD_ID:-?}"
curl -s -X POST "https://api.runpod.io/graphql" -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
  -d "{\"query\":\"mutation { podStop(input:{podId:\\\"${RUNPOD_POD_ID:-}\\\"}) { id desiredStatus } }\"}"
echo
