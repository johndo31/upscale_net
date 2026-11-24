#!/bin/bash
# Keep the pod alive after run_gpu3.sh finishes, for at most HOLD_MIN minutes, so a follow-up queue
# can be started without losing the GPU. Dead-man switch: if no follow-up takes over (it must create
# runs_gpu/queue4.started), stop the pod when the time is up.
#   nohup bash run_hold.sh >> runs_gpu/run_hold.log 2>&1 &
set -u
cd "$(dirname "$0")" || exit 1
HOLD_MIN="${HOLD_MIN:-150}"
ts() { echo "$(date '+%Y-%m-%d %H:%M:%S') == $*"; }
ts "hold: waiting for run_gpu3.sh to finish its stages"
until grep -q "ALL DONE 3" runs_gpu/run_gpu3.log; do sleep 15; done
pkill -f "bash run_gpu3.sh" && ts "hold: cancelled the auto-stop of run_gpu3.sh"
ts "hold: keeping the pod up for at most $HOLD_MIN min unless runs_gpu/queue4.started appears"
for i in $(seq 1 $((HOLD_MIN * 2))); do
  [ -f runs_gpu/queue4.started ] && { ts "hold: follow-up queue took over, hold ends"; exit 0; }
  sleep 30
done
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_POD_ID|RUNPOD_API_KEY)=' | xargs)
ts "hold: time is up, stopping pod ${RUNPOD_POD_ID:-?}"
curl -s -X POST "https://api.runpod.io/graphql" -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
  -d "{\"query\":\"mutation { podStop(input:{podId:\\\"${RUNPOD_POD_ID:-}\\\"}) { id desiredStatus } }\"}"
echo
