#!/bin/bash
# Fourth queue ON THE POD: the self-set exploration experiment (model chooses its own temperature),
# with a random-choice control and a fixed-temperature control. Takes over from run_hold.sh,
# writes report4.md and stops the pod at the end.
#   nohup bash run_gpu4.sh >> runs_gpu/run_gpu4.log 2>&1 &
set -u
cd "$(dirname "$0")" || exit 1
export PATH=/workspace/venv/bin:$PATH HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
OUT=runs_gpu; SE=$OUT/selfexplore; LOGS=$OUT/logs
ARGS="--device cuda --seed 0 --rank 16 --lr 2e-4 --k 16 --epochs 40 --pool-cap 4 --max-steps 2000 --rounds 10 --out-dir $SE"
GRACE="${GRACE:-600}"
FAILED=""
ts() { echo "$(date '+%Y-%m-%d %H:%M:%S') == $*"; }
touch $OUT/queue4.started
ts "run_gpu4.sh start (hold released)"
while pgrep -f "python (wsp|selfexplore|random_arm).py" >/dev/null; do sleep 10; done

stage() {
  local name=$1; shift
  if [ -f "$LOGS/$name.done" ]; then ts "skip $name (done)"; return 0; fi
  ts "start $name: python selfexplore.py $*"
  local t0=$SECONDS
  python selfexplore.py "$@" >> "$LOGS/$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$LOGS/$name.done"; ts "done $name in $(( (SECONDS - t0) / 60 )) min: $(grep -E '^\[' "$LOGS/$name.log" | tail -1)"
  else FAILED="$FAILED $name"; ts "FAILED $name (exit $rc) after $(( (SECONDS - t0) / 60 )) min; last lines:"; tail -5 "$LOGS/$name.log"; fi
}
stage se_self    --policy self      $ARGS
stage se_uniform --policy uniform   $ARGS
stage se_fixed14 --policy fixed:1.4 $ARGS

ts "report"
python report.py $SE > $OUT/report4.md 2> $LOGS/report4.log
ts "ALL DONE 4; failed stages:${FAILED:- none}"
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_POD_ID|RUNPOD_API_KEY)=' | xargs)
ts "stopping pod ${RUNPOD_POD_ID:-?} in $GRACE s (kill this script to prevent it)"
sleep "$GRACE"
ts "stopping pod ${RUNPOD_POD_ID:-?}"
curl -s -X POST "https://api.runpod.io/graphql" -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
  -d "{\"query\":\"mutation { podStop(input:{podId:\\\"${RUNPOD_POD_ID:-}\\\"}) { id desiredStatus } }\"}"
echo
