#!/bin/bash
# Third queue ON THE POD: wait for run_gpu2.sh to finish its stages, cancel its auto-stop,
# run the random-guess arm, write report3.md, then stop the pod.
#   nohup bash run_gpu3.sh >> runs_gpu/run_gpu3.log 2>&1 &
set -u
cd "$(dirname "$0")" || exit 1
export PATH=/workspace/venv/bin:$PATH HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
OUT=runs_gpu; MAIN=$OUT/main; LOGS=$OUT/logs
GRACE="${GRACE:-600}"
FAILED=""
ts() { echo "$(date '+%Y-%m-%d %H:%M:%S') == $*"; }

ts "run_gpu3.sh waiting for run_gpu2.sh to finish its stages"
until grep -q "ALL DONE 2" $OUT/run_gpu2.log; do sleep 20; done
pkill -f "bash run_gpu2.sh" && ts "cancelled the auto-stop of run_gpu2.sh"
while pgrep -f "python wsp.py" >/dev/null; do sleep 10; done

name=random_R2
if [ -f "$LOGS/$name.done" ]; then ts "skip $name (done)"; else
  ts "start $name"
  t0=$SECONDS
  python random_arm.py --device cuda --seed 0 --rank 16 --lr 2e-4 --k 16 --epochs 40 --pool-cap 4 --max-steps 2000 \
    --rounds 10 --rand-range 2 --out-dir $MAIN >> "$LOGS/$name.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then touch "$LOGS/$name.done"; ts "done $name in $(( (SECONDS - t0) / 60 )) min: $(grep -E '^\[' "$LOGS/$name.log" | tail -1)"
  else FAILED="$FAILED $name"; ts "FAILED $name (exit $rc); last lines:"; tail -5 "$LOGS/$name.log"; fi
fi

ts "report"
python report.py $MAIN $OUT/explore_T13 $OUT/explore_k64 $OUT/teacher_e10 $OUT/teacher_e40 $OUT/teacher_e150 > $OUT/report3.md 2> $LOGS/report3.log
python random_baseline.py --runs $MAIN --out $OUT/random_baseline.json > $OUT/random_baseline.md 2>&1
ts "ALL DONE 3; failed stages:${FAILED:- none}"

export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_POD_ID|RUNPOD_API_KEY)=' | xargs)
ts "stopping pod ${RUNPOD_POD_ID:-?} in $GRACE s (kill this script to prevent it)"
sleep "$GRACE"
ts "stopping pod ${RUNPOD_POD_ID:-?}"
curl -s -X POST "https://api.runpod.io/graphql" -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
  -d "{\"query\":\"mutation { podStop(input:{podId:\\\"${RUNPOD_POD_ID:-}\\\"}) { id desiredStatus } }\"}"
echo
