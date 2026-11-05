#!/bin/bash
# Run ON THE POD: wait for run_all.sh to finish, wait a grace period, then stop this pod.
# Models and results live on the /workspace volume, which survives a stop.
#   nohup ./auto_stop.sh [grace_seconds] > /workspace/auto_stop.log 2>&1 &
GRACE="${1:-600}"
# ssh sessions don't inherit the container env; read it from PID 1
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_POD_ID|RUNPOD_API_KEY)=' | xargs)
echo "$(date) watching pod $RUNPOD_POD_ID; grace ${GRACE}s after run_all.sh exits"
sleep 60                                   # let run_all.sh start if launched just before
while pgrep -f "^/bin/bash ./run_all.sh|^/bin/bash ./after_run.sh" >/dev/null; do sleep 60; done
echo "$(date) run_all.sh and after_run.sh finished; results: $(ls /workspace/upscale_net/runs/results-*.md 2>/dev/null)"
sleep "$GRACE"
echo "$(date) stopping pod $RUNPOD_POD_ID"
curl -s -X POST "https://api.runpod.io/graphql" -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}"
echo
