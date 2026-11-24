#!/bin/bash
# Run 04d natural_graft_exp.py (Part F natural-notation addition, G1 mid-layer write-out,
# G2 hidden-state read-in) on Colab as a DETACHED remote process with polling and auto-stop.
#   bash run_colab_natural.sh                 # full run, merges into runs/natural_graft_results.json
#   SMOKE=1 bash run_colab_natural.sh         # tiny end-to-end check, separate smoke files
#   PARTS=f MAX_HOURS=2 bash run_colab_natural.sh
# The cleanup trap ALWAYS downloads the results JSON and log, stops the session and prints usage.
set -uo pipefail
cd "$(dirname "$0")"
COLAB="${COLAB:-$HOME/.local/bin/colab}"
SESSION="${SESSION:-graft04d}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PARTS="${PARTS:-f,g1,g2}"
SEEDS="${SEEDS:-0,1,2}"
MAX_HOURS="${MAX_HOURS:-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SMOKE="${SMOKE:-0}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
RETRY_WAIT="${RETRY_WAIT:-45}"
POLL_S=60
SYNC_EVERY=10   # polls between JSON downloads (= 10 minutes)

if [ "$SMOKE" = "1" ]; then
  NAME="natural_graft_smoke"; SMOKE_FLAG="--smoke"
else
  NAME="natural_graft"; SMOKE_FLAG=""
fi
REMOTE=/content/graft04d
L_JSON="runs/${NAME}_results.json"      # full run: runs/natural_graft_results.json
L_LOG="runs/${NAME}.log"                # full run: runs/natural_graft.log
R_JSON="$REMOTE/$L_JSON"
R_LOG="$REMOTE/$L_LOG"
R_DONE="$REMOTE/runs/${NAME}.DONE"
ALLOCATED=0
ts() { date '+%Y-%m-%d %H:%M:%S'; }

sync_results() {
  mkdir -p runs
  "$COLAB" download -s "$SESSION" "$R_JSON" "$L_JSON" 2>&1 | tail -1 || true
  "$COLAB" download -s "$SESSION" "$R_LOG" "$L_LOG" 2>&1 | tail -1 || true
}

cleanup() {
  if [ "$ALLOCATED" -eq 1 ]; then
    echo "$(ts) == [cleanup] downloading $L_JSON and $L_LOG, stopping '$SESSION' =="
    sync_results
    "$COLAB" stop -s "$SESSION" 2>&1 || true
  fi
  "$COLAB" usage 2>&1 || true
}
trap cleanup EXIT
trap 'echo "$(ts) == interrupted =="; exit 130' INT TERM

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  for GPU in ${GPUS:-A100 L4 T4}; do
    echo "$(ts) == Trying $GPU for '$SESSION' (attempt $attempt/$MAX_ATTEMPTS) =="
    if COLAB_CLI_ASSIGN_RETRIES=0 "$COLAB" new -s "$SESSION" --gpu "$GPU"; then
      ALLOCATED=1
      break 2
    fi
  done
  echo "$(ts) == A100/L4/T4 unavailable; retrying in ${RETRY_WAIT}s =="
  sleep "$RETRY_WAIT"
done
[ "$ALLOCATED" -eq 1 ] || exit 1
sleep 4

echo "import os; os.makedirs('/content/tw', exist_ok=True); os.makedirs('$REMOTE/runs', exist_ok=True)" | "$COLAB" exec -s "$SESSION" --timeout 60 || exit 1
for f in target_tf.py tasks.py verify_realizability.py writer_loop.py; do
  "$COLAB" upload -s "$SESSION" "../03-transformer-writer/$f" "/content/tw/$f" || exit 1
done
"$COLAB" upload -s "$SESSION" real_graft_exp.py "$REMOTE/real_graft_exp.py" || exit 1
"$COLAB" upload -s "$SESSION" natural_graft_exp.py "$REMOTE/natural_graft_exp.py" || exit 1
if [ -f "$L_JSON" ]; then
  echo "$(ts) == uploading existing $L_JSON so finished cells are skipped =="
  "$COLAB" upload -s "$SESSION" "$L_JSON" "$R_JSON" || exit 1
fi

echo "$(ts) == launching detached run (model=$MODEL parts=$PARTS seeds=$SEEDS smoke=$SMOKE) =="
cat <<EOF | "$COLAB" exec -s "$SESSION" --timeout 120 || exit 1
import os, shlex, subprocess, sys
for p in ("$R_DONE",):
    if os.path.exists(p):
        os.remove(p)
cmd = [sys.executable, "-u", "$REMOTE/natural_graft_exp.py", "--model", "$MODEL", "--parts", "$PARTS",
       "--seeds", "$SEEDS", "--g-seeds", "$SEEDS", "--out", "$R_JSON"] + shlex.split("$SMOKE_FLAG $EXTRA_ARGS")
line = " ".join(shlex.quote(c) for c in cmd)
wrapper = f"cd $REMOTE && {line} >> $R_LOG 2>&1; echo \$? > $R_DONE"
subprocess.Popen(["bash", "-c", wrapper], start_new_session=True,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
print("LAUNCHED:", line, flush=True)
EOF

START=$(date +%s)
DEADLINE=$(( START + MAX_HOURS * 3600 ))
POLLS=0
FAILS=0
while true; do
  sleep "$POLL_S"
  POLLS=$(( POLLS + 1 ))
  OUT=$(cat <<EOF | "$COLAB" exec -s "$SESSION" --timeout 90 2>&1
import os
log = "$R_LOG"
if os.path.exists(log):
    lines = open(log, errors="replace").read().splitlines()
    print("\n".join(l[:300] for l in lines[-4:]))
print("DONE_RC=" + open("$R_DONE").read().strip() if os.path.exists("$R_DONE") else "STATUS=RUNNING")
EOF
)
  RC=$?
  MIN=$(( ( $(date +%s) - START ) / 60 ))
  if [ $RC -ne 0 ]; then
    FAILS=$(( FAILS + 1 ))
    echo "$(ts) [+${MIN}m] poll failed ($FAILS/5): $(echo "$OUT" | tail -2)"
    [ "$FAILS" -ge 5 ] && { echo "$(ts) == session unreachable; giving up =="; exit 1; }
    continue
  fi
  FAILS=0
  echo "$(ts) [+${MIN}m] ----"
  echo "$OUT"
  if echo "$OUT" | grep -q 'DONE_RC=[0-9]'; then
    DRC=$(echo "$OUT" | grep -o 'DONE_RC=[0-9]*' | tail -1 | cut -d= -f2)
    echo "$(ts) == remote process finished with exit code $DRC =="
    [ "$DRC" = "0" ] && exit 0 || exit 3     # trap downloads and stops
  fi
  if [ $(( POLLS % SYNC_EVERY )) -eq 0 ]; then
    echo "$(ts) == periodic download of results and log =="
    sync_results
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "$(ts) == MAX_HOURS=$MAX_HOURS reached; stopping (finished cells are in the JSON; rerun resumes) =="
    exit 2
  fi
done
