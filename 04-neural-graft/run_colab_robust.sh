#!/bin/bash
# Run 04c robust_graft_exp.py (Parts A-E) on Colab. The run takes longer than one `colab exec`, so the
# Python process is started detached on the Colab machine (output -> log file) and this script polls:
# every POLL_S seconds it prints new log lines, every SYNC_EVERY_S seconds it downloads the results
# JSON, and it finishes when the DONE marker appears or the process is gone. A hard wall-clock cap
# (MAX_HOURS, counted from GPU allocation) stops the session regardless. The cleanup trap ALWAYS
# downloads the JSON and the log and stops the session.
#
#   ./run_colab_robust.sh                 # full run -> runs/robust_graft_results.json, runs/robust_graft.log
#   SMOKE=1 ./run_colab_robust.sh         # cheap end-to-end check (--smoke) -> runs/robust_graft_smoke_*.{json,log}
#   PARTS=b,a MAX_HOURS=2 ./run_colab_robust.sh
#   EXTRA_ARGS="--seeds 0,1,2" ./run_colab_robust.sh
#
# A rerun uploads the existing local results JSON first, so finished cells are skipped (checkpoints of
# seed-0 models live only on the Colab machine; Parts C/E retrain them if a new session needs them).
set -uo pipefail
cd "$(dirname "$0")"
COLAB="${COLAB:-$HOME/.local/bin/colab}"
SESSION="${SESSION:-graft04c}"
MAX_HOURS="${MAX_HOURS:-4}"
SMOKE="${SMOKE:-0}"
PARTS="${PARTS:-a,b,c,d,e}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
ROUND_JSON="${ROUND_JSON:-../03-transformer-writer/runs/coder3b_hindsight/round5.json}"
POLL_S="${POLL_S:-60}"
SYNC_EVERY_S="${SYNC_EVERY_S:-600}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
RETRY_WAIT="${RETRY_WAIT:-45}"
REMOTE=/content/graft04
TW=/content/tw

if [ "$SMOKE" = "1" ]; then
  JSON_NAME=robust_graft_smoke_results.json
  LOG_NAME=robust_graft_smoke.log
  SMOKE_FLAG=--smoke
else
  JSON_NAME=robust_graft_results.json
  LOG_NAME=robust_graft.log
  SMOKE_FLAG=
fi

for f in robust_graft_exp.py real_graft_exp.py graft_exp.py \
         ../03-transformer-writer/target_tf.py ../03-transformer-writer/tasks.py \
         ../03-transformer-writer/verify_realizability.py ../03-transformer-writer/writer_loop.py "$ROUND_JSON"; do
  [ -f "$f" ] || { echo "missing local file: $f"; exit 1; }
done
mkdir -p runs
ALLOCATED=0

fetch_json() {
  rm -f "runs/$JSON_NAME.part"
  if "$COLAB" download -s "$SESSION" "$REMOTE/runs/$JSON_NAME" "runs/$JSON_NAME.part" >/dev/null 2>&1 \
     && [ -s "runs/$JSON_NAME.part" ]; then
    mv "runs/$JSON_NAME.part" "runs/$JSON_NAME"
    echo "$(date '+%H:%M:%S') == synced runs/$JSON_NAME =="
  else
    rm -f "runs/$JSON_NAME.part"
    echo "$(date '+%H:%M:%S') == (no results JSON to download yet) =="
  fi
}

fetch_log() {
  rm -f "runs/$LOG_NAME.part"
  if "$COLAB" download -s "$SESSION" "$REMOTE/runs/$LOG_NAME" "runs/$LOG_NAME.part" >/dev/null 2>&1 \
     && [ -s "runs/$LOG_NAME.part" ]; then
    mv "runs/$LOG_NAME.part" "runs/$LOG_NAME"
    echo "$(date '+%H:%M:%S') == downloaded runs/$LOG_NAME =="
  else
    rm -f "runs/$LOG_NAME.part"
  fi
}

cleanup() {
  if [ "$ALLOCATED" -eq 1 ]; then
    echo "== [cleanup] downloading results + log and stopping '$SESSION' =="
    fetch_json
    fetch_log
    "$COLAB" stop -s "$SESSION" 2>&1 || true
    "$COLAB" usage 2>&1 || true
  fi
}
trap cleanup EXIT
trap 'echo "== interrupted =="; exit 130' INT TERM

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  for GPU in ${GPUS:-A100 L4 T4}; do
    echo "$(date '+%H:%M:%S') == Trying $GPU for '$SESSION' (attempt $attempt/$MAX_ATTEMPTS) =="
    if COLAB_CLI_ASSIGN_RETRIES=0 "$COLAB" new -s "$SESSION" --gpu "$GPU"; then
      ALLOCATED=1
      break 2
    fi
  done
  echo "$(date '+%H:%M:%S') == A100/L4/T4 unavailable; retrying in ${RETRY_WAIT}s =="
  sleep "$RETRY_WAIT"
done
[ "$ALLOCATED" -eq 1 ] || exit 1
START=$(date +%s)
CAP_S=$(awk -v h="$MAX_HOURS" 'BEGIN { printf "%d", h * 3600 }')
DEADLINE=$((START + CAP_S))
echo "== allocated; hard cap MAX_HOURS=$MAX_HOURS -> stop at $(date -r "$DEADLINE" '+%H:%M:%S' 2>/dev/null || echo "+${CAP_S}s") =="
sleep 4

echo "import os; os.makedirs('$TW', exist_ok=True); os.makedirs('$REMOTE/runs', exist_ok=True)" | "$COLAB" exec -s "$SESSION" --timeout 60 || exit 1
for f in target_tf.py tasks.py verify_realizability.py writer_loop.py; do
  "$COLAB" upload -s "$SESSION" "../03-transformer-writer/$f" "$TW/$f" || exit 1
done
for f in robust_graft_exp.py real_graft_exp.py graft_exp.py; do
  "$COLAB" upload -s "$SESSION" "$f" "$REMOTE/$f" || exit 1
done
"$COLAB" upload -s "$SESSION" "$ROUND_JSON" "$REMOTE/round5.json" || exit 1
if [ -f "runs/$JSON_NAME" ]; then
  echo "== uploading existing runs/$JSON_NAME so finished cells are skipped =="
  "$COLAB" upload -s "$SESSION" "runs/$JSON_NAME" "$REMOTE/runs/$JSON_NAME" || exit 1
fi

# ---- launch detached ------------------------------------------------------------------------
cat <<'PYEOF' | "$COLAB" exec -s "$SESSION" --timeout 600 \
    --env "ROBUST_REMOTE=$REMOTE" --env "ROBUST_PARTS=$PARTS" --env "ROBUST_SMOKE=$SMOKE_FLAG" \
    --env "ROBUST_EXTRA=$EXTRA_ARGS" --env "ROBUST_JSON=$JSON_NAME" --env "ROBUST_LOG=$LOG_NAME" || exit 1
import os, shlex, subprocess, sys
R = os.environ["ROBUST_REMOTE"]
runs = os.path.join(R, "runs")
os.makedirs(runs, exist_ok=True)
try:
    import datasets  # noqa: F401
except Exception:
    subprocess.call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
for f in ("DONE", "EXIT_CODE", "PID"):
    p = os.path.join(runs, f)
    if os.path.exists(p):
        os.remove(p)
cmd = [sys.executable, "-u", os.path.join(R, "robust_graft_exp.py"),
       "--parts", os.environ["ROBUST_PARTS"],
       "--round-json", os.path.join(R, "round5.json"),
       "--out", os.path.join(runs, os.environ["ROBUST_JSON"]),
       "--ckpt-dir", os.path.join(R, "ckpt")]
if os.environ.get("ROBUST_SMOKE"):
    cmd.append(os.environ["ROBUST_SMOKE"])
cmd += shlex.split(os.environ.get("ROBUST_EXTRA", ""))
log = os.path.join(runs, os.environ["ROBUST_LOG"])
wrapper = (f"{shlex.join(cmd)} >> {shlex.quote(log)} 2>&1; echo $? > {shlex.quote(os.path.join(runs, 'EXIT_CODE'))}; "
           f"touch {shlex.quote(os.path.join(runs, 'DONE'))}")
p = subprocess.Popen(["bash", "-c", wrapper], cwd=R, start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
open(os.path.join(runs, "PID"), "w").write(str(p.pid))
print("LAUNCHED pid", p.pid)
print("CMD", shlex.join(cmd))
PYEOF

# ---- poll -----------------------------------------------------------------------------------
POLL_PY=$(cat <<'PYEOF'
import os
R = os.environ["ROBUST_REMOTE"]; runs = os.path.join(R, "runs")
off = int(os.environ.get("ROBUST_OFF", "0"))
log = os.path.join(runs, os.environ["ROBUST_LOG"])
if os.path.exists(log):
    with open(log, "rb") as f:
        f.seek(off)
        data = f.read(200000)
    if data:
        print(data.decode("utf-8", "replace"), end="" if data.endswith(b"\n") else "\n")
    off += len(data)
state, code = "running", ""
if os.path.exists(os.path.join(runs, "DONE")):
    state = "done"
    try:
        code = open(os.path.join(runs, "EXIT_CODE")).read().strip()
    except Exception:
        code = "?"
else:
    try:
        pid = int(open(os.path.join(runs, "PID")).read().strip())
        st = open(f"/proc/{pid}/stat").read().split(")")[-1].split()[0]
        if st == "Z":
            state = "gone"
    except Exception:
        state = "gone"
print(f"__ROBUST_OFFSET__ {off}")
print(f"__ROBUST_STATE__ {state} {code}")
PYEOF
)

OFF=0
LAST_SYNC=$(date +%s)
FAILS=0
FINAL_RC=0
while true; do
  NOW=$(date +%s)
  if [ "$NOW" -ge "$DEADLINE" ]; then
    echo "$(date '+%H:%M:%S') == MAX_HOURS=$MAX_HOURS reached; stopping the session regardless =="
    FINAL_RC=3
    break
  fi
  OUT=$(printf '%s\n' "$POLL_PY" | "$COLAB" exec -s "$SESSION" --timeout 120 \
          --env "ROBUST_REMOTE=$REMOTE" --env "ROBUST_LOG=$LOG_NAME" --env "ROBUST_OFF=$OFF" 2>&1)
  RC=$?
  NEW_OFF=$(printf '%s\n' "$OUT" | sed -n 's/^__ROBUST_OFFSET__ \([0-9][0-9]*\).*/\1/p' | tail -1)
  STATE_LINE=$(printf '%s\n' "$OUT" | grep '^__ROBUST_STATE__' | tail -1)
  if [ "$RC" -ne 0 ] || [ -z "$NEW_OFF" ] || [ -z "$STATE_LINE" ]; then
    FAILS=$((FAILS + 1))
    echo "$(date '+%H:%M:%S') == poll failed ($FAILS/5, rc=$RC): $(printf '%s' "$OUT" | tail -3) =="
    if [ "$FAILS" -ge 5 ]; then
      echo "== 5 consecutive poll failures; giving up (cleanup will try to download and stop) =="
      FINAL_RC=4
      break
    fi
    sleep "$POLL_S"
    continue
  fi
  FAILS=0
  printf '%s\n' "$OUT" | grep -v '^__ROBUST_' | sed '/^$/d'
  OFF=$NEW_OFF
  STATE=$(echo "$STATE_LINE" | awk '{print $2}')
  CODE=$(echo "$STATE_LINE" | awk '{print $3}')
  if [ "$STATE" = "done" ]; then
    echo "$(date '+%H:%M:%S') == remote run finished (exit code ${CODE:-?}) =="
    [ "${CODE:-1}" = "0" ] || FINAL_RC=5
    break
  fi
  if [ "$STATE" = "gone" ]; then
    echo "$(date '+%H:%M:%S') == remote process is gone without DONE marker =="
    FINAL_RC=6
    break
  fi
  NOW=$(date +%s)
  if [ $((NOW - LAST_SYNC)) -ge "$SYNC_EVERY_S" ]; then
    fetch_json
    LAST_SYNC=$NOW
  fi
  sleep "$POLL_S"
done
exit "$FINAL_RC"
