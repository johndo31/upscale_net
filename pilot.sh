#!/bin/bash
# Run ON THE POD, whole experiment, unattended:
#   1. grow the source with three inits (zero / split / guided), verify each is function-preserving
#   2. heal each on the FULL 1B-token schedule but stop after $PILOT steps (state saved), benchmark
#   3. pick the winner (pick.py: held-out loss at $PILOT, ties within 0.01 go to zero)
#   4. resume the winner to the end of the schedule, then train the control (source model, same tokens)
#   5. final benchmarks, results table, stop this pod (volume survives; models/results live there)
# Re-running skips finished steps and resumes interrupted heals from runs/<name>/state.pt.
#   nohup ./pilot.sh > /workspace/pilot.log 2>&1 &
set -u
cd "$(dirname "$0")"
export PATH=/workspace/venv/bin:$PATH HF_HOME=/workspace/hf PY=/workspace/venv/bin
SRC=Qwen/Qwen2.5-0.5B
TOKENS="${TOKENS:-1000000000}"; SEQ=2048; BATCH=8; ACC=4; LR=3e-5; WARM=100
STEPS=$((TOKENS / (SEQ * BATCH * ACC))); PILOT="${PILOT:-2000}"
GROW="--hidden 1280 --intermediate 6912 --layers 28 --heads 20 --kv-heads 4 --dtype fp32"
HEAL="--device cuda --steps $STEPS --seq-len $SEQ --batch $BATCH --grad-accum $ACC --optimizer adamw \
      --lr $LR --warmup $WARM --eval-every 100 --save-every 1000 --resume"
INITS="${INITS:-zero split guided llm}"
echo "$(date) == pilot: $PILOT of $STEPS steps per init ($INITS), then resume the winner"
mkdir -p runs/bench
# a heal.py started by an earlier copy of this script may still be running: let it finish first
while pgrep -f "python heal.py" >/dev/null; do sleep 30; done

[ -d runs/bench/small-source ] || ./bench.sh $SRC small-source 0 cuda

for INIT in $INITS; do
  DST=models/qwen05-grown-1b-$INIT
  EXTRA=""; [ "$INIT" = guided ] && EXTRA="--stats runs/stats-qwen05.json"
  if [ "$INIT" = llm ]; then                      # plan written by hand after reading the model (plan_llm.py)
    while [ ! -f runs/plan-llm.json ]; do echo "$(date) waiting for runs/plan-llm.json"; sleep 120; done
    EXTRA="--plan runs/plan-llm.json"; INITARG=plan
  else INITARG=$INIT; fi
  [ -f $DST/config.json ] || python grow.py --src $SRC --dst $DST $GROW --init $INITARG $EXTRA 2>&1 | tail -3
  [ -f runs/verify-small-$INIT.log ] || python verify.py --src $SRC --dst $DST --dtype fp32 --device cuda 2>&1 | grep SUMMARY | tee runs/verify-small-$INIT.log
  [ "$INIT" = zero ] && [ ! -d runs/bench/small-grown-step0 ] && ./bench.sh $DST small-grown-step0 0 cuda
  if [ ! -f runs/heal-small-$INIT/checkpoint-$PILOT/config.json ] && [ ! -f runs/heal-small-$INIT/model/config.json ]; then
    echo "$(date) == heal $INIT to step $PILOT"
    python heal.py --model $DST --name heal-small-$INIT $HEAL --stop-after $PILOT 2>&1 | grep -E '^\{|Error|error' | tee -a runs/heal-small-$INIT.log | grep -v '"step"'
  fi
  [ -d runs/bench/small-$INIT-pilot ] || ./bench.sh runs/heal-small-$INIT/checkpoint-$PILOT small-$INIT-pilot 0 cuda
done

WIN=$(python pick.py --step $PILOT --inits $INITS | tee runs/pick.md | tail -1 | awk '{print $NF}')
echo "$(date) == winner: $WIN"
echo "$WIN" > runs/winner.txt

if [ ! -f runs/heal-small-$WIN/model/config.json ]; then
  echo "$(date) == resume $WIN to step $STEPS"
  python heal.py --model models/qwen05-grown-1b-$WIN --name heal-small-$WIN $HEAL 2>&1 | grep -E '^\{|Error|error' | tee -a runs/heal-small-$WIN.log | grep -v '"step"'
fi
ln -sfn heal-small-$WIN runs/heal-small          # results.py reads runs/heal-small/
[ -d runs/bench/small-healed ] || ./bench.sh runs/heal-small-$WIN/model small-healed 0 cuda

if [ ! -f runs/control-small/model/config.json ]; then
  echo "$(date) == control: source model, same $STEPS steps"
  python heal.py --model $SRC --name control-small $HEAL 2>&1 | grep -E '^\{|Error|error' | tee -a runs/control-small.log | grep -v '"step"'
fi
[ -d runs/bench/small-control ] || ./bench.sh runs/control-small/model small-control 0 cuda

python results.py --mode small --tag small | tee runs/results-small.md
echo "$(date) == all done"

# stop this pod (the volume with models/ and runs/ survives a stop)
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_POD_ID|RUNPOD_API_KEY)=' | xargs)
sleep "${GRACE:-600}"
echo "$(date) stopping pod $RUNPOD_POD_ID"
curl -s -X POST "https://api.runpod.io/graphql" -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}"
echo
