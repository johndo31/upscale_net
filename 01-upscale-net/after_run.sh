#!/bin/bash
# Queue, ON THE POD: after run_all.sh finishes, run the initialization ablation.
# Each variant: grow with a different --init, heal 100M tokens with the main run's
# hyperparameters, benchmark. All variants share the same short schedule (the main run's
# first 100M tokens are NOT a fair comparison: its cosine decay is 10x slower).
#   nohup ./after_run.sh > /workspace/after_run.log 2>&1 &
set -u
cd "$(dirname "$0")"
export PATH=/workspace/venv/bin:$PATH HF_HOME=/workspace/hf
TOKENS="${TOKENS:-100000000}"; SEQ=2048; BATCH=8; ACC=4; LR=3e-5; WARM=100
STEPS=$((TOKENS / (SEQ * BATCH * ACC)))
GROW="--hidden 1280 --intermediate 6912 --layers 28 --heads 20 --kv-heads 4 --dtype fp32"
SRC=Qwen/Qwen2.5-0.5B

sleep 60
while pgrep -f "^/bin/bash ./run_all.sh" >/dev/null; do sleep 60; done
echo "$(date) main run finished; starting init ablation ($STEPS steps per variant)"

for INIT in ${INITS:-zero split guided}; do   # zero gets its own 100M run: same cosine schedule as the others
  DST=models/qwen05-grown-1b-$INIT
  EXTRA=""; [ "$INIT" = guided ] && EXTRA="--stats runs/stats-qwen05.json"
  [ -f $DST/config.json ] || python grow.py --src $SRC --dst $DST $GROW --init $INIT $EXTRA 2>&1 | tail -3
  [ -f runs/verify-small-$INIT.log ] || python verify.py --src $SRC --dst $DST --dtype fp32 --device cuda 2>&1 | grep SUMMARY | tee runs/verify-small-$INIT.log
  if [ ! -f runs/heal-small-$INIT/model/config.json ]; then
    python heal.py --model $DST --name heal-small-$INIT --device cuda --steps $STEPS --seq-len $SEQ \
      --batch $BATCH --grad-accum $ACC --optimizer adamw --lr $LR --warmup $WARM --eval-every 100 \
      2>&1 | grep -E '^\{' > runs/heal-small-$INIT.log
  fi
  [ -d runs/bench/small-$INIT-healed-100M ] || ./bench.sh runs/heal-small-$INIT/model small-$INIT-healed-100M 0 cuda
done
echo "$(date) ablation done"
