#!/bin/bash
# Full pipeline on a GPU box: grow -> verify -> benchmark -> heal -> benchmark -> results table.
#
#   ./run_all.sh small [TOKENS]   Qwen2.5-0.5B -> 1.05B   (fits any 24 GB+ GPU, AdamW)
#   ./run_all.sh big   [TOKENS]   DeepSeek-R1-Distill-Qwen-1.5B -> 8.2B  (80 GB+ GPU, Adafactor)
#
# Also trains the un-grown source on the same tokens as a control (CONTROL=0 to skip).
# HEAL_DATA=self  heals on verified solutions the *source model* wrote (self_distill.py)
#                 instead of fineweb-edu; a sensible TOKENS for that is 30000000-100000000.
# TOKENS = healing budget, default 1e9. Everything lands in runs/ and models/.
# Re-running skips steps whose outputs already exist; an interrupted heal resumes from runs/<name>/state.pt.
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-small}"; TOKENS="${2:-1000000000}"
export PY="${PY:-$(dirname "$(command -v python)")}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-0}"            # 0 = full benchmark tasks

case "$MODE" in
  small)
    SRC=Qwen/Qwen2.5-0.5B; DST=models/qwen05-grown-1b
    GROW="--hidden 1280 --intermediate 6912 --layers 28 --heads 20 --kv-heads 4"
    SEQ=2048; BATCH=8; ACC=4; OPT=adamw; LR=3e-5; WARM=100; MIX=""   # batch 8 fits 80 GB; 1e-4 was too high for a 65k-token batch
    GROW_DTYPE=fp32 ;;          # 4 GB checkpoint; keeps the sqrt(d/D) norm scaling exact
  med)
    SRC=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; DST=models/deepseek-r1-qwen-grown-3.8b
    GROW="--hidden 2560 --intermediate 10240 --layers 32 --heads 20 --kv-heads 4"
    SEQ=2048; BATCH=1; ACC=32; OPT=adafactor; LR=1e-3; WARM=100
    GROW_DTYPE=fp32             # 15.2 GB checkpoint; exact fp32 norm scaling, fits A100 40GB cleanly
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export BENCH_REASONING=1
    MIX="--mix-dataset open-r1/OpenR1-Math-220k --mix-subset default --mix-ratio 0.2" ;;
  big)
    SRC=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; DST=models/deepseek-r1-qwen-grown-8b
    GROW="--hidden 4096 --intermediate 14336 --layers 32 --heads 32 --kv-heads 8"
    SEQ=2048; BATCH=4; ACC=8; OPT=adafactor; LR=1e-3; WARM=100
    GROW_DTYPE=bf16             # 16 GB checkpoint; fp32 would be 33 GB
    export BENCH_REASONING=1                      # source is a reasoning distill: measure GSM8K too
    MIX="--mix-dataset open-r1/OpenR1-Math-220k --mix-subset default --mix-ratio 0.2" ;;
  *) echo "mode must be small, med, or big"; exit 1 ;;
esac
TPS=$((SEQ * BATCH * ACC)); STEPS=$((TOKENS / TPS))
HEAL_DATA="${HEAL_DATA:-fineweb}"     # fineweb | self   (self = verified solutions written by the source model)
TAG="$MODE"; DATA_ARGS="$MIX"
if [ "$HEAL_DATA" = "self" ]; then
  TAG="$MODE-self"; DATA_ARGS="--dataset runs/self-$MODE/data.jsonl"
  export BENCH_REASONING=1
fi
HEAL="heal-$TAG"
echo "== mode=$MODE  data=$HEAL_DATA  tokens=$TOKENS  tokens/step=$TPS  steps=$STEPS  optimizer=$OPT lr=$LR"
mkdir -p runs

if [ ! -f "$DST/config.json" ]; then
  echo "== 1/6 grow"
  python grow.py --src "$SRC" --dst "$DST" $GROW --dtype "$GROW_DTYPE" 2>&1 | tee runs/grow-$MODE.log
fi

if [ ! -f runs/verify-$MODE.log ]; then
  echo "== 2/6 verify function preservation"
  # fp32 compare: bf16 rounding of the rescaled norm weights alone gives |Δlogit| ~ 1 while argmax still agrees
  python verify.py --src "$SRC" --dst "$DST" --dtype fp32 --device "$DEVICE" 2>&1 | tee runs/verify-$MODE.log
fi

echo "== 3/6 baseline benchmarks"
[ -d runs/bench/$MODE-source ]      || ./bench.sh "$SRC" "$MODE-source"      "$LIMIT" "$DEVICE"
[ -d runs/bench/$MODE-grown-step0 ] || ./bench.sh "$DST" "$MODE-grown-step0" "$LIMIT" "$DEVICE"

if [ "$HEAL_DATA" = "self" ] && [ ! -f runs/self-$MODE/data.jsonl ]; then
  echo "== 3b/7 self-distillation data from the source model"
  python self_distill.py --model "$SRC" --out runs/self-$MODE/data.jsonl --device "$DEVICE" \
    --engine "${SD_ENGINE:-hf}" --samples "${SD_SAMPLES:-4}" --limit "${SD_LIMIT:-0}" 2>&1 | tee runs/self-$MODE.log
fi

if [ ! -f runs/$HEAL/model/config.json ]; then
  echo "== 4/7 heal ($STEPS steps)"
  python heal.py --model "$DST" --name "$HEAL" --device "$DEVICE" \
    --steps "$STEPS" --seq-len "$SEQ" --batch "$BATCH" --grad-accum "$ACC" \
    --optimizer "$OPT" --lr "$LR" --warmup "$WARM" --eval-every 100 --save-every 250 --resume $DATA_ARGS \
    2>&1 | tee -a runs/$HEAL.log
fi

echo "== 5/7 final benchmarks"
[ -d runs/bench/$TAG-healed ] || ./bench.sh "runs/$HEAL/model" "$TAG-healed" "$LIMIT" "$DEVICE"

# Control: the *source* model trained on the same tokens with the same settings.
# If the grown model does not beat this, growth did not help. CONTROL=0 skips it.
if [ "${CONTROL:-1}" = "1" ]; then
  if [ ! -f runs/control-$TAG/model/config.json ]; then
    echo "== 6/7 control: heal the source model itself ($STEPS steps)"
    python heal.py --model "$SRC" --name "control-$TAG" --device "$DEVICE" \
      --steps "$STEPS" --seq-len "$SEQ" --batch "$BATCH" --grad-accum "$ACC" \
      --optimizer "$OPT" --lr "$LR" --warmup "$WARM" --eval-every 100 --save-every 250 --resume $DATA_ARGS \
      2>&1 | tee -a runs/control-$TAG.log
  fi
  [ -d runs/bench/$TAG-control ] || ./bench.sh "runs/control-$TAG/model" "$TAG-control" "$LIMIT" "$DEVICE"
fi

echo "== 7/7 results"
python results.py --mode "$MODE" --tag "$TAG" | tee runs/results-$TAG.md
