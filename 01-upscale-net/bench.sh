#!/bin/sh
# Usage: ./bench.sh <model_dir_or_hf_id> <output_name> [limit] [device]
# Runs a small lm-eval suite plus wikitext perplexity. limit=0 means full tasks.
# max_length is capped at 4096: the models advertise 32k-131k context and
# wikitext would otherwise be scored in huge windows that blow up memory.
set -e
MODEL="$1"; NAME="$2"; LIMIT="${3:-300}"; DEVICE="${4:-cuda}"
PY="${PY:-.venv/bin}"
mkdir -p runs/bench
LIM=""; [ "$LIMIT" != "0" ] && LIM="--limit $LIMIT"
run() {  # tasks batch
  "$PY/lm_eval" --model hf \
    --model_args "pretrained=$MODEL,dtype=bfloat16,max_length=4096" \
    --tasks "$1" --device "$DEVICE" --batch_size "$2" $LIM \
    --output_path "runs/bench/$NAME" 2>&1 | tee -a "runs/bench/$NAME.log" | grep -E '^\|'
}
: > "runs/bench/$NAME.log"
run hellaswag,arc_easy,piqa,winogrande,arc_challenge 16
run wikitext 1
# Reasoning check (generative, slower). Needed for reasoning-tuned sources like
# DeepSeek-R1-Distill, whose main skill the other tasks do not measure.
if [ "${BENCH_REASONING:-0}" = "1" ]; then
  GLIM="--limit ${REASONING_LIMIT:-250}"; [ "$LIMIT" != "0" ] && GLIM="$LIM"
  "$PY/lm_eval" --model hf \
    --model_args "pretrained=$MODEL,dtype=bfloat16,max_length=4096" \
    --tasks gsm8k --device "$DEVICE" --batch_size 16 $GLIM \
    --gen_kwargs max_gen_toks=1024 \
    --output_path "runs/bench/$NAME" 2>&1 | tee -a "runs/bench/$NAME.log" | grep -E '^\|'
fi
