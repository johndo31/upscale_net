#!/bin/sh
# Benchmark open ~1B reference models on the same suite, for scale in the write-up.
#   ./bench_refs.sh [limit] [device]
set -e
LIMIT="${1:-0}"; DEVICE="${2:-cuda}"
export PY="${PY:-$(dirname "$(command -v python)")}"   # bench.sh needs the venv bin dir
REFS="${REFS:-Qwen/Qwen2.5-1.5B HuggingFaceTB/SmolLM2-1.7B allenai/OLMo-2-0425-1B TinyLlama/TinyLlama_v1.1 EleutherAI/pythia-1b}"
for m in $REFS; do
  name="ref-$(echo "$m" | tr '/' '__')"
  [ -d "runs/bench/$name" ] && { echo "skip $name (done)"; continue; }
  echo "== reference: $m"
  ./bench.sh "$m" "$name" "$LIMIT" "$DEVICE" || echo "FAILED: $m"
done
echo "== references done"
