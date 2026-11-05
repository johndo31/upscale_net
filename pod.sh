#!/bin/sh
# Helper for talking to the RunPod pod from the Mac.
#   ./pod.sh ssh 'command'          run a command on the pod (no command = interactive shell)
#   ./pod.sh put file1 file2 ...    copy files into /workspace/upscale_net on the pod (POD_DIR)
#   ./pod.sh setup                  venv + deps under /workspace (one time per volume)
#   ./pod.sh run 'cmd'              run cmd inside /workspace/upscale_net with venv + HF cache on the volume
#   ./pod.sh get remote_path local  copy a file/dir back (recursive)
#   ./pod.sh sync [once]            mirror runs/ (logs, json, tables; no weights) to ./runs-pod/ every hour
#   ./pod.sh get-models [small|big] copy healed + control model weights to ./models/pod-<mode>/
# Set POD_HOST / POD_PORT for a different pod (shown under "SSH over exposed TCP" in the RunPod console).
POD_HOST="${POD_HOST:?set POD_HOST to your pod address}"; POD_PORT="${POD_PORT:?set POD_PORT to your pod SSH port}"; POD_KEY="${POD_KEY:-$HOME/.ssh/runpod_ed25519}"
POD_DIR="${POD_DIR:-/workspace/upscale_net}"      # on the persistent volume, survives pod stops
OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -i $POD_KEY"
cmd="$1"; shift
case "$cmd" in
  ssh) exec ssh $OPTS -p "$POD_PORT" "root@$POD_HOST" "$@" ;;
  setup) exec ssh $OPTS -p "$POD_PORT" "root@$POD_HOST" "mkdir -p $POD_DIR /workspace/hf && cd $POD_DIR && chmod +x *.sh && \
      ([ -x /workspace/venv/bin/python ] || python -m venv --system-site-packages /workspace/venv) && \
      /workspace/venv/bin/pip install -q transformers safetensors huggingface_hub accelerate datasets 'lm-eval>=0.4.8' 2>&1 | grep -v -i warn | tail -2; \
      /workspace/venv/bin/python -c 'import torch,transformers,lm_eval;print(\"ok\",torch.__version__,transformers.__version__,lm_eval.__version__,torch.cuda.get_device_name(0))'; df -h /workspace | tail -1" ;;
  run) exec ssh $OPTS -p "$POD_PORT" "root@$POD_HOST" "cd $POD_DIR && export PATH=/workspace/venv/bin:\$PATH HF_HOME=/workspace/hf && $1" ;;
  put) ssh $OPTS -p "$POD_PORT" "root@$POD_HOST" "mkdir -p $POD_DIR" && exec scp -q $OPTS -P "$POD_PORT" "$@" "root@$POD_HOST:$POD_DIR/" ;;
  get) exec scp -q -r $OPTS -P "$POD_PORT" "root@$POD_HOST:$1" "$2" ;;
  sync) # copy logs, benchmark json and results tables (not model weights) to ./runs-pod/, repeatedly
    while :; do
      rsync -az -e "ssh $OPTS -p $POD_PORT" --include='*/' --include='*.md' --include='*.log' --include='*.jsonl' \
        --include='*.json' --exclude='*' "root@$POD_HOST:$POD_DIR/runs/" ./runs-pod/ && echo "synced $(date +%H:%M)"
      [ "${1:-}" = "once" ] && exit 0
      sleep "${SYNC_EVERY:-3600}"
    done ;;
  get-models) # copy the trained models (healed + control, bf16) for a mode into ./models/pod-<mode>/
    MODE="${1:-small}"; mkdir -p "./models/pod-$MODE"
    for d in heal-$MODE heal-$MODE-self control-$MODE control-$MODE-self; do
      ssh $OPTS -p "$POD_PORT" "root@$POD_HOST" "test -d $POD_DIR/runs/$d/model" 2>/dev/null || continue
      echo "fetching $d/model ..."
      rsync -rltz -e "ssh $OPTS -p $POD_PORT" "root@$POD_HOST:$POD_DIR/runs/$d/model/" "./models/pod-$MODE/$d/"
    done
    du -sh "./models/pod-$MODE"/* 2>/dev/null ;;
  *) echo "usage: $0 ssh|put|get|sync|get-models ..."; exit 1 ;;
esac
