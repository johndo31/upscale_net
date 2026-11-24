#!/bin/sh
# Helper for running weight_writer on a RunPod pod (PyTorch template, root user) from the Mac.
#   ./pod.sh ssh ['command']   run a command on the pod (no command = interactive shell)
#   ./pod.sh put               copy the project files (*.py *.json *.sh, top level only, no runs*) to POD_DIR
#   ./pod.sh setup             venv (system torch) + transformers, pre-download the model into /workspace/hf
#   ./pod.sh run 'cmd'         run cmd inside POD_DIR with the venv and HF_HOME set
#   ./pod.sh sync [once]       mirror the pod's runs_gpu/ (results + logs; adapters only with SYNC_ADAPTERS=1)
#                              to ./runs_gpu/, every SYNC_EVERY seconds (default 600) unless "once"
# Set POD_HOST / POD_PORT (shown under "SSH over exposed TCP" in the RunPod console).
POD_HOST="${POD_HOST:?set POD_HOST to your pod address}"; POD_PORT="${POD_PORT:?set POD_PORT to your pod SSH port}"
POD_KEY="${POD_KEY:-$HOME/.ssh/runpod_ed25519}"
POD_DIR="${POD_DIR:-/workspace/weight_writer}"   # on the persistent volume, survives pod stops
TRANSFORMERS="${TRANSFORMERS:-transformers==5.22.1}"   # the version wsp.py was written and tested against
OPTS="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -i $POD_KEY"
RSH="ssh $OPTS -p $POD_PORT"
cd "$(dirname "$0")" || exit 1
cmd="$1"; [ $# -gt 0 ] && shift
case "$cmd" in
  ssh) exec ssh $OPTS -p "$POD_PORT" "root@$POD_HOST" "$@" ;;
  put) # top-level *.py *.json *.sh only: runs*/ and __pycache__/ are excluded by the final rule
    ssh $OPTS -p "$POD_PORT" "root@$POD_HOST" "mkdir -p $POD_DIR" && \
    exec rsync -rltz -e "$RSH" --include='*.py' --include='*.json' --include='*.sh' --exclude='*' \
      ./ "root@$POD_HOST:$POD_DIR/" ;;
  setup) exec ssh $OPTS -p "$POD_PORT" "root@$POD_HOST" "mkdir -p $POD_DIR /workspace/hf && cd $POD_DIR && chmod +x *.sh 2>/dev/null; \
      ([ -x /workspace/venv/bin/python ] || python -m venv --system-site-packages /workspace/venv) && \
      /workspace/venv/bin/pip install -q '$TRANSFORMERS' numpy 2>&1 | grep -v -i warn | tail -3; \
      HF_HOME=/workspace/hf HF_HUB_OFFLINE=0 /workspace/venv/bin/python -c 'from huggingface_hub import snapshot_download as s; print(\"model at\", s(\"Qwen/Qwen2.5-0.5B\"))' && \
      /workspace/venv/bin/python -c 'import torch,transformers,numpy;v=tuple(int(x) for x in torch.__version__.split(\"+\")[0].split(\".\")[:2]);print(\"ok torch\",torch.__version__,\"transformers\",transformers.__version__,\"numpy\",numpy.__version__,\"cuda\",torch.version.cuda,torch.cuda.get_device_name(0));v>=(2,6) or print(\"WARNING: transformers 5 needs torch>=2.6; pick a newer PyTorch template\")'; \
      nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; df -h /workspace | tail -1" ;;
  run) exec ssh $OPTS -p "$POD_PORT" "root@$POD_HOST" "cd $POD_DIR && export PATH=/workspace/venv/bin:\$PATH HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 && $1" ;;
  sync) # results JSON, logs and report tables; the LoRA adapters (~35 MB each) stay on the pod volume by default
    PT="--exclude=*.pt"; [ "${SYNC_ADAPTERS:-0}" = 1 ] && PT="--include=*.pt"
    mkdir -p ./runs_gpu
    while :; do
      rsync -rltz -e "$RSH" --exclude='*.tmp' "$PT" "root@$POD_HOST:$POD_DIR/runs_gpu/" ./runs_gpu/ && echo "synced $(date +%H:%M)"
      [ "${1:-}" = "once" ] && exit 0
      sleep "${SYNC_EVERY:-600}"
    done ;;
  *) echo "usage: $0 ssh|put|setup|run|sync ..."; exit 1 ;;
esac
