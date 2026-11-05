#!/bin/sh
# Pack the scripts into a single self-extracting shell snippet (bundle.sh).
# Paste the contents of bundle.sh into the pod's terminal to recreate the repo there.
set -e
cd "$(dirname "$0")"
FILES="grow.py verify.py heal.py self_distill.py results.py bench.sh run_all.sh requirements.txt runpod.md README.md"
{
  echo "mkdir -p ~/upscale_net && cd ~/upscale_net && base64 -d <<'__BUNDLE__' | tar xz"
  tar cz $FILES | base64
  echo "__BUNDLE__"
  echo "chmod +x *.sh && ls -la && echo 'upscale_net unpacked'"
} > bundle.sh
echo "wrote bundle.sh ($(wc -c < bundle.sh) bytes). Paste its contents into the pod terminal."
