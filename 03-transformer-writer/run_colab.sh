#!/bin/bash
# Run 03-transformer-writer on an active Colab session (-s tw), pull back all JSON results,
# and ALWAYS stop the Colab VM on exit (success, error, or interrupt) so it never idles.
set -uo pipefail
cd "$(dirname "$0")"
COLAB="${COLAB:-$HOME/.local/bin/colab}"
SESSION="${SESSION:-tw}"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"

cleanup() {
  echo "== [cleanup] Downloading results tarball (if any) and stopping Colab session '$SESSION' =="
  echo 'import tarfile, os
if os.path.isdir("/content/tw/runs"):
    with tarfile.open("/content/tw/runs_json.tar.gz", "w:gz") as tf:
        for root, _, files in os.walk("/content/tw/runs"):
            for fn in files:
                if fn.endswith(".json"):
                    p = os.path.join(root, fn)
                    tf.add(p, arcname=os.path.relpath(p, "/content/tw"))
    print("packed", os.path.getsize("/content/tw/runs_json.tar.gz"), "bytes")' | "$COLAB" exec -s "$SESSION" 2>&1 || true
  mkdir -p runs
  "$COLAB" download -s "$SESSION" /content/tw/runs_json.tar.gz runs/colab_runs_json.tar.gz 2>&1 || true
  if [ -f runs/colab_runs_json.tar.gz ]; then
    tar -xzf runs/colab_runs_json.tar.gz -C . 2>&1 || true
  fi
  "$COLAB" stop -s "$SESSION" 2>&1 || true
  "$COLAB" usage 2>&1 || true
}
trap cleanup EXIT

echo "== Running hindsight arm (6 rounds) on $SESSION ($MODEL) =="
cat <<EOF | "$COLAB" exec -s "$SESSION"
import subprocess, sys
cmd = [
    sys.executable, "-u", "/content/tw/writer_loop.py",
    "--model", "$MODEL",
    "--arm", "hindsight",
    "--rounds", "6",
    "--k", "16",
    "--heldout-k", "8",
    "--batch-size", "32",
    "--train-steps", "120",
    "--out-dir", "/content/tw/runs/coder3b_hindsight",
]
rc = subprocess.call(cmd)
print("HINDSIGHT_EXIT_CODE:", rc)
if rc != 0:
    sys.exit(rc)
EOF

echo "== Running unverified control arm (3 rounds) on $SESSION ($MODEL) =="
cat <<EOF | "$COLAB" exec -s "$SESSION"
import subprocess, sys
cmd = [
    sys.executable, "-u", "/content/tw/writer_loop.py",
    "--model", "$MODEL",
    "--arm", "unverified",
    "--rounds", "3",
    "--k", "16",
    "--heldout-k", "8",
    "--batch-size", "32",
    "--train-steps", "120",
    "--out-dir", "/content/tw/runs/coder3b_unverified",
]
rc = subprocess.call(cmd)
print("UNVERIFIED_EXIT_CODE:", rc)
if rc != 0:
    sys.exit(rc)
EOF
echo "== All remote arms finished cleanly =="
