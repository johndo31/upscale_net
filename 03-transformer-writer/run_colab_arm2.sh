#!/bin/bash
# Finish Exp 03: Arm 3 (3B unverified control, 2 rounds) + Arm 2b (7B debug+distill, 3 rounds)
# Uses live-streaming Popen (no silence timeouts), syncs JSONs after each arm, auto-stops on exit.
set -uo pipefail
cd "$(dirname "$0")"
COLAB="${COLAB:-$HOME/.local/bin/colab}"
SESSION="${SESSION:-tw3}"

sync_json() {
  echo 'import tarfile, os
if os.path.isdir("/content/tw/runs"):
    with tarfile.open("/content/tw/runs_arm3.tar.gz", "w:gz") as tf:
        for root, _, files in os.walk("/content/tw/runs"):
            for fn in files:
                if fn.endswith(".json"):
                    p = os.path.join(root, fn)
                    tf.add(p, arcname=os.path.relpath(p, "/content/tw"))
    print("packed", os.path.getsize("/content/tw/runs_arm3.tar.gz"), "bytes")' | "$COLAB" exec -s "$SESSION" --timeout 120 2>&1 || true
  mkdir -p runs
  "$COLAB" download -s "$SESSION" /content/tw/runs_arm3.tar.gz runs/colab_runs_arm3.tar.gz 2>&1 || true
  if [ -f runs/colab_runs_arm3.tar.gz ]; then
    tar -xzf runs/colab_runs_arm3.tar.gz -C . 2>&1 || true
  fi
}

cleanup() {
  echo "== [cleanup] Syncing Exp 03 results and stopping '$SESSION' =="
  sync_json
  "$COLAB" stop -s "$SESSION" 2>&1 || true
  "$COLAB" usage 2>&1 || true
}
trap cleanup EXIT

ALLOCATED=0
for GPU in G4 A100 L4; do
  echo "== Trying $GPU for '$SESSION' =="
  if "$COLAB" new -s "$SESSION" --gpu "$GPU"; then
    ALLOCATED=1
    break
  fi
done
[ "$ALLOCATED" -eq 1 ] || exit 1

echo 'import os; os.makedirs("/content/tw", exist_ok=True)' | "$COLAB" exec -s "$SESSION" --timeout 60 || exit 1
for f in target_tf.py tasks.py verify_realizability.py writer_loop.py debug_loop.py; do
  "$COLAB" upload -s "$SESSION" "$f" "/content/tw/$f" || exit 1
done

echo "== 1/2: Arm 3 — Qwen2.5-Coder-3B-Instruct (unverified control, 2 rounds) =="
cat <<'EOF' | "$COLAB" exec -s "$SESSION" --timeout 1800
import subprocess, sys
p = subprocess.Popen(
    [
        sys.executable, "-u", "/content/tw/writer_loop.py",
        "--model", "Qwen/Qwen2.5-Coder-3B-Instruct",
        "--arm", "unverified",
        "--rounds", "2",
        "--k", "16",
        "--heldout-k", "8",
        "--batch-size", "32",
        "--train-steps", "120",
        "--out-dir", "/content/tw/runs/coder3b_unverified",
    ],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print("ARM3_UNVERIF_EXIT:", rc, flush=True)
EOF
sync_json

echo "== 2/2: Arm 2b — Qwen2.5-Coder-7B-Instruct (debug + distill, 3 rounds) =="
cat <<'EOF' | "$COLAB" exec -s "$SESSION" --timeout 1800
import subprocess, sys
p = subprocess.Popen(
    [
        sys.executable, "-u", "/content/tw/debug_loop.py",
        "--model", "Qwen/Qwen2.5-Coder-7B-Instruct",
        "--rounds", "3",
        "--trajs", "6",
        "--debug-turns", "3",
        "--heldout-trajs", "4",
        "--batch-size", "16",
        "--train-steps", "140",
        "--out-dir", "/content/tw/runs/coder7b_debug",
    ],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print("ARM2B_7B_EXIT:", rc, flush=True)
EOF
sync_json

echo "== Exp 03 remaining arms finished =="
