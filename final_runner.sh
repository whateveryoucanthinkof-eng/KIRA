#!/usr/bin/env bash
# One command to launch (or resume) the full training plan on this machine.
#
#   ./final_runner.sh              # launch; resumes from the last finished epoch if any
#   RUN_DRY_RUN=1 ./final_runner.sh  # run the synthetic end-to-end dry run first
#
# Settings come from ~/.config/cyberworld/plan_env.sh (template:
# scripts/ops/plan_env.example.sh): dataset paths, LANES, and
# PLAN_ENCODER_EXTRA (production: --fast_step --fast_step_level 3 --batch_planner;
# see how_to_test_and_apply_level4.md before switching to level 4).
#
# No resource caps: jobs may use all RAM, CPU and GPU. The one thing kept is
# MemorySwapMax=0 on the training jobs: this laptop's only swap is zram, and a
# job spilling into it livelocked the desktop twice (no OOM kill ever fired).
# With it, a job that truly runs out of RAM is stopped cleanly and the plan
# resumes from its last finished epoch.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
ENV_FILE="${ENV_FILE:-$HOME/.config/cyberworld/plan_env.sh}"
[ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE (copy scripts/ops/plan_env.example.sh there)" >&2; exit 2; }

if systemctl --user is-active -q train-plan; then
    echo "train-plan is already running. Monitor it:  systemctl --user status train-plan" >&2
    exit 1
fi
# A stopped plan can leave its per-job scopes behind; never run two copies.
if pgrep -f "bita/train.py|scripts/retrain_" >/dev/null; then
    echo "training processes are still running (pgrep -af 'bita/train.py|scripts/retrain_'); stop them first" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$ENV_FILE"
export MEM_MAX="${MEM_MAX_OVERRIDE:-infinity}"            # no memory cap
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[runner] building Rust components"
(cd rust/pcap_fast && cargo build --release -q)
(cd rust/tgn_host && cargo build --release -q)

if [ "${RUN_DRY_RUN:-0}" = 1 ]; then
    echo "[runner] dry run (synthetic corpus, whole plan)"
    PYTHONPATH="$REPO:$REPO/bita" "$PYTHON" scripts/dry_run_plan.py
fi

echo "=== launch $(date) commit $(git log --oneline -1) extra=[${PLAN_ENCODER_EXTRA:-}] lanes=${LANES:-1} ===" >> plan.log
systemctl --user reset-failed train-plan 2>/dev/null || true
systemd-run --user --unit=train-plan --working-directory="$REPO" \
    --setenv=PATH="$PATH" --setenv=MEM_MAX="$MEM_MAX" --setenv=ENV_FILE="$ENV_FILE" \
    --setenv=PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    -p StandardOutput=append:"$REPO/plan.log" -p StandardError=append:"$REPO/plan.log" \
    bash -c 'source "$ENV_FILE" && export MEM_MAX PYTORCH_CUDA_ALLOC_CONF && SKIP_DRY_RUN=1 exec bash scripts/run_training_plan.sh all'

cat <<EOF

[runner] launched as the user service 'train-plan' (survives closing the terminal).
  status:    systemctl --user status train-plan
  log:       tail -f $REPO/plan.log
  per-model: ls $REPO/results/training_plan/*/logs/
  watchdog:  python3 -u $REPO/scripts/ops/watchdog.py
  GPU:       nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  stop:      systemctl --user stop train-plan; then stop leftover 'run-p*.scope' units
             (systemctl --user list-units --type=scope); rerun this script to resume.
EOF
