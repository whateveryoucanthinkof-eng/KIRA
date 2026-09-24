#!/usr/bin/env bash
# Full live retrain launcher.
#
# Dependency graph:
#   Branch A ─┐
#             ├─ run concurrently (independent consumers of the TGNE encoder)
#   Branch B ─┴─> DeepOP (loads Branch B's new host_wdt.pt)
#
# The previous 2026-09-22 run failed at TrajectoryStore.finalize(), not from an
# OOM. Its spill files stopped at exactly 513 MiB while the stores had grown
# past 22M rows. `LimitFSIZE=infinity` is required: a full feature spill is
# multiple GiB. Every active trainer also receives its own spill directory.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="/var/home/samito/.pyenv/versions/3.12.14/bin/python3"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="$REPO/.spill/retrain-$RUN_ID"
LOG_ROOT="$REPO/logs"

mkdir -p "$RUN_ROOT/branch-a" "$RUN_ROOT/branch-b" "$RUN_ROOT/deepop" "$LOG_ROOT"

A_LOG="$LOG_ROOT/branch_a_$RUN_ID.out"
B_LOG="$LOG_ROOT/branch_b_$RUN_ID.out"
D_LOG="$LOG_ROOT/deepop_$RUN_ID.out"

if ! systemctl --user show-environment >/dev/null 2>&1; then
    echo "ERROR: cannot reach the user systemd manager. Run this from your normal desktop terminal." >&2
    exit 1
fi

echo "=== full retrain $RUN_ID ==="
echo "repo:  $REPO"
echo "spill: $RUN_ROOT"
df -h "$REPO"

launch() {
    local unit="$1" max_mem="$2" high_mem="$3" log="$4"
    shift 4
    echo "[$(date -Is)] starting $unit -> $log"
    systemd-run --user --wait --collect \
        --unit="${unit}-${RUN_ID}" \
        --description="${unit} full retrain ${RUN_ID}" \
        -p "MemoryMax=$max_mem" \
        -p "MemoryHigh=$high_mem" \
        -p "MemorySwapMax=0" \
        -p "LimitFSIZE=infinity" \
        -p "WorkingDirectory=$REPO" \
        -p "StandardOutput=append:$log" \
        -p "StandardError=append:$log" \
        -p "Environment=PYTHONPATH=$REPO" \
        "$@"
}

# Risk target: hazard, not severity. The severity target is ~82% exact zeros
# and describes the CURRENT window, so both risk heads were learning detection
# with a one-step delay (Branch B's risk head was worse than predicting zero in
# every epoch). hazard = exp(-seconds_to_next_attack / tau) is continuous and
# forward-looking. soft_bce is its proper scoring rule; plain bce would
# binarise it and throw the timing away. The previous run used
# `--risk-objective bce --risk-target severity`; compare against it.
#
# DeepOP only starts if Branch B beats persistence (MIN_BRANCH_B_SKILL in
# scripts/retrain_future_models_live.py). If it refuses, fix Branch B first.
#
# The measured peaks were ~6.7 GiB (A) and ~7.0 GiB (B). 10 GiB each keeps
# their combined hard caps below the machine's 22 GiB physical memory.
launch branch-a-retrain 10G 9G "$A_LOG" \
    "$PY" -u scripts/retrain_branch_a_live.py \
    --cic-dir /var/home/samito/Documents/SIH/DATA/CSV \
    --ctu-dir /var/home/samito/Documents/SIH/CTU-13-Dataset \
    --output "$REPO/saved_models/branch_a/branch_a_lstm.pt" \
    --epochs 8 --patience 3 --risk-objective soft_bce --risk-target hazard \
    --spill-dir "$RUN_ROOT/branch-a" --num-workers 4 &
A_PID=$!

launch branch-b-retrain 10G 9G "$B_LOG" \
    "$PY" -u scripts/retrain_future_models_live.py \
    --cic-dir /var/home/samito/Documents/SIH/DATA/CSV \
    --ctu-dir /var/home/samito/Documents/SIH/CTU-13-Dataset \
    --tgne "$REPO/saved_models/bita_bigru_transformer-unified_final.pth" \
    --out-dir "$REPO/saved_models" \
    --stages branch_b --epochs 6 --patience 2 --risk-target hazard \
    --spill-dir "$RUN_ROOT/branch-b" --num-workers 4 &
B_PID=$!

A_OK=0
B_OK=0
if wait "$A_PID"; then A_OK=1; else echo "Branch A failed; see $A_LOG" >&2; fi
if wait "$B_PID"; then B_OK=1; else echo "Branch B failed; see $B_LOG" >&2; fi

if (( ! B_OK )); then
    echo "Not starting DeepOP because Branch B, its prerequisite, failed." >&2
    echo "Logs: $A_LOG  $B_LOG" >&2
    exit 1
fi
if (( ! A_OK )); then
    echo "WARNING: Branch A failed, but it is independent of Branch B/DeepOP; continuing." >&2
fi

# DeepOP reads saved_models/branch_b/host_wdt.pt, so it must follow Branch B.
launch deepop-retrain 13G 11G "$D_LOG" \
    "$PY" -u scripts/retrain_future_models_live.py \
    --cic-dir /var/home/samito/Documents/SIH/DATA/CSV \
    --ctu-dir /var/home/samito/Documents/SIH/CTU-13-Dataset \
    --tgne "$REPO/saved_models/bita_bigru_transformer-unified_final.pth" \
    --out-dir "$REPO/saved_models" \
    --stages deepop --epochs 6 --patience 2 \
    --spill-dir "$RUN_ROOT/deepop" --num-workers 4

echo "=== all stages completed | $(date -Is) ==="
printf 'Logs:\n  %s\n  %s\n  %s\n' "$A_LOG" "$B_LOG" "$D_LOG"
echo "Spill root retained for diagnosis: $RUN_ROOT"
df -h "$REPO"

if (( ! A_OK )); then
    echo "Branch B + DeepOP completed, but Branch A failed; see $A_LOG" >&2
    exit 1
fi
