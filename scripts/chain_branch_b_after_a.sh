#!/usr/bin/env bash
# Run Branch B once Branch A is done -- sequentially, not alongside it.
#
# Both at full density do NOT fit on this 22 GiB box. Measured 2026-09-22 with
# both live plus a review agent: Branch B was throttled 3,775,949 times, pinned
# at its MemoryHigh, its workers had each read 748 GiB from a 2.3 GiB feature
# block, and it reported 10.5 batch/s against its own 52.5 baseline -- 197
# minutes for ONE of six epochs. Branch A was throttled 1,500,062 times in the
# same window. Stopping Branch B took Branch A straight back to 128.7 batch/s
# with 0.00% stall.
#
# They are independent (both only consume TGNE), so running them concurrently
# is *correct* -- it is just slower than running them one after the other on
# this much RAM.
set -uo pipefail
REPO=/var/home/samito/Documents/cyber_network_predictor-2f6ff8a69d8845ddb9bf9f3d72366af6dd5b5875
PY=/var/home/samito/.pyenv/versions/3.12.14/bin/python3
A_LOG="$REPO/logs/clean_branch_a.out"
RID="$(date +%Y%m%d-%H%M%S)"

echo "[chain-b] waiting for Branch A ($(date -Is))"
while systemctl --user list-units --state=running --no-legend 2>/dev/null \
      | grep -q 'clean-branch-a-'; do sleep 60; done
echo "[chain-b] Branch A idle ($(date -Is))"

# Only the most recent run: StandardOutput=append accumulates across runs.
START=$(grep -n "^train_records=" "$A_LOG" | tail -1 | cut -d: -f1)
if [ -z "$START" ] || ! tail -n "+$START" "$A_LOG" | grep -q "^saved="; then
    echo "[chain-b] ABORT: Branch A did not reach its save step."
    tail -n "+${START:-1}" "$A_LOG" | tail -20
    exit 1
fi
echo "[chain-b] Branch A completed:"
tail -n "+$START" "$A_LOG" | grep -E "^epoch=|^HELD-OUT|^saved=|early stop" | tail -12

mkdir -p "$REPO/.spill/run-$RID/branch-b"
systemd-run --user --unit="clean-branch-b-$RID" \
    --description="Branch B, hazard target, Huber risk loss (sequential)" \
    -p MemoryMax=15G -p MemoryHigh=13G -p MemorySwapMax=0 -p LimitFSIZE=infinity \
    -p WorkingDirectory="$REPO" \
    -p StandardOutput="append:$REPO/logs/clean_branch_b.out" \
    -p StandardError="append:$REPO/logs/clean_branch_b.out" \
    -p Environment="PYTHONPATH=$REPO" \
    "$PY" -u scripts/retrain_future_models_live.py \
      --cic-dir /var/home/samito/Documents/SIH/DATA/CSV \
      --ctu-dir /var/home/samito/Documents/SIH/CTU-13-Dataset \
      --tgne "$REPO/saved_models/bita_bigru_transformer-unified_final.pth" \
      --out-dir "$REPO/saved_models" --stages branch_b --risk-target hazard \
      --epochs 6 --patience 2 --num-workers 4 \
      --spill-dir "$REPO/.spill/run-$RID/branch-b"
echo "[chain-b] started clean-branch-b-$RID ($(date -Is))"
