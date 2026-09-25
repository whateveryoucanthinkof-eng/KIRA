#!/usr/bin/env bash
# Launch DeepOP as soon as Branch A and Branch B are both finished.
#
# Runs unattended. DeepOP trains on Branch B's rollouts, so it must wait for
# Branch B; it is chained rather than run concurrently because each job holds
# ~8 GiB for its trajectory store and this box has 22 GiB.
#
# It refuses to start unless Branch B actually completed -- a half-trained
# world model would silently become DeepOP's conditioning signal.
set -uo pipefail

REPO=/var/home/samito/Documents/cyber_network_predictor-2f6ff8a69d8845ddb9bf9f3d72366af6dd5b5875
PY=/var/home/samito/.pyenv/versions/3.12.14/bin/python3
BB_LOG="$REPO/logs/downstream_retrain.out"
OUT="$REPO/logs/deepop_retrain.out"

echo "[chain-deepop] waiting for branch-a-retrain and downstream-retrain ($(date -Is))"
while systemctl --user is-active --quiet branch-a-retrain.service \
   || systemctl --user is-active --quiet downstream-retrain.service; do
    sleep 60
done
echo "[chain-deepop] both idle ($(date -Is))"

# Only look at the most recent run's output: StandardOutput=append accumulates
# across runs, so a completion line from an earlier run must not count.
START=$(grep -n "^frozen split:" "$BB_LOG" | tail -1 | cut -d: -f1)
if [ -z "$START" ]; then
    echo "[chain-deepop] ABORT: no 'frozen split:' line in $BB_LOG"
    exit 1
fi
RUN=$(tail -n "+$START" "$BB_LOG")

if ! printf '%s\n' "$RUN" | grep -q "served checkpoint updated.*host_wdt.pt"; then
    echo "[chain-deepop] ABORT: Branch B did not finish. Last 25 lines:"
    printf '%s\n' "$RUN" | tail -25
    exit 1
fi

echo "[chain-deepop] Branch B completed:"
printf '%s\n' "$RUN" | grep -E "^Branch B epoch=|embeddings:|risk:|early stop" | tail -20

systemd-run --user --unit=deepop-retrain \
    --description="DeepOP: cached Branch-B rollouts, token baselines, early stop" \
    -p MemoryMax=13G -p MemoryHigh=11G -p MemorySwapMax=0 \
    -p WorkingDirectory="$REPO" \
    -p StandardOutput="append:$OUT" -p StandardError="append:$OUT" \
    -p Environment="PYTHONPATH=$REPO" \
    "$PY" -u scripts/retrain_future_models_live.py \
      --cic-dir /var/home/samito/Documents/SIH/DATA/CSV \
      --ctu-dir /var/home/samito/Documents/SIH/CTU-13-Dataset \
      --tgne "$REPO/saved_models/bita_bigru_transformer-unified_final.pth" \
      --out-dir "$REPO/saved_models" --stages deepop \
      --epochs 6 --patience 2 --num-workers 4 --spill-dir "$REPO/.spill"

echo "[chain-deepop] started deepop-retrain.service ($(date -Is)) -> $OUT"
