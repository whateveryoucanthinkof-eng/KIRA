#!/usr/bin/env bash
# Start the Branch B + DeepOP retrain as soon as Branch A finishes cleanly.
#
# Branch A holds ~7.6 GiB for its trajectory store and the downstream run
# builds a comparable one, so the two must not overlap on a 22 GiB box -- hence
# waiting rather than launching both. Branch A is the only producer of the
# Branch A checkpoint the adapter needs, so if it fails there is nothing to
# chain to and this exits without starting anything.
set -uo pipefail

REPO=/var/home/samito/Documents/cyber_network_predictor-2f6ff8a69d8845ddb9bf9f3d72366af6dd5b5875
LOG="$REPO/logs/branch_a_full.out"
UNIT=branch-a-retrain.service
OUT="$REPO/logs/downstream_retrain.out"

echo "[chain] waiting for $UNIT to finish ($(date -Is))"
while systemctl --user is-active --quiet "$UNIT"; do
    sleep 60
done
echo "[chain] $UNIT is no longer active ($(date -Is))"

# "saved=" is the last line main() prints, after the held-out test evaluation
# and the credibility stamp. Its absence means the run died part way.
if ! grep -q "^saved=" "$LOG"; then
    echo "[chain] ABORT: no 'saved=' line in $LOG -- Branch A did not complete."
    tail -25 "$LOG"
    exit 1
fi
echo "[chain] Branch A completed:"
grep -E "^epoch=|^HELD-OUT TEST|^saved=" "$LOG" | tail -12

systemd-run --user --unit=downstream-retrain \
    --description="Branch B + DeepOP retrain, full density, canonical encoder" \
    -p MemoryMax=17G -p MemoryHigh=15G -p MemorySwapMax=0 \
    -p WorkingDirectory="$REPO" \
    -p StandardOutput="append:$OUT" -p StandardError="append:$OUT" \
    -p Environment="PYTHONPATH=$REPO" \
    /var/home/samito/.pyenv/versions/3.12.14/bin/python3 -u \
    scripts/retrain_future_models_live.py \
      --cic-dir /var/home/samito/Documents/SIH/DATA/CSV \
      --ctu-dir /var/home/samito/Documents/SIH/CTU-13-Dataset \
      --tgne "$REPO/saved_models/bita_bigru_transformer-unified_final.pth" \
      --out-dir "$REPO/saved_models" \
      --epochs 6 --spill-dir "$REPO/.spill" --num-workers 4

echo "[chain] started downstream-retrain.service ($(date -Is)) -> $OUT"
