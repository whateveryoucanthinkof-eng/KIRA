#!/usr/bin/env bash
# K.I.R.A. -- train every model from the datasets under data/ (see README.md).
#
#   ./train.sh              check the datasets, build the Rust parser, train everything
#   ./train.sh check        only verify the datasets and the toolchain
#   ./train.sh dryrun       run the whole pipeline on a tiny synthetic corpus (minutes)
#   ./train.sh promote      install the trained models into saved_models/ for the console
#   ./train.sh <stage>      any stage of scripts/run_training_plan.sh
#
# Linux only. Heavy steps run under a systemd memory cap (MEM_MAX, default 17G).
# Dataset locations default to data/; override with PCAP_ROOT, CIC2018_CSV_DIR,
# CIC2017_DIR and CTU_DIR. Outputs go to results/training_plan/ (OUT=...).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHON="${PYTHON:-python3}"

if [ "$(uname -s)" != Linux ]; then
    echo "K.I.R.A. training runs on Linux only." >&2; exit 2
fi
if command -v cargo >/dev/null 2>&1; then
    echo "[train] building the Rust components (PCAP parser, neighbour sampler)"
    (cd rust/pcap_fast && cargo build --release -q)
    (cd rust/tgn_host && cargo build --release -q)
else
    echo "[train] cargo not found: PCAP parsing falls back to Python (~12x slower)" >&2
    export CYBERWORLD_PCAP_PARSER=python
fi

# Encoder fast path (CUDA graphs + batch planner) when a CUDA GPU is present;
# the same model either way, only faster.
if "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    export PLAN_ENCODER_EXTRA="${PLAN_ENCODER_EXTRA:---fast_step --fast_step_level 4 --batch_planner}"
else
    echo "[train] no CUDA GPU: training on CPU (much slower; a GPU with 8 GB is recommended)" >&2
fi

case "${1:-all}" in
    check)  exec bash scripts/run_training_plan.sh preflight ;;
    all)    SKIP_DRY_RUN="${SKIP_DRY_RUN:-1}" exec bash scripts/run_training_plan.sh all ;;
    *)      exec bash scripts/run_training_plan.sh "$1" ;;
esac
