#!/usr/bin/env bash
# Runs the training plan agreed in claude_latest_analysis/30_training_decisions.md.
#
#   PCAP_ROOT=... CIC2018_CSV_DIR=... CIC2017_DIR=... CTU_DIR=... \
#       scripts/run_training_plan.sh [preflight|dryrun|compare|seeds|downstream|summary|all]
#
# Stages, in order ("all" runs them in sequence and stops at the first failure):
#
#   preflight   every capture present and correctly named (scripts/check_datasets.py)
#   dryrun      the whole plan on a tiny synthetic corpus (scripts/dry_run_plan.py);
#               SKIP_DRY_RUN=1 to skip it
#   compare     encoder (TGN memory on, 10 most-recent neighbours) + Branch A,
#               with and without IP-identity features, seed 42, scheme
#               cross_year_ctu: train on CIC-2018 PCAP + CTU-13, tune on
#               2018 validation days, score ONCE on all of CIC-2017.
#   seeds       the chosen variant (cross_network, decided in advance so the
#               2017 test never picks it) again with seeds 123 and 2024.
#   downstream  Branch B then DeepOP on the seed-42 cross_network encoder.
#               DeepOP is skipped if Branch B does not beat persistence by 2%.
#   summary     mean / spread over the three seeds, and the promotion commands.
#
# Nothing here writes to saved_models/. Promoting a result to serving is a
# deliberate step, printed by `summary`.
#
# Every heavy step runs under a memory cap (MemoryMax, no swap): an uncapped
# job has frozen the training machine twice (analysis 27). Re-running a stage
# skips any step whose output already exists.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

: "${PCAP_ROOT:?set PCAP_ROOT to the CIC-2018 <day>_pcap directories}"
: "${CIC2018_CSV_DIR:?set CIC2018_CSV_DIR to the CIC-2018 <day>_csv.csv label files}"
: "${CIC2017_DIR:?set CIC2017_DIR to the CIC-2017 TrafficLabelling CSVs}"
: "${CTU_DIR:?set CTU_DIR to the CTU-13 <scenario>/*.binetflow directory}"
OUT="${OUT:-results/training_plan}"
MEM_MAX="${MEM_MAX:-17G}"
PYTHON="${PYTHON:-python}"
NUM_WORKERS="${NUM_WORKERS:-4}"

SCHEME=cross_year_ctu
CHOSEN_IP=cross_network            # decided in advance; see analysis 30, "winner"
SEEDS_EXTRA=(123 2024)             # with 42 from `compare`: cyberworld_v4.config.SEEDS[:3]
# Every model trains under cyberworld_v4/training_guard.py: LR warmup, gradient
# clipping, non-finite steps skipped; after 2 epochs without improvement it
# restores the best weights and halves the LR, after 3 it stops. The epoch
# counts below are ceilings, not targets -- the guard ends each run once it
# stops learning.
#
# PLAN_ENCODER_EXTRA / PLAN_BRANCH_A_EXTRA / PLAN_DOWNSTREAM_EXTRA are appended
# to each stage's arguments (later flags win). scripts/dry_run_plan.py uses
# them to run this exact plan on a tiny synthetic corpus.
ENCODER_ARGS="--use_memory --n_degree 10 --n_epoch 50 --patience 3 --step_back_after 2 ${PLAN_ENCODER_EXTRA:-}"
BRANCH_A_ARGS="--architecture paper --risk-objective soft_bce --risk-target hazard \
--epochs 15 --patience 3 --step-back-after 2 --operating-point-criterion budgeted_f1 --alert-budget 2.0 \
--spill-dir $OUT/.spill --num-workers $NUM_WORKERS ${PLAN_BRANCH_A_EXTRA:-}"

compare_dir() { echo "$OUT/compare_seed$1"; }
encoder_of() {  # seed ip -> encoder checkpoint written by run_encoder_comparison.py
    echo "$(compare_dir "$1")/encoders/cic2018__$2/enc_cic2018__$2-cic2018.pth"
}

capped() {
    if command -v systemd-run >/dev/null 2>&1; then
        systemd-run --user --scope --quiet -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 -- "$@"
    elif [ "${ALLOW_UNCAPPED:-0}" = 1 ]; then
        echo "[plan] WARNING: no systemd-run; running UNCAPPED because ALLOW_UNCAPPED=1" >&2
        "$@"
    else
        echo "[plan] systemd-run not found: refusing to start a heavy job without a memory cap." >&2
        echo "       Set ALLOW_UNCAPPED=1 to override knowingly." >&2
        exit 2
    fi
}

preflight() {
    local bad=0
    for v in PCAP_ROOT CIC2018_CSV_DIR CIC2017_DIR CTU_DIR; do
        if [ ! -d "${!v}" ]; then echo "[plan] $v=${!v} is not a directory" >&2; bad=1; fi
    done
    [ "$bad" = 0 ] || exit 2
    "$PYTHON" -c "import torch; print('[plan] torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    # Every capture by its lock name, each PCAP day paired with its own label
    # CSV, the right CIC-2017 variant, and which known-corrupt PCAPs remain.
    "$PYTHON" scripts/check_datasets.py --scheme "$SCHEME" \
        --pcap-root "$PCAP_ROOT" --cic2018-csv-dir "$CIC2018_CSV_DIR" \
        --cic2017-dir "$CIC2017_DIR" --ctu-dir "$CTU_DIR"
    echo "[plan] out=$OUT mem_max=$MEM_MAX scheme=$SCHEME chosen_ip=$CHOSEN_IP"
    echo "[plan] preflight ok"
}

comparison() {  # seed ip-flags...
    local seed="$1"; shift
    capped "$PYTHON" scripts/run_encoder_comparison.py \
        --arms cic2018 "$@" --split-scheme "$SCHEME" \
        --pcap-root "$PCAP_ROOT" --cic2018-csv-dir "$CIC2018_CSV_DIR" \
        --cic2017-dir "$CIC2017_DIR" --ctu-dir "$CTU_DIR" \
        --out "$(compare_dir "$seed")" --seed "$seed" \
        --encoder-args "$ENCODER_ARGS" --branch-a-args "$BRANCH_A_ARGS" \
        --python "$PYTHON"
}

compare() {
    echo "[plan] compare: both IP variants, seed 42"
    comparison 42 --ip-ablation
}

seeds() {
    for s in "${SEEDS_EXTRA[@]}"; do
        echo "[plan] seeds: $CHOSEN_IP, seed $s"
        comparison "$s" --ip-variants "$CHOSEN_IP"
    done
}

downstream() {
    local enc; enc="$(encoder_of 42 "$CHOSEN_IP")"
    [ -f "$enc" ] || { echo "[plan] $enc missing: run 'compare' first" >&2; exit 2; }
    mkdir -p "$OUT/downstream"
    echo "[plan] downstream: Branch B (+ DeepOP if Branch B clears the skill gate) on $enc"
    # The encoder was trained with the cross_network node-feature ablation; the
    # loader refuses to serve it under any other setting, so set it to match.
    capped env CYBERWORLD_ABLATE_NODE_FEATURES="$CHOSEN_IP" \
        "$PYTHON" scripts/retrain_future_models_live.py \
        --tgne "$enc" --out-dir "$OUT/downstream" \
        --pcap-root "$PCAP_ROOT" --cic2018-csv-dir "$CIC2018_CSV_DIR" \
        --ctu-dir "$CTU_DIR" --cic2017-dir "$CIC2017_DIR" \
        --split-scheme "$SCHEME" --risk-target hazard \
        --epochs 12 --patience 3 --step-back-after 2 \
        --results-json "$OUT/downstream/cross_year.json" \
        --spill-dir "$OUT/.spill" --num-workers "$NUM_WORKERS" \
        ${PLAN_DOWNSTREAM_EXTRA:-} \
        2>&1 | tee "$OUT/downstream/downstream.log"
}

summary() {
    "$PYTHON" - "$OUT" "$CHOSEN_IP" <<'EOF'
import json, math, sys
from pathlib import Path
out, ip = Path(sys.argv[1]), sys.argv[2]
keys = ("test_2017_macro_f1_seen", "test_2017_macro_f1_all", "test_2017_risk_auc",
        "test_2017_risk_brier", "val_2018_macro_f1")
print("\n== both IP variants, seed 42 (reported, not selected on) ==")
cmp42 = out / "compare_seed42" / "comparison.md"
print(cmp42.read_text() if cmp42.exists() else f"  {cmp42} missing")
rows = []
for seed in (42, 123, 2024):
    p = out / f"compare_seed{seed}" / "comparison.json"
    if not p.exists():
        print(f"  seed {seed}: {p} missing"); continue
    for r in json.loads(p.read_text())["rows"]:
        if r["encoder"] == "cic2018" and r["ip_features"] == ip and r["status"] == "ok":
            rows.append((seed, r))
print(f"\n== {ip}: {len(rows)} seed(s) ==")
for k in keys:
    v = [r[k] for _s, r in rows if isinstance(r.get(k), (int, float)) and r[k] == r[k]]
    if not v:
        print(f"  {k:28s} n/a"); continue
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else float("nan")
    print(f"  {k:28s} mean {m:.4f}  sd {sd:.4f}  n={len(v)}  " + " ".join(f"{x:.4f}" for x in v))
print("  credible:", {s: r.get("credible") for s, r in rows})
EOF
    local enc; enc="$(encoder_of 42 "$CHOSEN_IP")"
    cat <<EOF

== To serve these (deliberate, not done by this script) ==
  cp "$enc" saved_models/bita_bigru_transformer-unified_final.pth
  cp "${enc%.pth}_config.json" saved_models/bita_bigru_transformer-unified_final_config.json
  cp "$(compare_dir 42)/branch_a/cic2018__$CHOSEN_IP.pt" saved_models/branch_a/branch_a_lstm.pt
  cp "$OUT/downstream/branch_b/host_wdt.pt" saved_models/branch_b/host_wdt.pt
  cp "$OUT/downstream/deepop/cwa_forecast_decoder.pt" saved_models/deepop/cwa_forecast_decoder.pt   # only if DeepOP trained
  export CYBERWORLD_ABLATE_NODE_FEATURES=$CHOSEN_IP     # serving must match the encoder
  $PYTHON scripts/write_model_manifests.py
  then remove the strict xfail markers in tests/test_served_checkpoint_contracts.py
EOF
}

dryrun() {
    # The whole plan on a tiny synthetic corpus in the real formats, before the
    # real run spends hours: a crash here costs minutes. It found the TGN-memory
    # crash that would have ended the first real validation pass.
    if [ "${SKIP_DRY_RUN:-0}" = 1 ]; then
        echo "[plan] dry run skipped (SKIP_DRY_RUN=1)"
        return
    fi
    echo "[plan] dry run: the whole plan on a tiny synthetic corpus first"
    "$PYTHON" scripts/dry_run_plan.py
}

stage="${1:-all}"
case "$stage" in
    preflight)  preflight ;;
    dryrun)     dryrun ;;
    compare)    preflight; compare ;;
    seeds)      seeds ;;
    downstream) downstream ;;
    summary)    summary ;;
    all)        preflight; dryrun; compare; seeds; downstream; summary ;;
    *) echo "usage: $0 [preflight|dryrun|compare|seeds|downstream|summary|all]" >&2; exit 2 ;;
esac
