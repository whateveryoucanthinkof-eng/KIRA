#!/bin/bash
# run_a.sh REPO OUTDIR [extra args]  -- Branch A on the dry-run corpus, CPU single-thread
REPO=$1; OUT=$2; shift 2
C=${CORPUS:?set CORPUS to a dry-run corpus (scripts/dry_run_plan.py build_corpus)}
mkdir -p $OUT
cd $REPO
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 \
PYTHONPATH="$REPO:$REPO/bita" ${PYTHON:-python} -u scripts/retrain_branch_a_live.py \
  --split-scheme cross_year_ctu --pcap-root $C/pcap --cic2018-csv-dir $C/csv --cic2017-dir $C/cic2017 --ctu-dir $C/ctu13 \
  --tgne ${TGNE:?set TGNE to an encoder checkpoint} \
  --output $OUT/branch_a.pt --results-json $OUT/res.json --no-resume \
  --architecture paper --risk-objective soft_bce --risk-target hazard \
  --epochs 3 --patience 3 --step-back-after 2 --operating-point-criterion budgeted_f1 --alert-budget 2.0 \
  --spill-dir $OUT/spill --num-workers 0 --min-history-steps 3 --log-every 50 "$@" > $OUT/log.txt 2>&1
echo "exit $?"
