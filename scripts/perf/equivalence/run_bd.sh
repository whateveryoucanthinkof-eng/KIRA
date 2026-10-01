#!/bin/bash
# run_bd.sh REPO OUTDIR [extra]  -- Branch B + DeepOP on the dry-run corpus
REPO=$1; OUT=$2; shift 2
C=${CORPUS:?set CORPUS to a dry-run corpus (scripts/dry_run_plan.py build_corpus)}
mkdir -p $OUT
cd $REPO
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH="$REPO:$REPO/bita" \
${PYTHON:-python} -u scripts/retrain_future_models_live.py \
  --tgne ${TGNE:?set TGNE to an encoder checkpoint} \
  --out-dir $OUT/out --pcap-root $C/pcap --cic2018-csv-dir $C/csv --ctu-dir $C/ctu13 --cic2017-dir $C/cic2017 \
  --split-scheme cross_year_ctu --risk-target hazard --epochs 3 --patience 3 --step-back-after 2 \
  --results-json $OUT/cross_year.json --spill-dir $OUT/spill --num-workers 0 --no-resume \
  --allow-noncredible-branch-b "$@" > $OUT/log.txt 2>&1
echo "exit $?"
