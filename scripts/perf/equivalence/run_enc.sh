#!/bin/bash
# run_enc.sh REPO OUTDIR [extra]  -- encoder on the dry-run corpus, production fast flags
REPO=$1; OUT=$2; shift 2
C=${CORPUS:?set CORPUS to a dry-run corpus (scripts/dry_run_plan.py build_corpus)}
mkdir -p $OUT/{save,ckpt,logs,setup}
cd $REPO
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=0 PYTHONPATH="$REPO:$REPO/bita" \
${PYTHON:-python} -u bita/train.py --prefix enc_eq --data_name cic2018 \
  --save_dir $OUT/save --checkpoint_dir $OUT/ckpt --log_dir $OUT/logs --seed 42 \
  --pcap2018_root $C/pcap --pcap2018_label_dir $C/csv --split_scheme cross_year_ctu --train_splits train \
  --ctu13_dir $C/ctu13 --use_memory --n_degree 10 --n_epoch 2 --batch_size 200 --patience 3 --step_back_after 2 \
  --ingest_workers 2 --ingest_cache $OUT/ingest --shared_setup_dir $OUT/setup \
  --fast_step --fast_step_level 4 --batch_planner "$@" > $OUT/log.txt 2>&1
echo "exit $?"
