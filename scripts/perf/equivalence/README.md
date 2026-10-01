# Old-vs-new equivalence harness

Runs a real trainer end to end on the synthetic dry-run corpus with the
production flags, for two checkouts, and compares the checkpoints leaf by leaf
with exact equality (`cmp_ckpt.py`; wall-clock fields skipped).

    python -c "import sys; sys.path.insert(0,'scripts'); from pathlib import Path; \
        from dry_run_plan import build_corpus; build_corpus(Path('/var/tmp/corpus'))"
    git worktree add --detach /var/tmp/base <baseline-commit>
    export CORPUS=/var/tmp/corpus TGNE=<encoder .pth> PYTHON=$(which python)
    CUDA_VISIBLE_DEVICES="" scripts/perf/equivalence/run_a.sh /var/tmp/base /var/tmp/eq/old
    CUDA_VISIBLE_DEVICES="" scripts/perf/equivalence/run_a.sh "$PWD"       /var/tmp/eq/new
    python scripts/perf/equivalence/cmp_ckpt.py /var/tmp/eq/old/branch_a.pt /var/tmp/eq/new/branch_a.pt

`run_bd.sh` does the same for Branch B + DeepOP (compare `out/branch_b/host_wdt.pt`
and `out/deepop/cwa_forecast_decoder.pt`), `run_enc.sh` for the encoder (which
is not GPU-deterministic: compare against two baseline runs, not one). Extra
arguments are passed to the trainer (e.g. `--num-workers 2`). Use absolute
output paths: the scripts cd into the checkout.
