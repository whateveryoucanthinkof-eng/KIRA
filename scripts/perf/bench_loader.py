"""Throughput of a downstream training DataLoader alone, on a synthetic store.

    python scripts/perf/bench_loader.py --dataset a|b|deepop --rows 77468077 --hosts 3491797 \
        --feats /path/on/disk/feats.bin --batches 3000 [--repo /other/checkout]

`--repo` imports the dataset from another checkout (an A/B against the
baseline). Reports batch/s, and the major page faults and I/O wait of the
loader workers, which is what the full-scale run was waiting on.
Run under `systemd-run --scope -p MemorySwapMax=0 -p MemoryMax=...` to give it
the page cache the production trainer actually had.
"""

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _majflt(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[9])
    except OSError:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2_000_000)
    ap.add_argument("--hosts", type=int, default=90_000)
    ap.add_argument("--windows", type=int, default=600_000)
    ap.add_argument("--feats", default=None, help="feature block on disk (memmap); RAM if unset")
    ap.add_argument("--batches", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=None, help="default: the trainer's (128, 128, 64)")
    ap.add_argument("--dataset", choices=("a", "b", "deepop"), default="a")
    ap.add_argument("--repo", default=str(HERE.parent.parent))
    ap.add_argument("--evict", action="store_true", help="drop the feature file from page cache first")
    ap.add_argument("--batched", action="store_true", help="new code: batched gather, host-major features")
    a = ap.parse_args()

    repo = os.path.abspath(a.repo)
    sys.path.insert(0, repo)
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from synth_store import make_store
    import branch_a_gnn_lstm.sequence_dataset as sd
    assert os.path.abspath(sd.__file__).startswith(repo), sd.__file__

    torch.manual_seed(42)
    t = time.time()
    store = make_store(a.rows, a.hosts, n_windows=a.windows, seed=7, feats_path=a.feats)
    store.use_hazard_target(10.0)
    if a.dataset == "a":
        ds = sd.LazyHostSequenceDataset(store, seq_len=15, min_trajectory_len=1)
    elif a.dataset == "b":
        from branch_b_world_model.train_branch_b import LazyHostRolloutDataset
        ds = LazyHostRolloutDataset(store, T=15, K=5)
    else:
        from deepop_decoder.joint_vocab import get_joint_vocab
        from deepop_decoder.train_cwa_decoder import LazyCWADataset
        ds = LazyCWADataset(store, get_joint_vocab(network_observable_only=True), K=5, T=15)
    a.batch_size = a.batch_size or {"a": 128, "b": 128, "deepop": 64}[a.dataset]
    print(f"index built {time.time() - t:.0f}s", flush=True)
    dset, kw = ds, dict(batch_size=a.batch_size, shuffle=True)
    if a.batched:
        t1 = time.time()
        ds.enable_batched()
        print(f"enable_batched {time.time() - t1:.0f}s", flush=True)
        from data_unification.host_major import BatchedView, PermutationBatchSampler, collate_prebatched
        dset = BatchedView(ds)
        kw = dict(batch_sampler=PermutationBatchSampler(len(ds), a.batch_size),
                  collate_fn=collate_prebatched)
    print(f"setup {time.time() - t:.0f}s, {len(ds):,} samples, feats "
          f"{type(store.feats).__name__} {store.feats.nbytes / 1e9:.1f} GB", flush=True)
    if a.evict and a.feats:
        fd = os.open(a.feats, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)

    dl = DataLoader(dset, num_workers=a.workers,
                    pin_memory=torch.cuda.is_available(), persistent_workers=a.workers > 0,
                    prefetch_factor=4 if a.workers else None, **kw)
    it = iter(dl)
    pids = [w.pid for w in getattr(it, "_workers", [])]
    next(it)
    f0 = sum(_majflt(p) for p in pids)
    c0 = os.times()
    t0 = time.perf_counter()
    n = 0
    report_every = max(1, a.batches // 5)
    for _ in range(a.batches):
        next(it)
        n += 1
        if n % report_every == 0:
            el = time.perf_counter() - t0
            print(f"  {n} batches  {n / el:7.1f} batch/s  worker majflt/batch "
                  f"{(sum(_majflt(p) for p in pids) - f0) / n:7.1f}", flush=True)
    el = time.perf_counter() - t0
    print(f"RESULT {a.dataset} {'batched' if a.batched else 'legacy'} batch/s {n / el:.1f}  majflt/batch {(sum(_majflt(p) for p in pids) - f0) / n:.1f}  "
          f"main cpu {100 * (os.times().user - c0.user + os.times().system - c0.system) / el:.0f}%",
          flush=True)


if __name__ == "__main__":
    main()
