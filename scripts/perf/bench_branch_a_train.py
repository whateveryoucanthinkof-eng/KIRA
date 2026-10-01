"""Branch A training-loop throughput on a synthetic store (loader + step).

    python scripts/perf/bench_branch_a_train.py [--repo CHECKOUT] [--legacy] \
        [--rows N --hosts H --feats PATH] [--batches 3000] [--profile]

Mirrors the loop of scripts/retrain_branch_a_live.py with the production
model (PAPER_ARCH, soft_bce), optimizer and guard. `--legacy` uses the
per-sample loader and the syncing guard step (the pre-optimisation loop);
`--repo` imports the model/dataset from another checkout. `--profile` runs
torch.profiler over 50 steps and prints GPU busy time, kernels and syncs.
"""

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4_000_000)
    ap.add_argument("--hosts", type=int, default=180_000)
    ap.add_argument("--windows", type=int, default=600_000)
    ap.add_argument("--feats", default=None)
    ap.add_argument("--batches", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--repo", default=str(HERE.parent.parent))
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--cuda-graph", action="store_true")
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)
    sys.path.insert(0, repo)
    sys.path.insert(0, str(HERE))
    import torch
    from torch.utils.data import DataLoader
    from synth_store import make_store
    import branch_a_gnn_lstm.sequence_dataset as sd
    from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
    from cyberworld_v4.training_guard import TrainingGuard
    assert os.path.abspath(sd.__file__).startswith(repo)

    torch.manual_seed(42)
    dev = "cuda"
    st = make_store(a.rows, a.hosts, n_windows=a.windows, seed=7, feats_path=a.feats)
    st.use_hazard_target(10.0)
    ds = sd.LazyHostSequenceDataset(st, seq_len=15, min_trajectory_len=1)
    kw = dict(num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0,
              prefetch_factor=4 if a.workers else None)
    if a.legacy:
        dl = DataLoader(ds, batch_size=128, shuffle=True, **kw)
    else:
        ds.enable_batched()
        dl = DataLoader(sd.BatchedSequenceView(ds), collate_fn=sd.collate_prebatched,
                        batch_sampler=sd.PermutationBatchSampler(len(ds), 128), **kw)
    model = MultiTaskLSTM(input_dim=27, num_techniques=14, num_gradations=4,
                          risk_objective="soft_bce", **MultiTaskLSTM.PAPER_ARCH).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    guard = TrainingGuard("branch_a", [model], opt, mode="max", patience=3, step_back_after=2,
                          warmup_steps=500, clip_norm=1.0, log=print)
    step = guard.backward_step if (a.legacy or not guard.deferred_supported()) \
        else guard.backward_step_deferred
    loss_sum = torch.zeros((), device=dev, dtype=torch.float64)
    nb = torch.zeros((), device=dev, dtype=torch.long)
    model.train()

    def one(batch):
        nonlocal loss_sum
        x = batch["features"].to(dev, non_blocking=True)
        tg = {k: batch[k].to(dev, non_blocking=True) for k in ("risk", "technique", "gradation")}
        th = batch["t_history"].to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        p = model(x, t_history=th)
        loss, _ = model.compute_loss(p, tg)
        ok = step(loss)
        if ok is False:
            return
        model.uncertainty_loss.project_()
        if ok is True:
            loss_sum += loss.detach().double().sum()
        else:
            loss_sum += torch.where(ok, loss.detach().double().sum(),
                                    torch.zeros((), device=dev, dtype=torch.float64))

    it = iter(dl)
    for _ in range(100):
        one(next(it))
    torch.cuda.synchronize()
    if a.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(50):
                one(next(it))
            torch.cuda.synchronize()
        ev = prof.key_averages()
        kern = [e for e in ev if e.device_type.name == "CUDA"]
        gpu_ms = sum(e.self_device_time_total for e in kern) / 50 / 1e3
        n_k = sum(e.count for e in kern) / 50
        n_sync = sum(e.count for e in ev if "Synchronize" in e.key) / 50
        print(f"PROFILE GPU busy {gpu_ms:.3f} ms/step, {n_k:.0f} kernels/step, "
              f"{n_sync:.1f} syncs/step")
        print(ev.table(sort_by="self_cpu_time_total", row_limit=25))
    t0 = time.perf_counter()
    n = 0
    for _ in range(a.batches):
        one(next(it))
        n += 1
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    print(f"RESULT {'legacy' if a.legacy else 'new'} {n / el:.1f} batch/s "
          f"({1e3 * el / n:.2f} ms/batch)", flush=True)


if __name__ == "__main__":
    main()
