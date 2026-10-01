"""Branch B / DeepOP training-loop throughput on a synthetic store (loader + step).

    python scripts/perf/bench_downstream_train.py --model b|deepop [--no-graph] \
        [--legacy --repo CHECKOUT] [--batches 3000] [--workers 4]

Mirrors the training loops of scripts/retrain_future_models_live.py with the
production models, optimizer and guard. DeepOP is fed a random h_rollout of
the right shape (the cached Branch B rollout, which costs a gather here).
"""

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("b", "deepop"), required=True)
    ap.add_argument("--rows", type=int, default=4_000_000)
    ap.add_argument("--hosts", type=int, default=180_000)
    ap.add_argument("--batches", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--repo", default=str(HERE.parent.parent))
    ap.add_argument("--legacy", action="store_true", help="per-sample loader + syncing guard step")
    ap.add_argument("--no-graph", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.repo))
    sys.path.insert(0, str(HERE))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from synth_store import make_store
    from cyberworld_v4.training_guard import TrainingGuard

    torch.manual_seed(42)
    dev = "cuda"
    st = make_store(a.rows, a.hosts, n_windows=600_000, seed=7)
    st.use_hazard_target(10.0)
    if a.model == "b":
        from branch_b_world_model.train_branch_b import LazyHostRolloutDataset
        from branch_b_world_model.rollout_encoder_decoder import HostWorldDynamicsTransformer
        from branch_b_world_model.infiltration_head import InfiltrationRiskHead
        ds = LazyHostRolloutDataset(st, T=15, K=5)
        bs = 128
        wdt = HostWorldDynamicsTransformer(d_latent=27, d_model=64, n_heads=4, n_layers=3).to(dev)
        risk = InfiltrationRiskHead(d_latent=27, hidden_dim=32).to(dev)
        mods = [wdt, risk]
        opt = torch.optim.Adam(list(wdt.parameters()) + list(risk.parameters()), lr=1e-3, weight_decay=1e-4)

        def lossfn(h, tgt, tr, th, tf):
            pred = wdt.rollout(h, K=5, t_history=th, t_future=tf)
            pr, _ = risk.forward_trajectory(pred)
            return sum((0.9 ** k) * F.mse_loss(pred[:, k], tgt[:, k]) for k in range(5)) + risk.risk_loss(pr, tr)

        def inputs(b):
            return [b[k].to(dev, non_blocking=True) for k in ("h_history", "h_future", "risk_future", "t_history", "t_future")]
    else:
        from deepop_decoder.joint_vocab import get_joint_vocab
        from deepop_decoder.train_cwa_decoder import LazyCWADataset
        from deepop_decoder.forecast_decoder import DeepOPForecastDecoder, smoothed_and_plain_ce
        vocab = get_joint_vocab(network_observable_only=True)
        ds = LazyCWADataset(st, vocab, K=5, T=15)
        bs = 64
        dec = DeepOPForecastDecoder(d_latent=27, d_model=72, vocab_size=vocab.vocab_size, n_heads=6,
                                    num_layers=2, window_sizes=[2, 4, 8], dim_feedforward=144).to(dev)
        mods = [dec]
        opt = torch.optim.AdamW(dec.parameters(), lr=5e-4, weight_decay=1e-4)
        sup = torch.ones(vocab.vocab_size, dtype=torch.bool, device=dev)
        sup[:3] = False
        kw = {}
        if not a.legacy:
            kw = dict(support_index=sup.nonzero().squeeze(1))

        def lossfn(h, inp, tgt, obs):
            l, p = smoothed_and_plain_ce(dec(h, inp, obs_tokens=obs), tgt, label_smoothing=0.04,
                                         support=sup, **kw)
            return l, p.detach()

        def inputs(b):
            h = b["h_future"].to(dev, non_blocking=True)
            obs = b["obs_tokens"].to(dev, non_blocking=True)
            drop = (torch.rand(obs.shape, device=dev) < 0.1) & (obs != vocab.pad_idx) & (obs != vocab.bos_idx)
            return [h + 0.01, b["input_tokens"].to(dev, non_blocking=True),
                    b["target_tokens"].to(dev, non_blocking=True), obs.masked_fill(drop, vocab.pad_idx)]

    kwl = dict(num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0,
               prefetch_factor=4 if a.workers else None)
    if a.legacy:
        dl = DataLoader(ds, batch_size=bs, shuffle=True, **kwl)
    else:
        from data_unification.host_major import batched_loader
        ds.enable_batched()
        dl = batched_loader(ds, bs, True, **kwl)
    guard = TrainingGuard("x", mods, opt, mode="min", patience=3, step_back_after=2,
                          warmup_steps=500, clip_norm=1.0, log=print,
                          **({} if a.legacy else {'graph_undo': not getattr(a, 'no_graph', False)}))
    step = guard.backward_step if (a.legacy or not guard.deferred_supported()) else guard.backward_step_deferred
    fn = lossfn
    if not (a.legacy or a.no_graph):
        from cyberworld_v4.graphed_step import GraphedLoss
        fn = GraphedLoss(lossfn, mods)
    for m in mods:
        m.train()
    acc = torch.zeros((), device=dev, dtype=torch.float64)

    def one(b):
        opt.zero_grad(set_to_none=True)
        out = fn(*inputs(b))
        loss = out[0] if isinstance(out, tuple) else out
        ok = step(loss)
        if ok is True:
            acc.add_(loss.detach().double())

    it = iter(dl)
    for _ in range(100):
        one(next(it))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.batches):
        one(next(it))
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    tag = "legacy" if a.legacy else ("new" + ("" if a.no_graph else "+graph"))
    print(f"RESULT {a.model} {tag} w{a.workers} {a.batches / el:.1f} batch/s "
          f"({1e3 * el / a.batches:.2f} ms/batch)", flush=True)


if __name__ == "__main__":
    main()
