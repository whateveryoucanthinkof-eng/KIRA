"""Validation-pass throughput of Branch A, Branch B and DeepOP on a synthetic store.

    python scripts/perf/bench_validation.py --model a|b|deepop [--batches 2000]

Calls the trainers' own validation code where it is a function (Branch A's
`_evaluate`) and otherwise times the same per-batch body via
`retrain_future_models_live.validate_branch_b` / `validate_deepop`
(the functions the training loops call). Reports batch/s and ms/batch.
"""

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("a", "b", "deepop"), required=True)
    ap.add_argument("--rows", type=int, default=3_000_000)
    ap.add_argument("--hosts", type=int, default=150_000)
    ap.add_argument("--batches", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    for p in (str(REPO), str(REPO / "bita"), str(REPO / "scripts"), str(HERE)):
        sys.path.insert(0, p)
    import torch
    from synth_store import make_store
    from data_unification.host_major import batched_loader

    torch.manual_seed(0)
    dev = "cuda"
    st = make_store(a.rows, a.hosts, n_windows=600_000, seed=7)
    st.use_hazard_target(10.0)
    kw = dict(num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    if a.model == "a":
        import retrain_branch_a_live as T
        from branch_a_gnn_lstm.sequence_dataset import LazyHostSequenceDataset
        from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
        ds = LazyHostSequenceDataset(st, seq_len=15, min_trajectory_len=1)
        ds.enable_batched()
        dl = batched_loader(ds, 128, False, pack=True, **kw)
        m = MultiTaskLSTM(input_dim=27, num_techniques=14, num_gradations=4,
                          risk_objective="soft_bce", **MultiTaskLSTM.PAPER_ARCH).to(dev)

        def run():
            return T._evaluate(m, _Cap(dl, a.batches), dev, num_techniques=14,
                               risk_positive_above=0.36)
    elif a.model == "b":
        import retrain_future_models_live as T
        from branch_b_world_model.train_branch_b import LazyHostRolloutDataset
        from branch_b_world_model.rollout_encoder_decoder import HostWorldDynamicsTransformer
        from branch_b_world_model.infiltration_head import InfiltrationRiskHead
        ds = LazyHostRolloutDataset(st, T=15, K=5)
        ds.enable_batched()
        dl = batched_loader(ds, 128, False, pack=True, **kw)
        wdt = HostWorldDynamicsTransformer(d_latent=27, d_model=64, n_heads=4, n_layers=3).to(dev)
        risk = InfiltrationRiskHead(d_latent=27, hidden_dim=32).to(dev)

        def run():
            return T.validate_branch_b(wdt, risk, _Cap(dl, a.batches), dev, 5)
    else:
        import retrain_future_models_live as T
        from deepop_decoder.joint_vocab import get_joint_vocab
        from deepop_decoder.train_cwa_decoder import LazyCWADataset
        from deepop_decoder.forecast_decoder import DeepOPForecastDecoder
        vocab = get_joint_vocab(network_observable_only=True)
        ds = LazyCWADataset(st, vocab, K=5, T=0, oversample=False)
        ds.enable_batched()
        dl = batched_loader(ds, 64, False, pack=True, **kw)
        dec = DeepOPForecastDecoder(d_latent=27, d_model=72, vocab_size=vocab.vocab_size, n_heads=6,
                                    num_layers=2, window_sizes=[2, 4, 8], dim_feedforward=144).to(dev)

        def run():
            return T.validate_deepop(dec, None, _Cap(dl, a.batches), dev, vocab)

    run()                                   # warm-up (workers, cuDNN, graphs)
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = run()
    torch.cuda.synchronize()
    el = time.perf_counter() - t
    print(f"RESULT val {a.model}: {a.batches / el:.1f} batch/s ({1e3 * el / a.batches:.2f} ms/batch) "
          f"check={_digest(out)}", flush=True)


class _Cap:
    """The first n batches of a loader (iterated afresh each time)."""

    def __init__(self, dl, n):
        self.dl, self.n = dl, n

    def __iter__(self):
        for i, b in enumerate(self.dl):
            if i >= self.n:
                return
            yield b

    def __len__(self):
        return self.n


def _digest(out):
    import numpy as np
    import torch
    vals = []
    items = out.items() if isinstance(out, dict) else enumerate(out if isinstance(out, (list, tuple)) else [out])
    for _k, v in items:
        if torch.is_tensor(v):
            vals.append(float(v.double().sum()))
        elif isinstance(v, np.ndarray) and v.dtype.kind in "fiu":
            vals.append(float(v.astype(np.float64).sum()))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append(float(v))
    return f"{sum(vals):.10g}"


if __name__ == "__main__":
    main()
