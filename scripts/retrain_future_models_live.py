"""Retrain Branch B and DeepOP on the canonical live TGNE latent space."""

import argparse
import gc
import math
import shutil
import tempfile
import time
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from branch_a_gnn_lstm.train_branch_a import build_or_load_tgne_ta
from branch_b_world_model.infiltration_head import InfiltrationRiskHead
from branch_b_world_model.rollout_encoder_decoder import HostWorldDynamicsTransformer
from branch_b_world_model.train_branch_b import (
    HostRolloutDataset, LazyHostRolloutDataset, create_rollout_samples,
)
from data_unification.attack_windows import derive_windows
from data_unification.cic2018_adapter import CIC2018Adapter
from data_unification.ctu13_adapter import CTU13Adapter
from data_unification.multi_dataset_stream import HostTrajectoryExtractor
from data_unification.pcap_bridge import iter_day_records
from data_unification.density import require_full_density
from data_unification.split_policy import partition_paths, split_of


def _pcap_day_split(day_dir) -> str:
    """Frozen split of one PCAP capture day, from splits.lock.json.

    Day directory names are the lock's PCAP2018 capture keys verbatim, including
    the two that are misspelled in the corpus itself (fri_23_pacap,
    thu_15_pacap). A day the lock does not name is an error rather than a guess:
    silently defaulting it to train is how an evaluation day leaks.
    """
    from pathlib import Path as _P
    name = _P(day_dir).name
    try:
        return split_of("PCAP2018", name)
    except KeyError:
        raise KeyError(
            f"PCAP day {name!r} is not in the frozen split. Add it via "
            f"scripts/freeze_splits.py rather than assigning it on the fly."
        )
from deepop_decoder.forecast_decoder import DeepOPForecastDecoder
from deepop_decoder.joint_vocab import get_joint_vocab
from deepop_decoder.forecast_decoder import DeepOPTokenScorer, smoothed_and_plain_ce
from deepop_decoder.train_cwa_decoder import (
    CWASequenceDataset, LazyCWADataset, create_cwa_training_samples,
)
from cyberworld_v4.config import get_contract


def _strided(gen, stride: int, want=None):
    """Samples every Nth record across a wider read instead of a plain file-prefix
    (see scripts/retrain_branch_a_live.py::_strided for why -- D6, row-prefix sampling)."""
    out = []
    for i, r in enumerate(gen):
        if i % stride == 0:
            out.append(r)
            # want is None at FULL DENSITY -- keep everything. Comparing an
            # int to None raises, and defaulting it to 0 would silently
            # return an empty list, which is worse.
            if want is not None and len(out) >= want:
                break
    return out


def read_one_capture(path, cic, ctu, rows_per_file, stride=1):
    """Parse a single capture file. Kept separate so a caller can stream."""
    _cap = None if rows_per_file is None else rows_per_file * stride
    gen = (cic.parse_file(str(path), max_rows=_cap)
           if path.suffix == ".csv"
           else ctu.parse_netflow_csv(str(path), max_rows=_cap))
    return _strided(gen, stride, rows_per_file)


def split_capture_files(cic_dir, ctu_dir):
    """The frozen split's train/val capture paths."""
    files = sorted(cic_dir.glob("*.csv")) + sorted(ctu_dir.glob("*/*.binetflow"))
    part = partition_paths(files)
    return part["train"], part["val"]


def load_records(cic_dir, ctu_dir, rows_per_file, stride=1):
    records = []
    cic = CIC2018Adapter()
    ctu = CTU13Adapter()
    files = sorted(cic_dir.glob("*.csv")) + sorted(ctu_dir.glob("*/*.binetflow"))
    # The frozen lock decides this, not an 80/20 slice of a sorted list. The
    # previous version put whichever captures sorted last into validation, so
    # the assignment moved whenever a file was added -- and it disagreed with
    # splits.lock.json, which every other stage is now bound to.
    part = partition_paths(files)

    def _read(paths):
        out = []
        for path in paths:
            # None at full density; must propagate as "no cap", not become 0.
            _cap = None if rows_per_file is None else rows_per_file * stride
            gen = (cic.parse_file(str(path), max_rows=_cap)
                   if path.suffix == ".csv"
                   else ctu.parse_netflow_csv(str(path), max_rows=_cap))
            out.extend(_strided(gen, stride, rows_per_file))
        return out

    return _read(part["train"]), _read(part["val"])


def load_pcap_records(pcap_root, csv_label_dir, window_seconds, max_windows_per_day=None, max_packets_per_host=None, window_stride=1):
    """Real per-host host-trajectory data (fixes D1: CIC-2018 CSV fabricates host IPs by row-index
    for 9/10 days -- see claude_latest_analysis/08_why_the_csv_path_cannot_benchmark.md).

    pcap_root holds one subdirectory per capture day, named "<day>_pcap" (e.g. tue_20_pcap),
    matching the corpus layout under ~/Documents/SIH/DATA/pcap/. csv_label_dir holds the paired
    CIC-2018 CSVs named "<day>_csv.csv" -- PCAP packets carry no label of their own, so each day's
    attack windows are derived from its CSV via attack_windows.derive_windows() and applied by
    timestamp. An 80/20 split is taken by day (not by window), so train/val never share a day.
    """
    # Two of the ten real capture-day directories are misspelled "_pacap" rather
    # than "_pcap" in the corpus itself (verified: fri_23_pacap, thu_15_pacap) --
    # match both so those days aren't silently dropped.
    day_re = re.compile(r"^(?P<day>.+)_pa?cap$")
    day_dirs = sorted((p, day_re.match(p.name)) for p in Path(pcap_root).iterdir() if p.is_dir())
    day_dirs = sorted((p, m.group("day")) for p, m in day_dirs if m)
    # Frozen per-day assignment, not an 80/20 slice (see splits.lock.json).
    train_days = [d for d in day_dirs if _pcap_day_split(d) == "train"]
    val_days = [d for d in day_dirs if _pcap_day_split(d) == "val"]

    def _load(days):
        records = []
        for day_dir, day in days:
            csv_path = Path(csv_label_dir) / f"{day}_csv.csv"
            if not csv_path.exists():
                # The PCAP and CSV corpora name one day differently ("wed_28" vs
                # "wed_29" -- verified, not a typo in this code). Fall back to
                # matching on weekday prefix before giving up on the day.
                prefix = day.rsplit("_", 1)[0]
                candidates = sorted(Path(csv_label_dir).glob(f"{prefix}_*_csv.csv"))
                if candidates:
                    csv_path = candidates[0]
                else:
                    print(f"skipping {day_dir.name}: no matching label CSV at {csv_path}")
                    continue
            dw = derive_windows(str(csv_path))
            if not dw.ok:
                print(f"skipping {day_dir.name}: implausible attack-window derivation ({dw.evidence})")
                continue
            n_windows = 0
            # window_stride keeps every Nth window across the WHOLE day rather than
            # truncating to a prefix. A prefix would drop late-starting campaigns
            # entirely -- wed_14's FTP-BruteForce begins ~2h18m into the capture --
            # which is the same time-coverage bias as D6 on the CSV path. Striding
            # cuts peak memory proportionally while keeping the full time range.
            for w_idx, (_start, _end, window_records) in enumerate(iter_day_records(
                day_dir, dw, scenario_id=day, window_seconds=window_seconds,
                max_packets_per_host=max_packets_per_host,
            )):
                if window_stride > 1 and (w_idx % window_stride) != 0:
                    continue
                records.extend(window_records)
                n_windows += 1
                if max_windows_per_day is not None and n_windows >= max_windows_per_day:
                    break
            print(f"{day_dir.name}: {n_windows} windows, {len(records)} cumulative records")
        return records

    return _load(train_days), _load(val_days)


def iter_pcap_day_records(pcap_root, csv_label_dir, window_seconds, max_windows_per_day=None,
                          max_packets_per_host=None, window_stride=1):
    """Yields (split, day_name, records) one capture day at a time.

    load_pcap_records() accumulates every day's records before returning, which
    at full density is ~9.9M UnifiedFlowRecord objects held at once. Yielding
    per day lets the caller extract and free each day, so peak memory is one
    day rather than the corpus -- the same reason Branch A now loads per file.
    """
    day_re = re.compile(r"^(?P<day>.+)_pa?cap$")
    day_dirs = sorted((p, day_re.match(p.name)) for p in Path(pcap_root).iterdir() if p.is_dir())
    day_dirs = sorted((p, m.group("day")) for p, m in day_dirs if m)
    # Frozen per-day assignment; see splits.lock.json / _pcap_day_split.

    for i, (day_dir, day) in enumerate(day_dirs):
        split = _pcap_day_split(day_dir)
        csv_path = Path(csv_label_dir) / f"{day}_csv.csv"
        if not csv_path.exists():
            prefix = day.rsplit("_", 1)[0]
            cands = sorted(Path(csv_label_dir).glob(f"{prefix}_*_csv.csv"))
            if not cands:
                print(f"skipping {day_dir.name}: no label CSV", flush=True)
                continue
            csv_path = cands[0]
        dw = derive_windows(str(csv_path))
        if not dw.ok:
            print(f"skipping {day_dir.name}: implausible windows ({dw.evidence})", flush=True)
            continue
        recs = []
        n_win = 0
        for w_idx, (_s, _e, wr) in enumerate(iter_day_records(
                day_dir, dw, scenario_id=day, window_seconds=window_seconds,
                max_packets_per_host=max_packets_per_host)):
            if window_stride > 1 and (w_idx % window_stride) != 0:
                continue
            recs.extend(wr)
            n_win += 1
            if max_windows_per_day is not None and n_win >= max_windows_per_day:
                break
        print(f"  [{split}] {day_dir.name}: {n_win} windows, {len(recs)} records", flush=True)
        yield split, day, recs


def _loader_kwargs(device, num_workers: int):
    """DataLoader settings that keep the GPU fed.

    Both lazy datasets trade memory for per-item work (38 GiB -> 280 MB for
    Branch B, 52 GiB -> 340 MB for DeepOP). That trade only pays if the work
    overlaps GPU compute. With the PyTorch default of num_workers=0 it does
    not: profiling Branch A's full-density run put 35.7% of wall clock in the
    dataset/collate path, all of it in the main process.
    """
    kw = dict(num_workers=num_workers, pin_memory=(str(device) == "cuda"))
    if num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return kw


def train_branch_b_live(train_traj, val_traj, output, epochs, device, num_workers: int = 4, patience: int = 2):
    _c = get_contract()
    # Lazy: create_rollout_samples materialises h_history [15,12],
    # h_future [5,12] and risk_future [5] per sample -- 1,164 bytes each, and
    # ~38 GiB at the 35M samples full corpus density produces. The store
    # already holds every embedding in one memmapped block.
    train_ds = LazyHostRolloutDataset(train_traj, T=_c.history_steps, K=_c.forecast_steps)
    val_ds = LazyHostRolloutDataset(val_traj, T=_c.history_steps, K=_c.forecast_steps)
    print(f"Branch B samples: train={len(train_ds)} val={len(val_ds)}", flush=True)
    _lk = _loader_kwargs(device, num_workers)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, **_lk)
    val_loader = DataLoader(val_ds, batch_size=128, **_lk)
    wdt = HostWorldDynamicsTransformer(d_latent=12, d_model=64, n_heads=4, n_layers=3).to(device)
    risk = InfiltrationRiskHead(d_latent=12, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(list(wdt.parameters()) + list(risk.parameters()), lr=1e-3, weight_decay=1e-4)
    best = float("inf")
    best_epoch, _since_improve = 0, 0
    _history = []
    best_state = {k: v.detach().clone() for k, v in wdt.state_dict().items()}
    _nb_total = len(train_loader)
    _nblk = (str(device) == "cuda")
    for epoch in range(epochs):
        wdt.train(); risk.train()
        # Accumulated on device and read once per epoch: a `.item()` per batch
        # forces a host-device sync that stalls the prefetch queue. The train
        # loss was previously not tracked at all here, so an epoch reported
        # nothing until validation finished.
        _tr_sum = torch.zeros((), device=device, dtype=torch.float64); _nb = 0
        _t0 = time.time()
        for batch in train_loader:
            h = batch["h_history"].to(device, non_blocking=_nblk)
            target = batch["h_future"].to(device, non_blocking=_nblk)
            target_risk = batch["risk_future"].to(device, non_blocking=_nblk)
            optimizer.zero_grad(set_to_none=True)
            pred = wdt.rollout(h, K=_c.forecast_steps)
            pred_risk, _ = risk.forward_trajectory(pred)
            loss = sum((0.9 ** k) * F.mse_loss(pred[:, k], target[:, k]) for k in range(_c.forecast_steps))
            loss = loss + F.binary_cross_entropy(pred_risk, target_risk)
            loss.backward(); optimizer.step()
            _tr_sum += loss.detach().double().sum(); _nb += 1
            if _nb % 2000 == 0:
                _el = time.time() - _t0; _r = _nb / max(_el, 1e-9)
                print(f"  Branch B epoch={epoch+1} batch={_nb}/{_nb_total} "
                      f"({100.0*_nb/max(_nb_total,1):.1f}%) {_r:.1f} batch/s "
                      f"eta={(_nb_total-_nb)/max(_r,1e-9)/60:.1f}m", flush=True)
        wdt.eval(); risk.eval()
        # Validation, measured against baselines that cost nothing to beat.
        #
        # A world model that predicts the next 5 host embeddings has an obvious
        # null hypothesis: copy the last observed embedding forward ("nothing
        # changes in 10 seconds"). On 2-second windows that is a strong
        # baseline, and a loss value on its own cannot say whether the model
        # beats it. Branch B's first run moved train loss 1.9% across 6 epochs
        # with its best validation at epoch 1 -- consistent either with a model
        # that converged instantly or with one that never learned anything, and
        # nothing reported could tell those apart.
        _v_sum = torch.zeros((), device=device, dtype=torch.float64); _vn = 0
        _mse_model = torch.zeros((), device=device, dtype=torch.float64)
        _mse_persist = torch.zeros((), device=device, dtype=torch.float64)
        _bce_model = torch.zeros((), device=device, dtype=torch.float64)
        _risk_mae_model = torch.zeros((), device=device, dtype=torch.float64)
        _risk_sum = torch.zeros((), device=device, dtype=torch.float64)
        _risk_n = 0
        with torch.no_grad():
            for batch in val_loader:
                h=batch["h_history"].to(device, non_blocking=_nblk)
                target=batch["h_future"].to(device, non_blocking=_nblk)
                target_risk=batch["risk_future"].to(device, non_blocking=_nblk)
                pred=wdt.rollout(h,K=_c.forecast_steps); pred_risk,_=risk.forward_trajectory(pred)
                _v_sum += (F.mse_loss(pred,target)+F.binary_cross_entropy(pred_risk,target_risk)).double().sum()
                _vn += 1
                # persistence: repeat the last observed step across the horizon
                _last = h[:, -1:, :].expand(-1, target.shape[1], -1)
                _mse_model += F.mse_loss(pred, target).double()
                _mse_persist += F.mse_loss(_last, target).double()
                _bce_model += F.binary_cross_entropy(pred_risk, target_risk).double()
                _risk_mae_model += (pred_risk - target_risk).abs().double().mean()
                _risk_sum += target_risk.double().mean()
                _risk_n += 1
        score=float((_v_sum/max(_vn,1)).item())
        _trl=float((_tr_sum/max(_nb,1)).item())
        _n = max(_risk_n, 1)
        _mm, _mp = float((_mse_model/_n).item()), float((_mse_persist/_n).item())
        _bm = float((_bce_model/_n).item())
        _rmae = float((_risk_mae_model/_n).item())
        _rbar = float((_risk_sum/_n).item())
        # BCE of predicting the empirical mean risk for everything, and the MAE
        # of predicting 0.0 -- the two trivial risk predictors.
        _p = min(max(_rbar, 1e-7), 1 - 1e-7)
        _bce_base = -(_rbar * math.log(_p) + (1 - _rbar) * math.log(1 - _p))
        _skill = (1.0 - _mm / _mp) if _mp > 0 else float("nan")
        print(f"Branch B epoch={epoch+1} train_loss={_trl:.4f} val_loss={score:.4f} "
              f"wall={(time.time()-_t0)/60:.1f}m", flush=True)
        print(f"  embeddings: mse_model={_mm:.6f} mse_persistence={_mp:.6f} "
              f"skill={_skill:+.3f}"
              f"{'  <-- WORSE THAN COPYING THE LAST STEP' if _mm >= _mp else ''}", flush=True)
        print(f"  risk: bce_model={_bm:.4f} bce_constant={_bce_base:.4f} "
              f"mae_model={_rmae:.4f} mae_predict_zero={_rbar:.4f}"
              f"{'  <-- WORSE THAN PREDICTING ZERO' if _rmae >= _rbar else ''}", flush=True)
        _history.append({"epoch": epoch + 1, "train_loss": _trl, "val_loss": score,
                         "mse_model": _mm, "mse_persistence": _mp, "skill": _skill,
                         "bce_model": _bm, "bce_constant": _bce_base,
                         "risk_mae_model": _rmae, "risk_mae_zero": _rbar})
        if score < best:
            best=score; best_epoch=epoch+1; _since_improve=0
            output.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"wdt_state_dict":wdt.state_dict(),"risk_head_state_dict":risk.state_dict(),"epoch":epoch+1,"history_steps":_c.history_steps,"forecast_steps":_c.forecast_steps,"window_seconds":_c.window_seconds,"epoch_history":list(_history),"baselines":{"mse_persistence":_mp,"risk_mae_zero":_rbar}},output)
            best_state={k:v.detach().clone() for k,v in wdt.state_dict().items()}
        else:
            _since_improve += 1
            # The best weights are already saved, so stopping here cannot cost
            # quality -- it only stops spending hours on epochs that do not
            # improve validation. Run 1 went 6 epochs and its best was epoch 1.
            if patience and _since_improve >= patience:
                print(f"Branch B: early stop at epoch {epoch+1}; no improvement "
                      f"in {_since_improve} epochs, best was epoch {best_epoch} "
                      f"(val_loss {best:.4f})", flush=True)
                break
    wdt.load_state_dict(best_state)
    wdt.eval()
    return wdt


class _WithRollout(torch.utils.data.Dataset):
    """Serves a precomputed Branch-B rollout alongside each sample.

    Wrapping rather than changing LazyCWADataset keeps the cache out of the
    dataset's own contract, and the lookup happens in the DataLoader workers
    rather than the main process.
    """

    def __init__(self, base, cache):
        self.base, self.cache = base, cache

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        d = self.base[i]
        d["h_rollout"] = torch.from_numpy(np.ascontiguousarray(self.cache[i]))
        return d


def _precompute_rollouts(wdt, ds, device, spill_dir, label, K, batch=1024, num_workers=3):
    """Run the frozen Branch-B rollout once and memmap the result.

    DeepOP conditions every batch on `wdt.rollout(h_history)`. The WDT is
    frozen and in eval mode there, so that call is a pure function of
    h_history: profiling the 2026-09-21 run put **25.4% of DeepOP's wall
    clock** in a rollout whose output is identical on every epoch. Verified
    bit-identical across repeated calls and invariant to batch order (max abs
    diff exactly 0.0), and *not* identical in train mode, where dropout is
    live -- which is why this asserts eval below rather than trusting it.

    Computing it once costs one inference pass; a 6-epoch run pays for it six
    times. The pass itself has no gradients, so its batch size carries no
    optimisation semantics and can be wide.
    """
    if wdt.training:
        raise RuntimeError(
            "refusing to cache rollouts from a WDT in train mode: dropout is "
            "live, so the rollout is not a function of its input and every "
            "epoch would otherwise see a different conditioning signal")

    n = len(ds)
    path = os.path.join(spill_dir or tempfile.gettempdir(),
                        f"rollout_{label}_{os.getpid()}.f32")
    cache = np.memmap(path, dtype=np.float32, mode="w+", shape=(n, K, 12))
    try:
        os.unlink(path)          # reclaimed when the mapping is dropped
    except OSError:
        pass

    loader = DataLoader(ds, batch_size=batch, shuffle=False,
                        num_workers=num_workers, pin_memory=(str(device) == "cuda"))
    t0, done = time.time(), 0
    with torch.no_grad():
        for b in loader:
            h = b["h_history"].to(device, non_blocking=(str(device) == "cuda"))
            out = wdt.rollout(h, K=K).detach().float().cpu().numpy()
            cache[done:done + len(out)] = out
            done += len(out)
            if done % (batch * 200) == 0:
                r = done / max(time.time() - t0, 1e-9)
                print(f"  rollout cache [{label}] {done:,}/{n:,} "
                      f"({100.0*done/n:.1f}%) {r:,.0f} samples/s "
                      f"eta={(n-done)/max(r,1e-9)/60:.1f}m", flush=True)
    cache.flush()
    print(f"  rollout cache [{label}]: {n:,} samples in "
          f"{(time.time()-t0)/60:.1f}m ({cache.nbytes/2**30:.2f} GiB, unlinked)",
          flush=True)
    return cache


def train_deepop_live(train_traj, val_traj, output, epochs, device, wdt=None, num_workers: int = 4, patience: int = 2):
    vocab=get_joint_vocab(network_observable_only=True)
    _c=get_contract()
    # T>0 makes samples carry h_history so we can condition on Branch B's own
    # rollout instead of the oracle future (audit E1). At serve time DeepOP
    # only ever sees WDT output; training it on ground truth is a train/serve
    # mismatch that noise augmentation only approximates.
    _T=_c.history_steps if wdt is not None else 0
    # Lazy: create_cwa_training_samples materialises h_future, h_history and
    # two token arrays per sample (~1,250 B), one per snapshot plus a
    # duplicate per attack window -- ~52 GiB at full corpus density.
    train_ds=LazyCWADataset(train_traj,vocab,K=_c.forecast_steps,T=_T)
    # Validation is NOT oversampled. LazyCWADataset duplicates every window
    # containing a non-Benign token, which is a training-time class-balance
    # device; applying it to the eval split inflates the attack rate there
    # (measured Benign share 0.8571 -> 0.7692) so every DeepOP validation
    # number reported before this was read against a distribution that does
    # not exist. It moved the printed lift by +0.024 on its own.
    val_ds=LazyCWADataset(val_traj,vocab,K=_c.forecast_steps,T=_T,oversample=False)
    print(f"DeepOP samples: train={len(train_ds)} val={len(val_ds)}",flush=True)
    print(f"DeepOP conditioning: {'Branch-B rollouts (E1 fixed)' if wdt is not None else 'oracle + noise (interim)'}",flush=True)
    # Precompute the frozen rollout once instead of recomputing it every epoch.
    if wdt is not None and _T > 0:
        _sp = os.environ.get("CYBERWORLD_SPILL_DIR")
        train_ds = _WithRollout(train_ds, _precompute_rollouts(
            wdt, train_ds, device, _sp, "train", _c.forecast_steps,
            num_workers=max(1, num_workers)))
        val_ds = _WithRollout(val_ds, _precompute_rollouts(
            wdt, val_ds, device, _sp, "val", _c.forecast_steps,
            num_workers=max(1, num_workers)))

    _lk=_loader_kwargs(device,num_workers)
    train_loader=DataLoader(train_ds,batch_size=64,shuffle=True,**_lk)
    val_loader=DataLoader(val_ds,batch_size=64,**_lk)
    decoder=DeepOPForecastDecoder(d_latent=12,d_model=72,vocab_size=vocab.vocab_size,n_heads=6,num_layers=2,window_sizes=[2,4,8],dim_feedforward=144).to(device)
    optimizer=torch.optim.AdamW(decoder.parameters(),lr=5e-4,weight_decay=1e-4)
    best=float("inf"); best_epoch=0; _since_improve=0; _history=[]
    _nb_total=len(train_loader); _nblk=(str(device)=="cuda")
    for epoch in range(epochs):
        decoder.train()
        # On-device accumulation: `loss.item()` per batch synced the host to
        # the GPU on every step and defeated the worker prefetch queue.
        _tr_sum=torch.zeros((),device=device,dtype=torch.float64)
        _tr_plain=torch.zeros((),device=device,dtype=torch.float64); _nb=0
        _t0=time.time()
        for batch in train_loader:
            h=batch["h_future"].to(device,non_blocking=_nblk)
            inp=batch["input_tokens"].to(device,non_blocking=_nblk)
            tgt=batch["target_tokens"].to(device,non_blocking=_nblk)
            if "h_rollout" in batch:
                h_aug=batch["h_rollout"].to(device,non_blocking=_nblk)
            elif wdt is not None and "h_history" in batch:
                # Condition on what Branch B actually predicts, which is what
                # DeepOP receives in production.
                with torch.no_grad():
                    h_aug=wdt.rollout(batch["h_history"].to(device,non_blocking=_nblk), K=h.shape[1]).detach()
            else:
                step_sigma=torch.linspace(0.015,0.055,steps=h.shape[1],device=device).unsqueeze(0).unsqueeze(-1)
                h_aug=h+torch.randn_like(h)*step_sigma
            optimizer.zero_grad(set_to_none=True); logits=decoder(h_aug,inp)
            # Train used smoothed CE while validation used plain CE, so the two
            # printed numbers were different functions and their gap was not a
            # generalisation gap. Identity: L_smooth = 0.96*plain + 0.04*U with
            # U >= ln(10), which put the real degradation at >= 0.2087 nats
            # against a printed 0.1404. Both are now reported.
            loss, _plain = smoothed_and_plain_ce(logits, tgt, label_smoothing=0.04)
            loss.backward(); torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.0); optimizer.step()
            _tr_sum+=loss.detach().double().sum(); _tr_plain+=_plain.detach().double().sum(); _nb+=1
            if _nb % 2000 == 0:
                _el=time.time()-_t0; _r=_nb/max(_el,1e-9)
                print(f"  DeepOP epoch={epoch+1} batch={_nb}/{_nb_total} "
                      f"({100.0*_nb/max(_nb_total,1):.1f}%) {_r:.1f} batch/s "
                      f"eta={(_nb_total-_nb)/max(_r,1e-9)/60:.1f}m",flush=True)
        decoder.eval()
        # Validation against the two predictors that require no model at all.
        #
        # DeepOP emits a sequence of (category, technique) tokens. The corpus
        # is ~90% Benign and a host's label rarely changes inside a 10-second
        # horizon, so both "repeat the last observed token" and "always emit
        # the most common token" score well. Cross-entropy alone cannot show
        # that, exactly as Branch A's 0.88 accuracy could not show a head
        # sitting on the class prior. Token accuracy, macro F1 over the tokens
        # actually present, and both baselines are reported together.
        V = vocab.vocab_size
        _v_sum=torch.zeros((),device=device,dtype=torch.float64); _vn=0
        _hit_model = torch.zeros((), device=device, dtype=torch.long)
        _hit_persist = torch.zeros((), device=device, dtype=torch.long)
        _tok_total = 0
        _conf = torch.zeros(V * V, device=device, dtype=torch.long)
        try:
            _scorer = DeepOPTokenScorer(V, device=device)
        except Exception as _e:
            print(f"  token scorer unavailable: {_e}", flush=True); _scorer = None
        _tgt_hist = torch.zeros(V, device=device, dtype=torch.long)
        with torch.no_grad():
            for batch in val_loader:
                hv=batch["h_future"].to(device,non_blocking=_nblk)
                if "h_rollout" in batch:
                    hv=batch["h_rollout"].to(device,non_blocking=_nblk)
                elif wdt is not None and "h_history" in batch:
                    hv=wdt.rollout(batch["h_history"].to(device,non_blocking=_nblk), K=hv.shape[1]).detach()
                tgt=batch["target_tokens"].to(device,non_blocking=_nblk)
                logits=decoder(hv,batch["input_tokens"].to(device,non_blocking=_nblk))
                _v_sum+=F.cross_entropy(logits.reshape(-1,V),tgt.reshape(-1)).double().sum()
                _vn+=1
                pred = logits.argmax(dim=-1)
                if _scorer is not None:
                    _obs = batch.get("obs_token")
                    if _obs is not None:
                        _free, _ = decoder.forecast_sequence(
                            hv, max_steps=hv.shape[1],
                            observed_token=_obs.to(device, non_blocking=_nblk),
                            continuity_bonus=0.0)
                        _scorer.update(tgt, batch["input_tokens"].to(device, non_blocking=_nblk),
                                       _obs.to(device, non_blocking=_nblk),
                                       pred_tf=pred, pred_free=_free)
                flat_p, flat_t = pred.reshape(-1), tgt.reshape(-1)
                _hit_model += (flat_p == flat_t).sum()
                _tok_total += int(flat_t.numel())
                _conf += torch.bincount(flat_t * V + flat_p, minlength=V * V)
                _tgt_hist += torch.bincount(flat_t, minlength=V)
                # persistence: the last token actually observed, repeated
                obs = batch.get("obs_token")
                if obs is not None:
                    obs = obs.to(device, non_blocking=_nblk)
                    _hit_persist += (obs.unsqueeze(1).expand_as(tgt) == tgt).sum()
        score=float((_v_sum/max(_vn,1)).item())
        _tt = max(_tok_total, 1)
        acc = float(_hit_model.item()) / _tt
        acc_persist = float(_hit_persist.item()) / _tt
        _hist_np = _tgt_hist.cpu().numpy()
        acc_majority = float(_hist_np.max()) / _tt if _hist_np.sum() else 0.0
        cm = _conf.reshape(V, V).cpu().numpy()
        _sup, _pred_n, _tp = cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
        with np.errstate(divide="ignore", invalid="ignore"):
            _pr = np.where(_pred_n > 0, _tp / np.maximum(_pred_n, 1), 0.0)
            _rc = np.where(_sup > 0, _tp / np.maximum(_sup, 1), 0.0)
            _dn = _pr + _rc
            _f1 = np.where(_dn > 0, 2 * _pr * _rc / np.maximum(_dn, 1e-12), 0.0)
        _present = _sup > 0
        macro_f1 = float(_f1[_present].mean()) if _present.any() else 0.0
        _best_base = max(acc_persist, acc_majority)
        print(f"DeepOP epoch={epoch+1} train_loss={float((_tr_sum/max(_nb,1)).item()):.4f} "
              f"val_loss={score:.4f} wall={(time.time()-_t0)/60:.1f}m",flush=True)
        print(f"  tokens: acc={acc:.4f} macro_f1={macro_f1:.4f} "
              f"| baselines persistence={acc_persist:.4f} majority={acc_majority:.4f} "
              f"| lift={acc - _best_base:+.4f} "
              f"| classes_pred={int((_pred_n > 0).sum())}/{int(_present.sum())}"
              f"{'  <-- NO BETTER THAN A CONSTANT' if acc <= _best_base else ''}", flush=True)
        # The line above compares a TEACHER-FORCED model (handed y_{s-1} at
        # every step) against a FREE-RUNNING baseline (handed y_{-1} and held
        # for 5 steps). Those see different information, so the comparison
        # flatters the model: a zero-parameter model that echoes its input
        # token scores +0.0914 by it. Deployment is free-running
        # (model_adapter calls forecast_sequence), so the free-running row is
        # the one that describes what is actually served.
        if _scorer is not None:
            print(DeepOPTokenScorer.format(_scorer.result()), flush=True)
        _plain_tr = float((_tr_plain/max(_nb,1)).item())
        print(f"  train CE plain={_plain_tr:.4f} (smoothed={float((_tr_sum/max(_nb,1)).item()):.4f}) "
              f"vs val CE {score:.4f} -> comparable gap {score - _plain_tr:+.4f}", flush=True)
        _history.append({"epoch": epoch + 1, "val_loss": score, "token_acc": acc,
                         "macro_f1": macro_f1, "acc_persistence": acc_persist,
                         "acc_majority": acc_majority, "lift": acc - _best_base})
        if score < best:
            best=score; best_epoch=epoch+1; _since_improve=0
            output.parent.mkdir(parents=True,exist_ok=True); torch.save({"decoder_state_dict":decoder.state_dict(),"epoch":epoch+1,"history_steps":_c.history_steps,"forecast_steps":_c.forecast_steps,"window_seconds":_c.window_seconds,"vocab_size":vocab.vocab_size,"epoch_history":list(_history),"baselines":{"acc_persistence":acc_persist,"acc_majority":acc_majority}},output)
        else:
            _since_improve += 1
            # Best weights are already on disk; stopping cannot cost quality.
            if patience and _since_improve >= patience:
                print(f"DeepOP: early stop at epoch {epoch+1}; no improvement in "
                      f"{_since_improve} epochs, best was epoch {best_epoch} "
                      f"(val_loss {best:.4f})", flush=True)
                break


def _pcap_trajectories_per_day(args, extractor):
    """Per-day extraction into two shared stores -- see iter_pcap_day_records.

    Peak memory is one capture day rather than the whole corpus (~9.9M records
    held at once otherwise), the same reason Branch A now loads per file.
    """
    import gc, time
    from data_unification.trajectory_store import TrajectoryStoreBuilder
    spill = str(args.spill_dir) if args.spill_dir else None
    # All three splits get a store. The frozen lock assigns thu_1_pcap to test,
    # and without a store for it the day would either crash on a missing key or
    # -- worse, if defaulted -- be folded into training. It is extracted and
    # kept separate so a held-out PCAP evaluation is possible, and it is never
    # returned to the trainers.
    builders = {k: TrajectoryStoreBuilder(spill_dir=spill)
                for k in ("train", "val", "test")}
    wbase = {"train": 0, "val": 0, "test": 0}
    for split, day, recs in iter_pcap_day_records(
            args.pcap_root, args.cic2018_csv_dir, get_contract().window_seconds,
            args.pcap_max_windows_per_day, window_stride=args.pcap_window_stride):
        b = builders[split]
        t = time.time()
        extractor.extract_trajectories(recs, builder=b, window_idx_base=wbase[split])
        if b._window_idx.n:
            wbase[split] = int(b._window_idx.buf[: b._window_idx.n].max()) + 1
        print(f"    -> {split} store: {b._n} snapshots ({time.time()-t:.1f}s)", flush=True)
        del recs
        gc.collect()
    out = {}
    for k, b in builders.items():
        st = b.finalize()
        print(f"{k}: {st.n_snapshots} snapshots over {len(st)} hosts | {st.memory_report()}", flush=True)
        out[k] = st
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cic-dir",type=Path); parser.add_argument("--ctu-dir",type=Path)
    parser.add_argument("--pcap-root",type=Path,help="Directory of <day>_pcap subdirs; real per-host trajectories")
    parser.add_argument("--cic2018-csv-dir",type=Path,help="Directory of <day>_csv.csv label files, required with --pcap-root")
    parser.add_argument("--pcap-max-windows-per-day",type=int,default=None)
    parser.add_argument("--pcap-window-stride",type=int,default=1,help="Keep every Nth window across the full day")
    parser.add_argument("--tgne",type=Path,required=True); parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--rows-per-file",type=int,default=None,
                        help="Cap records kept per capture. Default None = FULL DENSITY.")
    parser.add_argument("--stride",type=int,default=1)
    parser.add_argument("--spill-dir",type=Path,default=None,help="Write the bulk trajectory feature block here instead of RAM (np.memmap)")
    parser.add_argument("--epochs",type=int,default=3)
    parser.add_argument("--risk-target",choices=("severity","hazard"),default="severity",
                        help="Which risk target to train against. 'severity' is "
                             "base_severity(tactic) for an attack window and 0.0 "
                             "otherwise -- bimodal, not a forecast, and close to a "
                             "function of the category another head predicts; "
                             "Branch B's risk head was worse than predicting zero "
                             "against it in every epoch. 'hazard' is "
                             "exp(-seconds_to_next_attack / horizon): continuous, "
                             "forward-looking, tactic-independent. Default stays "
                             "severity so the change is measured, not silent.")
    parser.add_argument("--stages",choices=("both","branch_b","deepop"),default="both",
                        help="Which models to train. DeepOP trains on Branch B's "
                             "rollouts and cannot be better than it, so "
                             "'branch_b' lets you check Branch B against its "
                             "persistence baseline before committing hours to "
                             "DeepOP. 'deepop' loads the Branch B already on disk.")
    parser.add_argument("--patience",type=int,default=2,
                        help="Stop a model after N epochs without a validation "
                             "improvement. The best checkpoint is written every "
                             "time it improves, so this cannot cost quality -- it "
                             "only stops paying for epochs that do nothing. Run 1 "
                             "took 6 Branch B epochs and its best was epoch 1. "
                             "0 disables.")
    parser.add_argument("--num-workers",type=int,default=4,
                        help="DataLoader worker processes; 0 loads in the main "
                             "process and serialises loading with GPU compute.")
    args=parser.parse_args(); random.seed(42); np.random.seed(42); torch.manual_seed(42)
    import time
    t0=time.time()

    tgn=build_or_load_tgne_ta(checkpoint_path=str(args.tgne))
    extractor=HostTrajectoryExtractor(
        tgne_ta_model=tgn, window_size_sec=get_contract().window_seconds,
        spill_dir=str(args.spill_dir) if args.spill_dir else None)

    require_full_density(
        'Branch B + DeepOP retrain',
        stride=args.stride,
        rows_per_file=args.rows_per_file,
        pcap_window_stride=args.pcap_window_stride,
        pcap_max_windows_per_day=args.pcap_max_windows_per_day,
    )

    if args.pcap_root:
        if not args.cic2018_csv_dir:
            parser.error("--pcap-root requires --cic2018-csv-dir (PCAP packets carry no label of their own)")
        _stores = _pcap_trajectories_per_day(args, extractor)
        train_traj, val_traj = _stores["train"], _stores["val"]
        # _stores["test"] is the frozen held-out capture day. It is deliberately
        # not handed to the trainers; score it once, after the model is frozen.
        print(f"held-out test store: {_stores['test'].n_snapshots} snapshots "
              f"(not used for training or model selection)", flush=True)
    else:
        if not (args.cic_dir and args.ctu_dir):
            parser.error("either --pcap-root/--cic2018-csv-dir or --cic-dir/--ctu-dir is required")
        # Stream one capture at a time.
        #
        # This used to call load_records(), which returned every record of a
        # split as one Python list, and only then built the trajectory store on
        # top of it -- so the 32,461,461-record list (~9.7 GiB at ~300 B each)
        # and the store were resident together. Measured on the 2026-09-21
        # run: the job sat at memory.current == memory.max == 17 GiB, was
        # throttled 983,985 times, and spent 57.7% of its wall clock fully
        # stalled in reclaim. It was not going to finish.
        #
        # Branch A already loads per capture for exactly this reason; that is
        # what this mirrors. Peak becomes the largest single capture instead of
        # a whole split, and the feature block goes to the spill memmap.
        #
        # Building the TGNE neighbour graph per capture is also more faithful:
        # a CIC-2018 day and a CTU-13 scenario are unrelated networks, and a
        # merged graph would make their hosts each other's temporal neighbours.
        _train_files, _val_files = split_capture_files(args.cic_dir, args.ctu_dir)
        for _n, _f in (("train", _train_files), ("val", _val_files)):
            if not _f:
                parser.error(f"frozen split '{_n}' matched no capture files under "
                             f"{args.cic_dir} / {args.ctu_dir}")
        print(f"frozen split: {len(_train_files)} train / {len(_val_files)} val captures",
              flush=True)

        from data_unification.trajectory_store import TrajectoryStoreBuilder
        _cic, _ctu = CIC2018Adapter(), CTU13Adapter()
        _spill = str(args.spill_dir) if args.spill_dir else None

        def _store_per_capture(files, label):
            shared = TrajectoryStoreBuilder(spill_dir=_spill)
            widx_base, total = 0, 0
            for i, f in enumerate(files):
                t = time.time()
                recs = read_one_capture(f, _cic, _ctu, args.rows_per_file, args.stride)
                total += len(recs)
                extractor.extract_trajectories(recs, builder=shared,
                                               window_idx_base=widx_base)
                if shared._window_idx.n:
                    widx_base = int(shared._window_idx.buf[: shared._window_idx.n].max()) + 1
                print(f"  [{label} {i+1}/{len(files)}] {f.name}: {len(recs)} recs, "
                      f"store={shared._n} snaps, {time.time()-t:.1f}s", flush=True)
                del recs
                gc.collect()
            store = shared.finalize()
            print(f"{label}: {total} records -> {store.n_snapshots} snapshots", flush=True)
            return store

        t0=time.time(); train_traj = _store_per_capture(_train_files, "train")
        print(f"train trajectories extracted in {time.time()-t0:.1f}s ({len(train_traj)} hosts)", flush=True)
        t0=time.time(); val_traj = _store_per_capture(_val_files, "val")
        print(f"val trajectories extracted in {time.time()-t0:.1f}s ({len(val_traj)} hosts)", flush=True)

    # Gate the data before spending hours training on it. The v4 trainer has
    # had this since three runs reporting PR-AUC 0.9998 were discarded; the
    # three-branch trainers never adopted it.
    try:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from credibility_check import evaluate_store, gate as _gate, report as _report
        _stats = evaluate_store(train_traj, val_traj)
        _problems = _gate(_stats)
        _report(_stats, _problems)
    except Exception as _e:
        print(f"credibility check skipped: {_e}", flush=True)

    device="cuda" if torch.cuda.is_available() else "cpu"
    if args.spill_dir:
        os.environ["CYBERWORLD_SPILL_DIR"] = str(args.spill_dir)

    # Write where the adapter actually loads from.
    #
    # These used to be written as "<out_dir>/host_wdt.canonical-tgne.pt" and
    # "<out_dir>/cwa_forecast_decoder.canonical-tgne.pt". Nothing reads either
    # name -- control_backend/model_adapter.py loads
    # saved_models/branch_b/host_wdt.pt and
    # saved_models/deepop/cwa_forecast_decoder.pt. So a full downstream retrain
    # could succeed and leave the served models untouched: the v3 checkpoints
    # would stay in place, the adapter would keep refusing to compose them with
    # a v4 Branch A, and the parity check would stay blocked -- with nothing
    # anywhere reporting a failure. Branch A's live script already writes
    # straight to its served path; these now match it.
    #
    # The existing checkpoint is copied aside first, once, before training
    # starts. Both trainers save on every improving epoch, so a backup taken
    # any later would capture a partly-retrained model rather than the model
    # being replaced.
    bb_out = args.out_dir / "branch_b" / "host_wdt.pt"
    dp_out = args.out_dir / "deepop" / "cwa_forecast_decoder.pt"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for _p in (bb_out, dp_out):
        _p.parent.mkdir(parents=True, exist_ok=True)
        if _p.exists():
            _bak = _p.with_name(f"{_p.stem}.superseded-{stamp}{_p.suffix}")
            shutil.copy2(_p, _bak)
            print(f"backed up {_p} -> {_bak}", flush=True)

    # Optionally replace the risk target before any dataset is built.
    #
    # Branch B's risk head was worse than predicting zero in all five epochs
    # of the 2026-09-22 run -- MAE 0.2412-0.2548 against 0.1401 -- the same
    # defect Branch A's had, from the same cause. `risk_score` as written
    # during extraction is base_severity(tactic) for an attack window and
    # exactly 0.0 otherwise: bimodal, not a forecast, and close to a function
    # of the coarse category. The hazard form is continuous and forward
    # looking. Both stores are swapped together, or the train and validation
    # targets would be on different scales.
    if args.risk_target == "hazard":
        _tau = _c.forecast_steps * _c.window_seconds
        for _nm, _st in (("train", train_traj), ("val", val_traj)):
            _info = _st.use_hazard_target(_tau)
            print(f"risk target [{_nm}]: severity -> hazard(tau={_tau}s) | "
                  f"zeros {_info['zero_fraction_before']:.3f} -> "
                  f"{_info['zero_fraction_after']:.3f} | "
                  f"distinct {_info['distinct_before']} -> {_info['distinct_after']} | "
                  f"mean {_info['mean_before']:.4f} -> {_info['mean_after']:.4f}",
                  flush=True)

    # DeepOP is trained on Branch B's rollouts, so it cannot be better than
    # Branch B: if the world model does not beat persistence, DeepOP is
    # conditioning on a near-constant signal and can only learn the label
    # prior. Splitting the stages means Branch B can be validated against its
    # baseline first, and DeepOP run only once that is worth doing.
    wdt = None
    if args.stages in ("both", "branch_b"):
        wdt = train_branch_b_live(train_traj,val_traj,bb_out,args.epochs,device,
                                  num_workers=args.num_workers,patience=args.patience)
        print(f"served checkpoint updated: {bb_out}", flush=True)

    if args.stages in ("both", "deepop"):
        if wdt is None:
            # deepop-only: condition on the Branch B already on disk rather
            # than retraining it. eval() matters -- the rollout has dropout,
            # and a WDT left in train mode would feed DeepOP a different
            # (randomly perturbed) conditioning signal every epoch.
            if not bb_out.exists():
                parser.error(f"--stages deepop needs a trained Branch B at {bb_out}")
            _bb = torch.load(bb_out, map_location=device, weights_only=False)
            wdt = HostWorldDynamicsTransformer(d_latent=12, d_model=64, n_heads=4,
                                               n_layers=3).to(device)
            wdt.load_state_dict(_bb["wdt_state_dict"])
            wdt.eval()
            print(f"loaded Branch B from {bb_out} (epoch {_bb.get('epoch')}) for "
                  f"DeepOP conditioning", flush=True)
        train_deepop_live(train_traj,val_traj,dp_out,args.epochs,device,wdt=wdt,
                          num_workers=args.num_workers,patience=args.patience)
        print(f"served checkpoint updated: {dp_out}", flush=True)


if __name__ == "__main__":
    main()
