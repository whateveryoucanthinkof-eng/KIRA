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
from data_unification.split_policy import is_cross_year, partition_paths, split_of
from cyberworld_v4.training_guard import (IMPROVED, STOP, ResumePoint, TrainingGuard,
                                          default_warmup_steps, run_fingerprint)


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
from cyberworld_v4.device_hist import device_hist
from cyberworld_v4.graphed_step import GraphedBody, GraphedLoss, WholeStepGraph, graphs_enabled
from data_unification.host_major import batched_loader, unpack_batch


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
        from data_unification.training_sources import _pcap_label_csv, check_label_day
        for day_dir, day in days:
            # Shared resolver: exact name, then the wed_28 -> wed_29 alias. The
            # weekday-prefix fallback that used to live here labelled wed_28
            # from wed_14's CSV (see training_sources.PCAP_LABEL_ALIASES).
            csv_path = _pcap_label_csv(day_dir, csv_label_dir)
            if csv_path is None:
                print(f"skipping {day_dir.name}: no matching label CSV under {csv_label_dir}")
                continue
            dw = derive_windows(str(csv_path))
            if not dw.ok:
                print(f"skipping {day_dir.name}: implausible attack-window derivation ({dw.evidence})")
                continue
            check_label_day(day, dw.intervals)
            # Scope intervals the CSV could not (no IP columns) from the participant
            # map. This had no caller, so a filled-in map changed nothing.
            from data_unification.attack_participants import apply_participants
            apply_participants(dw, day)
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
                          max_packets_per_host=None, window_stride=1, scheme="frozen"):
    """Yields (split, day_name, records) one capture day at a time.

    load_pcap_records() accumulates every day's records before returning, which
    at full density is ~9.9M UnifiedFlowRecord objects held at once. Yielding
    per day lets the caller extract and free each day, so peak memory is one
    day rather than the corpus -- the same reason Branch A now loads per file.
    Discovery and reading are data_unification/training_sources.py, shared with
    Branch A and the encoder.
    """
    from data_unification.training_sources import discover_captures, iter_pcap_day_windows

    caps = discover_captures(scheme=scheme, pcap2018_root=pcap_root)
    for cap in sorted((c for sp in caps.values() for c in sp), key=lambda c: c.name):
        recs, n_win = [], 0
        for window in iter_pcap_day_windows(cap.path, csv_label_dir, window_seconds,
                                            max_windows_per_day, window_stride,
                                            max_packets_per_host):
            recs.extend(window)
            n_win += 1
        # UNKNOWN rows (unattributable attack intervals, unmapped labels) are
        # not "benign": keep them out, as read_capture does for Branch A.
        from data_unification.label_filter import drop_unresolved
        recs, _cov = drop_unresolved(recs)
        print(f"  [{cap.split}] {cap.name}: {n_win} windows, {len(recs)} records", flush=True)
        yield cap.split, cap.name, recs


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


#: Set by main() from --legacy-loader: per-sample __getitem__ + default collate.
LEGACY_LOADER = False


def _make_loader(ds, batch_size, shuffle, device, num_workers):
    """The loader policy for Branch B, DeepOP and the rollout precompute.

    Batched by default: each batch is one vectorised `gather_batch` over a
    host-major copy of the feature block (data_unification/host_major.py)
    instead of `batch_size` per-sample gathers of scattered memmap rows. Same
    batches, same order, same RNG draws as DataLoader(ds, shuffle=...):
    tests/test_branch_b_deepop_batched_loader.py.
    """
    lk = _loader_kwargs(device, num_workers)
    base = getattr(ds, "base", ds)
    if LEGACY_LOADER or not hasattr(ds, "gather_batch"):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **lk)
    if not getattr(base, "_batched_ready", False):
        t = time.time()
        base.enable_batched(spill_dir=os.environ.get("CYBERWORLD_SPILL_DIR"))
        print(f"  batched loader ready in {time.time() - t:.1f}s (host-major features: "
              f"{getattr(base, '_feats_hm', None) is not None})", flush=True)
    # one buffer per batch (host_major.pack_batch); the loops unpack_batch it
    return batched_loader(ds, batch_size, shuffle, pack=True, **lk)


def _guard_step_fn(guard):
    """The guard's sync-free step when it can be used (TrainingGuard.
    backward_step_deferred: same decisions and weights bit for bit), else the
    syncing one. CYBERWORLD_GUARD_SYNC=1 forces the latter."""
    return guard.backward_step_deferred if guard.deferred_supported() else guard.backward_step


def _accumulate_ok(ok, acc_pairs, nb):
    """Add each (accumulator, value) for a step the guard took; count it in `nb`.

    `ok` is a bool (synchronous step) or a device bool (deferred step). For a
    skipped step this adds exactly 0.0 and 0, as the old `continue` did."""
    if ok is True:
        for acc, v in acc_pairs:
            acc += v
        nb += 1
    else:
        for acc, v in acc_pairs:
            acc += torch.where(ok, v, torch.zeros((), device=v.device, dtype=v.dtype))
        nb += ok.to(nb.dtype)


# --------------------------------------------------------------- validation
#
# Each validation batch is a forward pass plus in-place updates of a few
# device accumulators. The bodies below are exactly what the training loops
# ran inline; they are replayed as CUDA graphs (cyberworld_v4/graphed_step.
# GraphedBody) because a validation batch, like a training step, is many tiny
# kernels whose launch cost exceeded their GPU time. Host-side counters (batch
# and token counts) are kept outside the bodies: a graph replays device work
# only. Same kernels, same order: identical accumulators.

def _masked_mse(pred, target, valid):
    """MSE over the real (non-padded) forecast steps. pred/target [B, K, D] or
    [B, D] with valid [B, K] or [B]."""
    v = valid.unsqueeze(-1).to(pred.dtype)
    return ((pred - target) ** 2 * v).sum() / (v.sum() * pred.shape[-1]).clamp_min(1.0)


def _future_valid(batch, K, device, nblk):
    """The batch's [B, K] real-step mask (all ones for a loader that has none)."""
    v = batch.get("future_valid")
    if v is None:
        return torch.ones(batch["h_future"].shape[0], K, device=device)
    return v.to(device, non_blocking=nblk)


def _bb_variance_loss(pred, logvar, target, valid=None):
    """Gaussian NLL of the step-k residual under the predicted variance.

    The mean is detached (and logvar_head reads a detached trunk), so this
    trains ONLY the variance: the mean -- the forecast the persistence gate
    judges -- trains exactly as it did under MSE alone."""
    from branch_b_world_model.rollout_encoder_decoder import gaussian_nll
    if valid is None:
        return sum((0.9 ** k) * gaussian_nll(pred[:, k].detach(), logvar[:, k], target[:, k])
                   for k in range(pred.shape[1]))
    out = 0.0
    for k in range(pred.shape[1]):
        lv, mu = logvar[:, k], pred[:, k].detach()
        el = 0.5 * (lv + (target[:, k] - mu) ** 2 * torch.exp(-lv) + 1.8378770664093453)
        v = valid[:, k].unsqueeze(-1).to(el.dtype)
        out = out + (0.9 ** k) * (el * v).sum() / (v.sum() * el.shape[-1]).clamp_min(1.0)
    return out


#: z for a two-sided 90% Gaussian interval: coverage of mean +/- 1.645 sigma.
_Z90 = 1.6448536269514722


def _bb_val_body(wdt, risk, K, step_offset, positive_above=0.0):
    def body(acc, h, target, target_risk, t_hist, t_fut, valid):
        pred, logv = wdt.rollout(h, K=K, t_history=t_hist, t_future=t_fut, return_logvar=True)
        _v = valid.unsqueeze(-1).double()
        _el = 0.5 * (logv + (target - pred) ** 2 * torch.exp(-logv) + 1.8378770664093453)
        acc["nll_model"] += (_el.double() * _v).sum() / (_v.sum() * _el.shape[-1]).clamp_min(1.0)
        _inside = ((target - pred).abs() <= _Z90 * torch.exp(0.5 * logv)).double()
        acc["cov90_by_step"] += (_inside * _v).sum(dim=(0, 2)) / (
            _v.sum(dim=(0, 2)) * _inside.shape[-1]).clamp_min(1.0)
        pred_risk, _ = risk.forward_trajectory(pred)
        _rb = ((pred_risk - target_risk).abs().clamp(0, 1) * (FORECAST_RISK_BINS - 1)).long()
        acc["resid_hist"] += device_hist((_rb.view(-1, K) + step_offset).reshape(-1),
                                         K * FORECAST_RISK_BINS, valid.reshape(-1).long())
        _sb = (pred_risk.clamp(0, 1) * (FORECAST_RISK_BINS - 1)).long().reshape(-1)
        _pos = (target_risk > positive_above).reshape(-1)
        _vl = valid.reshape(-1) > 0
        acc["op_pos"] += device_hist(_sb, FORECAST_RISK_BINS, (_pos & _vl).long())
        acc["op_neg"] += device_hist(_sb, FORECAST_RISK_BINS, (~_pos & _vl).long())
        # Every metric over REAL future steps only: padded steps repeat the
        # last state, which persistence predicts perfectly by construction.
        _w = valid.double()
        _nw = _w.sum().clamp_min(1.0)
        _mm = _masked_mse(pred, target, valid).double()
        acc["v_sum"] += _mm + risk.risk_loss(pred_risk, target_risk, weight=valid).double()
        # persistence: repeat the last observed step across the horizon
        _last = h[:, -1:, :].expand(-1, target.shape[1], -1)
        acc["mse_model"] += _mm
        acc["mse_persist"] += _masked_mse(_last, target, valid).double()
        acc["bce_model"] += (F.binary_cross_entropy(pred_risk, target_risk, reduction="none").double()
                             * _w).sum() / _nw
        acc["risk_mae_model"] += ((pred_risk - target_risk).abs().double() * _w).sum() / _nw
        acc["risk_sum"] += (target_risk.double() * _w).sum() / _nw
    return body


def fit_forecast_operating_point(pos_hist, neg_hist) -> dict:
    """Branch B's own alert threshold: max F1 over its forecast risk scores on
    validation (real future steps; positive = the same event Branch A's
    threshold is fitted to). Serving used Branch A's threshold for these
    scores, though the two heads are calibrated independently."""
    ph = np.asarray(pos_hist, dtype=np.float64)
    nh = np.asarray(neg_hist, dtype=np.float64)
    P, N = ph.sum(), nh.sum()
    if P <= 0 or N <= 0:
        return {"fitted": False, "reason": f"validation has {int(P)} positive and {int(N)} "
                                           f"negative forecast steps; a threshold needs both"}
    tp = np.cumsum(ph[::-1])[::-1]
    fp = np.cumsum(nh[::-1])[::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        rec = tp / P
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    rate = (tp + fp) / (P + N)
    usable = (rate > 0) & (rate < 1)
    if not usable.any():
        return {"fitted": False, "reason": "the forecast risk head emits a constant"}
    b = int(np.argmax(np.where(usable, f1, -1.0)))
    bins = len(ph)
    return {"fitted": True, "alert_threshold": b / (bins - 1), "precision": float(prec[b]),
            "recall": float(rec[b]), "f1": float(f1[b]),
            "fpr": float(fp[b] / N), "criterion": "max F1, validation, real steps"}


def validate_branch_b(wdt, risk, loader, device, K, positive_above=0.0):
    """Branch B's validation pass: the accumulators the epoch summary reads."""
    z = lambda: torch.zeros((), device=device, dtype=torch.float64)
    acc = {"v_sum": z(), "mse_model": z(), "mse_persist": z(), "bce_model": z(),
           "risk_mae_model": z(), "risk_sum": z(),
           # the predictive distribution: mean NLL, and per-step coverage of the
           # nominal 90% interval (calibrated <=> ~0.90 at every step)
           "nll_model": z(), "cov90_by_step": torch.zeros(K, device=device, dtype=torch.float64),
           # forecast risk scores of positive / negative real steps, for the
           # head's OWN operating point (fit_forecast_operating_point)
           "op_pos": torch.zeros(FORECAST_RISK_BINS, device=device, dtype=torch.long),
           "op_neg": torch.zeros(FORECAST_RISK_BINS, device=device, dtype=torch.long),
           # per-step |risk residual| histograms for the forecast's conformal band
           "resid_hist": torch.zeros(K * FORECAST_RISK_BINS, device=device, dtype=torch.long)}
    step_offset = (torch.arange(K, device=device) * FORECAST_RISK_BINS).view(1, K)
    run = GraphedBody(_bb_val_body(wdt, risk, K, step_offset, positive_above), acc,
                      enabled=(str(device) == "cuda" and graphs_enabled()))
    nblk = (str(device) == "cuda")
    vn = risk_n = 0
    with torch.no_grad():
        for batch in loader:
            batch = unpack_batch(batch, device)
            h = batch["h_history"].to(device, non_blocking=nblk)
            target = batch["h_future"].to(device, non_blocking=nblk)
            target_risk = batch["risk_future"].to(device, non_blocking=nblk)
            t_hist = batch["t_history"].to(device, non_blocking=nblk) if "t_history" in batch else None
            t_fut = batch["t_future"].to(device, non_blocking=nblk) if "t_future" in batch else None
            run(h, target, target_risk, t_hist, t_fut, _future_valid(batch, K, device, nblk))
            vn += 1
            risk_n += 1
    return {**acc, "vn": vn, "risk_n": risk_n, "graphed_batches": run.n_graphed}


class _ScorerView:
    """A DeepOPTokenScorer's accumulators taken from a dict, so a graph body
    can update either the scorer's own tensors or warm-up copies of them."""

    def __init__(self, V, acc):
        self.V = V
        for k in _SCORER_KEYS:
            setattr(self, k, acc["sc" + k])


_SCORER_KEYS = ("_tgt_hist", "_hit_tf", "_conf_tf", "_hit_free", "_conf_free",
                "_hit_persist_free", "_hit_persist_fed")


def _dp_val_body(decoder, V, with_scorer):
    def body(acc, hv, inp, tgt, obs_seq, obs):
        logits = decoder(hv, inp, obs_tokens=obs_seq)
        acc["v_sum"] += F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1)).double().sum()
        pred = logits.argmax(dim=-1)
        if with_scorer:
            _free, _ = decoder.forecast_sequence(
                hv, max_steps=hv.shape[1], observed_token=obs, observed_sequence=obs_seq,
                continuity_bonus=0.0, decode_names=False)
            DeepOPTokenScorer.update_device(_ScorerView(V, acc), tgt, inp, obs,
                                            pred_tf=pred, pred_free=_free)
        flat_p, flat_t = pred.reshape(-1), tgt.reshape(-1)
        acc["hit_model"] += (flat_p == flat_t).sum()
        acc["conf"] += device_hist(flat_t * V + flat_p, V * V)
        acc["tgt_hist"] += device_hist(flat_t, V)
        # persistence: the last token actually observed, repeated
        acc["hit_persist"] += (obs.unsqueeze(1).expand_as(tgt) == tgt).sum()
    return body


def validate_deepop(decoder, wdt, loader, device, vocab):
    """DeepOP's validation pass: the accumulators the epoch summary reads."""
    V = vocab.vocab_size
    nblk = (str(device) == "cuda")
    try:
        scorer = DeepOPTokenScorer(V, device=device)
    except Exception as _e:
        print(f"  token scorer unavailable: {_e}", flush=True); scorer = None
    acc = {"v_sum": torch.zeros((), device=device, dtype=torch.float64),
           "hit_model": torch.zeros((), device=device, dtype=torch.long),
           "hit_persist": torch.zeros((), device=device, dtype=torch.long),
           "conf": torch.zeros(V * V, device=device, dtype=torch.long),
           "tgt_hist": torch.zeros(V, device=device, dtype=torch.long)}
    if scorer is not None:
        acc.update({"sc" + k: getattr(scorer, k) for k in _SCORER_KEYS})
    run = GraphedBody(_dp_val_body(decoder, V, scorer is not None), acc,
                      enabled=(str(device) == "cuda" and graphs_enabled()))
    vn = tok_total = 0
    with torch.no_grad():
        for batch in loader:
            batch = unpack_batch(batch, device)
            hv = batch["h_future"].to(device, non_blocking=nblk)
            if "h_rollout" in batch:
                hv = batch["h_rollout"].to(device, non_blocking=nblk)
            elif wdt is not None and "h_history" in batch:
                _th = batch["t_history"].to(device, non_blocking=nblk) if "t_history" in batch else None
                _tf = batch["t_future"].to(device, non_blocking=nblk) if "t_future" in batch else None
                hv = wdt.rollout(batch["h_history"].to(device, non_blocking=nblk), K=hv.shape[1],
                                 t_history=_th, t_future=_tf).detach()
            tgt = batch["target_tokens"].to(device, non_blocking=nblk)
            obs_seq = batch["obs_tokens"].to(device, non_blocking=nblk)
            inp = batch["input_tokens"].to(device, non_blocking=nblk)
            obs = batch.get("obs_token")
            if obs is None:
                # never produced by LazyCWADataset; kept for hand-built batches
                _eager_dp_val_no_obs(decoder, V, acc, hv, inp, tgt, obs_seq)
            else:
                obs = obs.to(device, non_blocking=nblk)
                run(hv, inp, tgt, obs_seq, obs)
                if scorer is not None:
                    scorer.update_host(tgt, has_free=True)
            vn += 1
            tok_total += int(tgt.numel())
    return {**acc, "vn": vn, "tok_total": tok_total, "scorer": scorer,
            "graphed_batches": run.n_graphed}


def _eager_dp_val_no_obs(decoder, V, acc, hv, inp, tgt, obs_seq):
    logits = decoder(hv, inp, obs_tokens=obs_seq)
    acc["v_sum"] += F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1)).double().sum()
    pred = logits.argmax(dim=-1)
    flat_p, flat_t = pred.reshape(-1), tgt.reshape(-1)
    acc["hit_model"] += (flat_p == flat_t).sum()
    acc["conf"] += device_hist(flat_t * V + flat_p, V * V)
    acc["tgt_hist"] += device_hist(flat_t, V)


def train_branch_b_live(train_traj, val_traj, output, epochs, device, num_workers: int = 4, patience: int = 3, risk_target: str = "severity",
                        lr: float = 1e-3, step_back_after: int = 2, clip_norm: float = 1.0,
                        resume: "ResumePoint | None" = None):
    _c = get_contract()
    # Lazy: create_rollout_samples materialises h_history [15,12],
    # h_future [5,12] and risk_future [5] per sample -- 1,164 bytes each, and
    # ~38 GiB at the 35M samples full corpus density produces. The store
    # already holds every embedding in one memmapped block.
    # min_history_steps=1: serving scores a host from its first window, so the
    # world model must learn short (edge-padded) histories too.
    train_ds = LazyHostRolloutDataset(train_traj, T=_c.history_steps, K=_c.forecast_steps,
                                      min_history_steps=1)
    val_ds = LazyHostRolloutDataset(val_traj, T=_c.history_steps, K=_c.forecast_steps,
                                    min_history_steps=1)
    print(f"Branch B samples: train={len(train_ds)} val={len(val_ds)}", flush=True)
    _lk = _loader_kwargs(device, num_workers)
    train_loader = _make_loader(train_ds, 128, True, device, num_workers)
    val_loader = _make_loader(val_ds, 128, False, device, num_workers)
    # World state = the full enriched tensor (TGNE latent + host attributes),
    # the same s(t) Branch A reads. Its width is the store's.
    d_state = int(train_traj.feats.shape[1])
    wdt = HostWorldDynamicsTransformer(d_latent=d_state, d_model=64, n_heads=4, n_layers=3).to(device)
    risk = InfiltrationRiskHead(d_latent=d_state, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(list(wdt.parameters()) + list(risk.parameters()), lr=lr, weight_decay=1e-4)
    best = float("inf")
    best_epoch = 0
    _history = []
    best_state = {k: v.detach().clone() for k, v in wdt.state_dict().items()}
    _nb_total = len(train_loader)
    _nblk = (str(device) == "cuda")
    # Shared policy (cyberworld_v4/training_guard.py): warmup, clipping,
    # non-finite steps skipped, step back to the best weights at half the LR
    # after `step_back_after` flat epochs, stop at `patience`.
    guard = TrainingGuard("branch_b", [wdt, risk], optimizer, mode="min",
                          patience=patience, step_back_after=step_back_after,
                          warmup_steps=default_warmup_steps(_nb_total),
                          clip_norm=clip_norm or None,
                          log=lambda m: print(m, flush=True),
                          graph_undo=(str(device) == "cuda" and graphs_enabled()))
    first_epoch = 0
    _rp = resume.load() if resume is not None else None
    if _rp is not None:
        wdt.load_state_dict(_rp["wdt"]); risk.load_state_dict(_rp["risk"])
        optimizer.load_state_dict(_rp["optimizer"]); guard.load_state_dict(_rp["guard"])
        best, best_epoch, best_state = _rp["best"], _rp["best_epoch"], _rp["best_state"]
        _history[:] = _rp["history"]
        first_epoch = epochs if guard.should_stop() else _rp["done_epochs"]
    _graphed = None
    for epoch in range(first_epoch, epochs):
        wdt.train(); risk.train()
        # Accumulated on device and read once per epoch: a `.item()` per batch
        # forces a host-device sync that stalls the prefetch queue. The train
        # loss was previously not tracked at all here, so an epoch reported
        # nothing until validation finished.
        _tr_sum = torch.zeros((), device=device, dtype=torch.float64); _nb = 0
        _nb_dev = torch.zeros((), device=device, dtype=torch.long); _k = 0
        _step = _guard_step_fn(guard)
        if _graphed is None:
            def _bb_loss(h_, target_, target_risk_, t_hist_, t_fut_, valid_):
                pred_, logv_ = wdt.rollout(h_, K=_c.forecast_steps, t_history=t_hist_,
                                           t_future=t_fut_, return_logvar=True)
                pred_risk_, _ = risk.forward_trajectory(pred_)
                loss_ = sum((0.9 ** k) * _masked_mse(pred_[:, k], target_[:, k], valid_[:, k])
                            for k in range(_c.forecast_steps))
                return (loss_ + _bb_variance_loss(pred_, logv_, target_, valid_)
                        + risk.risk_loss(pred_risk_, target_risk_, weight=valid_))
            _graphed = GraphedLoss(_bb_loss, [wdt, risk],
                                   enabled=(str(device) == "cuda" and graphs_enabled()))
        _t0 = time.time()
        for batch in train_loader:
            _k += 1
            batch = unpack_batch(batch, device)
            h = batch["h_history"].to(device, non_blocking=_nblk)
            target = batch["h_future"].to(device, non_blocking=_nblk)
            target_risk = batch["risk_future"].to(device, non_blocking=_nblk)
            valid = _future_valid(batch, _c.forecast_steps, device, _nblk)
            # Real elapsed times. LazyHostRolloutDataset has emitted these all
            # along and rollout() encodes them, but this -- the trainer that
            # produces the SERVED checkpoint -- never passed them, so the
            # model was told every step was 2 s apart when the measured
            # median is 14 s and 64% of CTU-13 steps are over 10 s. The
            # standalone trainer's comment said it outright: "Passing these
            # two tensors is the whole change it needs."
            t_hist = batch["t_history"].to(device, non_blocking=_nblk) if "t_history" in batch else None
            t_fut = batch["t_future"].to(device, non_blocking=_nblk) if "t_future" in batch else None
            optimizer.zero_grad(set_to_none=True)
            if t_hist is not None and t_fut is not None:
                # forward + loss + backward replayed as a CUDA graph (cyberworld_v4/
                # graphed_step.py): identical losses and weights, a fraction of the
                # launch overhead. Same computation as the eager lines below.
                loss = _graphed(h, target, target_risk, t_hist, t_fut, valid)
                _ok = _step(loss)
                if _ok is False:
                    continue
                _accumulate_ok(_ok, [(_tr_sum, loss.detach().double().sum())], _nb_dev)
                if _k % 2000 == 0:
                    _el = time.time() - _t0; _r = _k / max(_el, 1e-9)
                    print(f"  Branch B epoch={epoch+1} batch={_k}/{_nb_total} "
                          f"({100.0*_k/max(_nb_total,1):.1f}%) {_r:.1f} batch/s "
                          f"eta={(_nb_total-_k)/max(_r,1e-9)/60:.1f}m", flush=True)
                continue
            pred, logv = wdt.rollout(h, K=_c.forecast_steps, t_history=t_hist, t_future=t_fut,
                                     return_logvar=True)
            pred_risk, _ = risk.forward_trajectory(pred)
            loss = sum((0.9 ** k) * _masked_mse(pred[:, k], target[:, k], valid[:, k])
                       for k in range(_c.forecast_steps))
            loss = loss + _bb_variance_loss(pred, logv, target, valid)
            # Huber, not BCE.
            #
            # BCE(p,t) is linear in t, so its minimiser is E[t|x] -- the same
            # conditional mean MSE finds. Both are mean-seeking, while the
            # metric this head is judged by (MAE against predict-zero) is
            # median-seeking, which is why the head lost to the constant 0 in
            # all five epochs of the 2026-09-22 run. BCE is also the wrong
            # noise model: it is the likelihood of a Bernoulli coin with bias
            # t, and hazard_risk is a deterministic decay, not a coin flip.
            # Measured on a synthetic hazard target: BCE 0.244, MSE 0.244,
            # Huber(beta=0.1) 0.216 against a zero baseline of 0.230 -- only
            # Huber beats it.
            loss = loss + risk.risk_loss(pred_risk, target_risk, weight=valid)
            _ok = _step(loss)
            if _ok is False:
                continue
            _accumulate_ok(_ok, [(_tr_sum, loss.detach().double().sum())], _nb_dev)
            if _k % 2000 == 0:
                _el = time.time() - _t0; _r = _k / max(_el, 1e-9)
                print(f"  Branch B epoch={epoch+1} batch={_k}/{_nb_total} "
                      f"({100.0*_k/max(_nb_total,1):.1f}%) {_r:.1f} batch/s "
                      f"eta={(_nb_total-_k)/max(_r,1e-9)/60:.1f}m", flush=True)
        guard.flush()
        _nb = int(_nb_dev)
        if _graphed is not None and (_graphed.enabled or _graphed.error):
            print(f"  cuda graph: {_graphed.n_graphed} graphed steps of {_k}"
                  + (f" ({_graphed.error})" if _graphed.error else ""), flush=True)
            _graphed.n_graphed = 0
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
        _K = _c.forecast_steps
        _va = validate_branch_b(wdt, risk, val_loader, device, _K,
                                positive_above=(math.exp(-1.0) - 1e-6 if risk_target == "hazard" else 0.0))
        _v_sum, _vn = _va["v_sum"], _va["vn"]
        _mse_model, _mse_persist = _va["mse_model"], _va["mse_persist"]
        _bce_model, _risk_mae_model = _va["bce_model"], _va["risk_mae_model"]
        _risk_sum, _risk_n, _resid_hist = _va["risk_sum"], _va["risk_n"], _va["resid_hist"]
        print(f"  validation: {_va['graphed_batches']} of {_va['vn']} batches as a cuda graph", flush=True)
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
        _nll = float((_va["nll_model"] / _n).item())
        _cov = [round(float(c), 4) for c in (_va["cov90_by_step"] / _n).tolist()]
        print(f"Branch B epoch={epoch+1} train_loss={_trl:.4f} val_loss={score:.4f} "
              f"wall={(time.time()-_t0)/60:.1f}m", flush=True)
        print(f"  predictive distribution: nll={_nll:.4f} "
              f"90%-interval coverage by step={_cov} (calibrated ~0.90)", flush=True)
        print(f"  embeddings: mse_model={_mm:.6f} mse_persistence={_mp:.6f} "
              f"skill={_skill:+.3f}"
              f"{'  <-- WORSE THAN COPYING THE LAST STEP' if _mm >= _mp else ''}", flush=True)
        print(f"  risk: bce_model={_bm:.4f} bce_constant={_bce_base:.4f} "
              f"mae_model={_rmae:.4f} mae_predict_zero={_rbar:.4f}"
              f"{'  <-- WORSE THAN PREDICTING ZERO' if _rmae >= _rbar else ''}", flush=True)
        _history.append({"epoch": epoch + 1, "train_loss": _trl, "val_loss": score,
                         "nll_model": _nll, "coverage90_by_step": _cov,
                         "mse_model": _mm, "mse_persistence": _mp, "skill": _skill,
                         "bce_model": _bm, "bce_constant": _bce_base,
                         "risk_mae_model": _rmae, "risk_mae_zero": _rbar})
        _bb_health = []
        if not (_mp > 0 and _mm < _mp):
            _bb_health.append(f"world model is not better than copying the last state "
                              f"forward (skill {_skill:+.3f})")
        if _rmae >= _rbar:
            _bb_health.append(f"risk head MAE {_rmae:.4f} is not better than predicting "
                              f"zero ({_rbar:.4f})")
        _action = guard.end_epoch(score, train_loss=_trl, health=_bb_health)
        _history[-1]["guard_action"] = _action
        if _action == IMPROVED:
            best=score; best_epoch=epoch+1
            output.parent.mkdir(parents=True,exist_ok=True)
            # Fitted from THIS epoch's validation residuals, so the band always
            # belongs to the weights saved beside it.
            _conf = _forecast_risk_conformal(_resid_hist.view(_K, FORECAST_RISK_BINS).cpu().numpy())
            _op = fit_forecast_operating_point(_va["op_pos"].cpu().numpy(), _va["op_neg"].cpu().numpy())
            print(f"  forecast risk operating point: {_op}", flush=True)
            for _k, _c_k in enumerate(_conf["by_step"], start=1):
                print(f"  forecast risk band step {_k}: "
                      + (f"+/-{_c_k['half_width']:.4f} (empirical {_c_k['empirical_coverage']:.3f}, "
                         f"n={_c_k['n']:,})" if _c_k["fitted"] else f"NOT FITTED -- {_c_k['reason']}"),
                      flush=True)
            torch.save({"wdt_state_dict":wdt.state_dict(),"risk_head_state_dict":risk.state_dict(),"epoch":epoch+1,"history_steps":_c.history_steps,"forecast_steps":_c.forecast_steps,"window_seconds":_c.window_seconds,"d_state":d_state,"epoch_history":list(_history),"risk_target":risk_target,"baselines":{"mse_persistence":_mp,"risk_mae_zero":_rbar},"forecast_risk_conformal":_conf,"operating_point":_op},output)
            best_state={k:v.detach().clone() for k,v in wdt.state_dict().items()}
        if resume is not None:
            resume.save(epoch + 1, wdt=wdt.state_dict(), risk=risk.state_dict(),
                        optimizer=optimizer.state_dict(), guard=guard.state_dict(), best=best,
                        best_epoch=best_epoch, best_state=best_state, history=list(_history))
        if _action == STOP:
            # The best weights are already saved, so stopping here cannot cost
            # quality -- it only stops spending hours on epochs that do not
            # improve validation. Run 1 went 6 epochs and its best was epoch 1.
            print(f"Branch B: early stop at epoch {epoch+1}; {guard.stop_reason} "
                  f"(best epoch {best_epoch}, val_loss {best:.4f})", flush=True)
            break
    wdt.load_state_dict({k: v.to(next(wdt.parameters()).device) for k, v in best_state.items()})
    wdt.eval()

    # Stamp the verdict into the served checkpoint, where the adapter and the
    # DeepOP stage can both read it, instead of leaving it in a log line.
    cred = branch_b_credibility(_history, best_epoch)
    if output.exists():
        _ck = torch.load(output, map_location="cpu", weights_only=False)
        _ck["credibility"] = cred
        _ck["training_guard"] = guard.summary()
        torch.save(_ck, output)
    print(f"Branch B credibility: {'CREDIBLE' if cred['credible'] else 'NOT CREDIBLE'}"
          + (f" -- {'; '.join(cred['problems'])}" if cred["problems"] else ""), flush=True)
    return wdt


#: Bins for the per-step |forecast risk residual| histograms; one bin is 5e-4.
FORECAST_RISK_BINS = 2000
FORECAST_RISK_ALPHA = 0.05


def _forecast_risk_conformal(step_hists, alpha: float = FORECAST_RISK_ALPHA):
    """Per-step split-conformal half-widths for the served forecast risk.

    The dashboard used to draw each forecast step's band as
    risk +/- 20 * (DeepOP token confidence): not a measurement of anything, and
    backwards -- a more confident token gave a WIDER band. This is the band the
    served risk actually earns: at step k, the ceil((n+1)(1-alpha))-th smallest
    |risk_pred - risk_true| over the validation windows, which covers
    1 - alpha of held-out windows under exchangeability. Wider at later steps
    when the model is less sure there, which is what a band should show.

    Fitted on validation, like Branch A's conformal width: that split also
    picks the epoch, so the band is mildly optimistic. It is marginal over all
    windows, not per class; see cyberworld_v4.conformal.coverage_by_group.
    """
    from cyberworld_v4.conformal import halfwidth_from_histogram
    by_step = [halfwidth_from_histogram(h, alpha) for h in np.asarray(step_hists)]
    return {
        "alpha": float(alpha),
        "fitted_on": "validation",
        "by_step": by_step,
        "half_width_by_step": [c["half_width"] if c["fitted"] else None for c in by_step],
    }


#: Minimum fractional MSE improvement over persistence ("copy the last
#: embedding forward") for Branch B to count as having learned dynamics.
#: Below it, DeepOP would be trained on rollouts that are the last observed
#: state repeated, and could only learn the label prior. Tuning DeepOP cannot
#: fix that, so the pipeline stops instead of spending hours on it.
MIN_BRANCH_B_SKILL = 0.02


def branch_b_credibility(history, best_epoch):
    """Verdict on whether the saved Branch B beats persistence."""
    entry = next((h for h in history if h.get("epoch") == best_epoch), None)
    if entry is None:
        return {"checked": False, "credible": False,
                "problems": ["no validation history for the saved epoch"]}
    skill = float(entry.get("skill", float("nan")))
    problems = []
    if not math.isfinite(skill):
        problems.append("skill vs persistence is undefined (persistence MSE was 0)")
    elif skill < MIN_BRANCH_B_SKILL:
        problems.append(
            f"embedding skill vs persistence is {skill:+.4f} (< {MIN_BRANCH_B_SKILL}); "
            f"the world model is not measurably better than copying the last step")
    if entry.get("risk_mae_model", 0.0) >= entry.get("risk_mae_zero", float("inf")):
        problems.append("risk head is no better than predicting zero")
    return {
        "checked": True,
        "credible": not problems,
        "problems": problems,
        "stats": {"best_epoch": best_epoch, "skill": skill,
                  "mse_model": entry.get("mse_model"),
                  "mse_persistence": entry.get("mse_persistence"),
                  "risk_mae_model": entry.get("risk_mae_model"),
                  "risk_mae_zero": entry.get("risk_mae_zero"),
                  "min_skill": MIN_BRANCH_B_SKILL},
    }


def require_credible_branch_b(ckpt, allow: bool) -> None:
    cred = ckpt.get("credibility") or {}
    if cred.get("checked") and cred.get("credible"):
        return
    why = "; ".join(cred.get("problems") or []) or (
        "the checkpoint carries no skill verdict (it predates the check)")
    msg = (f"Branch B is not credible: {why}. DeepOP trains on Branch B's "
           f"rollouts, so it would be learning from a near-copy of the input. "
           f"Fix Branch B first, or pass --allow-noncredible-branch-b.")
    if not allow:
        raise SystemExit(f"REFUSING TO TRAIN DEEPOP. {msg}")
    print(f"WARNING (overridden): {msg}", flush=True)


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

    def gather_batch(self, indices):
        """`default_collate([self[i] for i in indices])`, from the base's gather_batch."""
        d = self.base.gather_batch(indices)
        d["h_rollout"] = torch.from_numpy(np.ascontiguousarray(
            self.cache[np.asarray(indices, dtype=np.int64)]))
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

    if not (LEGACY_LOADER or not hasattr(ds, "gather_batch") or not hasattr(ds, "_host_idx")):
        return _precompute_rollouts_unique(wdt, ds, device, spill_dir, label, K, batch, num_workers)
    n = len(ds)
    # The rollout has the width of its input: 27 for the world state
    # s(t) = [TGNE ; attributes], not the bare 12-D embedding. A hard-coded 12
    # here crashed every Branch-B-conditioned DeepOP run.
    d = int(ds[0]["h_history"].shape[-1])
    path = os.path.join(spill_dir or tempfile.gettempdir(),
                        f"rollout_{label}_{os.getpid()}.f32")
    cache = np.memmap(path, dtype=np.float32, mode="w+", shape=(n, K, d))
    try:
        os.unlink(path)          # reclaimed when the mapping is dropped
    except OSError:
        pass

    loader = (DataLoader(ds, batch_size=batch, shuffle=False,
                         num_workers=num_workers, pin_memory=(str(device) == "cuda"))
              if LEGACY_LOADER or not hasattr(ds, "gather_batch")
              else _make_loader(ds, batch, False, device, num_workers))
    t0, done = time.time(), 0
    with torch.no_grad():
        for b in loader:
            b = unpack_batch(b, device)
            _nb = (str(device) == "cuda")
            h = b["h_history"].to(device, non_blocking=_nb)
            # Same real elapsed times Branch B is trained with, so the cached
            # rollouts are the ones Branch B actually produces.
            t_h = b["t_history"].to(device, non_blocking=_nb) if "t_history" in b else None
            t_f = b["t_future"].to(device, non_blocking=_nb) if "t_future" in b else None
            out = wdt.rollout(h, K=K, t_history=t_h, t_future=t_f).detach().float().cpu().numpy()
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


class _RolloutCache:
    """Rollout rows plus a per-sample map: sample i's rollout is rows[map[i]]."""

    def __init__(self, rows, idx_map):
        self.rows, self.map = rows, idx_map
        self.nbytes = rows.nbytes + idx_map.nbytes

    def __len__(self):
        return len(self.map)

    def __getitem__(self, i):
        return self.rows[self.map[i]]


class _Indexed:
    """`ds` restricted to the sample indices `idx`, for gather_batch loaders."""

    _batched_ready = True          # delegates to `ds`, prepared by the caller

    def __init__(self, ds, idx):
        self.ds, self.idx = ds, np.asarray(idx, dtype=np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        return self.ds[int(self.idx[j])]

    def gather_batch(self, js):
        return self.ds.gather_batch(self.idx[np.asarray(js, dtype=np.int64)])


def _precompute_rollouts_unique(wdt, ds, device, spill_dir, label, K, batch, num_workers):
    """_precompute_rollouts without computing (or storing) any window twice.

    DeepOP's training set lists every window with an attack token twice (the
    2x oversampling), so the cache computed and stored those rollouts twice:
    at full scale the cache is ~0.5 KB per sample, tens of GB on disk.

    Bit-identical to the original, which ran samples 0..n-1 in batches of
    `batch` and a final partial batch of r = n % batch. Measured on the WDT
    (tests/test_rollout_cache_unique.py): a sample's rollout does not depend
    on WHICH samples share its batch at batch size 1024, but it can depend on
    the batch SIZE (777, 64 and 1 differ by up to 4e-6). So:

      * the full-batch region's distinct windows run in batches of exactly
        `batch` -- the last one padded with repeats, discarded -- as before;
      * the original final partial batch runs as it was: same samples, same
        order, same size;
      * each sample (duplicates included) maps to its row.
    """
    if not getattr(ds, "_batched_ready", False):
        ds.enable_batched(spill_dir=os.environ.get("CYBERWORLD_SPILL_DIR"))
    n = len(ds)
    r = n % batch
    full = n - r
    d = int(ds[0]["h_history"].shape[-1])
    key = (ds._host_idx[:full].astype(np.int64) << 32) | ds._pos[:full].astype(np.int64)
    _u, first, inv = np.unique(key, return_index=True, return_inverse=True)
    del key, _u
    order = np.argsort(first, kind="stable")       # distinct windows in first-occurrence order
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    uniq = first[order]
    nu = len(uniq)
    pad = (-nu) % batch
    todo = np.concatenate([uniq, np.repeat(uniq[-1:], pad)]) if nu else uniq
    path = os.path.join(spill_dir or tempfile.gettempdir(),
                        f"rollout_{label}_{os.getpid()}.f32")
    rows = np.memmap(path, dtype=np.float32, mode="w+", shape=(max(nu + r, 1), K, d))
    try:
        os.unlink(path)          # reclaimed when the mapping is dropped
    except OSError:
        pass
    t0, done = time.time(), 0
    nb = (str(device) == "cuda")

    def run(idx, dst):
        nonlocal done
        if len(idx) == 0:
            return
        put = dst
        for b in _make_loader(_Indexed(ds, idx), batch, False, device, num_workers):
            b = unpack_batch(b, device)
            h = b["h_history"].to(device, non_blocking=nb)
            t_h = b["t_history"].to(device, non_blocking=nb) if "t_history" in b else None
            t_f = b["t_future"].to(device, non_blocking=nb) if "t_future" in b else None
            out = wdt.rollout(h, K=K, t_history=t_h, t_future=t_f).detach().float().cpu().numpy()
            keep = min(len(out), dst + (nu if dst == 0 else r) - put)
            rows[put:put + keep] = out[:keep]
            put += keep
            done += len(out)
            if done % (batch * 200) == 0:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  rollout cache [{label}] {done:,}/{len(todo) + r:,} "
                      f"{rate:,.0f} samples/s", flush=True)

    # The original made ONE DataLoader pass, and creating a DataLoader iterator
    # draws its base seed from torch's global RNG. Two passes here would draw
    # twice and shift every later draw (DeepOP's shuffle order): restore the
    # state afterwards and make the original's single draw.
    rng = torch.get_rng_state()
    with torch.no_grad():
        run(todo, 0)
        run(np.arange(full, n), nu)
    torch.set_rng_state(rng)
    torch.empty((), dtype=torch.int64).random_()
    rows.flush()
    idx_map = np.concatenate([rank[inv], nu + np.arange(r, dtype=np.int64)])
    idx_map = idx_map.astype(np.int32 if nu + r < 2 ** 31 else np.int64)
    print(f"  rollout cache [{label}]: {n:,} samples, {nu + r:,} distinct rollouts computed "
          f"in {(time.time()-t0)/60:.1f}m ({rows.nbytes/2**30:.2f} GiB instead of "
          f"{n * K * d * 4 / 2**30:.2f}, unlinked)", flush=True)
    return _RolloutCache(rows, idx_map)


#: Probability of dropping each observed technique from DeepOP's encoder input
#: during training. At serve time that sequence is Branch A's predictions, not
#: labels, so some entries will be wrong or missing. DeepOP's own robustness
#: experiment (Section 4.4, Fig. 7) removes a fraction of the observed
#: techniques to simulate detection failure; this trains against the same.
OBS_TOKEN_DROPOUT = 0.1
#: Probability of REPLACING an observed technique with another real token.
#: Branch A's errors at serve time are mostly substitutions -- an attack
#: window read as Benign, or the wrong technique -- not omissions, and dropout
#: alone only ever showed DeepOP clean-or-missing history.
OBS_TOKEN_SUBSTITUTION = 0.1


def _drop_observed(obs, vocab, p: float, p_sub: float = OBS_TOKEN_SUBSTITUTION):
    if p <= 0.0 and p_sub <= 0.0:
        return obs
    special = torch.tensor([vocab.pad_idx, vocab.bos_idx, vocab.eos_idx], device=obs.device)
    droppable = ~torch.isin(obs, special)
    if p_sub > 0.0:
        real = torch.tensor([i for i in range(vocab.vocab_size)
                             if i not in (vocab.pad_idx, vocab.bos_idx, vocab.eos_idx)],
                            device=obs.device)
        sub = (torch.rand(obs.shape, device=obs.device) < p_sub) & droppable
        repl = real[torch.randint(0, len(real), obs.shape, device=obs.device)]
        obs = torch.where(sub, repl, obs)
    if p <= 0.0:
        return obs
    drop = (torch.rand(obs.shape, device=obs.device) < p) & droppable
    return obs.masked_fill(drop, vocab.pad_idx)


def train_deepop_live(train_traj, val_traj, output, epochs, device, wdt=None, num_workers: int = 4, patience: int = 3,
                      lr: float = 5e-4, step_back_after: int = 2, clip_norm: float = 1.0,
                      resume: "ResumePoint | None" = None):
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
    # Recorded in the checkpoint so an evaluation on another dataset can tell
    # tokens training never contained from tokens it got wrong.
    _train_token_counts=[int(x) for x in train_ds.target_token_histogram()]
    print(f"DeepOP samples: train={len(train_ds)} val={len(val_ds)}",flush=True)
    print(f"DeepOP conditioning: {'Branch-B rollouts (E1 fixed)' if wdt is not None else 'oracle + noise (interim)'}",flush=True)
    # Precompute the frozen rollout once instead of recomputing it every epoch.
    if wdt is not None and _T > 0:
        _sp = os.environ.get("CYBERWORLD_SPILL_DIR")
        train_ds = _WithRollout(train_ds, _precompute_rollouts(
            wdt, train_ds, device, _sp, "train", _c.forecast_steps,
            num_workers=num_workers))
        val_ds = _WithRollout(val_ds, _precompute_rollouts(
            wdt, val_ds, device, _sp, "val", _c.forecast_steps,
            num_workers=num_workers))

    _lk=_loader_kwargs(device,num_workers)
    train_loader=_make_loader(train_ds,64,True,device,num_workers)
    val_loader=_make_loader(val_ds,64,False,device,num_workers)
    # DeepOP as published: encoder over the observed technique sequence,
    # causal-window decoder (h=6, n_cw=3). Conditioned on the full world state.
    d_state=int(train_traj.feats.shape[1])
    decoder=DeepOPForecastDecoder(d_latent=d_state,d_model=72,vocab_size=vocab.vocab_size,n_heads=6,num_layers=2,window_sizes=[2,4,8],dim_feedforward=144).to(device)
    optimizer=torch.optim.AdamW(decoder.parameters(),lr=lr,weight_decay=1e-4)
    best=float("-inf"); best_epoch=0; _history=[]   # maximised: see selection below
    # Label smoothing only over tokens that occur as a target. Six of the ten
    # vocabulary tokens never do; smoothing over all ten pushed 2.4% of every
    # target's mass onto impossible answers. See smoothed_and_plain_ce.
    _base_ds = getattr(train_ds, "base", train_ds)
    _hist = np.asarray(_base_ds.target_token_histogram())
    _support = torch.as_tensor(_hist > 0, dtype=torch.bool, device=device)
    # the same support as class ids, computed once (see smoothed_and_plain_ce)
    _support_idx = (torch.as_tensor(np.flatnonzero(_hist > 0), dtype=torch.long, device=device)
                    if (_hist > 0).any() else None)
    print(f"DeepOP label-smoothing support: {int(_support.sum())} of {len(_hist)} "
          f"tokens occur as a target", flush=True)

    _nb_total=len(train_loader); _nblk=(str(device)=="cuda")
    guard = TrainingGuard("deepop", [decoder], optimizer, mode="max",
                          patience=patience, step_back_after=step_back_after,
                          warmup_steps=default_warmup_steps(_nb_total),
                          clip_norm=clip_norm or None,
                          log=lambda m: print(m, flush=True),
                          graph_undo=(str(device) == "cuda" and graphs_enabled()))
    first_epoch = 0
    _rp = resume.load() if resume is not None else None
    if _rp is not None:
        decoder.load_state_dict(_rp["decoder"]); optimizer.load_state_dict(_rp["optimizer"])
        guard.load_state_dict(_rp["guard"])
        best, best_epoch = _rp["best"], _rp["best_epoch"]
        _history[:] = _rp["history"]
        first_epoch = epochs if guard.should_stop() else _rp["done_epochs"]
    _graphed = None
    for epoch in range(first_epoch, epochs):
        decoder.train()
        # On-device accumulation: `loss.item()` per batch synced the host to
        # the GPU on every step and defeated the worker prefetch queue.
        _tr_sum=torch.zeros((),device=device,dtype=torch.float64)
        _tr_plain=torch.zeros((),device=device,dtype=torch.float64); _nb=0
        _nb_dev=torch.zeros((),device=device,dtype=torch.long); _k=0
        _step=_guard_step_fn(guard)
        if _graphed is None:
            def _dp_loss(h_aug_, inp_, tgt_, obs_seq_):
                logits_ = decoder(h_aug_, inp_, obs_tokens=obs_seq_)
                l_, p_ = smoothed_and_plain_ce(logits_, tgt_, label_smoothing=0.04,
                                               support=_support, support_index=_support_idx)
                return l_, p_.detach()
            # the whole step (forward, loss, backward, clip, guard checks and
            # snapshot) as one CUDA graph: TrainingGuard.deferred_step_graphed
            _graphed = WholeStepGraph(_dp_loss, [decoder],
                                      enabled=(str(device) == "cuda" and graphs_enabled()
                                               and guard.deferred_supported()))
        _t0=time.time()
        for batch in train_loader:
            _k+=1
            batch = unpack_batch(batch, device)
            h=batch["h_future"].to(device,non_blocking=_nblk)
            inp=batch["input_tokens"].to(device,non_blocking=_nblk)
            tgt=batch["target_tokens"].to(device,non_blocking=_nblk)
            if "h_rollout" in batch:
                h_aug=batch["h_rollout"].to(device,non_blocking=_nblk)
            elif wdt is not None and "h_history" in batch:
                # Condition on what Branch B actually predicts, which is what
                # DeepOP receives in production.
                with torch.no_grad():
                    _th=batch["t_history"].to(device,non_blocking=_nblk) if "t_history" in batch else None
                    _tf=batch["t_future"].to(device,non_blocking=_nblk) if "t_future" in batch else None
                    h_aug=wdt.rollout(batch["h_history"].to(device,non_blocking=_nblk), K=h.shape[1],
                                      t_history=_th, t_future=_tf).detach()
            else:
                step_sigma=torch.linspace(0.015,0.055,steps=h.shape[1],device=device).unsqueeze(0).unsqueeze(-1)
                h_aug=h+torch.randn_like(h)*step_sigma
            obs_seq=_drop_observed(batch["obs_tokens"].to(device,non_blocking=_nblk),vocab,OBS_TOKEN_DROPOUT)
            optimizer.zero_grad(set_to_none=True)
            # forward + loss + backward as a CUDA graph (cyberworld_v4/graphed_step.py);
            # the observed-token dropout above stays eager, so its RNG draws are unchanged.
            # Train used smoothed CE while validation used plain CE, so the two
            # printed numbers were different functions and their gap was not a
            # generalisation gap. Identity: L_smooth = 0.96*plain + 0.04*U with
            # U >= ln(10), which put the real degradation at >= 0.2087 nats
            # against a printed 0.1404. Both are now reported.
            if _graphed.enabled:
                loss, _ok = guard.deferred_step_graphed(_graphed, [h_aug, inp, tgt, obs_seq])
            else:
                loss = _graphed.eager([h_aug, inp, tgt, obs_seq])
                _ok = _step(loss)
            _plain = _graphed.last_extras[0]
            if _ok is False:
                continue
            _accumulate_ok(_ok, [(_tr_sum, loss.detach().double().sum()),
                                 (_tr_plain, _plain.detach().double().sum())], _nb_dev)
            if _k % 2000 == 0:
                _el=time.time()-_t0; _r=_k/max(_el,1e-9)
                print(f"  DeepOP epoch={epoch+1} batch={_k}/{_nb_total} "
                      f"({100.0*_k/max(_nb_total,1):.1f}%) {_r:.1f} batch/s "
                      f"eta={(_nb_total-_k)/max(_r,1e-9)/60:.1f}m",flush=True)
        guard.flush()
        _nb=int(_nb_dev)
        if _graphed is not None and (_graphed.enabled or _graphed.error):
            print(f"  cuda graph: {_graphed.n_graphed} graphed steps of {_k}"
                  + (f" ({_graphed.error})" if _graphed.error else ""), flush=True)
            _graphed.n_graphed = 0
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
        _va = validate_deepop(decoder, wdt, val_loader, device, vocab)
        _v_sum, _vn, _hit_model, _hit_persist = _va["v_sum"], _va["vn"], _va["hit_model"], _va["hit_persist"]
        _tok_total, _conf, _tgt_hist, _scorer = _va["tok_total"], _va["conf"], _va["tgt_hist"], _va["scorer"]
        print(f"  validation: {_va['graphed_batches']} of {_va['vn']} batches as a cuda graph", flush=True)
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
        # Select on FREE-RUNNING macro F1, not validation cross-entropy.
        #
        # Serving is free-running (model_adapter calls forecast_sequence), and
        # with ~93% of target tokens Benign the validation CE is dominated by
        # how well the model fits that prior -- the minimum-CE epoch is the
        # one that best predicts "Benign", not the one that forecasts attacks.
        # Macro F1 over the tokens present weights the rare ones equally and
        # is scored on the decode an operator actually sees. Falls back to CE
        # (negated, since this is maximised) if the scorer is unavailable.
        _sel = None
        if _scorer is not None:
            _sel = _scorer.result().get("macro_f1_free")
        _sel_metric = "macro_f1_free" if _sel is not None else "neg_val_ce"
        if _sel is None:
            _sel = -score
        _history[-1]["selection_metric"] = _sel_metric
        _history[-1]["selection_score"] = float(_sel)
        _dp_health = []
        if acc - _best_base <= 0.0:
            _dp_health.append(f"token accuracy {acc:.4f} is not better than repeating the last "
                              f"token or the majority token ({_best_base:.4f})")
        _action = guard.end_epoch(_sel, train_loss=float((_tr_sum/max(_nb,1)).item()),
                                  health=_dp_health)
        _history[-1]["guard_action"] = _action
        if _action == IMPROVED:
            best=_sel; best_epoch=epoch+1
            output.parent.mkdir(parents=True,exist_ok=True); torch.save({"decoder_state_dict":decoder.state_dict(),"epoch":epoch+1,"history_steps":_c.history_steps,"forecast_steps":_c.forecast_steps,"window_seconds":_c.window_seconds,"vocab_size":vocab.vocab_size,"d_state":d_state,"arch":decoder.arch_config(),"train_token_counts":_train_token_counts,"epoch_history":list(_history),"selection_metric":_sel_metric,"label_smoothing_support":_support.cpu().tolist(),"baselines":{"acc_persistence":acc_persist,"acc_majority":acc_majority}},output)
        if resume is not None:
            resume.save(epoch + 1, decoder=decoder.state_dict(), optimizer=optimizer.state_dict(),
                        guard=guard.state_dict(), best=best, best_epoch=best_epoch,
                        history=list(_history))
        if _action == STOP:
            # Best weights are already on disk; stopping cannot cost quality.
            print(f"DeepOP: early stop at epoch {epoch+1}; {guard.stop_reason} "
                  f"(best epoch {best_epoch}, selection score {best:.4f})", flush=True)
            break
    if output.exists():
        _ck = torch.load(output, map_location="cpu", weights_only=False)
        _ck["training_guard"] = guard.summary()
        torch.save(_ck, output)


def _downstream_column_jobs(args):
    """[(split, namespace, ColumnSpec)] in exactly the order, and with exactly
    the reads, of the record path in _pcap_trajectories_per_day: every PCAP day
    sorted by name (iter_pcap_day_records: windows concatenated, unresolved
    labels dropped), then under cross_year_ctu the CTU-13 train and val scenarios
    (read_one_capture)."""
    from data_unification.capture_columns import ColumnSpec
    from data_unification.training_sources import discover_captures
    from data_unification.trajectory_store import capture_namespace
    ws = float(get_contract().window_seconds)
    jobs = []
    caps = discover_captures(scheme=args.split_scheme, pcap2018_root=args.pcap_root)
    for cap in sorted((c for sp in caps.values() for c in sp), key=lambda c: c.name):
        jobs.append((cap.split, f"pcap/{cap.name}", ColumnSpec(
            "pcap_windows", cap.dataset, cap.name, str(cap.path), ws,
            str(args.cic2018_csv_dir), None, 1, args.pcap_max_windows_per_day,
            int(args.pcap_window_stride), None)))
    if args.split_scheme == "cross_year_ctu":
        ctu_caps = discover_captures(scheme=args.split_scheme, ctu13_dir=args.ctu_dir)
        assert not ctu_caps["test"], "cross_year_ctu must never test on CTU-13"
        for split in ("train", "val"):
            for cap in ctu_caps[split]:
                jobs.append((split, capture_namespace(cap), ColumnSpec(
                    "one_capture", cap.dataset, cap.name, str(cap.path), ws,
                    None, args.rows_per_file, int(args.stride))))
    return jobs


def _columns_extract_in_order(jobs, plan, args, extractor, builders, wbase):
    """Extract each job's columns into builders[split], in job order.

    Yields (split, namespace, n_records, seconds) after each capture."""
    from data_unification.capture_columns import iter_capture_columns
    stream = iter_capture_columns(
        [spec for _s, _n, spec in jobs], cache_dir=plan.cache_dir,
        scratch_dir=plan.scratch_dir, workers=args.ingest_workers)
    if plan.cache_dir is not None and args.extract_workers > 1:
        # Independent captures, extracted at once, appended in job order:
        # the same stores (data_unification/parallel_extract.py).
        from data_unification.parallel_extract import extract_parallel
        items = ((n, cols, ns) for n, ((split, ns, _spec), (_sp, cols))
                 in enumerate(zip(jobs, stream)))
        for n, info in extract_parallel(
                items, extractor=extractor, builder_for=lambda k: builders[jobs[k][0]],
                tgne=str(args.tgne), part_dir=Path(args.spill_dir or plan.cache_dir) / "parts_downstream",
                workers=args.extract_workers):
            yield jobs[n][0], jobs[n][1], info["n_records"], info["seconds"]
        return
    for (split, ns, _spec), (_sp, cols) in zip(jobs, stream):
        t = time.time()
        b = builders[split]
        b.set_namespace(ns)
        extractor.extract_trajectories_columns(cols, builder=b, window_idx_base=wbase[split])
        if b._window_idx.n:
            wbase[split] = b.next_window_base()
        n = len(cols)
        del cols
        gc.collect()
        yield split, ns, n, time.time() - t


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
    builders = {k: TrajectoryStoreBuilder(spill_dir=spill, feat_dim=12 + extractor.n_temporal_attrs)
                for k in ("train", "val", "test")}
    wbase = {"train": 0, "val": 0, "test": 0}
    from data_unification.capture_columns import columns_plan
    plan = columns_plan(args.capture_cache, args.spill_dir, extractor)
    print(plan.describe(args.ingest_workers), flush=True)
    if plan.enabled:
        # The same captures, in the same order, into the same stores as the
        # record path below -- as cached columns (data_unification/
        # capture_columns.py), because a PCAP day as records is ~12 GB.
        for split, ns, n_recs, _t in _columns_extract_in_order(
                _downstream_column_jobs(args), plan, args, extractor, builders, wbase):
            print(f"  [{split}] {ns}: {n_recs} records -> {builders[split]._n} snapshots "
                  f"({_t:.1f}s)", flush=True)
        out = {}
        for k, b in builders.items():
            st = b.finalize()
            print(f"{k}: {st.n_snapshots} snapshots over {len(st)} hosts | {st.memory_report()}", flush=True)
            out[k] = st
        return out
    for split, day, recs in iter_pcap_day_records(
            args.pcap_root, args.cic2018_csv_dir, get_contract().window_seconds,
            args.pcap_max_windows_per_day, window_stride=args.pcap_window_stride,
            scheme=args.split_scheme):
        b = builders[split]
        t = time.time()
        b.set_namespace(f"pcap/{day}")   # one trajectory per (host, capture day)
        extractor.extract_trajectories(recs, builder=b, window_idx_base=wbase[split])
        if b._window_idx.n:
            wbase[split] = b.next_window_base()
        print(f"    -> {split} store: {b._n} snapshots ({time.time()-t:.1f}s)", flush=True)
        del recs
        gc.collect()
    # cross_year_ctu: CTU-13 joins train/val (never test). Without this the
    # PCAP path read PCAP days only, so the scheme would have silently trained
    # Branch B and DeepOP on a different corpus than Branch A and the encoder.
    if args.split_scheme == "cross_year_ctu":
        from data_unification.training_sources import discover_captures
        from data_unification.trajectory_store import capture_namespace
        _cic, _ctu = CIC2018Adapter(), CTU13Adapter()
        ctu_caps = discover_captures(scheme=args.split_scheme, ctu13_dir=args.ctu_dir)
        assert not ctu_caps["test"], "cross_year_ctu must never test on CTU-13"
        for split in ("train", "val"):
            for cap in ctu_caps[split]:
                t = time.time()
                recs = read_one_capture(Path(cap.path), _cic, _ctu, args.rows_per_file, args.stride)
                b = builders[split]
                b.set_namespace(capture_namespace(cap))   # one trajectory per (host, capture)
                extractor.extract_trajectories(recs, builder=b, window_idx_base=wbase[split])
                if b._window_idx.n:
                    wbase[split] = b.next_window_base()
                print(f"  [CTU13 {split}] {cap.name}: {len(recs)} recs -> {b._n} snapshots "
                      f"({time.time()-t:.1f}s)", flush=True)
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
    parser.add_argument("--split-scheme",choices=("frozen","cross_year","cross_year_ctu"),default="frozen",
                        help="'cross_year': train/tune on CIC-2018 PCAP days, then score the best "
                             "checkpoints ONCE on all of CIC-2017 (--cic2017-dir). Needs --pcap-root. "
                             "'cross_year_ctu': the same, with CTU-13 in train/val only (--ctu-dir).")
    parser.add_argument("--cic2017-dir",type=Path,default=None,
                        help="CIC-IDS-2017 CSVs, the cross_year test set")
    parser.add_argument("--results-json",type=Path,default=None,
                        help="Write the cross_year CIC-2017 scores here")
    parser.add_argument("--tgne",type=Path,required=True); parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--rows-per-file",type=int,default=None,
                        help="Cap records kept per capture. Default None = FULL DENSITY.")
    parser.add_argument("--stride",type=int,default=1)
    parser.add_argument("--packet-features", action="store_true",
                        help="Append the 30 PCAP packet-level host attributes (TTL, IAT, TCP "
                             "window, retransmissions, SYN/scan signatures): state 27-D -> 57-D. "
                             "Needs PCAP-derived train AND test captures.")
    parser.add_argument("--spill-dir",type=Path,default=None,help="Write the bulk trajectory feature block here instead of RAM (np.memmap)")
    parser.add_argument("--epochs",type=int,default=12,
                        help="Upper bound; the training guard stops each model once "
                             "--patience epochs pass without improvement.")
    parser.add_argument("--step-back-after",type=int,default=2,
                        help="After N flat epochs, restore the best weights and halve "
                             "the LR (cyberworld_v4/training_guard.py). Must be < --patience.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore resume points left by a crashed run of the same command")
    parser.add_argument("--clip-norm",type=float,default=1.0,
                        help="Gradient-norm clip for Branch B and DeepOP; 0 disables")
    parser.add_argument("--lr-branch-b",type=float,default=1e-3)
    parser.add_argument("--lr-deepop",type=float,default=5e-4)
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
    parser.add_argument("--patience",type=int,default=3,
                        help="Stop a model after N epochs without a validation "
                             "improvement. The best checkpoint is written every "
                             "time it improves, so this cannot cost quality -- it "
                             "only stops paying for epochs that do nothing. Run 1 "
                             "took 6 Branch B epochs and its best was epoch 1. "
                             "0 disables.")
    parser.add_argument("--allow-noncredible-branch-b",action="store_true",
                        help="Train DeepOP even if Branch B does not beat the "
                             "persistence baseline (see MIN_BRANCH_B_SKILL).")
    parser.add_argument("--capture-cache",type=str,default=None,
                        help="Cache of captures parsed to compact columns "
                             "(data_unification/capture_columns.py). Default: "
                             "$CYBERWORLD_CAPTURE_CACHE, else <--spill-dir>/capture_cache, "
                             "shared with Branch A. 'off' reads records as before.")
    parser.add_argument("--ingest-workers",type=int,default=3,
                        help="Captures parsed at once, each in its own process")
    parser.add_argument("--extract-workers",type=int,default=3,
                        help="Captures extracted at once, each in its own CPU process "
                             "(data_unification/parallel_extract.py); 1 = in this process")
    parser.add_argument("--legacy-loader",action="store_true",
                        help="Per-sample __getitem__ + default collate (the pre-2026-10 "
                             "loaders). The default batched loaders yield bit-identical "
                             "batches in the same order; this is a fallback.")
    parser.add_argument("--num-workers",type=int,default=4,
                        help="DataLoader worker processes; 0 loads in the main "
                             "process and serialises loading with GPU compute.")
    args=parser.parse_args()
    global LEGACY_LOADER
    LEGACY_LOADER = bool(args.legacy_loader)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    import time
    t0=time.time()

    tgn=build_or_load_tgne_ta(checkpoint_path=str(args.tgne))
    from data_unification.fast_extract import enable_fast_extraction
    enable_fast_extraction(tgn)   # serial extraction paths; workers do the same
    extractor=HostTrajectoryExtractor(
        tgne_ta_model=tgn, window_size_sec=get_contract().window_seconds,
        spill_dir=str(args.spill_dir) if args.spill_dir else None,
        include_packet_features=args.packet_features)
    if args.packet_features:
        print("PACKET-LEVEL FEATURES ON: host state = 12 latent + 15 flow + 30 packet "
              "attributes (TTL, IAT, TCP window, retransmissions, SYN/scan signatures, ...). "
              "Only PCAP captures carry them; flow-only captures (CTU-13 NetFlow, "
              "CICFlowMeter CSVs) get an all-zero packet block, which a model can read as "
              "'which corpus is this'. Train and test on PCAP-derived captures.", flush=True)

    require_full_density(
        'Branch B + DeepOP retrain',
        stride=args.stride,
        rows_per_file=args.rows_per_file,
        pcap_window_stride=args.pcap_window_stride,
        pcap_max_windows_per_day=args.pcap_max_windows_per_day,
    )

    if is_cross_year(args.split_scheme) and not args.pcap_root:
        parser.error(f"--split-scheme {args.split_scheme} trains on CIC-2018 and needs --pcap-root: "
                     "9 of 10 CIC-2018 CSV days fabricate host IPs")
    if is_cross_year(args.split_scheme) and not args.cic2017_dir:
        parser.error(f"--split-scheme {args.split_scheme} needs --cic2017-dir (the test set)")
    if args.split_scheme == "cross_year_ctu" and not args.ctu_dir:
        parser.error("--split-scheme cross_year_ctu needs --ctu-dir")
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

        from data_unification.trajectory_store import TrajectoryStoreBuilder, capture_namespace
        _cic, _ctu = CIC2018Adapter(), CTU13Adapter()
        _spill = str(args.spill_dir) if args.spill_dir else None

        def _store_per_capture(files, label):
            shared = TrajectoryStoreBuilder(spill_dir=_spill, feat_dim=12 + extractor.n_temporal_attrs)
            widx_base, total = 0, 0
            for i, f in enumerate(files):
                t = time.time()
                recs = read_one_capture(f, _cic, _ctu, args.rows_per_file, args.stride)
                total += len(recs)
                shared.set_namespace(capture_namespace(f))   # one trajectory per (host, capture)
                extractor.extract_trajectories(recs, builder=shared,
                                               window_idx_base=widx_base)
                if shared._window_idx.n:
                    widx_base = shared.next_window_base()
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
    # Crash recovery. Kept until the whole run finishes, so a crash in DeepOP
    # does not retrain a Branch B that had already finished.
    _fp = run_fingerprint(args, ignore=("epochs", "num_workers", "capture_cache", "ingest_workers", "extract_workers", "legacy_loader"))
    _log = lambda m: print(m, flush=True)
    bb_resume = ResumePoint(bb_out.with_name(bb_out.stem + "_resume.pt"), {**_fp, "stage": "'branch_b'"},
                            enabled=not args.no_resume, log=_log)
    dp_resume = ResumePoint(dp_out.with_name(dp_out.stem + "_resume.pt"), {**_fp, "stage": "'deepop'"},
                            enabled=not args.no_resume, log=_log)
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
        # get_contract() here, not _c: that name is local to the trainer
        # functions, and using it in main() raised NameError after a 40-minute
        # extraction had already been paid for.
        _hc = get_contract()
        _tau = hazard_tau_seconds()
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
                                  num_workers=args.num_workers,patience=args.patience,
                                  risk_target=args.risk_target, lr=args.lr_branch_b,
                                  step_back_after=args.step_back_after,
                                  clip_norm=args.clip_norm, resume=bb_resume)
        print(f"served checkpoint updated: {bb_out}", flush=True)

    if args.stages in ("both", "deepop"):
        if bb_out.exists():
            require_credible_branch_b(
                torch.load(bb_out, map_location="cpu", weights_only=False),
                args.allow_noncredible_branch_b)
        if wdt is None:
            # deepop-only: condition on the Branch B already on disk rather
            # than retraining it. eval() matters -- the rollout has dropout,
            # and a WDT left in train mode would feed DeepOP a different
            # (randomly perturbed) conditioning signal every epoch.
            if not bb_out.exists():
                parser.error(f"--stages deepop needs a trained Branch B at {bb_out}")
            _bb = torch.load(bb_out, map_location=device, weights_only=False)
            _d_state = int(_bb.get("d_state") or _bb["wdt_state_dict"]["in_proj.weight"].shape[1])
            wdt = HostWorldDynamicsTransformer(d_latent=_d_state, d_model=64, n_heads=4,
                                               n_layers=3).to(device)
            wdt.load_state_dict(_bb["wdt_state_dict"])
            wdt.eval()
            print(f"loaded Branch B from {bb_out} (epoch {_bb.get('epoch')}) for "
                  f"DeepOP conditioning", flush=True)
        train_deepop_live(train_traj,val_traj,dp_out,args.epochs,device,wdt=wdt,
                          num_workers=args.num_workers,patience=args.patience,
                          lr=args.lr_deepop, step_back_after=args.step_back_after,
                          clip_norm=args.clip_norm, resume=dp_resume)
        print(f"served checkpoint updated: {dp_out}", flush=True)

    if is_cross_year(args.split_scheme):
        _score_cross_year(args, extractor, train_traj, bb_out, dp_out, device)
    # The whole run finished: nothing left to resume.
    bb_resume.clear()
    dp_resume.clear()


def hazard_tau_seconds() -> float:
    """The hazard target's decay scale: the contract's forecast horizon
    (forecast_steps x forecast_window_seconds), the same as Branch A's."""
    return float(get_contract().forecast_seconds)


def _score_cross_year(args, extractor, train_traj, bb_out, dp_out, device):
    """Score the best Branch B / DeepOP checkpoints ONCE on all of CIC-2017.

    Nothing here feeds back into training or selection; both models were frozen
    on CIC-2018 validation days before this runs.
    """
    import json
    from cyberworld_v4.cross_dataset import (
        branch_b_on_store, deepop_on_store, format_unseen_report)
    from data_unification.trajectory_store import TrajectoryStoreBuilder
    from data_unification.training_sources import discover_captures, read_capture

    _c = get_contract()
    caps = discover_captures(scheme="cross_year", cic2017_dir=args.cic2017_dir)["test"]
    b = TrajectoryStoreBuilder(spill_dir=str(args.spill_dir) if args.spill_dir else None, feat_dim=12 + extractor.n_temporal_attrs)
    wbase = 0
    from data_unification.capture_columns import ColumnSpec, columns_plan, iter_capture_columns
    plan = columns_plan(args.capture_cache, args.spill_dir, extractor)
    if plan.enabled:
        specs = [ColumnSpec.for_read_capture(cap, window_seconds=_c.window_seconds) for cap in caps]
        for cap, (_s, cols) in zip(caps, iter_capture_columns(
                specs, cache_dir=plan.cache_dir, scratch_dir=plan.scratch_dir,
                workers=args.ingest_workers)):
            extractor.extract_trajectories_columns(cols, builder=b, window_idx_base=wbase)
            if b._window_idx.n:
                wbase = b.next_window_base()
            print(f"  [test] {cap.label}: {len(cols)} records", flush=True)
            del cols
            gc.collect()
    for cap in (caps if not plan.enabled else ()):
        recs = read_capture(cap, window_seconds=_c.window_seconds)
        extractor.extract_trajectories(recs, builder=b, window_idx_base=wbase)
        if b._window_idx.n:
            wbase = b.next_window_base()
        print(f"  [test] {cap.label}: {len(recs)} records", flush=True)
        del recs
        gc.collect()
    test_store = b.finalize()
    if args.risk_target == "hazard":
        # The models were trained and selected on the hazard target; scoring
        # them against the extraction-time SEVERITY column (this store was
        # never converted) compared two different quantities.
        test_store.use_hazard_target(hazard_tau_seconds())
    out = {"split_scheme": "cross_year", "test_snapshots": int(test_store.n_snapshots),
           "risk_target": args.risk_target}

    if bb_out.exists():
        ck = torch.load(bb_out, map_location=device, weights_only=False)
        d_state = int(ck.get("d_state") or ck["wdt_state_dict"]["in_proj.weight"].shape[1])
        wdt = HostWorldDynamicsTransformer(d_latent=d_state, d_model=64, n_heads=4, n_layers=3).to(device)
        wdt.load_state_dict(ck["wdt_state_dict"])
        risk = InfiltrationRiskHead(d_latent=d_state, hidden_dim=32).to(device)
        risk.load_state_dict(ck["risk_head_state_dict"])
        out["branch_b"] = branch_b_on_store(wdt, risk, test_store, device,
                                            _c.history_steps, _c.forecast_steps)
        print(f"CIC-2017 Branch B: {out['branch_b']}", flush=True)
        ck["test_metrics_cross_year"] = out["branch_b"]
        torch.save(ck, bb_out)

        if dp_out.exists():
            vocab = get_joint_vocab(network_observable_only=True)
            dck = torch.load(dp_out, map_location=device, weights_only=False)
            decoder = DeepOPForecastDecoder.from_checkpoint(dck, device=device)
            train_counts = LazyCWADataset(train_traj, vocab, K=_c.forecast_steps,
                                          T=_c.history_steps).target_token_histogram()
            out["deepop"] = deepop_on_store(decoder, wdt, test_store, vocab, device,
                                            train_counts, _c.history_steps, _c.forecast_steps)
            print(format_unseen_report(out["deepop"], "CIC-2017 DeepOP tokens"), flush=True)
            print(f"  persistence token accuracy {out['deepop']['acc_persistence']:.4f}", flush=True)
            dck["test_metrics_cross_year"] = out["deepop"]
            torch.save(dck, dp_out)

    if args.results_json:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(out, indent=2, default=str))
        print(f"results written to {args.results_json}", flush=True)


if __name__ == "__main__":
    main()
