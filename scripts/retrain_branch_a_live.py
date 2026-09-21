"""Retrain Branch A on supplied 2-second SIH/CTU live-telemetry data."""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
from branch_a_gnn_lstm.sequence_dataset import (
    TECHNIQUE_VOCAB,
    TECH_TO_IDX,
    HostSequenceDataset,
    create_host_sequence_samples,
    LazyHostSequenceDataset,
)
from branch_a_gnn_lstm.train_branch_a import build_or_load_tgne_ta
from data_unification.cic2018_adapter import CIC2018Adapter
from data_unification.ctu13_adapter import CTU13Adapter
from data_unification.multi_dataset_stream import HostTrajectoryExtractor
from data_unification.split_policy import partition_paths
from data_unification.density import require_full_density
from cyberworld_v4.config import get_contract, DEFAULT_CONFIG
from cyberworld_v4.manifest import ExperimentManifest, set_all_seeds


def _strided(gen, stride: int, want=None):
    """Samples every Nth record across a wider read instead of a plain file-prefix.

    max_rows in the adapters is a prefix (pandas nrows); CIC-2018 CSVs are
    time-ordered with attacks in contiguous blocks, so a plain prefix is
    87-100% single-label (see claude_latest_analysis/07_v4_audit_and_migration_plan.md,
    D6). Reading stride*want rows and keeping every `stride`-th one instead
    spans much more of the file's time range for the same record budget.
    """
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


def _load_records(cic_dir: Path, ctu_dir: Path, rows_per_file: int, files: List[Path], stride: int = 1):
    import time
    cic = CIC2018Adapter()
    ctu = CTU13Adapter()
    records = []
    for i, path in enumerate(files):
        t0 = time.time()
        # rows_per_file is None at FULL DENSITY (the default since the
        # subsampling sweep), so `rows_per_file * stride` raised TypeError.
        # None must propagate as "no cap" rather than becoming 0.
        _cap = None if rows_per_file is None else rows_per_file * stride
        gen = cic.parse_file(str(path), max_rows=_cap) if path.suffix.lower() == ".csv" \
            else ctu.parse_netflow_csv(str(path), max_rows=_cap)
        got = _strided(gen, stride, rows_per_file)
        records.extend(got)
        print(f"  [{i+1}/{len(files)}] {path.name}: {len(got)} records in {time.time()-t0:.1f}s (cumulative {len(records)})", flush=True)
    return records


def _make_samples(records, extractor, seq_len: int):
    trajectories = extractor.extract_trajectories(records)
    return create_host_sequence_samples(
        trajectories,
        seq_len=seq_len,
        min_trajectory_len=1,
    )


def _selection_score(metrics, mode):
    """Higher is better. Which epoch's weights we keep.

    ## Why the default is not validation loss

    The objective is Kendall & Gal homoscedastic uncertainty weighting:

        L = sum_i [ exp(-s_i) * L_i + s_i ]

    where every `s_i` is a **learned parameter that moves during training**.
    Two epochs therefore do not report the same quantity: the loss is computed
    under a different weighting each time, and the bare `+ s_i` terms add a
    drifting constant. The 2026-09-21 checkpoint ended with
    s_risk = -3.0090 and s_tech = -3.0243 against an initialisation of 0, so
    that constant drifted from 0 to -7.04 over the run. A large part of the
    apparent fall in validation loss was the offset moving, not the model
    improving -- and picking the single lowest reading then selected epoch 6
    (0.6601, about half the median of the other epochs), which scored worst on
    the held-out test.

    A selection metric has to mean the same thing at every epoch. Macro F1 and
    risk MAE do; the weighted loss does not.

    - ``composite`` (default): ``0.5 * macro_f1 + 0.5 * (1 - min(risk_mae, 1))``
      -- balances the two heads that carry the task, both bounded in [0, 1] and
      both independent of the loss weighting.
    - ``macro_f1``: technique head only.
    - ``val_loss``: the previous behaviour, kept so a run can be reproduced.
      Negated here because this function is maximised.
    """
    if mode == "val_loss":
        return -float(metrics["loss"])
    if mode == "macro_f1":
        return float(metrics.get("tech_macro_f1", 0.0))
    f1 = float(metrics.get("tech_macro_f1", 0.0))
    mae = min(float(metrics.get("risk_mae", 1.0)), 1.0)
    return 0.5 * f1 + 0.5 * (1.0 - mae)


def _flag_outlier_selection(history, best):
    """Warn when the winning epoch's val loss is far off the run's own trend.

    The 2026-09-21 run produced val losses 1.88, 2.08, 1.39, 1.24, 1.39, 0.66,
    1.41, 1.21. Epoch 6 at 0.66 is roughly half the median of every other
    epoch, and picking the single best reading selected exactly that outlier --
    which then scored 26.36 on the held-out test against 0.66 on validation.

    A one-off low reading on a multi-task loss with learned log-variance terms
    is at least as likely to be noise as a genuinely better model. This does
    not change the selection -- that would be a modelling decision, not a
    reporting one -- it states when the result deserves suspicion.
    """
    if not history or not best or len(history) < 4:
        return
    losses = [h["loss"] for h in history if "loss" in h]
    chosen = best.get("loss")
    if chosen is None or len(losses) < 4:
        return
    others = sorted(l for l in losses if l != chosen)
    if not others:
        return
    median = others[len(others) // 2]
    if median <= 0:
        return
    if chosen < 0.6 * median:
        print(
            f"\nNOTE: the selected epoch ({best.get('epoch')}) had validation "
            f"loss {chosen:.4f}, against a median of {median:.4f} across the "
            f"other epochs -- {median / max(chosen, 1e-9):.1f}x lower. A single "
            f"outlier reading may be noise rather than a better model; compare "
            f"the held-out test result before quoting this checkpoint. Every "
            f"epoch's metrics are in ckpt['epoch_history'].",
            flush=True)


def _print_per_class(per_class, where):
    """Per-class precision/recall/f1/support, largest class first."""
    if not per_class:
        return
    inv = {v: k for k, v in TECH_TO_IDX.items()}
    print(f"  per-class ({where}):", flush=True)
    print(f"    {'technique':<28} {'prec':>6} {'recall':>7} {'f1':>6} "
          f"{'support':>9} {'predicted':>10}", flush=True)
    for c, m in sorted(per_class.items(), key=lambda kv: -kv[1]["support"]):
        print(f"    {inv.get(c, f'class_{c}'):<28} {m['precision']:>6.3f} "
              f"{m['recall']:>7.3f} {m['f1']:>6.3f} {m['support']:>9,} "
              f"{m['predicted']:>10,}", flush=True)


def _warn_if_head_collapsed(metrics, where):
    """Say plainly when accuracy is coming from the class prior, not the model.

    On this corpus 82.5% of validation samples are Benign, so a constant
    predictor scores ~0.825. Accuracy alone therefore cannot fail visibly. The
    TGNE category head scored 0.83 while emitting one class for every input and
    went unnoticed until macro F1 was computed -- these checks exist so that
    cannot happen quietly a second time.
    """
    pred = metrics.get("tech_classes_predicted", 0)
    present = metrics.get("tech_classes_present", 0)
    lift = metrics.get("tech_lift_over_baseline", 0.0)
    f1 = metrics.get("tech_macro_f1", 0.0)

    if pred <= 1 and present > 1:
        print(f"  WARNING [{where}]: technique head predicts a SINGLE class for "
              f"every input ({present} classes present in the data). Its "
              f"accuracy {metrics.get('tech_accuracy', 0):.3f} is the class "
              f"prior, not a result.", flush=True)
    elif lift < 0.01 and present > 1:
        print(f"  WARNING [{where}]: technique accuracy "
              f"{metrics.get('tech_accuracy', 0):.3f} is within 1 point of the "
              f"majority-class baseline "
              f"{metrics.get('tech_majority_baseline', 0):.3f} -- the head is "
              f"adding almost nothing.", flush=True)
    elif f1 < 0.2 and present > 2:
        print(f"  WARNING [{where}]: macro F1 {f1:.3f} over {present} classes -- "
              f"accuracy {metrics.get('tech_accuracy', 0):.3f} is carried by the "
              f"majority class while minority classes are largely missed.",
              flush=True)


def _evaluate(model, loader, device, num_techniques=None, num_gradations=4):
    """Validation pass.

    Accumulators live on the device and are read once at the end. An earlier
    version called `.item()` for the loss, `.cpu().numpy()` for the risk errors
    and `.item()` for the technique hits -- three host-device syncs per batch --
    and grew a Python list to one float per validation sample (1.02M of them).

    ## Why this reports more than accuracy

    82.5% of validation samples are Benign (val_positive_rate 0.1752). A model
    that predicts Benign for everything scores ~0.825 aggregate accuracy, so
    `tech_accuracy=0.880` is only 5.5 points above a constant classifier and
    cannot, by itself, distinguish a working head from a collapsed one. That is
    not hypothetical: the TGNE category head scored 0.83 while predicting a
    single class, and was only caught when macro F1 was computed.

    So this also returns macro F1, the majority-class baseline, the lift over
    it, how many distinct classes the model actually predicts, and per-class
    precision/recall/support. A confusion matrix is accumulated on-device with
    a single bincount per batch, which adds no host-device sync.
    """
    model.eval()
    nb = 0
    total = 0
    non_blocking = (device == "cuda")
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    abs_err_sum = torch.zeros((), device=device, dtype=torch.float64)
    n_risk = 0
    correct_tech_t = torch.zeros((), device=device, dtype=torch.long)

    task_sums = {k: torch.zeros((), device=device, dtype=torch.float64)
                 for k in ("loss_risk", "loss_tech", "loss_grad",
                           "weight_risk", "weight_tech", "weight_grad")}

    C = int(num_techniques) if num_techniques else len(TECHNIQUE_VOCAB)
    # confusion[t * C + p] -- flat so one bincount per batch suffices
    confusion = torch.zeros(C * C, device=device, dtype=torch.long)

    grad_correct = torch.zeros((), device=device, dtype=torch.long)
    grad_total = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device, non_blocking=non_blocking)
            targets = {
                "risk": batch["risk"].to(device, non_blocking=non_blocking),
                "technique": batch["technique"].to(device, non_blocking=non_blocking),
                "gradation": batch["gradation"].to(device, non_blocking=non_blocking),
            }
            predictions = model(x)
            loss, parts = model.compute_loss(predictions, targets)
            loss_sum += loss.detach().double().sum()
            # Per-task losses and their learned weights. Without these the
            # total is uninterpretable: the 2026-09-21 run reported a
            # validation loss of 0.66 against a test loss of 26.36 with no way
            # to see that two of the three weights had saturated at exp(3)=20
            # and were multiplying everything.
            for _k, _acc in task_sums.items():
                _v = parts.get(_k)
                if _v is not None:
                    _acc += _v.double().reshape(())
            nb += 1
            err = (predictions["risk_score"] - targets["risk"]).abs()
            abs_err_sum += err.double().sum()
            n_risk += int(err.numel())

            pred_t = predictions["technique_logits"].argmax(dim=-1)
            true_t = targets["technique"]
            correct_tech_t += (pred_t == true_t).sum()
            total += int(true_t.numel())
            confusion += torch.bincount(true_t * C + pred_t, minlength=C * C)

            if "gradation_logits" in predictions:
                pred_g = predictions["gradation_logits"].argmax(dim=-1)
                grad_correct += (pred_g == targets["gradation"]).sum()
                grad_total += int(targets["gradation"].numel())

    correct_tech = int(correct_tech_t.item())
    cm = confusion.reshape(C, C).cpu().numpy()

    support = cm.sum(axis=1)             # true count per class
    predicted = cm.sum(axis=0)           # predicted count per class
    tp = np.diag(cm)

    present = support > 0                # classes that actually occur
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / np.maximum(predicted, 1), 0.0)
        recall = np.where(support > 0, tp / np.maximum(support, 1), 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)

    macro_f1 = float(f1[present].mean()) if present.any() else 0.0
    # What "always predict the most common class" would score.
    baseline = float(support.max() / max(support.sum(), 1)) if support.sum() else 0.0
    accuracy = correct_tech / max(1, total)

    per_class = {
        int(c): {
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
            "support": int(support[c]),
            "predicted": int(predicted[c]),
        }
        for c in range(C) if support[c] > 0 or predicted[c] > 0
    }

    out_tasks = {k: (float((v / nb).item()) if nb else 0.0) for k, v in task_sums.items()}
    return {
        "loss": float((loss_sum / nb).item()) if nb else 0.0,
        "risk_mae": float((abs_err_sum / n_risk).item()) if n_risk else 0.0,
        "tech_accuracy": accuracy,
        **out_tasks,
        # -- the metrics that can tell a working head from a collapsed one --
        "tech_macro_f1": macro_f1,
        "tech_majority_baseline": baseline,
        "tech_lift_over_baseline": accuracy - baseline,
        "tech_classes_present": int(present.sum()),
        "tech_classes_predicted": int((predicted > 0).sum()),
        "tech_per_class": per_class,
        "gradation_accuracy": (float(grad_correct.item()) / grad_total) if grad_total else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cic-dir", type=Path, required=True)
    parser.add_argument("--ctu-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-file", type=int, default=None,
                        help="Cap records kept per capture. Default None = FULL DENSITY.")
    parser.add_argument("--spill-dir", type=Path, default=None, help="Write the bulk trajectory feature block here instead of RAM (np.memmap)")
    parser.add_argument("--stride", type=int, default=1, help="Sample every Nth record across a wider read, instead of a plain file-prefix (see _strided)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--select-on", choices=("composite", "macro_f1", "val_loss"),
                        default="composite",
                        help="Which validation metric picks the kept checkpoint. "
                             "Default 'composite' = 0.5*macro_f1 + 0.5*(1-risk_mae). "
                             "'val_loss' was the previous default but is not "
                             "comparable across epochs: the uncertainty-weighted "
                             "loss contains learned log-variance terms that drift "
                             "(0 -> -7.04 over the 2026-09-21 run), so part of its "
                             "fall is the weighting moving rather than the model "
                             "improving.")
    parser.add_argument("--eval-only", type=str, default=None,
                        metavar="CKPT",
                        help="Score an existing checkpoint and exit; no "
                             "training, no checkpoint is written. '-' means "
                             "the path given by --output.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes. 0 loads in the main "
                             "process, which serialises data loading with GPU "
                             "compute -- measured at 35.7%% of wall clock on the "
                             "full-density run, i.e. the GPU idled for a third "
                             "of every epoch. Default 4 is deliberately modest: "
                             "loading costs ~1.0x compute, so 2 already hide it, "
                             "and this box has ~9 GiB free under a 17 GiB cap.")
    parser.add_argument("--log-every", type=int, default=2000,
                        help="Print a progress line every N batches. At full "
                             "density an epoch is ~161k batches; with no "
                             "progress line a run is unobservable for an hour.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    _cfg = DEFAULT_CONFIG
    _manifest = ExperimentManifest.create(
        f"branch_a_v4_seed{args.seed}",
        seed=args.seed,
        config=_cfg,
        repo=Path(__file__).resolve().parent.parent,
        dataset_sources=[str(args.cic_dir), str(args.ctu_dir)],
    )
    set_all_seeds(args.seed)

    require_full_density(
        'Branch A retrain',
        stride=args.stride,
        rows_per_file=args.rows_per_file,
    )

    cic_files = sorted(args.cic_dir.glob("*.csv"))
    ctu_files = sorted(args.ctu_dir.glob("*/*.binetflow"))
    all_files = cic_files + ctu_files
    if len(all_files) < 4:
        raise RuntimeError(f"Expected supplied SIH/CTU files, found {len(all_files)}")

    # Three-way, capture-disjoint, and READ FROM THE FROZEN LOCK.
    #
    # This used to slice a *sorted* file list 70/15/15, so "train" meant
    # "alphabetically first" and the assignment shifted whenever a file was
    # added or renamed -- results were not comparable across retrains, which is
    # the entire reason data_unification/splits.lock.json exists. The lock had
    # zero consumers; now it has this one.
    #
    # Test is scored once, after the model is frozen, and never influences
    # training.
    partition = partition_paths(all_files)
    train_files, val_files, test_files = (
        partition["train"], partition["val"], partition["test"],
    )
    for _name, _files in (("train", train_files), ("val", val_files), ("test", test_files)):
        if not _files:
            raise RuntimeError(
                f"frozen split '{_name}' matched no files under {args.cic_dir} / "
                f"{args.ctu_dir}. Refusing to train on a split that does not exist."
            )
    print(f"frozen split: {len(train_files)} train / {len(val_files)} val / "
          f"{len(test_files)} test captures", flush=True)
    import time
    tgn = build_or_load_tgne_ta()
    # Contract-bound (v4). Previously 2.0s / seq_len=5 hardcoded, which matched
    # the v3 contract by coincidence rather than by construction. Under v4 this
    # produces history_steps=15, so it yields a v4 checkpoint, not a v3 one.
    _c = get_contract()
    extractor = HostTrajectoryExtractor(tgne_ta_model=tgn, window_size_sec=_c.window_seconds,
                                        spill_dir=str(args.spill_dir) if args.spill_dir else None)
    # Load -> extract -> free, one split at a time. Holding both record lists
    # at once costs an extra ~2.8 GB at full density for no reason: the val
    # records are not needed until the train split has already been reduced to
    # samples.
    record_counts: Dict[str, int] = {}

    def _store_per_capture(files, label):
        """Load -> extract -> free, one capture file at a time.

        Holding the whole split resident costs ~15.7 GB at full density (30.4M
        records x ~518 B measured), which does not fit alongside the snapshot
        store and adapter arrays. Per-file keeps peak at the largest single
        capture (~4.7M records, ~2.4 GB).

        Processing per file also builds the TGNE neighbour graph per capture
        rather than across all of them. That is more faithful, not less: a
        CIC-2018 capture day and a CTU-13 botnet scenario are unrelated
        networks, and a merged graph would make hosts from different captures
        each other's temporal neighbours, which they never were.
        """
        from data_unification.trajectory_store import TrajectoryStoreBuilder
        shared = TrajectoryStoreBuilder(spill_dir=str(args.spill_dir) if args.spill_dir else None)
        widx_base = 0
        total_recs = 0
        for i, f in enumerate(files):
            t = time.time()
            recs = _load_records(args.cic_dir, args.ctu_dir, args.rows_per_file, [f], args.stride)
            total_recs += len(recs)
            extractor.extract_trajectories(recs, builder=shared, window_idx_base=widx_base)
            if shared._window_idx.n:
                widx_base = int(shared._window_idx.buf[: shared._window_idx.n].max()) + 1
            print(f"  [{label} {i+1}/{len(files)}] {f.name}: {len(recs)} recs, "
                  f"store={shared._n} snaps, {time.time()-t:.1f}s", flush=True)
            del recs
            gc.collect()
        store = shared.finalize()
        print(f"{label}: {total_recs} records -> {store.n_snapshots} snapshots "
              f"over {len(store)} hosts | {store.memory_report()}", flush=True)
        # The record count is returned, not just printed: the run summary below
        # used to reference `train_records`/`val_records`, which the
        # load -> extract -> free refactor had already deleted. Every
        # non-credible run therefore died with NameError instead of reporting
        # the credibility verdict it had just computed.
        record_counts[label] = total_recs
        # Return the STORE, not materialised samples.
        #
        # create_host_sequence_samples builds a [15, 27] float32 array per
        # sample -- 2,053 bytes each. At full corpus density Branch A produces
        # roughly 42M samples (measured snapshot ratios: 1.99 per record for
        # CIC-2018, 0.72 for CTU-13), which is **80.3 GiB**. It does not fit,
        # and thinning the data is not an option.
        #
        # LazyHostSequenceDataset keeps two int32 columns (~8 B/sample, 336 MB
        # at 42M) and gathers each window from the memmapped feature block on
        # __getitem__ -- which is what DataLoader workers are for. Verified to
        # produce identical samples.
        return store

    import gc
    t0 = time.time()
    train_store = _store_per_capture(train_files, "train")
    train_ds = LazyHostSequenceDataset(train_store, seq_len=_c.history_steps, min_trajectory_len=1)
    print(f"train done in {time.time()-t0:.1f}s ({len(train_ds)} samples)", flush=True)
    t0 = time.time()
    val_store = _store_per_capture(val_files, "val")
    val_ds = LazyHostSequenceDataset(val_store, seq_len=_c.history_steps, min_trajectory_len=1)
    print(f"val done in {time.time()-t0:.1f}s ({len(val_ds)} samples)", flush=True)
    t0 = time.time()
    test_store = _store_per_capture(test_files, "test")
    test_ds = LazyHostSequenceDataset(test_store, seq_len=_c.history_steps, min_trajectory_len=1)
    print(f"test done in {time.time()-t0:.1f}s ({len(test_ds)} samples)", flush=True)

    # Gate the data before training on it.
    try:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from credibility_check import evaluate_store, gate as _gate, report as _report
        _stats = evaluate_store(train_store, val_store)
        _report(_stats, _gate(_stats))
    except Exception as _e:
        print(f"credibility check skipped: {_e}", flush=True)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("The 2-second pipeline produced no train/validation samples")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Workers matter more than anything else in this loop: profiling the
    # full-density run showed 35.7% of wall clock inside the dataset/collate
    # path, all of it in the main process and therefore serialised with the
    # GPU. The dataset is lazy by design (336 MB of indices instead of an
    # 80.3 GiB materialised array), which trades memory for per-item work --
    # that trade only pays if the work is overlapped.
    _loader_kw = dict(num_workers=args.num_workers, pin_memory=(device == "cuda"))
    if args.num_workers > 0:
        _loader_kw.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **_loader_kw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **_loader_kw,
    )

    model = MultiTaskLSTM(
        input_dim=27,
        hidden_dim=64,
        num_layers=2,
        num_techniques=len(TECHNIQUE_VOCAB),
        num_gradations=4,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("-inf")       # _selection_score is maximised
    _history: List[Dict[str, float]] = []
    best_metrics: Dict[str, float] = {}

    print(
        f"train_records={record_counts.get('train', 0)} "
        f"val_records={record_counts.get('val', 0)} "
        f"train_samples={len(train_ds)} val_samples={len(val_ds)} device={device}"
    )
    _n_train_batches = len(train_loader)
    if args.eval_only:
        # Re-score an existing checkpoint without retraining.
        #
        # The full-density Branch A run of 2026-09-21 completed before
        # _evaluate reported macro F1, the majority-class baseline or a
        # per-class breakdown, so its checkpoint carries only aggregate
        # accuracy -- which on an 82.5%-Benign corpus cannot distinguish a
        # working technique head from a collapsed one. Extraction is the
        # expensive part (~37 min) and is identical either way, so re-scoring
        # costs a fraction of a retrain.
        _src = args.eval_only if str(args.eval_only) != "-" else args.output
        print(f"EVAL-ONLY: scoring {_src} (no training)", flush=True)
        _ck = torch.load(_src, map_location=device, weights_only=False)
        model.load_state_dict(_ck["model_state_dict"])
        print(f"checkpoint epoch={_ck.get('epoch')} "
              f"recorded_metrics={_ck.get('metrics')}", flush=True)

        # Score BOTH splits. The 2026-09-21 run selected on validation loss and
        # then scored 0.6601 on validation against 26.36 on the held-out test --
        # a 40x gap. Reporting only one split cannot show whether that is a
        # collapsed head, a distribution shift between capture days, or a
        # checkpoint picked on an outlier epoch, so both are printed side by
        # side with the same metric set.
        _tl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         **_loader_kw)
        for _name, _ldr in (("validation", val_loader), ("held-out test", _tl)):
            _m = _evaluate(model, _ldr, device, num_techniques=len(TECHNIQUE_VOCAB))
            _pc = _m.pop("tech_per_class", {})
            print(f"\n{_name.upper()}: "
                  f"loss={_m['loss']:.4f} risk_mae={_m['risk_mae']:.4f} "
                  f"acc={_m['tech_accuracy']:.4f} "
                  f"macro_f1={_m['tech_macro_f1']:.4f} "
                  f"baseline={_m['tech_majority_baseline']:.4f} "
                  f"lift={_m['tech_lift_over_baseline']:+.4f} "
                  f"classes_pred={_m['tech_classes_predicted']}"
                  f"/{_m['tech_classes_present']} "
                  f"gradation_acc={_m['gradation_accuracy']}", flush=True)
            _warn_if_head_collapsed(_m, _name)
            _print_per_class(_pc, _name)
        return

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Loss is accumulated as a GPU tensor and read once at the end of the
        # epoch. `float(loss.item())` per batch forces a host-device sync on
        # every one of ~161k batches, which serialises the CPU against the GPU
        # and defeats the prefetching the workers above are there to provide.
        # The reported value is unchanged: still the unweighted mean over
        # batches.
        _nb = 0
        _loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        _t_epoch = time.time()
        _non_blocking = (device == "cuda")
        for batch in train_loader:
            x = batch["features"].to(device, non_blocking=_non_blocking)
            targets = {
                "risk": batch["risk"].to(device, non_blocking=_non_blocking),
                "technique": batch["technique"].to(device, non_blocking=_non_blocking),
                "gradation": batch["gradation"].to(device, non_blocking=_non_blocking),
            }
            optimizer.zero_grad(set_to_none=True)
            predictions = model(x)
            loss, _ = model.compute_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            # Keep the log-variances in range in the saved weights too: a step
            # can leave one epsilon outside the bound, and that is the value a
            # checkpoint written this epoch would record.
            model.uncertainty_loss.project_()
            _loss_sum += loss.detach().double().sum()
            _nb += 1
            if args.log_every and _nb % args.log_every == 0:
                _el = time.time() - _t_epoch
                _rate = _nb / max(_el, 1e-9)
                _eta = (_n_train_batches - _nb) / max(_rate, 1e-9)
                print(f"  epoch={epoch} batch={_nb}/{_n_train_batches} "
                      f"({100.0 * _nb / max(_n_train_batches, 1):.1f}%) "
                      f"{_rate:.1f} batch/s elapsed={_el / 60:.1f}m "
                      f"eta={_eta / 60:.1f}m", flush=True)

        metrics = _evaluate(model, val_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = float((_loss_sum / max(_nb, 1)).item())
        metrics["epoch_seconds"] = float(time.time() - _t_epoch)
        print(
            f"epoch={epoch} train_loss={metrics['train_loss']:.4f} "
            f"val_loss={metrics['loss']:.4f} risk_mae={metrics['risk_mae']:.4f} "
            f"tech_accuracy={metrics['tech_accuracy']:.3f} "
            f"tech_macro_f1={metrics['tech_macro_f1']:.3f} "
            f"lift={metrics['tech_lift_over_baseline']:+.3f} "
            f"classes_pred={metrics['tech_classes_predicted']}/{metrics['tech_classes_present']} "
            f"sel[{args.select_on}]={_selection_score(metrics, args.select_on):.4f} "
            f"| task_loss risk={metrics['loss_risk']:.4f} tech={metrics['loss_tech']:.4f} "
            f"grad={metrics['loss_grad']:.4f} "
            f"| weight risk={metrics['weight_risk']:.2f} tech={metrics['weight_tech']:.2f} "
            f"grad={metrics['weight_grad']:.2f} "
            f"wall={metrics['epoch_seconds'] / 60:.1f}m"
        )
        _warn_if_head_collapsed(metrics, f"epoch {epoch}")
        # Keep every epoch's validation metrics. Only the best-scoring weights
        # are written, so without this the other epochs are unrecoverable and
        # a selection decision cannot be revisited without a full retrain.
        _history.append({k: v for k, v in metrics.items() if k != "tech_per_class"})
        _score = _selection_score(metrics, args.select_on)
        metrics["selection_score"] = _score
        metrics["selection_metric"] = args.select_on
        if _score > best_loss:
            best_loss = _score
            best_metrics = metrics
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "epoch_history": list(_history),
                    # Written from the contract actually in force, not from
                    # literals. These were hardcoded to 2.0/5 regardless of what
                    # the run used, so the metadata could describe a model that
                    # was never trained.
                    "training_contract": {
                        "window_seconds": _c.window_seconds,
                        "window_size_sec": _c.window_seconds,  # v3 key, kept readable
                        "history_steps": _c.history_steps,
                        "forecast_steps": _c.forecast_steps,
                        "feature_dim": _cfg.state_dim,
                        "sources": [str(args.cic_dir), str(args.ctu_dir)],
                    },
                    "config": _cfg.to_dict(),
                    "manifest": _manifest.to_dict(),
                    "fingerprint": _manifest.fingerprint(),
                },
                args.output,
            )

    # Held-out test: scored once, on the restored best checkpoint, after
    # training is finished. This is the only number that is a generalisation
    _flag_outlier_selection(_history, best_metrics)

    # estimate rather than a selection artefact.
    ckpt = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loader = DataLoader(test_ds,
                             batch_size=args.batch_size, shuffle=False, **_loader_kw)
    test_metrics = _evaluate(model, test_loader, device)
    _per_class = test_metrics.pop("tech_per_class", {})
    print(f"HELD-OUT TEST (best epoch {ckpt.get('epoch')}): {test_metrics}", flush=True)
    _warn_if_head_collapsed(test_metrics, "held-out test")
    _print_per_class(_per_class, "held-out test")
    test_metrics["tech_per_class"] = _per_class
    # The credibility verdict travels WITH the checkpoint.
    #
    # It used to be computed, printed to stdout, and thrown away. The
    # checkpoint kept `test_metrics` but nothing recording that those numbers
    # might be meaningless, so a non-credible model could be loaded and served
    # with no trace -- and its metrics quoted as results. That is the same
    # "looks fine, means nothing" failure mode as a collapsed head scoring
    # 0.83 CatAcc, or a class with seven training samples.
    credibility = {"checked": False}
    try:
        from credibility_check import evaluate_store as _es, gate as _g, report as _r
        _ts = _es(train_store, test_store)
        _problems = _g(_ts, model_accuracy=test_metrics.get("tech_accuracy"))
        _r(_ts, _problems)
        credibility = {
            "checked": True,
            "credible": not _problems,
            "problems": list(_problems),
            "stats": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                      for k, v in _ts.items()},
        }
    except Exception as _e:
        credibility = {"checked": False, "error": str(_e)}
        print(f"test credibility check skipped: {_e}", flush=True)

    ckpt["test_metrics"] = test_metrics
    ckpt["credibility"] = credibility
    torch.save(ckpt, args.output)
    if credibility.get("checked") and not credibility.get("credible"):
        print("WARNING: checkpoint saved but marked NOT CREDIBLE -- "
              "its metrics must not be reported as results.", flush=True)

    print(f"saved={args.output} best_metrics={best_metrics} test_metrics={test_metrics}")


if __name__ == "__main__":
    main()