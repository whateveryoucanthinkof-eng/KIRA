"""Retrain Branch B and DeepOP on the canonical live TGNE latent space."""

import argparse
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
from deepop_decoder.train_cwa_decoder import CWASequenceDataset, create_cwa_training_samples
from cyberworld_v4.config import get_contract


def _strided(gen, stride: int, want: int):
    """Samples every Nth record across a wider read instead of a plain file-prefix
    (see scripts/retrain_branch_a_live.py::_strided for why -- D6, row-prefix sampling)."""
    out = []
    for i, r in enumerate(gen):
        if i % stride == 0:
            out.append(r)
            if len(out) >= want:
                break
    return out


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
            gen = (cic.parse_file(str(path), max_rows=rows_per_file * stride)
                   if path.suffix == ".csv"
                   else ctu.parse_netflow_csv(str(path), max_rows=rows_per_file * stride))
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


def train_branch_b_live(train_traj, val_traj, output, epochs, device):
    _c = get_contract()
    # Lazy: create_rollout_samples materialises h_history [15,12],
    # h_future [5,12] and risk_future [5] per sample -- 1,164 bytes each, and
    # ~38 GiB at the 35M samples full corpus density produces. The store
    # already holds every embedding in one memmapped block.
    train_ds = LazyHostRolloutDataset(train_traj, T=_c.history_steps, K=_c.forecast_steps)
    val_ds = LazyHostRolloutDataset(val_traj, T=_c.history_steps, K=_c.forecast_steps)
    print(f"Branch B samples: train={len(train_ds)} val={len(val_ds)}", flush=True)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    wdt = HostWorldDynamicsTransformer(d_latent=12, d_model=64, n_heads=4, n_layers=3).to(device)
    risk = InfiltrationRiskHead(d_latent=12, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(list(wdt.parameters()) + list(risk.parameters()), lr=1e-3, weight_decay=1e-4)
    best = float("inf")
    best_state = {k: v.detach().clone() for k, v in wdt.state_dict().items()}
    for epoch in range(epochs):
        wdt.train(); risk.train()
        for batch in train_loader:
            h = batch["h_history"].to(device); target = batch["h_future"].to(device); target_risk = batch["risk_future"].to(device)
            optimizer.zero_grad()
            pred = wdt.rollout(h, K=_c.forecast_steps)
            pred_risk, _ = risk.forward_trajectory(pred)
            loss = sum((0.9 ** k) * F.mse_loss(pred[:, k], target[:, k]) for k in range(_c.forecast_steps))
            loss = loss + F.binary_cross_entropy(pred_risk, target_risk)
            loss.backward(); optimizer.step()
        wdt.eval(); risk.eval(); losses=[]
        with torch.no_grad():
            for batch in val_loader:
                h=batch["h_history"].to(device); target=batch["h_future"].to(device); target_risk=batch["risk_future"].to(device)
                pred=wdt.rollout(h,K=_c.forecast_steps); pred_risk,_=risk.forward_trajectory(pred)
                losses.append((F.mse_loss(pred,target)+F.binary_cross_entropy(pred_risk,target_risk)).item())
        score=float(np.mean(losses)); print(f"Branch B epoch={epoch+1} val_loss={score:.4f}")
        if score < best:
            best=score; output.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"wdt_state_dict":wdt.state_dict(),"risk_head_state_dict":risk.state_dict(),"epoch":epoch+1,"history_steps":_c.history_steps,"forecast_steps":_c.forecast_steps,"window_seconds":_c.window_seconds},output)
            best_state={k:v.detach().clone() for k,v in wdt.state_dict().items()}
    wdt.load_state_dict(best_state)
    wdt.eval()
    return wdt


def train_deepop_live(train_traj, val_traj, output, epochs, device, wdt=None):
    vocab=get_joint_vocab(network_observable_only=True)
    _c=get_contract()
    # T>0 makes samples carry h_history so we can condition on Branch B's own
    # rollout instead of the oracle future (audit E1). At serve time DeepOP
    # only ever sees WDT output; training it on ground truth is a train/serve
    # mismatch that noise augmentation only approximates.
    _T=_c.history_steps if wdt is not None else 0
    train_samples=create_cwa_training_samples(train_traj,vocab,K=_c.forecast_steps,T=_T)
    val_samples=create_cwa_training_samples(val_traj,vocab,K=_c.forecast_steps,T=_T)
    print(f"DeepOP conditioning: {'Branch-B rollouts (E1 fixed)' if wdt is not None else 'oracle + noise (interim)'}",flush=True)
    train_loader=DataLoader(CWASequenceDataset(train_samples),batch_size=64,shuffle=True)
    val_loader=DataLoader(CWASequenceDataset(val_samples),batch_size=64)
    decoder=DeepOPForecastDecoder(d_latent=12,d_model=72,vocab_size=vocab.vocab_size,n_heads=6,num_layers=2,window_sizes=[2,4,8],dim_feedforward=144).to(device)
    optimizer=torch.optim.AdamW(decoder.parameters(),lr=5e-4,weight_decay=1e-4)
    best=float("inf")
    for epoch in range(epochs):
        decoder.train(); train_losses=[]
        for batch in train_loader:
            h=batch["h_future"].to(device); inp=batch["input_tokens"].to(device); tgt=batch["target_tokens"].to(device)
            if wdt is not None and "h_history" in batch:
                # Condition on what Branch B actually predicts, which is what
                # DeepOP receives in production.
                with torch.no_grad():
                    h_aug=wdt.rollout(batch["h_history"].to(device), K=h.shape[1]).detach()
            else:
                step_sigma=torch.linspace(0.015,0.055,steps=h.shape[1],device=device).unsqueeze(0).unsqueeze(-1)
                h_aug=h+torch.randn_like(h)*step_sigma
            optimizer.zero_grad(); logits=decoder(h_aug,inp); loss=F.cross_entropy(logits.reshape(-1,vocab.vocab_size),tgt.reshape(-1),label_smoothing=0.04); loss.backward(); torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.0); optimizer.step(); train_losses.append(loss.item())
        decoder.eval(); val_losses=[]
        with torch.no_grad():
            for batch in val_loader:
                hv=batch["h_future"].to(device)
                if wdt is not None and "h_history" in batch:
                    hv=wdt.rollout(batch["h_history"].to(device), K=hv.shape[1]).detach()
                logits=decoder(hv,batch["input_tokens"].to(device)); val_losses.append(F.cross_entropy(logits.reshape(-1,vocab.vocab_size),batch["target_tokens"].to(device).reshape(-1)).item())
        score=float(np.mean(val_losses)); print(f"DeepOP epoch={epoch+1} train_loss={np.mean(train_losses):.4f} val_loss={score:.4f}")
        if score < best:
            best=score; output.parent.mkdir(parents=True,exist_ok=True); torch.save({"decoder_state_dict":decoder.state_dict(),"epoch":epoch+1,"history_steps":_c.history_steps,"forecast_steps":_c.forecast_steps,"window_seconds":_c.window_seconds,"vocab_size":vocab.vocab_size},output)


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
        train_records, val_records = load_records(args.cic_dir, args.ctu_dir, args.rows_per_file, args.stride)
        print(f"loaded {len(train_records)} train + {len(val_records)} val records in {time.time()-t0:.1f}s", flush=True)
        t0=time.time(); train_traj = extractor.extract_trajectories(train_records)
        print(f"train trajectories extracted in {time.time()-t0:.1f}s ({len(train_traj)} hosts)", flush=True)
        t0=time.time(); val_traj = extractor.extract_trajectories(val_records)
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
    args.out_dir.mkdir(parents=True, exist_ok=True)
    wdt = train_branch_b_live(train_traj,val_traj,args.out_dir/"host_wdt.canonical-tgne.pt",args.epochs,device)
    train_deepop_live(train_traj,val_traj,args.out_dir/"cwa_forecast_decoder.canonical-tgne.pt",args.epochs,device,wdt=wdt)


if __name__ == "__main__":
    main()
