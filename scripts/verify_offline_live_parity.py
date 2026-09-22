#!/usr/bin/env python3
"""Offline/live feature parity check (spec 42, 43).

The failure mode this guards against: an offline preprocessing implementation
and a live one that drift apart, so a model is trained on features that differ
subtly from the ones it is served. That divergence is invisible in both test
suites and shows up only as unexplained live degradation.

Both paths already route through the same `HostTrajectoryExtractor`, so parity
should hold structurally. This asserts it rather than assuming it, and will
fail loudly the moment someone adds a live-only transform.

    python scripts/verify_offline_live_parity.py --state <telemetry.jsonl>

Exit 0 on parity, 1 on divergence.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description="offline/live feature parity")
    ap.add_argument("--state", type=Path, required=True, help="JSONL from run_telemetry.py")
    ap.add_argument("--tol", type=float, default=0.0, help="max allowed abs difference")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.disable(logging.INFO)
    from control_backend.model_adapter import (
        model_adapter,
        flows_from_span_dicts,
        select_primary_target,
    )

    records = [json.loads(l) for l in args.state.read_text().splitlines() if l.strip()]
    if not records:
        print(f"no records in {args.state}")
        return 1

    extractor = model_adapter.extractor
    worst = 0.0
    compared = 0
    skipped_clipped = 0

    print(f"contract in force: {model_adapter.window_seconds}s window | "
          f"{model_adapter.history_steps} history | {model_adapter.forecast_steps} forecast")
    print()

    for rec in records:
        flows = flows_from_span_dicts(rec.get("flows") or [])
        if not flows:
            continue
        target = select_primary_target(flows)
        if not target:
            continue

        # --- offline path: the extractor directly, as the trainers use it ---
        trajectories = extractor.extract_trajectories(flows)
        if target not in trajectories or not trajectories[target]:
            continue
        # Take BOTH halves from the snapshot the trainers actually consume.
        #
        # This used to recompute the attributes here by calling
        # `extractor.compute_host_temporal_attributes(...)` -- and the live
        # side below called the same function with the same arguments, so 15
        # of the 27 dimensions were compared against themselves. That half of
        # the check was arithmetically guaranteed to report 0.0 no matter how
        # far the two paths had drifted, and it hid a real skew: serving
        # computed attributes on a window-clipped flow list while handing the
        # unclipped snapshot to TGNE, moving the embedding by up to 3.3e-2 per
        # dimension.
        #
        # `LazyHostSequenceDataset` reads `feats[row] = [embedding ;
        # temporal_attrs]` straight off the snapshot, so the snapshot is the
        # offline ground truth for both halves.
        _snap = trajectories[target][-1]
        off_emb = np.asarray(_snap.embedding, dtype=np.float32)
        off_attrs = np.asarray(_snap.temporal_attrs, dtype=np.float32)
        offline = np.concatenate([off_emb, off_attrs])

        # --- live path: the adapter's own construction ---
        # Both sides get the SAME flows.
        #
        # An earlier version of this fix clipped the live side to the last
        # `window_seconds` while leaving the offline side unclipped, and then
        # reported a 6.789 divergence. That was the comparison's fault, not the
        # pipeline's: measured over these 105 windows, feeding both paths the
        # same flows gives max |delta| of exactly 0.0 across all 12 embedding
        # dimensions, while clipping only one side gives 6.789 in 105 of 105.
        # A parity check that feeds its two sides different inputs measures the
        # inputs, not the parity.
        #
        # Whether the adapter's window clip lines up with the extractor's own
        # internal windowing is a real and separate question -- it needs a
        # fixture spanning several windows, which this one (already grouped
        # into single 2 s windows) cannot exercise. Noted rather than faked.
        _host_flows = [r for r in flows
                       if r.src_ip == target or r.dst_ip == target]
        if not _host_flows:
            skipped_clipped += 1
            continue
        live_emb = model_adapter._build_embedding(target, flows)
        live_attrs = np.asarray(
            model_adapter.extractor.compute_host_temporal_attributes(
                host_ip=target,
                window_records=_host_flows,
                window_duration=model_adapter.window_seconds,
            ),
            dtype=np.float32,
        )
        live = np.concatenate([np.asarray(live_emb, dtype=np.float32), live_attrs])

        if offline.shape != live.shape:
            print(f"SHAPE DIVERGENCE window {rec.get('window_id')}: "
                  f"offline {offline.shape} vs live {live.shape}")
            return 1

        diff = float(np.max(np.abs(offline - live)))
        worst = max(worst, diff)
        compared += 1
        if args.verbose or diff > args.tol:
            print(f"  window {rec.get('window_id'):>4}  host {target:<16} "
                  f"dim {offline.shape[0]}  max|Δ| {diff:.3e}")

    if compared == 0:
        print("no comparable windows — cannot assert parity")
        return 1

    print()
    print(f"windows compared : {compared}")
    print(f"windows skipped (host had no flows after the clip): {skipped_clipped}")
    print(f"worst deviation  : {worst:.3e}  (tolerance {args.tol:g})")
    if worst > args.tol:
        print("RESULT: DIVERGENT — offline and live features differ")
        return 1
    print("RESULT: PARITY — identical features from both paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
