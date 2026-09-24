"""Full-density trajectory extraction with bounded memory.

`HostTrajectoryExtractor.extract_trajectories` needs its records as a list: it
sorts them, windows them, and hands each window's records to the temporal
attribute and behavioural-fingerprint code. At full corpus density that list is
~37M UnifiedFlowRecord objects (~529 B each, measured) = ~19 GB, which is the
remaining bottleneck after the snapshot store went columnar.

This processes the stream in time-ordered chunks instead, freeing each chunk
before the next is read, while every chunk appends into one shared
TrajectoryStoreBuilder. Peak memory becomes one chunk plus the (memmapped)
store rather than the whole corpus.

The chunks overlap by `history_seconds`, and snapshots whose window starts
inside that overlap are dropped because the previous chunk already emitted
them. The overlap is what makes this exact rather than approximate: TGNE is a
temporal graph encoder, so a host's embedding depends on the neighbour history
preceding its window. Replaying the trailing history into the next chunk gives
the encoder the same context it would have had processing the corpus in one
pass; without it, every chunk boundary would silently truncate that history.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator, List, Optional

from data_unification.trajectory_store import TrajectoryStore, TrajectoryStoreBuilder
from data_unification.unified_schema import UnifiedFlowRecord


def extract_trajectories_chunked(
    extractor,
    records: Iterable[UnifiedFlowRecord],
    *,
    chunk_size: int = 2_000_000,
    history_seconds: float = 30.0,
    progress: Optional[Callable[[str], None]] = None,
) -> TrajectoryStore:
    """Chunked, memory-bounded equivalent of extractor.extract_trajectories(list)."""
    builder = TrajectoryStoreBuilder(spill_dir=getattr(extractor, "spill_dir", None))
    carry: List[UnifiedFlowRecord] = []
    buf: List[UnifiedFlowRecord] = []
    chunk_no = 0
    total = 0
    window_idx_base = 0

    def _run(batch: List[UnifiedFlowRecord], emit_after: Optional[float]) -> None:
        nonlocal chunk_no, window_idx_base
        if not batch:
            return
        chunk_no += 1
        extractor.extract_trajectories(
            batch, builder=builder, emit_after=emit_after, window_idx_base=window_idx_base
        )
        # window_idx only ever feeds `sorted(snaps, key=...)` in the three
        # consumers -- nothing does arithmetic on it or requires consecutive
        # values -- so it just has to stay monotonic in time across chunks.
        # Deriving the next base from what was actually emitted is exact,
        # unlike estimating it from the chunk's time span.
        # _window_idx is a growable numpy column (_Col), not a list: indexing
        # it directly raised TypeError, so any run past its first chunk died.
        col = builder._window_idx
        if col.n:
            tail = col.buf[max(0, col.n - len(batch) * 4):col.n]
            window_idx_base = int(tail.max()) + 1
        if progress:
            progress(f"  chunk {chunk_no}: {len(batch)} records "
                     f"(cumulative {total}), store now {builder._n} snapshots")

    for rec in records:
        buf.append(rec)
        total += 1
        if len(buf) >= chunk_size:
            batch = carry + buf
            batch.sort(key=lambda r: r.start_time)
            cutoff = batch[-1].start_time - history_seconds
            # everything before this chunk's own start was already emitted
            emit_after = carry[0].start_time + history_seconds if carry else None
            _run(batch, emit_after)
            carry = [r for r in batch if r.start_time >= cutoff]
            buf = []
            del batch

    if buf or carry:
        batch = carry + buf
        batch.sort(key=lambda r: r.start_time)
        emit_after = carry[0].start_time + history_seconds if carry else None
        _run(batch, emit_after)

    return builder.finalize()
