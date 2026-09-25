//! Host-side hot loops of the TGN training step (see bita/utils/tgn_host.py).
//!
//! `tgn_recent_neighbors` is NeighborFinder.get_temporal_neighbor's
//! most-recent-n path (uniform=False) over the CSR arrays: per row, a binary
//! search for the first interaction at or after the cut time inside the
//! node's slice (numpy's searchsorted side="left", which the vectorised numpy
//! version emulates), then the n most recent interactions right-aligned, zeros
//! where there are fewer. Pure integer/copy work: the output equals the numpy
//! version bit for bit (tests/test_tgn_host_rust.py).

/// Returns 0 on success, -1 if a node id is outside the offsets table.
///
/// # Safety
/// All pointers must be valid for the lengths given; out_* hold n*k items.
#[no_mangle]
pub unsafe extern "C" fn tgn_recent_neighbors(
    flat_nbr: *const i32,
    flat_eidx: *const i32,
    flat_ts: *const f64,
    offsets: *const i64,
    n_offsets: i64,
    nodes: *const i64,
    cut: *const f64,
    n: i64,
    k: i64,
    out_nbr: *mut i32,
    out_eidx: *mut i32,
    out_ts: *mut f64,
) -> i32 {
    let n = n as usize;
    let k = k as i64;
    for i in 0..n {
        let v = *nodes.add(i);
        if v < 0 || v + 1 >= n_offsets {
            return -1;
        }
        let s = *offsets.add(v as usize);
        let e = *offsets.add(v as usize + 1);
        let c = *cut.add(i);
        let (mut lo, mut hi) = (s, e);
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if *flat_ts.add(mid as usize) < c {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let stop = lo;
        let row = i * k as usize;
        for col in 0..k {
            let src = stop - k + col;
            let o = row + col as usize;
            if src >= s && src < stop {
                *out_nbr.add(o) = *flat_nbr.add(src as usize);
                *out_eidx.add(o) = *flat_eidx.add(src as usize);
                *out_ts.add(o) = *flat_ts.add(src as usize);
            } else {
                *out_nbr.add(o) = 0;
                *out_eidx.add(o) = 0;
                *out_ts.add(o) = 0.0;
            }
        }
    }
    0
}
