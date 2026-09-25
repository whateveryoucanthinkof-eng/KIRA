//! Batch-planner kernels (bita/fast/planner.py, bita/fast/plan_host.py).
//!
//! Both are functions of the data and of the message pool's HOST bookkeeping
//! only (which rows hold which node's pending messages, their peers and
//! times), never of model values, so the planner runs them ahead of the
//! trainer. Each reproduces a numpy function in bita/fast/plan_host.py bit for
//! bit (tests/test_batch_planner.py):
//!
//! * `tgn_bita_*`: the BiTA grouping of fast_tgn._fast_bita
//!   (plan_host.bita_group_numpy): the batch's nodes with pending messages
//!   (np.unique order), their messages grouped into edges (one per peer, in
//!   order of the peer's first message), each edge's messages sorted by time
//!   (stable), the last max_seq_len kept, left-aligned and zero-padded; the
//!   per-edge time deltas, lengths and last times; the per-node edge counts
//!   and latest message times.
//! * `tgn_nbr_plan`: the neighbour block of fast_tgn._fast_nbr
//!   (plan_host.nbr_host_numpy): most-recent-k temporal neighbours, their time
//!   deltas as float32, the attention padding mask with all-padding rows fixed
//!   up, and (host edge features) the neighbours' edge-feature rows.
//!
//! Floating point: only copies, one f64 subtraction, numpy's `maximum`
//! (NaN-propagating, first operand on ties), a multiplication by 1.0 or 0.0
//! and f64->f32 rounding (`as f32` rounds to nearest-even, as astype does).
//! Orderings use numpy's float sort order (NaN last) with the same tiebreaks.

use std::cmp::Ordering;

/// numpy.maximum(a, b) for float64: `a >= b || isnan(a) ? a : b`.
#[inline]
fn np_maximum(a: f64, b: f64) -> f64 {
    if a >= b || a.is_nan() {
        a
    } else {
        b
    }
}

/// numpy's float sort order: NaN after every number, numbers by `<`.
#[inline]
fn np_float_cmp(a: f64, b: f64) -> Ordering {
    if a < b || (b.is_nan() && !a.is_nan()) {
        Ordering::Less
    } else if b < a || (a.is_nan() && !b.is_nan()) {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

pub struct BitaPlan {
    to_update: Vec<i64>,
    node_ts: Vec<f64>,
    counts: Vec<f32>,
    owner: Vec<i64>,
    klen: Vec<i64>,
    last_t: Vec<f64>,
    /// kept messages of edge e are entries e_off[e]..e_off[e+1] of rows/times
    e_off: Vec<usize>,
    rows: Vec<i64>,
    times: Vec<f64>,
    l_max: i64,
    r_total: i64,
}

/// Group the pending messages of the distinct nodes in `nodes` (BiTA).
///
/// Writes [n_upd, R, E, L] to `sizes_out` and returns a handle for
/// `tgn_bita_fill` / `tgn_bita_free`, or null if a node id or a pool row is
/// out of range (the caller then runs the numpy version, which raises).
///
/// # Safety
/// cnt/start hold n_nodes items, row_peer/row_t n_rows, nodes n, sizes_out 4.
#[no_mangle]
pub unsafe extern "C" fn tgn_bita_plan(
    cnt: *const i64,
    start: *const i64,
    n_nodes: i64,
    row_peer: *const i64,
    row_t: *const f64,
    n_rows: i64,
    nodes: *const i64,
    n: i64,
    max_seq_len: i64,
    sizes_out: *mut i64,
) -> *mut BitaPlan {
    if max_seq_len < 1 {
        return std::ptr::null_mut();
    }
    let msl = max_seq_len as usize;
    let mut cand: Vec<i64> = std::slice::from_raw_parts(nodes, n as usize).to_vec();
    cand.sort_unstable();
    cand.dedup();
    let mut to_update = Vec::with_capacity(cand.len());
    for &v in &cand {
        if v < 0 || v >= n_nodes {
            return std::ptr::null_mut();
        }
        if *cnt.add(v as usize) > 0 {
            to_update.push(v);
        }
    }
    let n_upd = to_update.len();
    let mut plan = BitaPlan {
        node_ts: Vec::with_capacity(n_upd),
        counts: vec![0.0f32; n_upd],
        owner: Vec::new(),
        klen: Vec::new(),
        last_t: Vec::new(),
        e_off: vec![0],
        rows: Vec::new(),
        times: Vec::new(),
        l_max: 0,
        r_total: 0,
        to_update: Vec::new(),
    };
    let mut peer: Vec<i64> = Vec::new();
    let mut t: Vec<f64> = Vec::new();
    let mut ord: Vec<usize> = Vec::new();
    let mut first_ins: Vec<usize> = Vec::new();
    for (p, &u) in to_update.iter().enumerate() {
        let c = *cnt.add(u as usize);
        let s = *start.add(u as usize);
        if s < 0 || s + c > n_rows {
            return std::ptr::null_mut();
        }
        let c = c as usize;
        plan.r_total += c as i64;
        peer.clear();
        t.clear();
        for i in 0..c {
            peer.push(*row_peer.add(s as usize + i));
            t.push(*row_t.add(s as usize + i));
        }
        // Latest message time over ALL pending messages (np.maximum.reduceat).
        let mut acc = t[0];
        for &x in &t[1..] {
            acc = np_maximum(acc, x);
        }
        plan.node_ts.push(acc);
        // Each message's edge: the first insertion index among its peer's.
        ord.clear();
        ord.extend(0..c);
        ord.sort_unstable_by(|&a, &b| (peer[a], a).cmp(&(peer[b], b)));
        first_ins.clear();
        first_ins.resize(c, 0);
        let mut g0 = 0usize;
        for j in 0..c {
            if j > 0 && peer[ord[j]] != peer[ord[j - 1]] {
                g0 = j;
            }
            first_ins[ord[j]] = ord[g0];
        }
        // Edges in order of first appearance; inside, by time, then insertion.
        ord.sort_unstable_by(|&a, &b| {
            first_ins[a]
                .cmp(&first_ins[b])
                .then_with(|| np_float_cmp(t[a], t[b]))
                .then_with(|| a.cmp(&b))
        });
        let mut j = 0usize;
        while j < c {
            let e0 = j;
            while j < c && first_ins[ord[j]] == first_ins[ord[e0]] {
                j += 1;
            }
            let elen = j - e0;
            let drop = elen.saturating_sub(msl);
            let klen = elen - drop;
            for &m in &ord[e0 + drop..j] {
                plan.rows.push(s + m as i64);
                plan.times.push(t[m]);
            }
            plan.e_off.push(plan.rows.len());
            plan.owner.push(p as i64);
            plan.klen.push(klen as i64);
            plan.last_t.push(t[ord[j - 1]]);
            plan.counts[p] += 1.0;
            if klen as i64 > plan.l_max {
                plan.l_max = klen as i64;
            }
        }
    }
    plan.to_update = to_update;
    let e = plan.owner.len() as i64;
    *sizes_out = n_upd as i64;
    *sizes_out.add(1) = plan.r_total;
    *sizes_out.add(2) = e;
    *sizes_out.add(3) = plan.l_max;
    Box::into_raw(Box::new(plan))
}

/// Write a plan's arrays, padded for the caller's layout:
///   idx [e_rows, l_cols] i64, dt [e_rows, l_cols] f32 (rows >= E and
///   columns >= L zero; columns in [klen, L) hold numpy's padded-dt value),
///   klen [e_rows] i32 (0 past E), owner [e_rows] i64 (owner_fill past E),
///   last_t [E] f64, to_update [n_upd] i64, counts [c_rows] f32 (1.0 past
///   n_upd), node_ts [n_upd] f64. Returns 0, or -1 if a padded size is
///   smaller than the plan.
///
/// # Safety
/// Every pointer valid for the sizes above; `h` from tgn_bita_plan.
#[no_mangle]
pub unsafe extern "C" fn tgn_bita_fill(
    h: *const BitaPlan,
    e_rows: i64,
    l_cols: i64,
    c_rows: i64,
    owner_fill: i64,
    idx: *mut i64,
    dt: *mut f32,
    klen: *mut i32,
    owner: *mut i64,
    last_t: *mut f64,
    to_update: *mut i64,
    counts: *mut f32,
    node_ts: *mut f64,
) -> i32 {
    let p = &*h;
    let e = p.owner.len();
    let l = p.l_max as usize;
    let n_upd = p.to_update.len();
    if (e_rows as usize) < e || (l_cols as usize) < l || (c_rows as usize) < n_upd {
        return -1;
    }
    let (e_rows, l_cols, c_rows) = (e_rows as usize, l_cols as usize, c_rows as usize);
    for r in 0..e_rows {
        let row_idx = std::slice::from_raw_parts_mut(idx.add(r * l_cols), l_cols);
        let row_dt = std::slice::from_raw_parts_mut(dt.add(r * l_cols), l_cols);
        if r >= e {
            row_idx.fill(0);
            row_dt.fill(0.0);
            *klen.add(r) = 0;
            *owner.add(r) = owner_fill;
            continue;
        }
        let (a, b) = (p.e_off[r], p.e_off[r + 1]);
        let k = b - a;
        let last = p.last_t[r];
        for col in 0..l_cols {
            if col >= l {
                row_idx[col] = 0;
                row_dt[col] = 0.0;
                continue;
            }
            let valid = col < k;
            let (row, tt) = if valid { (p.rows[a + col], p.times[a + col]) } else { (0, 0.0) };
            row_idx[col] = row;
            // ((np.maximum(last_t[:, None] - times, 0.0)) * valid).astype(np.float32)
            let m = np_maximum(last - tt, 0.0);
            let v = m * if valid { 1.0 } else { 0.0 };
            row_dt[col] = v as f32;
        }
        *klen.add(r) = k as i32;
        *owner.add(r) = p.owner[r];
        *last_t.add(r) = last;
    }
    for i in 0..n_upd {
        *to_update.add(i) = p.to_update[i];
        *node_ts.add(i) = p.node_ts[i];
    }
    for i in 0..c_rows {
        *counts.add(i) = if i < n_upd { p.counts[i] } else { 1.0 };
    }
    0
}

/// # Safety
/// `h` from tgn_bita_plan, freed once.
#[no_mangle]
pub unsafe extern "C" fn tgn_bita_free(h: *mut BitaPlan) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// The neighbour block of one batch (fast_tgn._fast_nbr), written in place:
///   all_nodes [n + n*k] i64 (the batch nodes, then the neighbours),
///   eidx [n*k] i64, deltas [n*k] f32 = f32(cut - edge_time),
///   mask [n*k] u8 (neighbour == 0, all-padding rows unmask column 0),
///   invalid [n] u8 (row is all padding),
///   ef_n [n*k*de] f32 = host_ef rows of eidx (skipped when host_ef is null).
/// Returns 0; -1 for a node outside the offsets table, -2 for an edge index
/// outside the edge-feature table (the numpy version then decides).
///
/// # Safety
/// All pointers valid for the sizes above.
#[no_mangle]
pub unsafe extern "C" fn tgn_nbr_plan(
    flat_nbr: *const i32,
    flat_eidx: *const i32,
    flat_ts: *const f64,
    offsets: *const i64,
    n_offsets: i64,
    nodes: *const i64,
    cut: *const f64,
    n: i64,
    k: i64,
    host_ef: *const f32,
    n_ef_rows: i64,
    de: i64,
    all_nodes: *mut i64,
    eidx: *mut i64,
    deltas: *mut f32,
    mask: *mut u8,
    invalid: *mut u8,
    ef_n: *mut f32,
) -> i32 {
    let (n, k, de) = (n as usize, k as i64, de as usize);
    let ku = k as usize;
    for i in 0..n {
        let v = *nodes.add(i);
        if v < 0 || v + 1 >= n_offsets {
            return -1;
        }
        *all_nodes.add(i) = v;
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
        let mut all_pad = true;
        for col in 0..k {
            let src = stop - k + col;
            let o = i * ku + col as usize;
            let (nb, ei, et) = if src >= s && src < stop {
                (
                    *flat_nbr.add(src as usize),
                    *flat_eidx.add(src as usize),
                    *flat_ts.add(src as usize),
                )
            } else {
                (0i32, 0i32, 0.0f64)
            };
            *all_nodes.add(n + o) = nb as i64;
            *eidx.add(o) = ei as i64;
            *deltas.add(o) = (c - et) as f32;
            *mask.add(o) = (nb == 0) as u8;
            all_pad &= nb == 0;
            if !host_ef.is_null() {
                let r = ei as i64;
                if r < 0 || r >= n_ef_rows {
                    return -2;
                }
                std::ptr::copy_nonoverlapping(host_ef.add(r as usize * de), ef_n.add(o * de), de);
            }
        }
        *invalid.add(i) = all_pad as u8;
        if all_pad && ku > 0 {
            *mask.add(i * ku) = 0;
        }
    }
    0
}
