import os
import sys
import time
import math
import glob
import logging
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.extentedtgn import ExtendedTGN
from utils.utils import RandEdgeSampler, get_neighbor_finder, EarlyStopMonitor
from evaluation.eval_edge_prediction_with_categories import eval_edge_prediction_with_categories
from data_unification.tgne_features import (
    EDGE_FEATURE_NAMES,
    SCHEMA_VERSION,
    ablated_edge_features as _ablated_edge_features,
    extract_canonical_edge_features,
)
from data_unification.density import require_full_density
from data_unification.ip_features import ablated_node_features as _ablated_node_features


class FocalLoss(nn.Module):
    """Multi-class focal loss with per-class alpha.

    The previous implementation applied *binary* focal-loss weighting to a
    6-class problem:

        at = ones_like(targets) * (1 - alpha)   # 0.75 for every class
        at[targets == 1] = alpha                # 0.25 for class index 1 only

    With classes {0: Benign, 1: C2, 2: Impact, 3: InitialAccess, 4: Recon,
    5: UNKNOWN} that up-weights Benign -- the majority class -- to 0.75 while
    down-weighting C2 to 0.25, so the loss rewarded predicting Benign for
    everything. Measured result: per-class accuracy {0: 1.0, 1..5: 0.0} and
    macro F1 0.158 on the test split, i.e. a collapsed head.

    alpha is now a per-class weight vector (inverse class frequency by
    default), which is what focal loss means for more than two classes.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean', num_classes=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            a = torch.as_tensor(alpha, dtype=torch.float)
            if a.ndim == 0:
                if num_classes is None:
                    raise ValueError("scalar alpha needs num_classes")
                a = torch.full((num_classes,), float(a))
            self.register_buffer("alpha", a)

    @staticmethod
    def inverse_frequency_alpha(labels, num_classes: int, *, power: float = 0.5,
                                clip: tuple = (0.2, 5.0)) -> torch.Tensor:
        """Damped, clipped inverse-frequency per-class weights.

        Plain 1/frequency is unstable on this corpus. Classes absent (or nearly
        absent) from a split get astronomically large raw weights, and after
        normalising by the mean every other class collapses to ~0 -- measured
        alpha was {Benign: 0.0, C2: 0.0, Impact: 2.0, ...}, i.e. the majority
        classes contributed no loss at all. That is the original collapse
        inverted, not fixed.

        So: weights are computed over *present* classes only (an absent class
        is never a target, so its weight is irrelevant and must not skew the
        normalisation), damped by `power` (sqrt by default) to keep the ratio
        sane, and clipped to a bounded range.
        """
        counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
        w = np.ones(num_classes, dtype=np.float64)
        present = counts > 0
        if present.any():
            inv = (counts[present].sum() / counts[present]) ** power
            # Normalise by the GEOMETRIC mean, not the arithmetic mean.
            #
            # These are multiplicative weights, so the arithmetic mean is the
            # wrong centre: one ultra-rare class dominates it and crushes every
            # other weight to the clip floor. Measured on the full corpus, the
            # arithmetic version returned
            #   {Benign: 0.2, C2: 0.2, Impact: 0.2, InitialAccess: 0.2, Recon: 4.911}
            # -- four of five classes pinned at the floor and therefore
            # weighted IDENTICALLY, i.e. no class balancing at all among the
            # four that carry the data. Recon is ~5 orders of magnitude rarer
            # than Benign, and that single outlier set the scale for everyone.
            #
            # The geometric mean centres the weights in log space, which is
            # where a multiplicative correction belongs, so a single extreme
            # class shifts the others by a bounded factor instead of collapsing
            # them.
            w[present] = inv / float(np.exp(np.mean(np.log(inv))))
        w_unclipped = w.copy()
        w = np.clip(w, clip[0], clip[1])

        # Report what the weights were actually built from. The class counts
        # are the single most diagnostic number here: a weight of exactly 1.0
        # means a class had ZERO training samples, and that is how a broken
        # split was found (3 of 5 categories were absent because the temporal
        # cut was splitting by corpus).
        logging.info(
            "focal alpha: counts=%s -> weights=%s",
            {i: int(c) for i, c in enumerate(counts)},
            {i: round(float(x), 3) for i, x in enumerate(w)},
        )
        n_pinned = int(np.sum((w_unclipped < clip[0]) | (w_unclipped > clip[1])))
        if n_pinned > num_classes // 2:
            logging.warning(
                "focal alpha: %d of %d classes are pinned at a clip bound %s. "
                "Pinned classes are weighted identically, so the loss is doing "
                "no balancing between them. Class frequencies span %.0fx.",
                n_pinned, num_classes, clip,
                float(counts[present].max() / max(1.0, counts[present].min())),
            )
        return torch.as_tensor(w, dtype=torch.float)

    def forward(self, inputs, targets):
        logpt = -F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.alpha is not None:
            at = self.alpha.to(inputs.device)[targets]
            loss = loss * at
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


def _contract_window_seconds() -> float:
    from cyberworld_v4.config import get_contract
    return get_contract().window_seconds


def _is_memory_state(key: str) -> bool:
    """TGN memory rows are per-node runtime state, sized to one graph's nodes."""
    return key.endswith("memory.memory") or key.endswith("memory.last_update")


def _init_from_checkpoint(tgn, path, device):
    """Start from another encoder's weights (e.g. Warden -> fine-tune on CIC-2018).

    Everything whose shape matches is copied: the graph attention, time
    encoding, message function, BiTA aggregator and memory updater. Skipped:
    per-node memory state (a different graph's nodes) and any head whose size
    depends on the label set (the category head: Warden alert categories are
    not CIC's coarse categories).
    """
    src = torch.load(path, map_location=device)
    own = tgn.state_dict()
    take, skipped = {}, []
    for k, v in src.items():
        if _is_memory_state(k) or k not in own or own[k].shape != v.shape:
            skipped.append(k)
            continue
        take[k] = v
    tgn.load_state_dict(take, strict=False)
    logging.info("init_from %s: loaded %d tensors, skipped %d (%s)", path, len(take),
                 len(skipped), ", ".join(sorted({k.split(".")[0] for k in skipped})) or "none")


def compute_time_statistics(sources, destinations, timestamps):
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources:
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst:
            last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp

    mean_time_shift_src = float(np.mean(all_timediffs_src))
    std_time_shift_src = float(np.std(all_timediffs_src)) if np.std(all_timediffs_src) > 0 else 1.0
    mean_time_shift_dst = float(np.mean(all_timediffs_dst))
    std_time_shift_dst = float(np.std(all_timediffs_dst)) if np.std(all_timediffs_dst) > 0 else 1.0

    return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst


def median_resample(df, label_col="Category", time_col="DetectTime", seed=0):
    """BiTA's Warden preprocessing: align every class to the MEDIAN class size.

    Section 5 of the paper: random undersampling of classes above the median,
    random oversampling (with replacement) of classes below it, each class
    independently, then re-sorted chronologically BEFORE the temporal split,
    so the split itself stays leakage-free.
    """
    counts = df[label_col].value_counts()
    target = int(np.median(counts.values))
    rng = np.random.RandomState(seed)
    parts = []
    for cat, n in counts.items():
        g = df[df[label_col] == cat]
        parts.append(g.iloc[rng.choice(len(g), size=target, replace=bool(n < target))])
    return (pd.concat(parts)
            .sort_values(time_col, kind="stable")
            .reset_index(drop=True))


def load_and_preprocess_dataset(dataset_dir="Dataset", embedding_dim=4, resample_to_median=True,
                                ip_node_features=True):
    csv_files = sorted(glob.glob(os.path.join(dataset_dir, "*March_e.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    logging.info(f"Loading {len(csv_files)} CSV files from {dataset_dir}...")
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    df['DetectTime'] = pd.to_datetime(df['DetectTime'], format='ISO8601', utc=True)
    df = df.sort_values('DetectTime').reset_index(drop=True)

    df['SourceIP'] = df['SourceIP'].astype(str)
    df['TargetIP'] = df['TargetIP'].astype(str)
    df['Proto'] = df['Proto'].astype(str)
    df['Port'] = df['Port'].astype(str)
    df['Category'] = df['Category'].astype(str)
    df['FlowCount'] = df['FlowCount'].fillna(0)
    if resample_to_median:
        before = df['Category'].value_counts().to_dict()
        df = median_resample(df)
        logging.info("BiTA median resampling: %s -> %d per class", before,
                     int(df['Category'].value_counts().iloc[0]))

    proto_encoder = LabelEncoder()
    attack_type_encoder = LabelEncoder()
    port_type_encoder = LabelEncoder()
    flow_count_encoder = LabelEncoder()

    df['Proto_encoded'] = proto_encoder.fit_transform(df['Proto'])
    df['Category_encoded'] = attack_type_encoder.fit_transform(df['Category'])
    df['Port_encoded'] = port_type_encoder.fit_transform(df['Port'])
    df['FlowCount_encoded'] = flow_count_encoder.fit_transform(df['FlowCount'].astype(str))

    category_mapping = {index: label for index, label in enumerate(attack_type_encoder.classes_)}
    logging.info(f"Detected alert categories: {category_mapping}")

    df['SourceIP_encoded'], src_ips = pd.factorize(df['SourceIP'])
    df['TargetIP_encoded'], dst_ips = pd.factorize(df['TargetIP'])

    u_list = df['SourceIP_encoded'].values
    i_list = df['TargetIP_encoded'].values
    ts_list = df['DetectTime'].apply(lambda x: x.timestamp()).values
    label_list = df['Category_encoded'].values
    idx_list = np.arange(len(df))

    # Bipartite reindexing (1-based, disjoint node ids)
    upper_u = u_list.max() + 1
    new_i = i_list + upper_u
    new_u = u_list + 1
    new_i = new_i + 1
    new_idx = idx_list + 1

    max_node_idx = max(new_u.max(), new_i.max())
    total_nodes = max_node_idx + 1

    graph_df = pd.DataFrame({
        'u': new_u,
        'i': new_i,
        'ts': ts_list,
        'label': label_list,
        'idx': new_idx
    })

    # Canonical deterministic edge features. The old implementation used
    # unsaved random embedding tables, so live inference could not reproduce
    # the TGNE input coordinate system.
    proto_values = df['Proto'].str.lower().map({'tcp': 6, 'udp': 17, 'icmp': 1}).fillna(6)
    port_values = pd.to_numeric(df['Port'], errors='coerce').fillna(0)
    flow_counts = pd.to_numeric(df['FlowCount'], errors='coerce').fillna(0)
    raw_edge_features = np.stack(
        [
            extract_canonical_edge_features(
                fwd_bytes=float(flow_count),
                bwd_bytes=0.0,
                fwd_packets=float(flow_count),
                bwd_packets=0.0,
                duration_sec=0.0,
                byte_rate=float(flow_count),
                packet_rate=float(flow_count),
                protocol=int(proto),
                dst_port=int(port),
            )
            for flow_count, proto, port in zip(flow_counts, proto_values, port_values)
        ],
        axis=0,
    ).astype(np.float32)

    # Prepend zero padding for index 0
    empty_edge = np.zeros((1, raw_edge_features.shape[1]), dtype=np.float32)
    edge_features = np.vstack([empty_edge, raw_edge_features])

    # Node features: the same intrinsic 12-D IP features the CIC/PCAP path and
    # serving use (data_unification/ip_features.py), instead of zeros. An
    # encoder trained here is applied to CIC traffic, whose hosts all carry
    # these features; training it on all-zero rows would hand it an input
    # distribution at extraction time it never saw. Bipartite ids: attacker
    # k -> k + 1, victim k -> upper_u + k + 1 (the same IP on both sides gets
    # two nodes with identical features).
    node_feat_dim = edge_features.shape[1]
    if ip_node_features:
        from data_unification.ip_features import build_node_feature_matrix
        node_features = (
            build_node_feature_matrix({ip: k + 1 for k, ip in enumerate(src_ips)}, total_nodes)
            + build_node_feature_matrix({ip: int(upper_u) + k + 1 for k, ip in enumerate(dst_ips)},
                                        total_nodes))
    else:
        node_features = np.zeros((total_nodes, node_feat_dim), dtype=np.float32)

    return graph_df, edge_features, node_features, category_mapping


def load_and_preprocess_unified_dataset(
    cic2017_dir=None,
    cic2018_dir=None,
    ctu13_dir=None,
    max_rows_per_file=None,
    stride=1,
    splits=("train",),
    parallel_workers=0,
    pcap2018_root=None,
    pcap2018_label_dir=None,
    scheme="frozen",
    allow_cic2018_csv=False,
    window_seconds=2.0,
):
    """Loads CIC-2017 + CIC-2018 + CTU-13 into TGN's (u, i, ts, label, idx) graph format.

    Unlike load_and_preprocess_dataset (Warden), node ids share ONE namespace across
    source and destination: a host seen as both a source and a destination gets the
    same node id. Warden's loader instead builds a bipartite graph (disjoint id ranges
    for source vs. destination), which does not match CIC/CTU/live traffic, where any
    host can be both. Edge features come from each record's real bidirectional
    bytes/packets/duration via extract_canonical_edge_features -- not the flow-count
    proxy Warden's loader uses, since these datasets carry real flow statistics.

    `splits` restricts which frozen captures are read. It defaults to
    ("train",) and that default is load-bearing.

    TGNE is not purely self-supervised: alongside link prediction it trains a
    category head on each edge's `coarse_category`. Training it over the whole
    corpus therefore pushed LABEL information from the captures the frozen lock
    reserves for validation and test into the encoder -- and Branch A, Branch B
    and DeepOP all consume its embeddings. Every downstream "held-out" number
    would have been contaminated at the encoder, where it is invisible.

    Restricting to the frozen train captures keeps val and test captures
    entirely unseen by the encoder. TGN's own 70/85 temporal split then runs
    *within* the training captures, which is what it is for.
    """
    from data_unification.cic2017_adapter import CIC2017Adapter
    from data_unification.cic2018_adapter import CIC2018Adapter
    from data_unification.ctu13_adapter import CTU13Adapter
    from data_unification.ip_features import build_node_feature_matrix
    from data_unification.split_policy import split_of_path

    # Stride is applied DURING ingestion, per file, not after.
    #
    # Two defects this fixes:
    #
    # 1. Peak memory. `records[::stride]` after loading everything still holds
    #    the full corpus in RAM first, so striding saved nothing at the peak --
    #    exactly the "measure the peak, not the final state" failure that once
    #    made a builder look like 151 B/snapshot while it hoarded tens of GB.
    #    Striding as records arrive means peak == the strided size.
    #
    # 2. Label bias. `max_rows_per_file` takes a chronological PREFIX, and these
    #    captures are benign in the morning with attacks later in the day.
    #    Measured on the frozen val split, a 2000-row prefix per capture gave
    #    0% attack. Striding within each file spans the whole day instead, so
    #    `max_rows_per_file` is now a cap on records KEPT after striding rather
    #    than a cap on rows read.
    def _take(stream):
        """Every `stride`-th record of one file, at most `max_rows_per_file`."""
        kept = 0
        for n, rec in enumerate(stream):
            if stride > 1 and (n % stride):
                continue
            yield rec
            kept += 1
            if max_rows_per_file is not None and kept >= max_rows_per_file:
                return

    wanted = set(splits) if splits else None
    from data_unification.split_policy import is_cross_year
    if is_cross_year(scheme) and cic2018_dir and not allow_cic2018_csv:
        raise ValueError(
            "cross_year trains on CIC-2018, and 9 of its 10 CSV days fabricate host IPs "
            "from the row number; an encoder is a model of the host graph, so train it on "
            "--pcap2018_root instead (or pass --allow_cic2018_csv knowingly).")

    def _in_split(path):
        """True when the split scheme assigns `path` to one of `splits`."""
        try:
            if str(path).endswith(("_pcap", "_pacap")):
                from data_unification.split_policy import split_of
                sp = split_of("PCAP2018", os.path.basename(str(path)), scheme)
            else:
                sp = split_of_path(path, scheme)
        except (KeyError, ValueError):
            logging.warning("%s is not in the frozen split -- skipped", path)
            return False
        if sp is None:              # this scheme does not use the corpus at all
            return False
        return wanted is None or sp in wanted

    # ---------------------------------------------------------------- columnar
    #
    # Records are consumed as they stream and never accumulated.
    #
    # The previous version built a Python list of every record just to turn it
    # into fixed-width numeric columns. At ~8M records that list alone is
    # ~2.4 GB, and it coexisted with the sort, four more list comprehensions
    # and the edge-feature pass -- peak reached 14.3 GiB and the job was
    # OOM-killed by its own cgroup. Nothing in the output needs the objects:
    # every column here is a scalar derived from one record.
    #
    # Columns grow by doubling, which keeps peak at 2x the live size during a
    # resize rather than holding a second full structure.
    _probe = extract_canonical_edge_features(
        fwd_bytes=0, bwd_bytes=0, fwd_packets=0, bwd_packets=0, duration_sec=0.0,
        byte_rate=0.0, packet_rate=0.0, protocol=6, dst_port=0,
    )
    edge_dim = len(_probe)

    class _Growable:
        """Append-only numpy column that doubles in place."""

        __slots__ = ("buf", "n")

        def __init__(self, dtype, width=None, cap=1 << 20):
            shape = (cap,) if width is None else (cap, width)
            self.buf = np.empty(shape, dtype=dtype)
            self.n = 0

        def append(self, v):
            if self.n == len(self.buf):
                bigger = np.empty((len(self.buf) * 2,) + self.buf.shape[1:], self.buf.dtype)
                bigger[: self.n] = self.buf[: self.n]
                self.buf = bigger
            self.buf[self.n] = v
            self.n += 1

        def extend(self, arr):
            """Bulk append. The parallel path adds a whole capture at once,
            and per-element append() would put the Python loop straight back."""
            arr = np.asarray(arr)
            need = self.n + len(arr)
            if need > len(self.buf):
                cap = len(self.buf)
                while cap < need:
                    cap *= 2
                bigger = np.empty((cap,) + self.buf.shape[1:], self.buf.dtype)
                bigger[: self.n] = self.buf[: self.n]
                self.buf = bigger
            self.buf[self.n:need] = arr
            self.n = need

        def done(self):
            return self.buf[: self.n]

    col_u = _Growable(np.int64)
    col_i = _Growable(np.int64)
    col_ts = _Growable(np.float64)
    col_lbl = _Growable(np.int32)
    col_edge = _Growable(np.float32, width=edge_dim)
    # Which corpus each edge came from. The three occupy DISJOINT absolute
    # time ranges (CTU-13 2011, CIC-2017 2017, CIC-2018 2018), so a global
    # temporal split silently becomes a split BY CORPUS. See split_data().
    col_src = _Growable(np.int8)
    # Capture id. The temporal cut is taken per CAPTURE, not per corpus:
    # CIC-2017 and CIC-2018 organise attack types BY DAY, so a per-corpus
    # cut segregates whole attack classes into the future half. Measured:
    # Recon had 7 training samples out of 11.8M because CIC-2017's 70%
    # cut fell at 13:13 and all 158,930 PortScan rows run 13:00-15:59.
    col_cap = _Growable(np.int16)

    ip_to_id: dict = {}
    cat_to_id: dict = {}
    src_to_id: dict = {}
    cap_to_id: dict = {}
    _cur_cap = [0]

    def _consume(stream):
        for r in stream:
            su = ip_to_id.get(r.src_ip)
            if su is None:
                su = ip_to_id[r.src_ip] = len(ip_to_id) + 1   # 1-based; 0 is padding
            di = ip_to_id.get(r.dst_ip)
            if di is None:
                di = ip_to_id[r.dst_ip] = len(ip_to_id) + 1
            ci = cat_to_id.get(r.coarse_category)
            if ci is None:
                ci = cat_to_id[r.coarse_category] = len(cat_to_id)
            si = src_to_id.get(r.raw_label_source)
            if si is None:
                si = src_to_id[r.raw_label_source] = len(src_to_id)
            col_src.append(si)
            col_cap.append(_cur_cap[0])
            col_u.append(su)
            col_i.append(di)
            col_ts.append(r.start_time)
            col_lbl.append(ci)
            col_edge.append(extract_canonical_edge_features(
                fwd_bytes=r.fwd_bytes, bwd_bytes=r.bwd_bytes,
                fwd_packets=r.fwd_packets, bwd_packets=r.bwd_packets,
                duration_sec=r.duration, byte_rate=r.byte_rate,
                packet_rate=r.packet_rate, protocol=r.protocol,
                dst_port=r.dst_port,
            ))

    # The capture list, in the FIXED order serial parsing would visit. Node ids
    # are assigned in encounter order, so this order is what makes the run
    # reproducible -- parallel parsing must merge in exactly this sequence.
    captures = []
    if cic2017_dir:
        captures += [("CIC2017", f) for f in sorted(glob.glob(os.path.join(cic2017_dir, "*.csv")))]
    if cic2018_dir:
        captures += [("CIC2018", f) for f in sorted(glob.glob(os.path.join(cic2018_dir, "*.csv")))]
    if ctu13_dir:
        captures += [("CTU13", f) for f in sorted(glob.glob(os.path.join(ctu13_dir, "*", "*.binetflow")))]
    if pcap2018_root:
        # Real host addresses; labels from the paired CSVs by timestamp
        # (data_unification/training_sources.py).
        if not pcap2018_label_dir:
            raise ValueError("--pcap2018_root needs --pcap2018_label_dir (PCAPs carry no labels)")
        captures += [("PCAP2018", os.path.join(pcap2018_root, d))
                     for d in sorted(os.listdir(pcap2018_root))
                     if os.path.isdir(os.path.join(pcap2018_root, d)) and d.endswith(("_pcap", "_pacap"))]

    skipped = [os.path.basename(f) for _k, f in captures if not _in_split(f)]
    captures = [(k, f) for k, f in captures if _in_split(f)]
    n_read = len(captures)

    if parallel_workers and any(k == "PCAP2018" for k, _f in captures):
        logging.info("PCAP captures are read serially (the parallel parser handles CSVs only)")
        parallel_workers = 0
    if parallel_workers and len(captures) > 1:
        # Parallel path. Workers return LOCAL vocabularies; the merge below
        # assigns global ids in capture order, so the result is identical to
        # the serial path. See data_unification/parallel_ingest.py.
        from data_unification.parallel_ingest import parse_captures_parallel
        import time as _t
        _t0 = _t.time()
        for res in parse_captures_parallel(
            captures, stride=stride, max_rows_per_file=max_rows_per_file,
            edge_dim=edge_dim, workers=parallel_workers,
        ):
            # local ip id -> global node id, in this capture's own order
            ip_map = np.empty(len(res["ips"]), dtype=np.int64)
            for local, ip in enumerate(res["ips"]):
                gid = ip_to_id.get(ip)
                if gid is None:
                    gid = ip_to_id[ip] = len(ip_to_id) + 1
                ip_map[local] = gid
            cat_map = np.empty(len(res["cats"]), dtype=np.int32)
            for local, c in enumerate(res["cats"]):
                cid = cat_to_id.get(c)
                if cid is None:
                    cid = cat_to_id[c] = len(cat_to_id)
                cat_map[local] = cid
            si = src_to_id.get(res["kind"])
            if si is None:
                si = src_to_id[res["kind"]] = len(src_to_id)

            col_u.extend(ip_map[res["u"]])
            col_i.extend(ip_map[res["i"]])
            col_ts.extend(res["ts"])
            col_lbl.extend(cat_map[res["lbl"]])
            col_edge.extend(res["edge"])
            col_src.extend(np.full(res["n"], si, dtype=np.int8))
            _cid = cap_to_id.setdefault(res["path"], len(cap_to_id))
            col_cap.extend(np.full(res["n"], _cid, dtype=np.int16))
            logging.info("  %s: %d records", os.path.basename(res["path"]), res["n"])
        logging.info("Parallel ingest finished in %.1fs", _t.time() - _t0)
    else:
        for kind, f in captures:
            _cur_cap[0] = cap_to_id.setdefault(f, len(cap_to_id))
            if kind == "CIC2017":
                _consume(_take(CIC2017Adapter().parse_file(f, max_rows=None)))
            elif kind == "CIC2018":
                _consume(_take(CIC2018Adapter().parse_file(f, max_rows=None)))
            elif kind == "PCAP2018":
                from data_unification.training_sources import iter_pcap_day_windows
                _consume(_take(r for window in iter_pcap_day_windows(
                    f, pcap2018_label_dir, window_seconds) for r in window))
            else:
                _consume(_take(CTU13Adapter().parse_netflow_csv(f, max_rows=None)))

    # Label-mapping coverage, reported beside the data it describes.
    #
    # LabelResolver tracks how many label strings it could not map -- those
    # become UNKNOWN with is_attack=False, which is the right call (it asserts
    # nothing) but is silent label noise if the rate is material. The tracker
    # existed and had ZERO callers, so the number was computed and discarded,
    # the same pattern as the credibility verdict and the two loss terms.
    #
    # Measured 0.0% across 450k records from all three corpora today, so this
    # is a safeguard rather than a fix -- it matters when new data arrives with
    # label strings the maps have not seen.
    try:
        from data_unification.label_resolver import get_default_resolver
        _cov = get_default_resolver().unresolved_report()
        if _cov["unresolved_calls"]:
            logging.warning(
                "Label coverage: %.4f%% UNRESOLVED (%d of %d). Unmapped labels "
                "become UNKNOWN/is_attack=False -- silent label noise. First few: %s",
                100.0 * _cov["unresolved_rate"], _cov["unresolved_calls"],
                _cov["resolve_calls"], _cov["distinct_unresolved_labels"][:10],
            )
        else:
            logging.info("Label coverage: 100%% of %d labels mapped",
                         _cov["resolve_calls"])
    except Exception as _e:
        logging.warning("label coverage unavailable: %s", _e)

    logging.info(
        "Frozen split %s: %d capture(s) read, %d held out from the encoder%s",
        ",".join(sorted(wanted)) if wanted else "ALL",
        n_read,
        len(skipped),
        (": " + ", ".join(skipped)) if skipped else "",
    )

    n = col_ts.n
    if n == 0:
        raise FileNotFoundError(
            f"No records loaded (cic2017_dir={cic2017_dir}, cic2018_dir={cic2018_dir}, ctu13_dir={ctu13_dir})"
        )

    # Chronological order, by permuting the columns rather than sorting objects.
    order = np.argsort(col_ts.done(), kind="stable")
    u_list = col_u.done()[order]
    i_list = col_i.done()[order]
    ts_list = col_ts.done()[order]
    label_list = col_lbl.done()[order]
    source_list = col_src.done()[order]
    capture_list = col_cap.done()[order]

    # Row 0 stays zero: it is the padding edge, which is why no vstack is needed.
    edge_features = np.zeros((n + 1, edge_dim), dtype=np.float32)
    edge_features[1:] = col_edge.done()[order]
    del col_u, col_i, col_ts, col_lbl, col_edge, col_src, col_cap, order

    idx_list = np.arange(1, n + 1)

    # Category ids must match LabelEncoder's semantics -- ALPHABETICAL order,
    # not first-seen. cat_to_id above is assigned in encounter order for speed,
    # so remap it here. Getting this wrong would silently renumber the classes
    # relative to every previous run and to the focal-loss alpha vector, which
    # is indexed by class id.
    ordered = sorted(cat_to_id)
    remap = np.empty(len(ordered), dtype=np.int32)
    for name, encounter_id in cat_to_id.items():
        remap[encounter_id] = ordered.index(name)
    label_list = remap[label_list]
    category_mapping = {i: name for i, name in enumerate(ordered)}
    logging.info("Loaded %d unified flow records from CIC-2017/CIC-2018/CTU-13", n)
    logging.info("Detected coarse categories: %s", category_mapping)

    graph_df = pd.DataFrame({'u': u_list, 'i': i_list, 'ts': ts_list,
                             'label': label_list, 'idx': idx_list,
                             'source': source_list, 'capture': capture_list})
    _src_names = {v: k for k, v in src_to_id.items()}
    logging.info(
        "Edges per corpus: %s",
        {_src_names[v]: int((source_list == v).sum()) for v in sorted(_src_names)},
    )

    # Node features were np.zeros(...). Every one of them. In TGN a node's
    # embedding is a function of its memory, its node features and its
    # neighbours; for a node never seen in training the memory is zero too, so
    # an unseen host carried NO signal at all and the link decoder scored it at
    # chance. Measured: transductive val AUC 0.9981 vs inductive val AUC 0.5043.
    #
    # These features are pure functions of the IP string, so they introduce no
    # label leakage and no temporal leakage across the 70/85 quantile split.
    # The octets are what buys inductive generalisation: an unseen host in a
    # /24 the model has already seen arrives close to its neighbours in feature
    # space. See data_unification/ip_features.py.
    node_features = build_node_feature_matrix(ip_to_id)
    if node_features.shape[1] != edge_features.shape[1]:
        raise ValueError(
            f"node feature width {node_features.shape[1]} != edge feature width "
            f"{edge_features.shape[1]}; tgn.py sets embedding_dimension from the "
            f"node width, so the 12-D latent contract would break."
        )
    logging.info(
        "Node features: %d hosts x %dD intrinsic IP features (was all-zero, "
        "which made inductive link prediction impossible)",
        node_features.shape[0] - 1, node_features.shape[1],
    )

    return graph_df, edge_features, node_features, category_mapping


def split_data(graph_df, edge_features, node_features, different_new_nodes=True, randomize_features=False):
    if randomize_features:
        node_features = np.random.rand(node_features.shape[0], node_features.shape[1]).astype(np.float32)

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    # -------------------------------------------------------------- the split
    #
    # The temporal cut is taken PER CORPUS, not over the concatenated timeline.
    #
    # The three corpora occupy disjoint absolute time ranges -- CTU-13 is 2011,
    # CIC-2017 is July 2017, CIC-2018 is Feb/Mar 2018. A global
    # `np.quantile(graph_df.ts, [0.70, 0.85])` over records sorted by absolute
    # time therefore does not split time at all: it splits by CORPUS. The model
    # trained on 2011 Czech university botnet traffic and was validated and
    # tested on 2018 AWS enterprise traffic.
    #
    # That single fact explained three symptoms at once:
    #   * the focal-loss alpha came back {Benign: .239, C2: 1.761, Impact: 1.0,
    #     InitialAccess: 1.0, Recon: 1.0} -- weights of exactly 1.0 are what
    #     inverse_frequency_alpha leaves for classes with ZERO training
    #     samples, i.e. 3 of 5 categories never appeared in training, because
    #     CTU-13's label vocabulary is only Benign and C2;
    #   * inductive val AUC sat at chance (0.5043) -- val hosts were 172.31.x
    #     and 192.168.10.x while training hosts were CTU-13's 147.32.x, an
    #     entirely disjoint host population;
    #   * val CatAcc was frozen to four decimals across epochs.
    #
    # Splitting within each corpus gives every corpus a 70/15/15 of its own
    # timeline, so all five categories and all host populations are represented
    # on both sides, and "past predicts future" is what is actually being
    # measured.
    # Prefer the CAPTURE tag over the corpus tag. CIC-2017 and CIC-2018
    # organise attack types BY DAY, so cutting a whole corpus at one quantile
    # segregates entire attack classes into the future half. Measured: Recon
    # ended up with SEVEN training samples out of 11.8M, because CIC-2017's
    # 70% cut landed at 13:13 and all 158,930 PortScan rows run 13:00-15:59
    # on the Friday afternoon capture. A class with 7 samples cannot be
    # learned, and Recon is the earliest attack stage -- the one a forecaster
    # most needs.
    #
    # Cutting inside each capture keeps the causal property that matters
    # (train precedes val precedes test within every capture) while letting
    # every attack type appear on all three sides.
    _tag_col = "capture" if "capture" in graph_df.columns else (
        "source" if "source" in graph_df.columns else None)
    if _tag_col is not None:
        src_tag = graph_df[_tag_col].values
        val_mask_t = np.zeros(len(timestamps), dtype=bool)
        test_mask_t = np.zeros(len(timestamps), dtype=bool)
        for tag in np.unique(src_tag):
            sel = src_tag == tag
            v_t, te_t = np.quantile(timestamps[sel], [0.70, 0.85])
            val_mask_t |= sel & (timestamps > v_t) & (timestamps <= te_t)
            test_mask_t |= sel & (timestamps > te_t)
            logging.debug(
                "  %s %s: %d edges, val cut %.0f, test cut %.0f",
                _tag_col, tag, int(sel.sum()), v_t, te_t,
            )
        train_mask_t = ~(val_mask_t | test_mask_t)
        logging.info(
            "Temporal split taken per %s across %d unit(s): train=%d val=%d test=%d",
            _tag_col, len(np.unique(src_tag)),
            int(train_mask_t.sum()), int(val_mask_t.sum()), int(test_mask_t.sum()),
        )
        # val_time is still needed below to pick the inductive node pool; use
        # the earliest per-corpus validation boundary so "held-out node" keeps
        # meaning "appears only after its own corpus's cut".
        val_time = float(min(np.quantile(timestamps[src_tag == t], 0.70)
                             for t in np.unique(src_tag)))
        test_time = float(min(np.quantile(timestamps[src_tag == t], 0.85)
                              for t in np.unique(src_tag)))
    else:
        # Untagged data (the Warden loader) -- a global cut is correct.
        val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))
        train_mask_t = timestamps <= val_time
        val_mask_t = (timestamps > val_time) & (timestamps <= test_time)
        test_mask_t = timestamps > test_time

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    np.random.seed(2020)
    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    _held_out = val_mask_t | test_mask_t
    test_node_set = set(sources[_held_out]).union(set(destinations[_held_out]))
    requested_new_nodes = max(1, int(0.1 * n_total_unique_nodes))
    requested_new_nodes = min(requested_new_nodes, len(test_node_set))
    new_test_node_set = set(
        np.random.choice(list(test_node_set), requested_new_nodes, replace=False)
    )

    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # A class must not be annihilated by the inductive node draw.
    #
    # Holding out a node removes EVERY edge it touches from training. When one
    # host carries an entire class that is all-or-nothing -- and CIC-2017's
    # Recon class is exactly that: all 158,930 PortScan records come from the
    # single source 172.16.0.1. Whether Recon appeared in training was
    # therefore a coin flip on one random draw (~22% of nodes are held out),
    # and on this seed it lost: Recon went to ZERO training samples.
    #
    # Re-draw, excluding the offending hosts, until no class is wiped out.
    # This does not manufacture generalisation -- see the warning below -- it
    # only stops the split from silently deleting a class.
    def _classes_lost(mask):
        in_train = set(np.unique(labels[np.logical_and(train_mask_t, mask)]))
        return set(np.unique(labels)) - in_train

    lost = _classes_lost(observed_edges_mask)
    if lost:
        protect = set()
        for c in lost:
            edges_c = labels == c
            protect |= set(np.unique(sources[edges_c]).tolist())
            protect |= set(np.unique(destinations[edges_c]).tolist())
        kept = new_test_node_set - protect
        logging.warning(
            "Inductive draw removed class(es) %s from training entirely; "
            "re-drawing without their %d carrier host(s).",
            sorted(lost), len(new_test_node_set) - len(kept),
        )
        new_test_node_set = kept
        new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
        new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values
        observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)
        still = _classes_lost(observed_edges_mask)
        if still:
            logging.error("Class(es) %s STILL absent from training after re-draw.", sorted(still))

    # Name any class carried by a single host. Such a class cannot be learned
    # as a behaviour -- a model that scores well on it has memorised that host
    # -- so its metrics must never be reported as detection performance.
    for c in np.unique(labels):
        srcs = np.unique(sources[labels == c])
        if len(srcs) <= 2 and c != 0:
            logging.warning(
                "Class %s is carried by only %d source host(s). Its metrics "
                "measure host memorisation, not generalisation, and must not "
                "be reported as detection performance.",
                category_mapping.get(int(c), int(c)) if "category_mapping" in dir() else int(c),
                len(srcs),
            )

    train_mask = np.logical_and(train_mask_t, observed_edges_mask)
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask], edge_idxs[train_mask], labels[train_mask])

    train_node_set = set(train_data.sources).union(train_data.destinations)
    new_node_set = node_set - train_node_set

    val_mask = val_mask_t
    test_mask = test_mask_t

    if different_new_nodes:
        n_new_nodes = len(new_test_node_set) // 2
        val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
        test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

        edge_contains_new_val_node_mask = np.array([(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
        edge_contains_new_test_node_mask = np.array([(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)
    else:
        edge_contains_new_node_mask = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask], edge_idxs[val_mask], labels[val_mask])
    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask], edge_idxs[test_mask], labels[test_mask])

    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask], timestamps[new_node_val_mask], edge_idxs[new_node_val_mask], labels[new_node_val_mask])
    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask], timestamps[new_node_test_mask], edge_idxs[new_node_test_mask], labels[new_node_test_mask])

    logging.info(f"Dataset split: Full={full_data.n_interactions} ({full_data.n_unique_nodes} nodes), "
                 f"Train={train_data.n_interactions}, Val={val_data.n_interactions}, Test={test_data.n_interactions}, "
                 f"Inductive Val={new_node_val_data.n_interactions}, Inductive Test={new_node_test_data.n_interactions}")

    return node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data


def train(args):
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Directories
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    # Setup Logging
    log_file = os.path.join(args.log_dir, f"{args.prefix}_train_{int(time.time())}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Arguments: {vars(args)}")

    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logging.info(f"Training on device: {device}")

    # Load and Preprocess Data
    if args.cic2017_dir or args.cic2018_dir or args.ctu13_dir or args.pcap2018_root:
        if args.data_name == 'warden_alerts':  # still the default; unified run wasn't given its own name
            args.data_name = 'unified_cic_ctu13'
        require_full_density(
            'TGNE encoder training',
            stride=args.stride,
            rows_per_file=args.rows_per_file,
        )
        graph_df, edge_features, node_features, category_mapping = load_and_preprocess_unified_dataset(
            cic2017_dir=args.cic2017_dir,
            cic2018_dir=args.cic2018_dir,
            ctu13_dir=args.ctu13_dir,
            max_rows_per_file=args.rows_per_file,
            stride=args.stride,
            splits=tuple(args.train_splits.split(",")) if args.train_splits else None,
            parallel_workers=args.ingest_workers,
            pcap2018_root=args.pcap2018_root,
            pcap2018_label_dir=args.pcap2018_label_dir,
            scheme=args.split_scheme,
            allow_cic2018_csv=args.allow_cic2018_csv,
            window_seconds=_contract_window_seconds(),
        )
    else:
        graph_df, edge_features, node_features, category_mapping = load_and_preprocess_dataset(
            dataset_dir=args.dataset_dir, embedding_dim=args.feature_dim,
            resample_to_median=not args.no_median_resample,
            ip_node_features=not args.warden_zero_node_features,
        )
    num_categories = len(category_mapping)

    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        split_data(graph_df, edge_features, node_features, different_new_nodes=args.different_new_nodes,
                   randomize_features=args.randomize_features)

    # Neighbor finders
    train_ngh_finder = get_neighbor_finder(train_data, uniform=args.uniform)
    full_ngh_finder = get_neighbor_finder(full_data, uniform=args.uniform)

    # Samplers
    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
    nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
    nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)

    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
        compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

    model_save_path = os.path.join(args.save_dir, f"{args.prefix}-{args.data_name}.pth")
    checkpoint_path_fn = lambda epoch: os.path.join(args.checkpoint_dir, f"{args.prefix}-{args.data_name}-{epoch}.pth")
    results_path = f"results/{args.prefix}.pkl"

    # Initialize ExtendedTGN Model
    tgn = ExtendedTGN(
        neighbor_finder=train_ngh_finder,
        node_features=node_features,
        edge_features=edge_features,
        device=device,
        n_layers=args.n_layer,
        n_heads=args.n_head,
        dropout=args.drop_out,
        use_memory=args.use_memory,
        message_dimension=args.message_dim,
        memory_dimension=args.memory_dim,
        memory_update_at_start=not args.memory_update_at_end,
        embedding_module_type=args.embedding_module,
        message_function=args.message_function,
        aggregator_type=args.aggregator,
        memory_updater_type=args.memory_updater,
        n_neighbors=args.n_degree,
        mean_time_shift_src=mean_time_shift_src,
        std_time_shift_src=std_time_shift_src,
        mean_time_shift_dst=mean_time_shift_dst,
        std_time_shift_dst=std_time_shift_dst,
        use_destination_embedding_in_message=args.use_destination_embedding_in_message,
        use_source_embedding_in_message=args.use_source_embedding_in_message,
        dyrep=args.dyrep,
        num_categories=num_categories
    ).to(device)

    if args.init_from:
        _init_from_checkpoint(tgn, args.init_from, device)

    # Loss Functions & Optimizer
    edge_criterion = nn.BCELoss()
    if args.focal_loss:
        _alpha = FocalLoss.inverse_frequency_alpha(train_data.labels, num_categories)
        logging.info(f"focal-loss per-class alpha (inverse frequency): "
                     f"{ {category_mapping.get(i, i): round(float(w), 3) for i, w in enumerate(_alpha)} }")
        category_criterion = FocalLoss(alpha=_alpha, gamma=2.0)
    else:
        category_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(tgn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / args.batch_size)
    logging.info(f"Training instances: {num_instance}, Batches per epoch: {num_batch}")

    val_aps, new_nodes_val_aps = [], []
    val_aucs, new_nodes_val_aucs = [], []
    val_accuracies, new_nodes_val_accuracies = [], []
    val_mrrs, new_nodes_val_mrrs = [], []
    train_losses, epoch_times = [], []

    early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=True)

    if args.shuffle_batches and args.use_memory:
        raise ValueError(
            "--shuffle_batches cannot be combined with --use_memory: TGN's memory "
            "module makes batch N depend on batch N-1, so shuffling would train on "
            "memory states that never existed. Use --no_shuffle_batches with memory."
            "\n\nThat trade is not free, and turning memory on without handling it "
            "can make the encoder WORSE, not better. Shuffling exists because TGN "
            "slices batches contiguously in time and attacks are time-localised: "
            "58.6%% of 128-sample batches on fri_16 hold a single class, and the "
            "category head collapsed under that.\n\n"
            "Raise --backprop_every instead. It accumulates gradients over that "
            "many CONSECUTIVE batches before stepping, so one update spans "
            "backprop_every x batch_size samples and far more time, which restores "
            "class contrast per update -- while batch ORDER stays chronological, so "
            "the memory state remains the one that actually existed.\n\n"
            "  python bita/train.py --use_memory --no_shuffle_batches "
            "--backprop_every 8 --batch_size 128   # 1024 samples per update\n\n"
            "Check the per-epoch `cat` loss: if it oscillates rather than falling, "
            "backprop_every is still too small."
        )
    if args.use_memory and args.backprop_every < 4:
        logging.warning(
            "--use_memory with --backprop_every %d: batches are time-ordered and "
            "mostly single-class, so each update sees little class contrast. "
            "8 or more is recommended; see the --shuffle_batches error text.",
            args.backprop_every,
        )
    if args.use_memory and args.backprop_every < 4:
        logging.warning(
            "--use_memory with --backprop_every %d: batches are time-ordered and "
            "mostly single-class, so each update sees little class contrast. "
            "8 or more is recommended; see the --shuffle_batches error text.",
            args.backprop_every,
        )
    logging.info("Batch sampling: %s",
                 "SHUFFLED (memory off)" if args.shuffle_batches else "time-ordered")
    logging.info("Starting training loop...")
    for epoch in range(args.n_epoch):
        start_epoch = time.time()
        tgn.train()

        if args.use_memory:
            tgn.memory.__init_memory__()

        tgn.set_neighbor_finder(train_ngh_finder)
        m_loss = []
        m_edge_loss, m_cat_loss = [], []

        # Sample order for this epoch.
        #
        # TGN slices batches contiguously in TIME, and attacks are
        # time-localised, so most batches contain a SINGLE class. Measured on
        # fri_16: **58.6% of 128-sample batches are single-class**. The
        # category head then receives "predict X for all of these" with no
        # contrast, and oscillates across batches -- which is why it collapsed
        # to one class while a tree on the same features, shuffled, reaches
        # 0.9997 balanced accuracy.
        #
        # Note this must permute SAMPLES, not batch order: reordering
        # contiguous batches leaves each one single-class and changes nothing.
        #
        # Valid here only because the memory module is disabled. TGN's memory
        # is what makes batch N depend on batch N-1; with use_memory=False the
        # remaining order-sensitive piece is the neighbour lookup, which cuts
        # on each edge's OWN timestamp and is therefore order-independent. The
        # negative sampler is random either way. With memory enabled this
        # would corrupt the memory state, so it is refused.
        if args.shuffle_batches and not args.use_memory:
            perm = np.random.permutation(num_instance)
        else:
            perm = np.arange(num_instance)

        for k in range(0, num_batch, args.backprop_every):
            loss = 0.0
            category_loss_total = 0.0
            optimizer.zero_grad()

            for j in range(args.backprop_every):
                batch_idx = k + j
                if batch_idx >= num_batch:
                    continue

                start_idx = batch_idx * args.batch_size
                end_idx = min(num_instance, start_idx + args.batch_size)
                sel = perm[start_idx:end_idx]

                sources_batch = train_data.sources[sel]
                destinations_batch = train_data.destinations[sel]
                edge_idxs_batch = train_data.edge_idxs[sel]
                timestamps_batch = train_data.timestamps[sel]
                categories_batch = train_data.labels[sel]

                size = len(sources_batch)
                _, negatives_batch = train_rand_sampler.sample(size)

                pos_prob, neg_prob, category_logits = tgn.compute_edge_probabilities_and_categories(
                    sources_batch, destinations_batch, negatives_batch,
                    timestamps_batch, edge_idxs_batch, n_neighbors=args.n_degree
                )

                pos_label = torch.ones(size, dtype=torch.float, device=device)
                neg_label = torch.zeros(size, dtype=torch.float, device=device)

                batch_edge_loss = edge_criterion(pos_prob.squeeze(-1), pos_label) + \
                                  edge_criterion(neg_prob.squeeze(-1), neg_label)

                categories_batch_tensor = torch.tensor(categories_batch, dtype=torch.long, device=device)
                batch_cat_loss = category_criterion(category_logits, categories_batch_tensor)

                loss += batch_edge_loss
                category_loss_total += batch_cat_loss

            # Weight the auxiliary category term.
            #
            # The two losses used to be summed with EQUAL weight, and never
            # logged separately, so the imbalance was invisible. Measured at
            # this corpus's class distribution:
            #
            #     edge loss  (AUC ~0.82)      0.5754
            #     category loss, mediocre     0.0420   ~14x smaller
            #     category loss, confident    0.0001   ~5700x smaller
            #
            # Focal loss with gamma=2 exists to down-weight easy examples, but
            # 89% of this corpus is one easy class, so it drives the whole
            # term toward zero next to a co-summed task. C2 compounds it at
            # 0.3% of samples -- a batch of 128 holds ~0.4 C2 examples.
            #
            # The category head therefore stopped learning: three of five
            # classes sat at exactly 0.0 recall while the head was perfectly
            # capable (the same architecture reaches 0.998 on these features
            # standalone).
            total_loss = (loss + args.cat_loss_weight * category_loss_total) / args.backprop_every
            total_loss.backward()
            optimizer.step()
            m_loss.append(total_loss.item())
            m_edge_loss.append(float(loss.item()) / args.backprop_every)
            m_cat_loss.append(float(category_loss_total.item()) / args.backprop_every)

            if k % 500 == 0:
                elapsed = time.time() - start_epoch
                rate = (k + 1) / elapsed if elapsed > 0 else 0.0
                remaining = (num_batch - k - 1) / rate if rate > 0 else float("nan")
                logging.info(
                    f"  epoch {epoch+1} batch {k}/{num_batch} "
                    f"loss={np.mean(m_loss[-500:]):.4f} "
                    f"{rate:.2f} batch/s, ~{remaining/60:.1f} min left this epoch"
                )

            if args.use_memory:
                tgn.memory.detach_memory()

        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)
        mean_train_loss = float(np.mean(m_loss))
        train_losses.append(mean_train_loss)

        # Validation phase
        tgn.eval()
        tgn.set_neighbor_finder(full_ngh_finder)

        if args.use_memory:
            train_memory_backup = tgn.memory.backup_memory()

        val_results = eval_edge_prediction_with_categories(
            model=tgn, negative_edge_sampler=val_rand_sampler,
            data=val_data, n_neighbors=args.n_degree,
            edge_criterion=edge_criterion, category_criterion=category_criterion
        )
        val_ap = val_results[0]
        val_auc = val_results[1]
        val_mrr = val_results[2]
        val_cat_acc = val_results[4]
        val_cat_loss = val_results[5]
        val_f1_macro = val_results[8]

        if args.use_memory:
            val_memory_backup = tgn.memory.backup_memory()
            tgn.memory.restore_memory(train_memory_backup)

        # Inductive validation (unseen nodes)
        nn_val_results = eval_edge_prediction_with_categories(
            model=tgn, negative_edge_sampler=nn_val_rand_sampler,
            data=new_node_val_data, n_neighbors=args.n_degree,
            edge_criterion=edge_criterion, category_criterion=category_criterion
        )
        nn_val_ap = nn_val_results[0]
        nn_val_auc = nn_val_results[1]
        nn_val_mrr = nn_val_results[2]
        nn_val_cat_acc = nn_val_results[4]
        nn_val_cat_loss = nn_val_results[5]

        if args.use_memory:
            tgn.memory.restore_memory(val_memory_backup)

        val_aps.append(val_ap)
        val_aucs.append(val_auc)
        val_accuracies.append(val_cat_acc)
        val_mrrs.append(val_mrr)
        new_nodes_val_aps.append(nn_val_ap)
        new_nodes_val_aucs.append(nn_val_auc)
        new_nodes_val_accuracies.append(nn_val_cat_acc)
        new_nodes_val_mrrs.append(nn_val_mrr)

        logging.info(f"Epoch {epoch:02d} [{epoch_time:.2f}s] Loss: {mean_train_loss:.4f} (edge {np.mean(m_edge_loss):.4f} cat {np.mean(m_cat_loss):.4f} x{args.cat_loss_weight:g}) | "
                     f"Val AUC: {val_auc:.4f}, AP: {val_ap:.4f}, CatAcc: {val_cat_acc:.4f}, MRR: {val_mrr:.4f} | "
                     f"Inductive Val AUC: {nn_val_auc:.4f}, AP: {nn_val_ap:.4f}, CatAcc: {nn_val_cat_acc:.4f}")

        # Per-class accuracy, every epoch.
        #
        # Aggregate CatAcc is dominated by Benign, which is ~80% of the data --
        # a collapsed head that predicts Benign for everything still scores
        # ~0.8 there. The minority classes are the whole point of the model,
        # and they are invisible in the aggregate. This is the number that
        # actually says whether the class weighting is working.
        _per_class = val_results[7] or {}
        _nn_per_class = nn_val_results[7] or {}
        _nm = {i: category_mapping.get(i, str(i)) for i in _per_class}
        logging.info(
            "           per-class val acc: %s | inductive: %s",
            {_nm[i]: round(float(a), 3) for i, a in sorted(_per_class.items())},
            {_nm.get(i, i): round(float(a), 3) for i, a in sorted(_nn_per_class.items())},
        )

        # Checkpoint current epoch
        torch.save(tgn.state_dict(), checkpoint_path_fn(epoch))

        # Model selection metric.
        #
        # This used val_ap alone -- link prediction on seen hosts. That task
        # converges in ONE epoch on this corpus and then saturates:
        #
        #     val_ap        0.9971 -> 0.9968 -> 0.9968
        #     inductive AP  0.9926 -> 0.9921 -> 0.9918
        #
        # while the classification head was still improving materially
        # (C2 inductive recall 0.325 -> 0.319 -> 0.517). Selecting on the
        # saturated metric restores epoch 0 and throws the improving one away.
        #
        # The encoder is judged on two things that both matter downstream:
        # embedding quality on UNSEEN hosts (inductive AP -- deployment meets
        # new hosts constantly, and the transductive figure has no
        # discriminative power left at 0.9968), and balanced classification
        # (macro F1, not aggregate accuracy, because Benign is ~90% of val and
        # dominates any unweighted average).
        _sel = 0.5 * float(nn_val_ap) + 0.5 * float(val_f1_macro)
        logging.info(
            "           selection=%.4f  (inductive AP %.4f, val macro-F1 %.4f)",
            _sel, nn_val_ap, val_f1_macro,
        )

        # Check early stopping
        if early_stopper.early_stop_check(_sel):
            logging.info(f"Early stopping triggered! No improvement over {early_stopper.max_round} epochs.")
            logging.info(f"Best model was at Epoch {early_stopper.best_epoch} with selection score: {early_stopper.last_best:.4f}")
            best_checkpoint = checkpoint_path_fn(early_stopper.best_epoch)
            tgn.load_state_dict(torch.load(best_checkpoint, map_location=device))
            break

    # Save Best Model to final model path
    torch.save(tgn.state_dict(), model_save_path)
    logging.info(f"Best model successfully saved to: {model_save_path}")

    # Serialize explicit architectural configuration
    import json
    config_data = {
        "n_layers": args.n_layer,
        "n_heads": args.n_head,
        "dropout": args.drop_out,
        "use_memory": args.use_memory,
        "message_dimension": args.message_dim,
        "memory_dimension": args.memory_dim,
        "embedding_module_type": args.embedding_module,
        "message_function": args.message_function,
        "aggregator_type": args.aggregator,
        "memory_updater_type": args.memory_updater,
        "num_categories": num_categories,
        "edge_feat_dim": int(edge_features.shape[1]),
        "node_feat_dim": int(node_features.shape[1]),
        "feature_schema_version": SCHEMA_VERSION,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        # How each node's temporal neighbourhood was sampled. Serving must use
        # the same count and rule: the graph-attention weights were fitted to
        # this many neighbours, chosen this way. build_or_load_tgne_ta hands
        # both to HostTrajectoryExtractor instead of a literal that happened
        # to match.
        "n_neighbors": args.n_degree,
        "neighbor_sampling": "uniform" if args.uniform else "most_recent",
        # Which edge features were zeroed while training. The ablation is an
        # env var read at feature extraction, so serving must apply the SAME
        # mask; build_or_load_tgne_ta refuses a mismatch using this field.
        "ablated_edge_features": _ablated_edge_features(),
        # Same contract for the IP node features (CYBERWORLD_ABLATE_NODE_FEATURES).
        "ablated_node_features": _ablated_node_features(),
        "split_scheme": args.split_scheme,
        "init_from": args.init_from,
        "model_name": getattr(args, "model_name", args.prefix),
        "dataset_name": getattr(args, "data", args.data_name),
    }
    config_path = os.path.splitext(model_save_path)[0] + "_config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    logging.info(f"Model configuration saved to: {config_path}")

    # ==================== FINAL TEST EVALUATION ====================
    logging.info("Running comprehensive test set evaluation...")
    tgn.eval()
    tgn.set_neighbor_finder(full_ngh_finder)

    if args.use_memory:
        tgn.memory.restore_memory(val_memory_backup)

    # Transductive Test (Old Nodes)
    test_res = eval_edge_prediction_with_categories(
        model=tgn, negative_edge_sampler=test_rand_sampler,
        data=test_data, n_neighbors=args.n_degree,
        edge_criterion=edge_criterion, category_criterion=category_criterion
    )

    if args.use_memory:
        tgn.memory.restore_memory(val_memory_backup)

    # Inductive Test (New Nodes)
    nn_test_res = eval_edge_prediction_with_categories(
        model=tgn, negative_edge_sampler=nn_test_rand_sampler,
        data=new_node_test_data, n_neighbors=args.n_degree,
        edge_criterion=edge_criterion, category_criterion=category_criterion
    )

    logging.info("=" * 60)
    logging.info("FINAL TEST RESULTS (Transductive - Old Nodes):")
    logging.info(f"  Link Prediction AUC: {test_res[1]:.4f}")
    logging.info(f"  Link Prediction AP:  {test_res[0]:.4f}")
    logging.info(f"  Link Prediction MRR: {test_res[2]:.4f}")
    logging.info(f"  Category Accuracy:   {test_res[4]:.4f}")
    logging.info(f"  Category Macro F1:   {test_res[8]:.4f}")
    logging.info(f"  Category Hits@1:     {test_res[13]:.4f}, Hits@3: {test_res[14]:.4f}, Hits@5: {test_res[15]:.4f}")
    logging.info(f"  Per-class Accuracy:  {test_res[7]}")
    logging.info(f"  Per-class Precision: {test_res[18]}")
    logging.info(f"  Per-class Recall:    {test_res[21]}")

    logging.info("-" * 60)
    logging.info("FINAL TEST RESULTS (Inductive - Unseen New Nodes):")
    logging.info(f"  Link Prediction AUC: {nn_test_res[1]:.4f}")
    logging.info(f"  Link Prediction AP:  {nn_test_res[0]:.4f}")
    logging.info(f"  Link Prediction MRR: {nn_test_res[2]:.4f}")
    logging.info(f"  Category Accuracy:   {nn_test_res[4]:.4f}")
    logging.info(f"  Category Macro F1:   {nn_test_res[8]:.4f}")
    logging.info(f"  Category Hits@1:     {nn_test_res[13]:.4f}, Hits@3: {nn_test_res[14]:.4f}, Hits@5: {nn_test_res[15]:.4f}")
    logging.info("=" * 60)

    # Save metrics to CSV and Pickle
    df_metrics = pd.DataFrame({
        "epoch": list(range(len(train_losses))),
        "train_loss": train_losses,
        "val_ap": val_aps,
        "val_auc": val_aucs,
        "val_accuracy": val_accuracies,
        "val_mrr": val_mrrs,
        "inductive_val_ap": new_nodes_val_aps,
        "inductive_val_auc": new_nodes_val_aucs,
        "inductive_val_accuracy": new_nodes_val_accuracies,
        "epoch_time": epoch_times
    })
    metrics_csv_path = "results/training_metrics.csv"
    df_metrics.to_csv(metrics_csv_path, index=False)
    logging.info(f"Training metrics saved to: {metrics_csv_path}")

    # Plot results
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(val_aps, 'g-', label='Transductive Val AP')
    plt.plot(new_nodes_val_aps, 'g--', label='Inductive Val AP')
    plt.title('Validation Average Precision (AP)')
    plt.xlabel('Epoch')
    plt.ylabel('AP')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(val_accuracies, 'r-', label='Transductive Cat Acc')
    plt.plot(new_nodes_val_accuracies, 'r--', label='Inductive Cat Acc')
    plt.title('Alert Category Prediction Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(val_aucs, 'm-', label='Transductive Val AUC')
    plt.plot(new_nodes_val_aucs, 'm--', label='Inductive Val AUC')
    plt.title('Validation ROC-AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plot_path = "results/training_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logging.info(f"Training curves saved to: {plot_path}")

    return {
        "test_auc": test_res[1],
        "test_ap": test_res[0],
        "test_cat_acc": test_res[4],
        "nn_test_auc": nn_test_res[1],
        "nn_test_ap": nn_test_res[0],
        "nn_test_cat_acc": nn_test_res[4],
        "model_path": model_save_path
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BiTA Temporal Graph Network for Network Alert Prediction")
    parser.add_argument('--dataset_dir', type=str, default='Dataset', help='Directory containing Warden dataset CSVs (ignored if any --cic*/--ctu13-dir is given)')
    parser.add_argument('--data_name', type=str, default='warden_alerts', help='Dataset identifier name')
    parser.add_argument('--pcap2018_root', type=str, default=None,
                        help='CIC-IDS2018 PCAP root (<day>_pcap dirs): real host addresses. '
                             'Use instead of --cic2018_dir, whose CSVs fabricate IPs on 9 of 10 days.')
    parser.add_argument('--pcap2018_label_dir', type=str, default=None,
                        help='CIC-2018 <day>_csv.csv files used only as labels for --pcap2018_root')
    parser.add_argument('--split_scheme', type=str, default='frozen',
                        choices=['frozen', 'cross_year', 'cross_year_ctu'],
                        help="'cross_year': the encoder reads only CIC-2018 train days; CIC-2017 is "
                             "never loaded (it is the downstream test set). 'cross_year_ctu': the "
                             "same, plus CTU-13's lock train scenarios (pass --ctu13_dir). "
                             "See split_policy.py.")
    parser.add_argument('--allow_cic2018_csv', action='store_true', default=False)
    parser.add_argument('--init_from', type=str, default=None,
                        help='Start from this encoder checkpoint (e.g. a Warden-trained one) and '
                             'fine-tune. Category head and per-node memory are re-initialised.')
    parser.add_argument('--warden_zero_node_features', action='store_true', default=False,
                        help='Warden only: all-zero node features (the old behaviour) instead of '
                             'the intrinsic IP features every other path uses.')
    parser.add_argument('--no_median_resample', action='store_true', default=False,
                        help='Warden only: skip the BiTA paper step that resamples every alert '
                             'category to the median class size before the temporal split.')
    parser.add_argument('--cic2017_dir', type=str, default=None, help='CIC-IDS2017 CSV directory (switches to the unified, non-bipartite loader)')
    parser.add_argument('--cic2018_dir', type=str, default=None, help='CIC-IDS2018 CSV directory (switches to the unified, non-bipartite loader)')
    parser.add_argument('--ctu13_dir', type=str, default=None, help='CTU-13 directory of <scenario>/*.binetflow files (switches to the unified, non-bipartite loader)')
    parser.add_argument('--rows_per_file', type=int, default=None, help='Max records KEPT per source file, after striding (not a row prefix)')
    parser.add_argument('--stride', type=int, default=1, help='Keep every Nth record DURING ingestion: spans the whole capture and bounds peak memory')
    parser.add_argument('--shuffle_batches', action='store_true', default=None,
                        help='Shuffle batch ORDER within an epoch. Valid only with '
                             '--no_memory; TGN batches are contiguous in time and '
                             '58%% of them are single-class, which collapses the category head. '
                             'Default: on without memory, off with it.')
    parser.add_argument('--no_shuffle_batches', dest='shuffle_batches', action='store_false',
                        help='Keep the original time-ordered batch sequence.')
    parser.add_argument('--cat_loss_weight', type=float, default=15.0,
                        help='Weight on the auxiliary category loss. The two terms were '
                             'summed equally, but focal loss drives the category term ~14x '
                             'below the edge loss (5700x once confident), so the head stopped '
                             'learning. 1.0 restores the old behaviour and is the BiTA paper value (lambda, Eq. 17); it '
                             'is NOT the default here because on this corpus it was measured '
                             'to starve the category head (tests/test_category_loss_is_not_swamped.py).')
    parser.add_argument('--ingest_workers', type=int, default=0,
                        help='Parse captures across N worker processes. 0 = serial. '
                             'Results are identical either way: workers return local '
                             'vocabularies and the parent merges in fixed capture order.')
    parser.add_argument('--train_splits', type=str, default='train', help="Frozen-lock splits to read (comma separated). Default 'train' keeps val/test captures unseen by the encoder; pass '' to read everything (leaks labels downstream).")
    parser.add_argument('--prefix', type=str, default='bita_bigru_transformer', help='Prefix for saved artifacts')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--n_epoch', type=int, default=50, help='Maximum number of epochs (BiTA paper: 50)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of GNN layers')
    parser.add_argument('--n_head', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--n_degree', type=int, default=10, help='Number of sampled neighbors')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (use -1 for CPU)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--feature_dim', type=int, default=4, help='Embedding dimension for categorical features')
    parser.add_argument('--node_dim', type=int, default=12, help='Node feature dimension')
    parser.add_argument('--time_dim', type=int, default=12, help='Time encoding dimension')
    parser.add_argument('--message_dim', type=int, default=100, help='Message dimension')
    # BiTA reports a memory width of 9. It cannot be 9 here: the embedding
    # module ADDS memory to the node features (embedding_module.py), so the two
    # widths must match, and the node features are the 12-D contract. 9 would
    # crash the first time memory was used.
    parser.add_argument('--memory_dim', type=int, default=12, help='Memory dimension (= node feature dim)')
    # Memory is ON by default. BiTA's contribution IS the aggregator that feeds
    # the TGN memory update; TGN only builds an aggregator when memory is on,
    # so every encoder trained with memory off contained no BiTA at all.
    parser.add_argument('--use_memory', dest='use_memory', action='store_true', default=True,
                        help='Enable the TGN memory module and the BiTA aggregator (default)')
    parser.add_argument('--no_memory', dest='use_memory', action='store_false',
                        help='Memoryless ablation: plain temporal graph attention, NO BiTA aggregator')
    parser.add_argument('--embedding_module', type=str, default='graph_attention', choices=['graph_attention', 'graph_sum', 'identity', 'time'])
    parser.add_argument('--message_function', type=str, default='identity', choices=['identity', 'mlp'])
    parser.add_argument('--memory_updater', type=str, default='gru', choices=['gru', 'rnn'])
    # Only the implemented aggregators. The paper's other ablation variants
    # used to be accepted here and silently trained as the BiGRU-Transformer.
    parser.add_argument('--aggregator', type=str, default='bigru_transformer',
                        choices=['bigru_transformer', 'mean', 'last'])
    parser.add_argument('--memory_update_at_end', action='store_true', default=False)
    parser.add_argument('--different_new_nodes', action='store_true', default=True)
    parser.add_argument('--uniform', action='store_true', default=False)
    parser.add_argument('--randomize_features', action='store_true', default=False)
    parser.add_argument('--use_destination_embedding_in_message', action='store_true', default=False)
    parser.add_argument('--use_source_embedding_in_message', action='store_true', default=False)
    parser.add_argument('--dyrep', action='store_true', default=False)
    parser.add_argument('--focal_loss', action='store_true', default=True)
    parser.add_argument('--backprop_every', type=int, default=None,
                        help='Consecutive batches per optimiser step. Default 8 with memory '
                             '(time-ordered batches are mostly single-class), 1 without.')
    parser.add_argument('--save_dir', type=str, default='saved_models')
    # Per-epoch snapshots are scratch, not models: nothing serves from here.
    # This used to be a top-level saved_checkpoints/ that sat beside
    # saved_models/ looking equally authoritative, and a retrain writing to the
    # wrong one of the two has already been shipped once (analysis doc 23).
    # Promote a winner with scripts/select_best_encoder.py --copy.
    parser.add_argument('--checkpoint_dir', type=str, default='.spill/encoder_epochs')
    parser.add_argument('--log_dir', type=str, default='logs')

    args = parser.parse_args()
    if args.shuffle_batches is None:
        args.shuffle_batches = not args.use_memory
    if args.backprop_every is None:
        args.backprop_every = 8 if args.use_memory else 1
    train(args)
