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
    extract_canonical_edge_features,
)


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
            w[present] = inv / inv.mean()
        w = np.clip(w, clip[0], clip[1])
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


def load_and_preprocess_dataset(dataset_dir="Dataset", embedding_dim=4):
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

    df['SourceIP_encoded'], _ = pd.factorize(df['SourceIP'])
    df['TargetIP_encoded'], _ = pd.factorize(df['TargetIP'])

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

    # Node features: (total_nodes, feature_dim)
    node_feat_dim = edge_features.shape[1]
    node_features = np.zeros((total_nodes, node_feat_dim), dtype=np.float32)

    return graph_df, edge_features, node_features, category_mapping


def load_and_preprocess_unified_dataset(
    cic2017_dir=None,
    cic2018_dir=None,
    ctu13_dir=None,
    max_rows_per_file=None,
    stride=1,
    splits=("train",),
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

    def _in_split(path):
        """True when the frozen lock assigns `path` to one of `splits`."""
        if wanted is None:
            return True
        try:
            return split_of_path(path) in wanted
        except (KeyError, ValueError):
            logging.warning("%s is not in the frozen split -- skipped", path)
            return False

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

        def done(self):
            return self.buf[: self.n]

    col_u = _Growable(np.int64)
    col_i = _Growable(np.int64)
    col_ts = _Growable(np.float64)
    col_lbl = _Growable(np.int32)
    col_edge = _Growable(np.float32, width=edge_dim)

    ip_to_id: dict = {}
    cat_to_id: dict = {}

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

    skipped = []
    n_read = 0
    if cic2017_dir:
        for f in sorted(glob.glob(os.path.join(cic2017_dir, "*.csv"))):
            if not _in_split(f):
                skipped.append(os.path.basename(f)); continue
            _consume(_take(CIC2017Adapter().parse_file(f, max_rows=None))); n_read += 1
    if cic2018_dir:
        for f in sorted(glob.glob(os.path.join(cic2018_dir, "*.csv"))):
            if not _in_split(f):
                skipped.append(os.path.basename(f)); continue
            _consume(_take(CIC2018Adapter().parse_file(f, max_rows=None))); n_read += 1
    if ctu13_dir:
        ctu_adapter = CTU13Adapter()
        for f in sorted(glob.glob(os.path.join(ctu13_dir, "*", "*.binetflow"))):
            if not _in_split(f):
                skipped.append(os.path.basename(f)); continue
            _consume(_take(ctu_adapter.parse_netflow_csv(f, max_rows=None))); n_read += 1

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

    # Row 0 stays zero: it is the padding edge, which is why no vstack is needed.
    edge_features = np.zeros((n + 1, edge_dim), dtype=np.float32)
    edge_features[1:] = col_edge.done()[order]
    del col_u, col_i, col_ts, col_lbl, col_edge, order

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
                             'label': label_list, 'idx': idx_list})

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

    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    np.random.seed(2020)
    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    test_node_set = set(sources[timestamps > val_time]).union(set(destinations[timestamps > val_time]))
    requested_new_nodes = max(1, int(0.1 * n_total_unique_nodes))
    requested_new_nodes = min(requested_new_nodes, len(test_node_set))
    new_test_node_set = set(
        np.random.choice(list(test_node_set), requested_new_nodes, replace=False)
    )

    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask], edge_idxs[train_mask], labels[train_mask])

    train_node_set = set(train_data.sources).union(train_data.destinations)
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time

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
    if args.cic2017_dir or args.cic2018_dir or args.ctu13_dir:
        if args.data_name == 'warden_alerts':  # still the default; unified run wasn't given its own name
            args.data_name = 'unified_cic_ctu13'
        graph_df, edge_features, node_features, category_mapping = load_and_preprocess_unified_dataset(
            cic2017_dir=args.cic2017_dir,
            cic2018_dir=args.cic2018_dir,
            ctu13_dir=args.ctu13_dir,
            max_rows_per_file=args.rows_per_file,
            stride=args.stride,
            splits=tuple(args.train_splits.split(",")) if args.train_splits else None,
        )
    else:
        graph_df, edge_features, node_features, category_mapping = load_and_preprocess_dataset(
            dataset_dir=args.dataset_dir, embedding_dim=args.feature_dim
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

    logging.info("Starting training loop...")
    for epoch in range(args.n_epoch):
        start_epoch = time.time()
        tgn.train()

        if args.use_memory:
            tgn.memory.__init_memory__()

        tgn.set_neighbor_finder(train_ngh_finder)
        m_loss = []

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

                sources_batch = train_data.sources[start_idx:end_idx]
                destinations_batch = train_data.destinations[start_idx:end_idx]
                edge_idxs_batch = train_data.edge_idxs[start_idx:end_idx]
                timestamps_batch = train_data.timestamps[start_idx:end_idx]
                categories_batch = train_data.labels[start_idx:end_idx]

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

            total_loss = (loss + category_loss_total) / args.backprop_every
            total_loss.backward()
            optimizer.step()
            m_loss.append(total_loss.item())

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

        logging.info(f"Epoch {epoch:02d} [{epoch_time:.2f}s] Loss: {mean_train_loss:.4f} | "
                     f"Val AUC: {val_auc:.4f}, AP: {val_ap:.4f}, CatAcc: {val_cat_acc:.4f}, MRR: {val_mrr:.4f} | "
                     f"Inductive Val AUC: {nn_val_auc:.4f}, AP: {nn_val_ap:.4f}, CatAcc: {nn_val_cat_acc:.4f}")

        # Checkpoint current epoch
        torch.save(tgn.state_dict(), checkpoint_path_fn(epoch))

        # Check early stopping
        if early_stopper.early_stop_check(val_ap):
            logging.info(f"Early stopping triggered! No improvement over {early_stopper.max_round} epochs.")
            logging.info(f"Best model was at Epoch {early_stopper.best_epoch} with Val AP: {early_stopper.last_best:.4f}")
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
    parser.add_argument('--cic2017_dir', type=str, default=None, help='CIC-IDS2017 CSV directory (switches to the unified, non-bipartite loader)')
    parser.add_argument('--cic2018_dir', type=str, default=None, help='CIC-IDS2018 CSV directory (switches to the unified, non-bipartite loader)')
    parser.add_argument('--ctu13_dir', type=str, default=None, help='CTU-13 directory of <scenario>/*.binetflow files (switches to the unified, non-bipartite loader)')
    parser.add_argument('--rows_per_file', type=int, default=None, help='Max records KEPT per source file, after striding (not a row prefix)')
    parser.add_argument('--stride', type=int, default=1, help='Keep every Nth record DURING ingestion: spans the whole capture and bounds peak memory')
    parser.add_argument('--train_splits', type=str, default='train', help="Frozen-lock splits to read (comma separated). Default 'train' keeps val/test captures unseen by the encoder; pass '' to read everything (leaks labels downstream).")
    parser.add_argument('--prefix', type=str, default='bita_bigru_transformer', help='Prefix for saved artifacts')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--n_epoch', type=int, default=30, help='Maximum number of epochs')
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
    parser.add_argument('--memory_dim', type=int, default=9, help='Memory dimension')
    parser.add_argument('--use_memory', action='store_true', default=False, help='Enable memory module')
    parser.add_argument('--embedding_module', type=str, default='graph_attention', choices=['graph_attention', 'graph_sum', 'identity', 'time'])
    parser.add_argument('--message_function', type=str, default='identity', choices=['identity', 'mlp'])
    parser.add_argument('--memory_updater', type=str, default='gru', choices=['gru', 'rnn'])
    parser.add_argument('--aggregator', type=str, default='bigru_transformer', choices=[
        'bigru_transformer', 'bitransformer', 'bitransformer_temporal', 'relative_transformer', 'stacked_bitransformer', 'mean', 'last'
    ])
    parser.add_argument('--memory_update_at_end', action='store_true', default=False)
    parser.add_argument('--different_new_nodes', action='store_true', default=True)
    parser.add_argument('--uniform', action='store_true', default=False)
    parser.add_argument('--randomize_features', action='store_true', default=False)
    parser.add_argument('--use_destination_embedding_in_message', action='store_true', default=False)
    parser.add_argument('--use_source_embedding_in_message', action='store_true', default=False)
    parser.add_argument('--dyrep', action='store_true', default=False)
    parser.add_argument('--focal_loss', action='store_true', default=True)
    parser.add_argument('--backprop_every', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='saved_models')
    parser.add_argument('--checkpoint_dir', type=str, default='saved_checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')

    args = parser.parse_args()
    train(args)
