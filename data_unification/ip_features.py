"""Intrinsic 12-D node features derived from an IP address alone.

## Why this exists

`bita/train.py` built `node_features = np.zeros((total_nodes, 12))`. Every node
feature was zero. In TGN a node's embedding is a function of its memory, its
node features, and its neighbours. For a node never seen during training
(the *inductive* case) the memory is zero too -- so a new host carried no
signal whatsoever and the link decoder scored it at chance.

Measured: transductive val AUC 0.9981 against inductive val AUC 0.5043. The
encoder had memorised the hosts it had seen and learned nothing transferable.
That is fatal for a forecasting deployment, which meets unseen hosts
constantly.

## Why these features are safe

Every value here is a pure function of the IP string. Nothing is aggregated
over the dataset, so there is no label leakage and -- critically -- no temporal
leakage across the 70/85 quantile split. A feature computed from observed
behaviour would have to be restricted to the training period; these need no
such care because they are properties of the identifier itself.

## Why they should help inductively

The octets are the point. Two hosts in the same /24 share their first three
octets, so an unseen host in a subnet the model has seen arrives with a feature
vector close to its neighbours'. That is exactly the generalisation the
zero-feature setup made impossible. In these corpora the structure is real:
CIC-2018 hosts are 172.31.x.x, CIC-2017 hosts are 192.168.10.x, and external
traffic is public space.

The dimension is 12 because `tgn.py:82` sets
`embedding_dimension = n_node_features`, and 12 is the latent width the whole
downstream contract is built on.
"""

from __future__ import annotations

import ipaddress
from functools import lru_cache
from typing import Dict, List

import numpy as np

IP_FEATURE_NAMES: List[str] = [
    "is_parseable",      # 0 -- 1.0 when the string is a real IP at all
    "is_ipv4",           # 1
    "is_private",        # 2 -- RFC1918 / RFC4193
    "is_global",         # 3 -- routable public space
    "is_multicast",      # 4
    "is_loopback_or_reserved",  # 5
    "is_link_local",     # 6
    "octet1",            # 7 -- /8  identity, scaled to [0, 1]
    "octet2",            # 8 -- /16 identity
    "octet3",            # 9 -- /24 identity
    "octet4",            # 10 -- host identity within the subnet
    "bias",              # 11 -- constant 1.0, so an all-zero row means "unknown"
]

IP_FEATURE_DIM: int = len(IP_FEATURE_NAMES)
assert IP_FEATURE_DIM == 12, "node feature width is pinned to the 12-D latent contract"


@lru_cache(maxsize=1 << 20)
def ip_node_features(ip: str) -> tuple:
    """12 intrinsic features for one IP. Cached -- callers hit the same hosts often.

    Returns a tuple so it is hashable and cacheable; wrap in np.asarray to use.
    An unparseable string yields all zeros except the bias, which is how the
    model distinguishes "no information" from a genuine 0.0.0.0.
    """
    f = [0.0] * IP_FEATURE_DIM
    f[11] = 1.0  # bias

    try:
        addr = ipaddress.ip_address(str(ip).strip())
    except (ValueError, AttributeError):
        return tuple(f)

    f[0] = 1.0
    is_v4 = addr.version == 4
    f[1] = 1.0 if is_v4 else 0.0
    f[2] = 1.0 if addr.is_private else 0.0
    f[3] = 1.0 if addr.is_global else 0.0
    f[4] = 1.0 if addr.is_multicast else 0.0
    f[5] = 1.0 if (addr.is_loopback or addr.is_reserved or addr.is_unspecified) else 0.0
    f[6] = 1.0 if addr.is_link_local else 0.0

    if is_v4:
        o = [int(p) for p in addr.packed]
        f[7] = o[0] / 255.0
        f[8] = o[1] / 255.0
        f[9] = o[2] / 255.0
        f[10] = o[3] / 255.0
    else:
        # Use the last four bytes so IPv6 hosts still separate from one another.
        o = list(addr.packed[-4:])
        f[7] = o[0] / 255.0
        f[8] = o[1] / 255.0
        f[9] = o[2] / 255.0
        f[10] = o[3] / 255.0

    return tuple(f)


def build_node_feature_matrix(ip_to_id: Dict[str, int], n_nodes: int = None) -> np.ndarray:
    """(n_nodes, 12) matrix indexed by node id, row 0 reserved for padding.

    `ip_to_id` is the 1-based shared namespace built in
    `bita/train.py::load_and_preprocess_unified_dataset` at training time, and
    by `FlowToTemporalEventAdapter` at inference time. Both must produce the
    SAME features for the same address, or the encoder sees a different input
    distribution than it was trained on.

    `n_nodes` pads the matrix out to a caller-chosen size (inference allocates
    slack rows for hosts that have not appeared yet). Unused rows stay zero,
    which is also what an unknown node should look like.
    """
    needed = len(ip_to_id) + 1
    total_nodes = needed if n_nodes is None else max(int(n_nodes), needed)
    m = np.zeros((total_nodes, IP_FEATURE_DIM), dtype=np.float32)
    for ip, node_id in ip_to_id.items():
        if 0 <= node_id < total_nodes:
            m[node_id] = ip_node_features(ip)
    return m


def subnet_cohesion(ip_a: str, ip_b: str) -> float:
    """Fraction of leading octets shared -- 0.75 means same /24. Diagnostics only."""
    a, b = np.asarray(ip_node_features(ip_a)), np.asarray(ip_node_features(ip_b))
    if not (a[0] and b[0]):
        return 0.0
    shared = 0
    for i in (7, 8, 9, 10):
        if a[i] == b[i]:
            shared += 1
        else:
            break
    return shared / 4.0
