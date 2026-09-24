"""
data_unification/host_attributes.py
Single-source naming for the 15 per-host temporal attributes.

THESE NAMES DESCRIBE REAL COMPUTED VALUES.

The previous version of this list was aspirational and wrong in every entry: it
named `bytes_in_rate`/`bytes_out_rate`, separate `*_in`/`*_out` peer counts, and
four TCP-flag counters (`tcp_flags_syn/ack/fin/rst`). None of those are computed
anywhere in this repository -- there is no TCP flag extraction in the pipeline at
all. Zero of the fifteen names matched the value at that index.

Because the dimension was right (15) and only the labels were wrong, nothing ever
crashed. The damage was silent and interpretive: every feature attribution, every
dashboard explainability panel and every report that named a feature was naming
the wrong quantity with complete confidence -- e.g. index 9 was presented as
"tcp_flags_syn" when it is in fact the TCP *protocol ratio*.

The authoritative computation is
`data_unification/multi_dataset_stream.py::HostTrajectoryExtractor.compute_host_temporal_attributes`.
This module and that function are kept in lockstep by
`tests/test_host_attributes_match_computation.py`, which fails if either drifts.

All fifteen values are clipped to [0, 1] so downstream models see a bounded input.
"""

import math
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Log-compression divisors
# ---------------------------------------------------------------------------
#
# These were literals inside compute_host_temporal_attributes. They are named
# here because they are the SINGLE source of truth -- the computation imports
# them and the spec table below renders them, so a divisor can no longer drift
# from its documentation, and a test can assert against the constant instead of
# re-typing the number.
#
# `min(1, log1p(x) / D)` saturates at `exp(D) - 1`. Choosing D too small throws
# away the top of the range, and for two of these the top of the range is
# exactly where attacks are:
#
#   unique_peers      D=5 saturated at 147 peers. A host sweeping a /24 or
#                     fanning out to a botnet exceeds that inside one 2s
#                     window, so every such host produced the identical
#                     value 1.0 as a host with 148 peers. D=10 (matching
#                     flow_count) saturates at ~22,026 instead.
#
#   unique_dst_ports  D=5 saturated at 147 ports. A vertical port scan covers
#                     hundreds to tens of thousands of ports in seconds, so the
#                     single most direct scan signature in the feature vector
#                     was flat across its whole discriminative range. D is now
#                     log1p(65535), i.e. the attribute reaches 1.0 exactly when
#                     the host has touched the entire port space and not before.
#
# Both changes alter the input distribution of every downstream model, so they
# require a retrain -- see SCHEMA_VERSION in data_unification/tgne_features.py.
PORT_SPACE = 65535

COUNT_LOG_SCALE: float = 10.0          # flows, packets
BYTE_LOG_SCALE: float = 20.0           # byte totals
BYTE_RATE_LOG_SCALE: float = 15.0      # bytes/second
PEER_COUNT_LOG_SCALE: float = 10.0     # was 5.0 -- saturated at 147 peers
PORT_COUNT_LOG_SCALE: float = math.log1p(PORT_SPACE)   # ~11.09; was 5.0
DURATION_SCALE_SECONDS: float = 300.0

#: (name, human description, normalization applied) in index order.
#: Index i of the attribute vector is HOST_ATTRIBUTE_SPEC[i].
HOST_ATTRIBUTE_SPEC: List[Tuple[str, str, str]] = [
    ("flow_count",        "flows observed for this host in the window",      f"min(1, log1p(n) / {COUNT_LOG_SCALE:g})"),
    ("fwd_bytes",         "bytes sent by this host",                          f"min(1, log1p(x) / {BYTE_LOG_SCALE:g})"),
    ("bwd_bytes",         "bytes received by this host",                      f"min(1, log1p(x) / {BYTE_LOG_SCALE:g})"),
    ("total_bytes",       "fwd + bwd bytes",                                  f"min(1, log1p(x) / {BYTE_LOG_SCALE:g})"),
    ("fwd_packets",       "packets sent by this host",                        f"min(1, log1p(x) / {COUNT_LOG_SCALE:g})"),
    ("bwd_packets",       "packets received by this host",                    f"min(1, log1p(x) / {COUNT_LOG_SCALE:g})"),
    ("total_packets",     "fwd + bwd packets",                                f"min(1, log1p(x) / {COUNT_LOG_SCALE:g})"),
    ("unique_peers",      "distinct peer IPs contacted or contacted by",      f"min(1, log1p(x) / {PEER_COUNT_LOG_SCALE:g})"),
    ("unique_dst_ports",  "distinct destination ports seen",                  f"min(1, log1p(x) / {PORT_COUNT_LOG_SCALE:.4f})"),
    ("tcp_ratio",         "fraction of flows with protocol == 6",             "already a ratio in [0, 1]"),
    ("udp_ratio",         "fraction of flows with protocol == 17",            "already a ratio in [0, 1]"),
    ("avg_duration",      "mean flow duration in seconds",                    f"min(1, mean / {DURATION_SCALE_SECONDS:g})"),
    ("byte_rate",         "total bytes per second over the window",           f"min(1, log1p(x) / {BYTE_RATE_LOG_SCALE:g})"),
    ("packet_rate",       "total packets per second over the window",         f"min(1, log1p(x) / {COUNT_LOG_SCALE:g})"),
    ("peer_density",      "unique peers per flow (fan-out concentration)",    "min(1, peers / max(1, n))"),
]

HOST_ATTRIBUTES: List[str] = [name for name, _desc, _norm in HOST_ATTRIBUTE_SPEC]

#: name -> index, for code that wants to read one attribute without hardcoding a number.
HOST_ATTRIBUTE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(HOST_ATTRIBUTES)}

HOST_ATTR_DIM: int = len(HOST_ATTRIBUTES)

assert HOST_ATTR_DIM == 15, f"host attribute count drifted from the contract: {HOST_ATTR_DIM}"


def describe(index: int) -> str:
    """Human-readable description of attribute `index`, for reports and dashboards."""
    name, desc, norm = HOST_ATTRIBUTE_SPEC[index]
    return f"{name}: {desc} [{norm}]"


# ---------------------------------------------------------------------------
# Packet-level attributes (optional 30-D extension)
# ---------------------------------------------------------------------------
#
# `telemetry/packet/pcap_engine.py` has computed these thirty values since the
# sensor was written, `data_unification/pcap_adapter.py` and
# `data_unification/pcap_bridge.py` both call it, and `pcap_bridge.py` then
# dropped the result on the floor:
#
#     for host_ip, (flows, _packet_features) in per_host.items():
#
# So the cost of computing them was paid on every PCAP run and no model has
# ever seen one. They are the features the problem statement names -- TTL
# variance, TCP window size, payload-size distribution, port-scan signatures,
# retransmission counts -- and they carry signal the 15 flow attributes
# structurally cannot:
#
#   pkt_iat_cv               periodicity. A C2 beacon is defined by its
#                            regularity, and no flow-level aggregate sees it.
#   syn_no_response_ratio    the direct signature of a scan against closed
#                            ports, as opposed to a busy client.
#   vertical/horizontal_scan_score
#                            which KIND of sweep, ports-on-one-host versus
#                            one-port-across-hosts.
#   tcp_retransmission_ratio path quality, and the tell of a flood.
#   dns_domain_entropy_mean  DGA-style generated domains.
#   ttl_std / ttl_unique_count
#                            several origins behind one address.
#
# This is an OPT-IN extension, not a redefinition: HOST_ATTRIBUTES stays 15
# wide and every existing checkpoint stays loadable. Building an extractor with
# `include_packet_features=True` produces a 45-wide attribute vector and a
# 57-D model input, which needs `CyberWorldConfig(host_attr_dim=45)` and a
# retrain. Only the PCAP path can populate it; the CSV corpora have no packets.

#: Names, in index order, taken from the engine so the two cannot drift.
def _packet_names():
    from telemetry.packet.pcap_engine import PCAP_BEHAVIORAL_COLUMNS
    return list(PCAP_BEHAVIORAL_COLUMNS)


PACKET_ATTRIBUTES: List[str] = _packet_names()
PACKET_ATTR_DIM: int = len(PACKET_ATTRIBUTES)
assert PACKET_ATTR_DIM == 30, f"packet feature count drifted: {PACKET_ATTR_DIM}"

EXTENDED_HOST_ATTRIBUTES: List[str] = HOST_ATTRIBUTES + PACKET_ATTRIBUTES
EXTENDED_HOST_ATTR_DIM: int = len(EXTENDED_HOST_ATTRIBUTES)

#: Per-feature scaling into [0, 1], matching the convention the flow
#: attributes already follow. `("ratio", None)` means the engine already emits
#: a value in [0, 1]; `("div", d)` is x/d clipped; `("log", d)` is
#: log1p(x)/d clipped.
#:
#: The divisors are the natural ceiling of each quantity rather than a
#: percentile of this corpus, so they do not have to be refitted when the
#: corpus changes: TTL is one byte, a TCP window is 16 bits, entropy over
#: a byte alphabet is at most 8 bits, and an inter-arrival time inside a 2s
#: window cannot exceed the window.
PACKET_ATTRIBUTE_SCALING: Dict[str, Tuple[str, float]] = {
    "ttl_mean":                          ("div", 255.0),
    "ttl_std":                           ("div", 128.0),
    "ttl_unique_count":                  ("log", math.log1p(64)),
    "tcp_window_mean":                   ("div", 65535.0),
    "tcp_window_std":                    ("div", 32768.0),
    "tcp_retransmission_estimate":       ("log", COUNT_LOG_SCALE),
    "tcp_retransmission_ratio":          ("ratio", 1.0),
    "pkt_iat_mean":                      ("div", 2.0),
    "pkt_iat_std":                       ("div", 2.0),
    "pkt_iat_cv":                        ("div", 4.0),
    "payload_size_mean":                 ("div", 1500.0),
    "payload_size_std":                  ("div", 750.0),
    "payload_zero_ratio":                ("ratio", 1.0),
    "unique_dst_ips_pcap":               ("log", PEER_COUNT_LOG_SCALE),
    "vertical_scan_score":               ("ratio", 1.0),
    "horizontal_scan_score":             ("ratio", 1.0),
    "syn_only_ratio":                    ("ratio", 1.0),
    "syn_ack_response_ratio":            ("ratio", 1.0),
    "syn_no_response_ratio":             ("ratio", 1.0),
    "rst_after_syn_ratio":               ("ratio", 1.0),
    "tcp_handshake_completion_ratio":    ("ratio", 1.0),
    "http_request_count":                ("log", COUNT_LOG_SCALE),
    "http_uri_entropy_mean":             ("div", 8.0),
    "http_special_char_ratio":           ("ratio", 1.0),
    "dns_query_count":                   ("log", COUNT_LOG_SCALE),
    "dns_unique_domains":                ("log", COUNT_LOG_SCALE),
    "dns_domain_entropy_mean":           ("div", 8.0),
    "unique_internal_destinations_pcap": ("log", PEER_COUNT_LOG_SCALE),
    "unique_external_destinations_pcap": ("log", PEER_COUNT_LOG_SCALE),
    "internal_fanout_entropy_pcap":      ("div", 16.0),
}

_missing = [n for n in PACKET_ATTRIBUTES if n not in PACKET_ATTRIBUTE_SCALING]
_extra = [n for n in PACKET_ATTRIBUTE_SCALING if n not in PACKET_ATTRIBUTES]
assert not _missing and not _extra, (
    f"PACKET_ATTRIBUTE_SCALING is out of sync with the engine: "
    f"missing={_missing} extra={_extra}"
)

PACKET_ATTRIBUTE_INDEX: Dict[str, int] = {n: i for i, n in enumerate(PACKET_ATTRIBUTES)}


def normalize_packet_features(features) -> "np.ndarray":
    """Engine output dict -> (30,) float32 in [0, 1], in PACKET_ATTRIBUTES order.

    A missing or non-finite value becomes 0.0, which is also what the engine
    emits for an empty window -- "nothing observed" and "no packets" are the
    same statement here.
    """
    import numpy as np

    out = np.zeros(PACKET_ATTR_DIM, dtype=np.float32)
    if not features:
        return out
    for i, name in enumerate(PACKET_ATTRIBUTES):
        try:
            v = float(features.get(name, 0.0))
        except (TypeError, ValueError):
            v = 0.0
        if not np.isfinite(v) or v <= 0.0:
            out[i] = 0.0
            continue
        kind, d = PACKET_ATTRIBUTE_SCALING[name]
        if kind == "ratio":
            out[i] = min(1.0, v)
        elif kind == "div":
            out[i] = min(1.0, v / d)
        else:  # log
            out[i] = min(1.0, float(np.log1p(v)) / d)
    return out


def describe_packet(index: int) -> str:
    name = PACKET_ATTRIBUTES[index]
    kind, d = PACKET_ATTRIBUTE_SCALING[name]
    norm = "already in [0,1]" if kind == "ratio" else (
        f"min(1, x / {d:g})" if kind == "div" else f"min(1, log1p(x) / {d:.4f})")
    return f"{name}: packet-level [{norm}]"
