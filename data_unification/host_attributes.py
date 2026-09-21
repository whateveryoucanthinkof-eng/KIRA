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

from typing import Dict, List, Tuple

#: (name, human description, normalization applied) in index order.
#: Index i of the attribute vector is HOST_ATTRIBUTE_SPEC[i].
HOST_ATTRIBUTE_SPEC: List[Tuple[str, str, str]] = [
    ("flow_count",        "flows observed for this host in the window",      "min(1, log1p(n) / 10)"),
    ("fwd_bytes",         "bytes sent by this host",                          "min(1, log1p(x) / 20)"),
    ("bwd_bytes",         "bytes received by this host",                      "min(1, log1p(x) / 20)"),
    ("total_bytes",       "fwd + bwd bytes",                                  "min(1, log1p(x) / 20)"),
    ("fwd_packets",       "packets sent by this host",                        "min(1, log1p(x) / 10)"),
    ("bwd_packets",       "packets received by this host",                    "min(1, log1p(x) / 10)"),
    ("total_packets",     "fwd + bwd packets",                                "min(1, log1p(x) / 10)"),
    ("unique_peers",      "distinct peer IPs contacted or contacted by",      "min(1, log1p(x) / 5)"),
    ("unique_dst_ports",  "distinct destination ports seen",                  "min(1, log1p(x) / 5)"),
    ("tcp_ratio",         "fraction of flows with protocol == 6",             "already a ratio in [0, 1]"),
    ("udp_ratio",         "fraction of flows with protocol == 17",            "already a ratio in [0, 1]"),
    ("avg_duration",      "mean flow duration in seconds",                    "min(1, mean / 300)"),
    ("byte_rate",         "total bytes per second over the window",           "min(1, log1p(x) / 15)"),
    ("packet_rate",       "total packets per second over the window",         "min(1, log1p(x) / 10)"),
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
