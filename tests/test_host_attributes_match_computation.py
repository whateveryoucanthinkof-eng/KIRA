"""The 15 host-attribute NAMES must describe the values actually computed.

This test exists because they once did not: every one of the fifteen names was
wrong (four of them named TCP flag counters that are not computed anywhere in
the repository). The dimension was correct, so nothing crashed -- the error was
purely interpretive, and therefore invisible until someone read both files side
by side.

Each assertion below constructs a window whose expected value at one index is
known analytically, so a rename or a reordering on either side fails here.
"""

import numpy as np
import pytest

from data_unification.host_attributes import (
    BYTE_LOG_SCALE,
    BYTE_RATE_LOG_SCALE,
    COUNT_LOG_SCALE,
    DURATION_SCALE_SECONDS,
    HOST_ATTRIBUTES,
    HOST_ATTRIBUTE_INDEX,
    HOST_ATTR_DIM,
    PEER_COUNT_LOG_SCALE,
    PORT_COUNT_LOG_SCALE,
    PORT_SPACE,
)
from data_unification.multi_dataset_stream import HostTrajectoryExtractor
from data_unification.unified_schema import LabelSource, UnifiedFlowRecord

HOST = "10.0.0.1"
WINDOW = 2.0


def _rec(peer="10.0.0.2", *, fb=0, bb=0, fp=0, bp=0, proto=6, dport=80, dur=1.0):
    return UnifiedFlowRecord(
        src_ip=HOST, dst_ip=peer, src_port=1234, dst_port=dport, protocol=proto,
        start_time=0.0, end_time=dur,
        fwd_bytes=fb, bwd_bytes=bb, fwd_packets=fp, bwd_packets=bp,
        raw_label="BENIGN", raw_label_source=LabelSource.CIC2018.value,
        is_attack=False, coarse_category="Benign", attck_technique_ids=[],
    )


def _attrs(records, window=WINDOW):
    # the TGNE model is irrelevant here -- this function reads only the records
    ex = HostTrajectoryExtractor(tgne_ta_model=None, window_size_sec=window)
    return ex.compute_host_temporal_attributes(HOST, records, window)


def test_names_are_unique_and_sized_to_the_contract():
    assert HOST_ATTR_DIM == 15
    assert len(HOST_ATTRIBUTES) == 15
    assert len(set(HOST_ATTRIBUTES)) == 15, "duplicate attribute names"
    assert _attrs([_rec()]).shape == (15,)


def test_no_attribute_claims_a_tcp_flag():
    """There is no TCP flag extraction in this pipeline; no name may imply one."""
    offenders = [n for n in HOST_ATTRIBUTES if "flag" in n.lower() or "syn" in n.lower()]
    assert offenders == [], f"names promise TCP flags that are never computed: {offenders}"


def test_empty_window_is_all_zero():
    assert np.array_equal(_attrs([]), np.zeros(15, dtype=np.float32))


def test_all_values_are_bounded_unit_interval():
    """Every attribute is documented as clipped to [0, 1]; verify under extreme input."""
    huge = [_rec(peer=f"10.1.{i // 256}.{i % 256}", fb=10**9, bb=10**9,
                 fp=10**6, bp=10**6, dport=i, dur=10**5) for i in range(50)]
    a = _attrs(huge)
    assert a.min() >= 0.0 and a.max() <= 1.0, f"outside [0,1]: {a}"


@pytest.mark.parametrize("name,expected", [
    ("flow_count",    lambda n, **_: min(1.0, np.log1p(n) / COUNT_LOG_SCALE)),
    ("fwd_bytes",     lambda n, fb, **_: min(1.0, np.log1p(fb * n) / BYTE_LOG_SCALE)),
    ("bwd_bytes",     lambda n, bb, **_: min(1.0, np.log1p(bb * n) / BYTE_LOG_SCALE)),
    ("total_bytes",   lambda n, fb, bb, **_: min(1.0, np.log1p((fb + bb) * n) / BYTE_LOG_SCALE)),
    ("fwd_packets",   lambda n, fp, **_: min(1.0, np.log1p(fp * n) / COUNT_LOG_SCALE)),
    ("bwd_packets",   lambda n, bp, **_: min(1.0, np.log1p(bp * n) / COUNT_LOG_SCALE)),
    ("total_packets", lambda n, fp, bp, **_: min(1.0, np.log1p((fp + bp) * n) / COUNT_LOG_SCALE)),
])
def test_volume_attributes_match_their_names(name, expected):
    """Each volume attribute equals its own formula over a controlled window."""
    n, fb, bb, fp, bp = 4, 1000, 500, 20, 10
    recs = [_rec(peer=f"10.0.0.{i + 2}", fb=fb, bb=bb, fp=fp, bp=bp) for i in range(n)]
    got = _attrs(recs)[HOST_ATTRIBUTE_INDEX[name]]
    assert got == pytest.approx(expected(n=n, fb=fb, bb=bb, fp=fp, bp=bp), rel=1e-5)


def test_unique_peers_counts_peers_not_flows():
    """Ten flows to two peers is two peers -- the name says peers."""
    recs = [_rec(peer="10.0.0.2") for _ in range(5)] + [_rec(peer="10.0.0.3") for _ in range(5)]
    got = _attrs(recs)[HOST_ATTRIBUTE_INDEX["unique_peers"]]
    assert got == pytest.approx(min(1.0, np.log1p(2) / PEER_COUNT_LOG_SCALE), rel=1e-5)


def test_unique_dst_ports_counts_distinct_ports():
    recs = [_rec(dport=p) for p in (80, 443, 80, 8080)]
    got = _attrs(recs)[HOST_ATTRIBUTE_INDEX["unique_dst_ports"]]
    assert got == pytest.approx(min(1.0, np.log1p(3) / PORT_COUNT_LOG_SCALE), rel=1e-5)


def test_tcp_and_udp_ratios_are_protocol_fractions():
    """Index 9/10 are protocol ratios -- they were previously named TCP flag counters."""
    recs = [_rec(proto=6), _rec(proto=6), _rec(proto=6), _rec(proto=17)]
    a = _attrs(recs)
    assert a[HOST_ATTRIBUTE_INDEX["tcp_ratio"]] == pytest.approx(0.75)
    assert a[HOST_ATTRIBUTE_INDEX["udp_ratio"]] == pytest.approx(0.25)


def test_tcp_ratio_ignores_non_tcp_protocols():
    recs = [_rec(proto=1), _rec(proto=1)]  # ICMP only
    a = _attrs(recs)
    assert a[HOST_ATTRIBUTE_INDEX["tcp_ratio"]] == pytest.approx(0.0)
    assert a[HOST_ATTRIBUTE_INDEX["udp_ratio"]] == pytest.approx(0.0)


def test_avg_duration_is_a_mean_normalised_by_300s():
    recs = [_rec(dur=10.0), _rec(dur=20.0)]
    got = _attrs(recs)[HOST_ATTRIBUTE_INDEX["avg_duration"]]
    assert got == pytest.approx(min(1.0, 15.0 / DURATION_SCALE_SECONDS), rel=1e-5)


def test_rates_divide_by_window_duration_not_flow_count():
    """byte_rate / packet_rate are per-second over the window; halving the window doubles them."""
    recs = [_rec(fb=1000, bb=1000, fp=10, bp=10) for _ in range(3)]
    slow = _attrs(recs, window=100.0)
    fast = _attrs(recs, window=10.0)
    for name in ("byte_rate", "packet_rate"):
        i = HOST_ATTRIBUTE_INDEX[name]
        assert fast[i] > slow[i], f"{name} did not scale with window duration"

    tot_b, tot_p, dur = 6000, 60, 100.0
    assert slow[HOST_ATTRIBUTE_INDEX["byte_rate"]] == pytest.approx(
        min(1.0, np.log1p(tot_b / dur) / BYTE_RATE_LOG_SCALE), rel=1e-5)
    assert slow[HOST_ATTRIBUTE_INDEX["packet_rate"]] == pytest.approx(
        min(1.0, np.log1p(tot_p / dur) / COUNT_LOG_SCALE), rel=1e-5)


def test_peer_density_is_peers_per_flow():
    """One peer per flow is density 1.0; many flows to one peer drives it down."""
    fan_out = [_rec(peer=f"10.0.0.{i + 2}") for i in range(6)]
    assert _attrs(fan_out)[HOST_ATTRIBUTE_INDEX["peer_density"]] == pytest.approx(1.0)

    hammer = [_rec(peer="10.0.0.2") for _ in range(6)]
    assert _attrs(hammer)[HOST_ATTRIBUTE_INDEX["peer_density"]] == pytest.approx(1.0 / 6.0)


def test_peer_is_resolved_from_whichever_side_is_not_the_host():
    """Inbound flows must attribute the peer to the remote end, not to the host itself."""
    inbound = UnifiedFlowRecord(
        src_ip="10.9.9.9", dst_ip=HOST, src_port=5555, dst_port=443, protocol=6,
        start_time=0.0, end_time=1.0, fwd_bytes=10, bwd_bytes=10,
        fwd_packets=1, bwd_packets=1, raw_label="BENIGN",
        raw_label_source=LabelSource.CIC2018.value, is_attack=False,
        coarse_category="Benign", attck_technique_ids=[],
    )
    got = _attrs([inbound])[HOST_ATTRIBUTE_INDEX["unique_peers"]]
    assert got == pytest.approx(min(1.0, np.log1p(1) / PEER_COUNT_LOG_SCALE), rel=1e-5)


# ---------------------------------------------------------------------------
# Nobody may keep a second copy of these names
# ---------------------------------------------------------------------------

def test_explainability_names_derive_from_the_single_source():
    """`explainability/unified_explanation.py` kept its own hardcoded copy and
    had already drifted: index 14 was "active_conn_density" when the value is
    unique_peers / flow_count -- peer fan-out, not a connection count.

    These strings sit next to attribution scores an operator reads, so a wrong
    one is a confidently-stated wrong explanation.
    """
    from explainability.unified_explanation import FEATURE_NAMES
    assert len(FEATURE_NAMES) == 27, "12 embedding dims + 15 attributes"
    assert FEATURE_NAMES[12:] == list(HOST_ATTRIBUTES), (
        "explainability has drifted from host_attributes again"
    )
    assert FEATURE_NAMES[:12] == [f"H_emb_{i}" for i in range(12)]


def test_no_module_hardcodes_a_rival_attribute_list():
    """Catch a future copy-paste before it drifts.

    Three separate copies of this list existed and two had already drifted:
    explainability said "active_conn_density"; model_adapter used both
    "active_conn_density" and "avg_flow_duration" as dict KEYS, names that do
    not exist, so those lookups silently missed.

    Uses ast so only real string literals count -- comments and docstrings
    legitimately name the old values when explaining what went wrong.
    """
    import ast
    import pathlib

    banned = {"active_conn_density", "avg_flow_duration", "tcp_flags_syn",
              "bytes_in_rate", "conn_duration_avg"}
    offenders = []
    for path in pathlib.Path(".").rglob("*.py"):
        if set(path.parts) & {".git", "tests", "cyberworld_v4", "node_modules"}:
            continue
        try:
            tree = ast.parse(path.read_text())
        except Exception:
            continue
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                d = ast.get_docstring(node, clean=False)
                if d:
                    docstrings.add(d)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node.value in banned and node.value not in docstrings):
                offenders.append(f"{path}:{node.lineno}:{node.value}")
    assert not offenders, (
        "stale host-attribute names used as real values; import "
        "HOST_ATTRIBUTES instead of copying: " + ", ".join(offenders)
    )


def test_the_dashboard_group_map_covers_every_attribute():
    """model_adapter's FEATURE_GROUP_MAP had two keys that do not exist, so
    two of the fifteen attributes showed no group in the UI."""
    src = open("control_backend/model_adapter.py").read()
    ns = {}
    exec(compile(src.split("class AntigravityModelAdapter")[0], "ma", "exec"), ns)
    group_map = ns["FEATURE_GROUP_MAP"]
    for name in HOST_ATTRIBUTES:
        assert name in group_map, f"{name} has no dashboard group"
    dangling = [k for k in group_map
                if not k.startswith("H_emb_") and k not in HOST_ATTRIBUTES]
    assert not dangling, f"group map keys that are not real attributes: {dangling}"


# ---------------------------------------------------------------------------
# Saturation: the fan-out attributes must stay discriminative over the range
# the corpora actually contain. Both used to divide by 5.0, i.e. reach 1.0 at
# 147 and stay there -- so a 148-port probe and a full 65,535-port sweep were
# the identical feature value.
# ---------------------------------------------------------------------------


def test_unique_peers_still_separates_hosts_above_the_old_saturation_point():
    def peers(k):
        recs = [_rec(peer=f"10.{i // 65536}.{(i // 256) % 256}.{i % 256}") for i in range(k)]
        return _attrs(recs)[HOST_ATTRIBUTE_INDEX["unique_peers"]]

    assert peers(200) < peers(1000) < peers(5000), "fan-out saturates inside the observed range"
    assert peers(5000) < 1.0


def test_unique_dst_ports_reaches_one_only_at_the_full_port_space():
    def ports(k):
        return _attrs([_rec(dport=p) for p in range(1, k + 1)])[
            HOST_ATTRIBUTE_INDEX["unique_dst_ports"]]

    assert ports(200) < ports(2000) < ports(20000), "port sweep saturates inside the observed range"
    # Exactly 1.0 when the whole port space has been touched, and not before.
    assert min(1.0, np.log1p(PORT_SPACE) / PORT_COUNT_LOG_SCALE) == pytest.approx(1.0)
    assert ports(20000) < 1.0


def test_scales_are_not_duplicated_as_literals_in_the_computation():
    """The computation must import the divisors, not re-type them."""
    import inspect
    from data_unification.multi_dataset_stream import HostTrajectoryExtractor

    src = inspect.getsource(HostTrajectoryExtractor.compute_host_temporal_attributes)
    body = src.split('"""', 2)[-1]
    for literal in ("/ 5.0", "/ 20.0", "/ 15.0", "/ 300.0"):
        assert literal not in body, (
            f"{literal!r} is a hardcoded normalisation divisor; import it from "
            "host_attributes so the spec table stays authoritative"
        )
