"""The destination port must arrive at the encoder as a usable feature.

It was `min(65535, max(0, dst_port)) / 65535.0`. A port is categorical and that
is a linear ordinal encoding, which would be merely imperfect -- except for
where the interesting values sit. Every service that carries signal in these
corpora lives below 1024, so the whole discriminative range was compressed into
the bottom 1.5% of the feature:

    22 -> 0.00034    53 -> 0.00081    80 -> 0.00122    443 -> 0.00676

SSH and HTTP were 0.0009 apart, while two meaningless ephemeral source ports
40,000 apart were 0.6 apart. The single most informative attribute of a flow
reached the graph encoder as near-constant noise.

Log scaling is the best use of one slot in a 12-D contract: monotone, so
nothing that assumed ordering breaks, and it spends its resolution where the
services are.
"""

import numpy as np
import pytest

from data_unification.tgne_features import (
    EDGE_FEATURE_NAMES,
    SCHEMA_VERSION,
    extract_canonical_edge_features,
)

PORT_IDX = EDGE_FEATURE_NAMES.index("dst_port_log_norm")

WELL_KNOWN = [22, 53, 80, 443]
EPHEMERAL = [49152, 55000, 60000, 65535]


def _port_feature(port):
    return float(extract_canonical_edge_features(
        fwd_bytes=100, bwd_bytes=100, fwd_packets=2, bwd_packets=2,
        duration_sec=1.0, byte_rate=100, packet_rate=2, protocol=6, dst_port=port,
    )[PORT_IDX])


def test_the_feature_is_named_for_what_it_now_holds():
    assert "dst_port_norm_65535" not in EDGE_FEATURE_NAMES
    assert EDGE_FEATURE_NAMES[PORT_IDX] == "dst_port_log_norm"
    assert len(EDGE_FEATURE_NAMES) == 12, "the 12-D edge contract is unchanged"


def test_well_known_services_are_separated_by_a_usable_margin():
    """Under the old encoding every pair here was under 0.007 apart."""
    vals = [_port_feature(p) for p in WELL_KNOWN]
    for a, b in zip(vals, vals[1:]):
        assert b - a > 0.03, f"adjacent services still collapsed: {vals}"


def test_the_well_known_range_gets_most_of_the_feature_not_one_percent():
    """22..443 should span a real fraction of [0, 1], not 0.6% of it."""
    span = _port_feature(443) - _port_feature(22)
    assert span > 0.25, f"well-known ports span only {span:.4f} of the range"
    old_span = (443 - 22) / 65535.0
    assert span > 40 * old_span


def test_ephemeral_ports_are_compressed_rather_than_dominant():
    """Ephemeral port numbers carry no service identity; they should not carry
    most of the feature's dynamic range either."""
    eph = [_port_feature(p) for p in EPHEMERAL]
    assert max(eph) - min(eph) < 0.05, f"ephemeral range still dominates: {eph}"


def test_the_encoding_is_still_monotone():
    ports = [0, 1, 22, 80, 443, 1024, 8080, 30000, 65535]
    vals = [_port_feature(p) for p in ports]
    assert vals == sorted(vals)


def test_it_stays_inside_the_unit_interval_including_at_the_edges():
    for p in (-5, 0, 1, 65535, 70000, 10**9):
        v = _port_feature(p)
        assert np.isfinite(v) and 0.0 <= v <= 1.0, (p, v)
    assert _port_feature(0) == pytest.approx(0.0)
    assert _port_feature(65535) == pytest.approx(1.0)


def test_the_schema_version_was_bumped_with_the_values():
    """A value change with an unchanged version is the silent train/serve
    mismatch the loader guard exists to catch; it can only catch it if the
    version actually moves."""
    assert SCHEMA_VERSION != "1.0.0"


def test_the_loader_refuses_a_checkpoint_from_the_previous_schema():
    import inspect

    from branch_a_gnn_lstm.train_branch_a import build_or_load_tgne_ta

    src = inspect.getsource(build_or_load_tgne_ta)
    assert '!= "1.0.0"' not in src, "the guard still pins a hardcoded version"
    assert "SCHEMA_VERSION" in src or "_want" in src
