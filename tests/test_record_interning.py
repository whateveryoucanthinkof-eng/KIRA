"""String interning must actually fire, including for numpy string scalars.

UnifiedFlowRecord's docstring calls interning "load-bearing at corpus scale".
It was guarded on `type(x) is str`, which is False for numpy.str_ -- and
numpy.str_ is precisely what the pandas-based adapters produce for src_ip and
dst_ip. So the guard silently skipped the two highest-cardinality,
highest-volume fields while appearing to work on the label fields.

Measured on wed_29 before the fix: raw_label collapsed to ONE object across
5,000 records, while src_ip kept 5,000 distinct objects for 250 unique values.
Fixing it took the record from 479 to 300 bytes.
"""

import sys

import numpy as np
import pytest

from data_unification.unified_schema import LabelSource, UnifiedFlowRecord, _interned


def rec(src="10.0.0.1", dst="10.0.0.2", label="BENIGN"):
    return UnifiedFlowRecord(
        src_ip=src, dst_ip=dst, src_port=1, dst_port=80, protocol=6,
        start_time=0.0, end_time=1.0, fwd_bytes=1, bwd_bytes=1,
        fwd_packets=1, bwd_packets=1, raw_label=label,
        raw_label_source=LabelSource.CIC2018.value, is_attack=False,
        coarse_category="Benign", attck_technique_ids=[],
    )


def test_numpy_str_is_interned_to_an_exact_str():
    """The regression: numpy.str_ is a str SUBCLASS, so `type(x) is str` is False."""
    v = np.array(["10.0.0.1"])[0]
    assert isinstance(v, str) and type(v) is not str, "precondition: numpy.str_"
    out = _interned(v)
    assert type(out) is str
    assert out is sys.intern("10.0.0.1")


def test_plain_str_is_interned():
    assert _interned("".join(["10.", "0.0.9"])) is sys.intern("10.0.0.9")


def test_non_strings_pass_through_untouched():
    for v in (None, 5, 3.2, b"bytes", ["a"]):
        assert _interned(v) is v


def test_records_built_from_numpy_strings_share_one_object():
    """250 unique IPs across 5,000 records must be 250 objects, not 5,000."""
    ips = np.array([f"10.0.0.{i % 250}" for i in range(5000)])
    peers = np.array([f"172.16.0.{i % 100}" for i in range(5000)])
    recs = [rec(src=ips[i], dst=peers[i]) for i in range(5000)]

    src_objs = {id(r.src_ip) for r in recs}
    dst_objs = {id(r.dst_ip) for r in recs}
    assert len(src_objs) == 250, f"src_ip kept {len(src_objs)} objects for 250 values"
    assert len(dst_objs) == 100, f"dst_ip kept {len(dst_objs)} objects for 100 values"


def test_every_interned_field_shares_across_records():
    a, b = rec(), rec()
    for fld in ("src_ip", "dst_ip", "raw_label", "raw_label_source", "coarse_category"):
        assert getattr(a, fld) is getattr(b, fld), f"{fld} not shared between records"


def test_interning_preserves_value_exactly():
    """Memory work must never change what the record says."""
    ips = np.array(["192.168.10.50", "8.8.8.8", "172.31.69.25"])
    for ip in ips:
        assert rec(src=ip).src_ip == str(ip)


def test_record_still_uses_slots():
    """slots=True is the other half of the memory story; no per-instance dict."""
    assert not hasattr(rec(), "__dict__")


def test_metadata_defaults_to_none_not_an_empty_dict():
    """An empty dict per record costs 64 B and is never populated on the CSV path."""
    assert rec().metadata is None
