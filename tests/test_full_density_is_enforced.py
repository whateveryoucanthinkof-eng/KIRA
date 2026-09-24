"""No training run may silently thin its data.

Eleven places in this pipeline subsampled by default -- row caps, strides, and
structural truncation (`snapshot_flows(max_flows=256)` dropped every flow past
the 256th for a host in a window; `max_packets_per_host=20000` stopped reading
a capture partway). Each was defensible alone as a smoke-test or memory
workaround, and together they meant no run was ever on the full corpus.

The memory pressure that justified them turned out to be a single 13.91 GiB
pandas call plus a 293-byte-per-edge Python adjacency list, both since fixed.
"""

import inspect

import pytest

from data_unification.density import (
    ENV_FLAG,
    active_caps,
    require_full_density,
    subsampling_allowed,
)


# ------------------------------------------------------- the guard itself

def test_full_density_passes():
    require_full_density("t", stride=1, rows_per_file=None, max_flows=None)


@pytest.mark.parametrize("caps", [
    {"stride": 2}, {"stride": 20}, {"rows_per_file": 1000},
    {"max_per_source": 500}, {"pcap_window_stride": 4},
    {"stride": 1, "rows_per_file": 40000},
])
def test_any_active_cap_raises(caps, monkeypatch):
    monkeypatch.delenv(ENV_FLAG, raising=False)
    with pytest.raises(ValueError, match="thinned data"):
        require_full_density("t", **caps)


def test_stride_one_is_not_a_cap():
    assert active_caps(stride=1) == {}


def test_none_is_not_a_cap():
    assert active_caps(rows_per_file=None, max_per_source=None) == {}


def test_the_escape_hatch_works_but_is_explicit(monkeypatch):
    monkeypatch.setenv(ENV_FLAG, "1")
    assert subsampling_allowed()
    require_full_density("smoke test", stride=50)   # must not raise


def test_the_escape_hatch_is_off_by_default(monkeypatch):
    monkeypatch.delenv(ENV_FLAG, raising=False)
    assert not subsampling_allowed()


# --------------------------------------------- defaults across the codebase

def test_split_manager_defaults_to_full_density():
    from data_unification.split_manager import ScientificSplitManager as S
    for name in ("get_train_records", "get_val_records", "get_heldout_test_records"):
        sig = inspect.signature(getattr(S, name))
        assert sig.parameters["max_per_source"].default is None, f"{name} caps rows"
        assert sig.parameters["stride"].default == 1, f"{name} strides"


def test_records_for_defaults_to_full_density():
    from data_unification.split_manager import ScientificSplitManager as S
    sig = inspect.signature(S.records_for)
    assert sig.parameters["max_per_source"].default is None
    assert sig.parameters["stride"].default == 1


def test_snapshot_flows_is_unbounded_by_default():
    """256 silently discarded every flow past the 256th for a host in a window."""
    from telemetry.flow.flow_table import LiveFlowTable
    assert inspect.signature(LiveFlowTable.snapshot_flows).parameters["max_flows"].default is None


def test_pcap_ingestion_takes_every_flow():
    """The PCAP bridge feeds training and must not truncate."""
    import data_unification.pcap_bridge as pb
    import data_unification.pcap_adapter as pa
    for mod in (pb, pa):
        src = inspect.getsource(mod)
        assert "max_flows=256" not in src, f"{mod.__name__} still truncates flows"


def test_pcap_packet_reading_is_unbounded_by_default():
    import scripts.retrain_future_models_live as m
    for fn in (m.load_pcap_records, m.iter_pcap_day_records):
        d = inspect.signature(fn).parameters["max_packets_per_host"].default
        assert d is None, f"{fn.__name__} stops reading at {d} packets"


def test_deepop_trainer_defaults_to_full_density():
    from deepop_decoder.train_cwa_decoder import train_cwa_decoder
    assert inspect.signature(train_cwa_decoder).parameters["max_per_source"].default is None


def test_live_retrain_scripts_default_to_no_row_cap():
    import re
    for path in ("scripts/retrain_branch_a_live.py",
                 "scripts/retrain_future_models_live.py",
                 "scripts/build_tgne_live_dataset.py"):
        src = open(path).read()
        m = re.search(r'"--rows-per-file"[^)]*?default=(\w+)', src, re.S)
        assert m, f"{path}: no --rows-per-file found"
        assert m.group(1) == "None", f"{path} caps rows at {m.group(1)} by default"


def test_all_three_trainers_call_the_guard():
    """A default can drift; the guard is what actually stops a thinned run."""
    for path in ("bita/train.py",
                 "scripts/retrain_branch_a_live.py",
                 "scripts/retrain_future_models_live.py"):
        assert "require_full_density(" in open(path).read(), f"{path} is unguarded"


# ---------------------------------------------------------------------------
# Full density (rows_per_file=None) must actually work
# ---------------------------------------------------------------------------

def test_no_script_multiplies_a_possibly_none_row_cap():
    """`rows_per_file * stride` raised TypeError the moment full density
    became the default -- the Branch A retrain died on launch.

    None means "no cap" and must propagate as None, not become 0 (which the
    adapters would read as "read nothing") and not raise.
    """
    import pathlib
    import re
    offenders = []
    for path in (pathlib.Path("scripts/retrain_branch_a_live.py"),
                 pathlib.Path("scripts/retrain_future_models_live.py")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if not re.search(r"rows_per_file\s*\*\s*stride", stripped):
                continue
            # The guarded form is correct:
            #   _cap = None if rows_per_file is None else rows_per_file * stride
            if "is None else" in stripped:
                continue
            offenders.append(f"{path}:{i}")
    assert not offenders, (
        "unguarded rows_per_file * stride (None at full density): "
        + ", ".join(offenders)
    )


def test_the_guard_keeps_none_as_none():
    """The shape of the fix: None in, None out; a real cap still multiplies."""
    for rows_per_file, stride, expected in [(None, 20, None), (100, 20, 2000), (50, 1, 50)]:
        cap = None if rows_per_file is None else rows_per_file * stride
        assert cap == expected


# Branch A reads through data_unification.training_sources (shared with the
# encoder and the Branch B/DeepOP PCAP path), so that is where its stride lives.
@pytest.mark.parametrize("script", ["data_unification.training_sources", "retrain_future_models_live"])
def test_strided_keeps_everything_when_want_is_none(script):
    """`want` is None at full density. Comparing int >= None raises, and
    defaulting it to 0 would silently return an empty list -- worse than the
    crash, because training would proceed on nothing.

    This crashed the Branch A retrain on launch, twice: once on
    `rows_per_file * stride` and once here on `len(out) >= want`.
    """
    import importlib
    import sys
    sys.path.insert(0, "scripts")
    _strided = importlib.import_module(script)._strided

    assert len(_strided(iter(range(100)), 1, None)) == 100
    assert len(_strided(iter(range(100)), 5, None)) == 20
    assert len(_strided(iter(range(100)), 1, 10)) == 10
    assert _strided(iter([]), 1, None) == []


@pytest.mark.parametrize("script", ["data_unification.training_sources", "retrain_future_models_live"])
def test_strided_defaults_want_to_none(script):
    """The default must be 'no cap', matching full density."""
    import importlib
    import inspect
    import sys
    sys.path.insert(0, "scripts")
    sig = inspect.signature(importlib.import_module(script)._strided)
    assert sig.parameters["want"].default is None
