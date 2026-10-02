"""
tests/test_serving_window_contract.py

Pin for the hardcoded-window-divisor defect in the serving-path review
of the serving path: several places in
telemetry/** defaulted their window size to a literal `2.0` instead of
`cyberworld_v4.config.get_contract().window_seconds`, the single
authoritative source (cyberworld_v4/config.py: "Anything that hardcodes a
window size... is a contract violation"). A literal default silently
diverges the moment the contract ever changes; a contract-derived default
cannot.

These tests monkeypatch the contract to a value that is NOT 2.0, so a
hardcoded `2.0` default is caught even though it happens to equal today's
real contract value -- equality alone would not distinguish "derived from
the contract" from "coincidentally the same number".
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cyberworld_v4.config as cfg_mod
from cyberworld_v4.config import TemporalContract


def _with_patched_contract(monkeypatch, window_seconds: float):
    fake = TemporalContract(window_seconds=window_seconds, history_steps=15, forecast_steps=5)
    monkeypatch.setattr(cfg_mod, "DEFAULT_CONTRACT", fake)


class TestStateBuilderDefaultsToTheContract:
    def test_default_window_sec_follows_a_patched_contract(self, monkeypatch):
        _with_patched_contract(monkeypatch, 7.25)
        from telemetry.state.state_builder import LiveStateBuilder

        sb = LiveStateBuilder()
        assert sb.window_sec == 7.25

    def test_explicit_window_sec_still_wins(self, monkeypatch):
        _with_patched_contract(monkeypatch, 7.25)
        from telemetry.state.state_builder import LiveStateBuilder

        sb = LiveStateBuilder(window_sec=3.0)
        assert sb.window_sec == 3.0


class TestFlowTableDefaultsToTheContract:
    def test_extract_window_features_default_follows_a_patched_contract(self, monkeypatch):
        _with_patched_contract(monkeypatch, 9.5)
        from telemetry.flow.flow_table import LiveFlowTable

        table = LiveFlowTable()
        table.process_packet({
            "src_ip": "10.0.0.5", "dst_ip": "10.0.0.9", "src_port": 51000,
            "dst_port": 443, "protocol": 6, "timestamp": 0.0,
            "packet_length": 100, "payload_length": 50, "tcp_flags": {},
        })
        # idle_mean = max(window_sec - duration, 0); duration ~= 0, so
        # idle_mean ~= window_sec when the contract default is honoured.
        feats = table.extract_window_features()
        assert feats["idle_mean"] == 9.5


class TestPcapEngineDefaultsToTheContract:
    def test_extract_features_accepts_no_window_sec_without_hardcoding_2(self, monkeypatch):
        _with_patched_contract(monkeypatch, 11.0)
        from telemetry.packet.pcap_engine import LivePCAPEngine
        import inspect

        sig = inspect.signature(LivePCAPEngine.extract_features)
        assert sig.parameters["window_sec"].default is None, (
            "extract_features's window_sec should default to the contract, "
            "not a hardcoded literal"
        )


class TestRunTelemetryUsesTheContractNotALiteral(object):
    def test_run_telemetry_source_has_no_hardcoded_window_sec_literal(self):
        import inspect
        import telemetry.run_telemetry as rt

        src = inspect.getsource(rt.main)
        assert "window_sec=2.0" not in src, (
            "run_telemetry.py still hardcodes window_sec=2.0 instead of "
            "reading get_contract().window_seconds"
        )
        assert "get_contract" in src
