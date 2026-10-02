"""
tests/test_serving_flow_snapshot.py

Pins for telemetry/flow/flow_table.py defects found in the serving-path
review ("Reported, not fixed"):

  * snapshot_flows() used to export CUMULATIVE bytes/packets since flow
    start, not the traffic seen in that window. A steady flow's exported
    volume therefore ramped window over window even though nothing on the
    wire changed -- a train/serve skew, since training counts each flow's
    bytes once, against the window it happened in. Measured on the live
    path: a steady 20 kB/2 s flow reported 20k, 40k, 60k, 80k, 100k across
    five consecutive windows.

  * snapshot_flows(max_flows=N) used to keep the first N flows in dict
    insertion (flow-creation) order. Under a port scan, the newest probes
    are exactly the ones constantly being created, so they landed at the
    END of insertion order and were dropped by the cap -- the discarded
    flows were the signal, not the noise.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from telemetry.flow.flow_table import LiveFlowTable


def _pkt(ts, src="10.0.0.5", dst="10.0.0.9", sport=51000, dport=443,
         length=1000, proto=6, payload=900):
    return {
        "src_ip": src, "dst_ip": dst, "src_port": sport, "dst_port": dport,
        "protocol": proto, "timestamp": ts, "packet_length": length,
        "payload_length": payload, "tcp_flags": {},
    }


class TestSnapshotReportsPerWindowDeltaNotCumulative:
    def test_steady_flow_reports_equal_volume_each_window(self):
        """20 kB/2 s of steady traffic must report ~20 kB every window, not
        a cumulative 20k, 40k, 60k, 80k, 100k ramp."""
        table = LiveFlowTable()
        reported_bytes = []
        t = 0.0
        for _window in range(5):
            for _pkt_i in range(20):
                t += 0.1
                table.process_packet(_pkt(t))
            snap = table.snapshot_flows()
            assert len(snap) == 1
            reported_bytes.append(snap[0]["fwd_bytes"])

        assert reported_bytes == [20000] * 5, (
            "expected equal per-window volume every window (traffic never "
            f"changed) but got {reported_bytes}"
        )

    def test_new_flow_first_export_is_its_own_traffic_not_zero(self):
        table = LiveFlowTable()
        for i in range(5):
            table.process_packet(_pkt(1.0 + i * 0.1))
        snap = table.snapshot_flows()
        assert snap[0]["fwd_bytes"] == 5000
        assert snap[0]["fwd_packets"] == 5

    def test_flow_dropped_by_the_cap_keeps_its_bytes_for_next_export(self):
        """A flow NOT included in a capped snapshot must not lose the bytes
        it carried that window -- they must appear whole in a later export,
        and an already-exported flow's next export must be delta-only."""
        table = LiveFlowTable()
        table.process_packet(_pkt(0.0, dst="10.0.0.1", dport=1))   # flow A: old
        table.process_packet(_pkt(1.0, dst="10.0.0.2", dport=2))   # flow B: newer

        snap1 = table.snapshot_flows(max_flows=1)
        assert len(snap1) == 1
        assert snap1[0]["dst_ip"] == "10.0.0.2", "the newer flow should survive the cap"

        table.process_packet(_pkt(2.0, dst="10.0.0.1", dport=1))
        table.process_packet(_pkt(2.0, dst="10.0.0.2", dport=2))

        snap2 = table.snapshot_flows(max_flows=None)
        by_dst = {f["dst_ip"]: f for f in snap2}
        assert by_dst["10.0.0.1"]["fwd_bytes"] == 2000, (
            "flow A was never exported before (dropped by the first cap) -- "
            "its full two-packet volume must show up whole here"
        )
        assert by_dst["10.0.0.2"]["fwd_bytes"] == 1000, (
            "flow B was already exported once; this export must be only the "
            "NEW packet since then, not the cumulative total"
        )


class TestSnapshotCapKeepsTheNewestFlowsUnderAScan:
    def test_port_scan_survives_the_cap_instead_of_the_oldest_flow(self):
        """One old, boring flow plus many brand-new single-packet probe
        flows in the same window, more than fit under a small cap. The scan
        probes -- the newest flows -- must be what survives, not the old
        flow that happened to be inserted first."""
        table = LiveFlowTable()
        table.process_packet(_pkt(0.0, dst="10.0.0.1", dport=1))  # inserted first

        n_probes = 10
        for i in range(n_probes):
            table.process_packet(
                _pkt(1.0 + i * 0.01, dst="10.0.0.9", sport=51000 + i, dport=2000 + i)
            )

        cap = 5
        snap = table.snapshot_flows(max_flows=cap)
        assert len(snap) == cap
        dports = {f["dst_port"] for f in snap}
        assert 1 not in dports, (
            "the old, first-inserted flow should have been dropped in "
            "favour of the newer scan probes"
        )
        assert all(p >= 2000 for p in dports), (
            f"expected only the newest port-scan probes to survive the cap, "
            f"got dst_ports={dports}"
        )
