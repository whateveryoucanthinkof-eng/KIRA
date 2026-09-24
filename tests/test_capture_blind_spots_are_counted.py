"""A flooded sensor must say it is blind, not look like a quiet network.

The live path is a single-threaded Python AF_PACKET loop. Past its throughput
ceiling the kernel drops frames before the process sees them, and the state
recorder -- the only route from sensor to models -- dropped whole windows
when its queue was full. Neither was counted, so a volumetric flood produced
windows that were incomplete or missing with nothing to distinguish them from
low traffic.
"""

import queue
import struct

import pytest

import telemetry.capture.sniffer as sniffer_mod
from control_backend.capture_accounting import CaptureAccounting
from telemetry.capture.sniffer import StreamingPacketSniffer
from telemetry.run_telemetry import (
    AsyncPcapRecorder,
    AsyncStateRecorder,
    _capture_completeness,
    _emit_window,
)


class _StatsSocket:
    """Stands in for an AF_PACKET socket; each read returns then resets, like Linux."""

    def __init__(self, reads):
        self.reads = list(reads)
        self.calls = []

    def getsockopt(self, level, opt, buflen=None):
        self.calls.append((level, opt, buflen))
        packets, drops = self.reads.pop(0) if self.reads else (0, 0)
        return struct.pack("II", packets, drops)


@pytest.fixture
def linux_socket_module(monkeypatch):
    # AF_PACKET does not exist on Windows/macOS; the counter only exists on Linux.
    monkeypatch.setattr(sniffer_mod.socket, "AF_PACKET", 17, raising=False)


def test_kernel_drops_are_read_per_interval_and_totalled(linux_socket_module):
    s = StreamingPacketSniffer("eth-test")
    s.sock = _StatsSocket([(1000, 37), (500, 0), (200, 200)])

    first = s.read_kernel_stats()
    assert first == {"available": True, "packets": 1000, "drops": 37,
                     "packets_total": 1000, "drops_total": 37}
    assert s.sock.calls[0] == (sniffer_mod.SOL_PACKET, sniffer_mod.PACKET_STATISTICS, 8)

    assert s.read_kernel_stats()["drops"] == 0
    third = s.read_kernel_stats()
    assert third["drops"] == 200 and third["drops_total"] == 237
    assert s.kernel_packets_total == 1700


def test_no_counter_is_reported_as_unavailable_not_as_zero_drops(linux_socket_module):
    s = StreamingPacketSniffer("eth-test")
    assert s.read_kernel_stats()["available"] is False  # no socket yet

    class _Broken:
        def getsockopt(self, *a):
            raise OSError("not a packet socket")

    s.sock = _Broken()
    assert s.read_kernel_stats()["available"] is False


@pytest.mark.parametrize("cls,args", [(AsyncStateRecorder, ({},)), (AsyncPcapRecorder, (0.0, b"x"))])
def test_recorders_count_what_they_drop(cls, args):
    rec = cls.__new__(cls)  # no worker thread: the queue must stay full
    rec.queue = queue.Queue(maxsize=2)
    rec.dropped = 0
    for _ in range(5):
        rec.record(*args)
    assert rec.dropped == 3
    assert rec.queue.qsize() == 2


def test_each_window_carries_its_capture_completeness(linux_socket_module):
    s = StreamingPacketSniffer("eth-test")
    s.sock = _StatsSocket([(4000, 1000)])
    s.loop_errors = 2
    rec = AsyncStateRecorder.__new__(AsyncStateRecorder)
    rec.queue, rec.dropped = queue.Queue(maxsize=1), 4

    cap = _capture_completeness(s, rec, None)
    assert cap["kernel_stats_available"] is True
    assert cap["kernel_drops"] == 1000 and cap["kernel_drop_ratio"] == 0.25
    assert cap["state_records_dropped"] == 4
    assert cap["capture_loop_errors"] == 2
    assert cap["pcap_frames_dropped"] == 0

    line = _emit_window({"flows": [], "capture": cap}, None)
    assert "KERNEL DROPPED 1000" in line and "incomplete" in line
    assert "4 windows never reached the backend" in line

    clean = _emit_window({"flows": [], "capture": {"kernel_drops": 0}}, None)
    assert "!!" not in clean


def test_backend_counts_missing_and_incomplete_windows():
    acc = CaptureAccounting()
    acc.observe({"window_id": 0})
    acc.observe({"window_id": 1})
    got = acc.observe({"window_id": 5, "capture": {"kernel_drops": 12, "kernel_drop_ratio": 0.1}})
    assert got == {"windows_missed": 3, "kernel_drops": 12}
    acc.observe({"window_id": 6})
    assert acc.status() == {"sensorKernelDrops": 12, "incompleteWindows": 1, "windowsMissed": 3}


def test_backend_sequence_restarts_with_a_new_sensor():
    acc = CaptureAccounting()
    acc.observe({"window_id": 40})
    acc.reset_sequence()
    acc.observe({"window_id": 0})
    acc.observe({"window_id": 1})
    assert acc.windows_missed == 0


def test_records_without_ids_or_capture_are_tolerated():
    acc = CaptureAccounting()
    acc.observe({})
    acc.observe({"window_id": True, "capture": None})
    assert acc.status() == {"sensorKernelDrops": 0, "incompleteWindows": 0, "windowsMissed": 0}
