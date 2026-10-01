"""
Multi-Dataset Stream & Host Trajectory Extractor.

Aggregates flows from CIC-IDS2017, CTU-13, and Warden into unified time-windowed graph snapshots,
runs TGNE-TA to compute per-host embeddings H_t[v], computes the 15 per-host window temporal attributes,
and produces structured host sequence dictionaries.
"""

import os
import collections
from typing import List, Dict, Tuple, Optional, Iterator, Any
from dataclasses import dataclass, field
import numpy as np
import torch
import pandas as pd

from data_unification.unified_schema import UnifiedFlowRecord, CoarseCategory
from data_unification.host_attributes import (
    EXTENDED_HOST_ATTR_DIM,
    HOST_ATTR_DIM,
    PACKET_ATTR_DIM,
    normalize_packet_features,
    BYTE_LOG_SCALE,
    BYTE_RATE_LOG_SCALE,
    COUNT_LOG_SCALE,
    DURATION_SCALE_SECONDS,
    PEER_COUNT_LOG_SCALE,
    PORT_COUNT_LOG_SCALE,
)
from data_unification.flow_to_temporal_event import FlowToTemporalEventAdapter, TemporalEventStream
from data_unification.cic2017_adapter import CIC2017Adapter
from data_unification.ctu13_adapter import CTU13Adapter
from data_unification.warden_adapter import WardenAdapter
from data_unification.auth_log_adapter import AuthEventRecord, AuthLogAdapter, AuthEventType, AuthLogSource
from data_unification.behavioral_fingerprint import BehavioralFlowFingerprinter, BehavioralProfile
from data_unification.trajectory_store import EMB_DIM, TrajectoryStore, TrajectoryStoreBuilder
from dataclasses import field


def _packet_features_of(records) -> Optional[Dict[str, float]]:
    """The packet-level feature dict attached to this host-window, if any.

    `pcap_bridge` attaches ONE shared dict to every record of a host-window, so
    reading the first record that carries one is both correct and cheap -- and
    the sharing is why this does not cost 30 floats per record at corpus scale.
    """
    for r in records:
        md = getattr(r, "metadata", None)
        if md:
            pf = md.get("packet_features")
            if pf:
                return pf
    return None


#: MITRE ATT&CK tactic -> base severity, used to rank which of several attack
#: categories seen in one window describes the host's state. Module level
#: because it is a constant: it used to be rebuilt inside the per-host loop,
#: i.e. once per host per window (~25M dict constructions on a full-corpus run).
TACTIC_BASE_SEVERITY = {
    "Benign": 0.0,
    "Recon": 0.35,
    "Reconnaissance": 0.35,
    "Discovery": 0.38,
    "InitialAccess": 0.60,
    "CredentialAccess": 0.65,
    "Execution": 0.72,
    "Persistence": 0.75,
    "PrivilegeEscalation": 0.78,
    "DefenseEvasion": 0.75,
    "C2": 0.82,
    "CommandAndControl": 0.82,
    "LateralMovement": 0.85,
    "Exfiltration": 0.92,
    "Impact": 0.96,
}

#: How a host earns the attack label from the flows in its window.
#:
#:   "either"  host is either endpoint of an attack flow (attacker OR victim)
#:   "target"  host is the DESTINATION of an attack flow (victim only)
#:   "source"  host is the SOURCE of an attack flow (attacker only)
#:
#: "either" is the historical behaviour and stays the default, but it was never
#: a decision -- `host_recs` simply matched on both endpoints. It is now named,
#: recorded, and selectable, because the three targets mean different things and
#: a benchmark has to say which one it scored.
ATTACK_ROLES = ("either", "target", "source")


def _window_attack_label(atk_recs):
    """Coarse category and the UNION of technique ids for one host-window.

    The previous form was `atk_recs[0].coarse_category` / `atk_recs[0].attck_technique_ids`
    -- whichever attack flow happened to sort first decided the label and every
    other technique in the window was discarded. That silently nullified the
    multilabel technique target `cyberworld_v4.targets` builds downstream, which
    can only be as multilabel as the snapshot it reads.

    The category is now the most severe present (deterministic, and the one an
    operator would triage on) and the technique list is the union, ordered by
    first appearance so the result is stable across runs.
    """
    coarse = max(
        (r.coarse_category for r in atk_recs),
        key=lambda c: (TACTIC_BASE_SEVERITY.get(c, 0.50), c),
    )
    techs, seen = [], set()
    for r in atk_recs:
        for t in r.attck_technique_ids or ():
            if t not in seen:
                seen.add(t)
                techs.append(t)
    return coarse, techs


def _attack_window_risk(coarse: str, n_attack: int, n_host: int) -> float:
    """Severity risk of an attack host-window: tactic severity + density + volume.

    Shared by extract_trajectories and extract_trajectories_columns so the two
    cannot drift apart.
    """
    base_sev = TACTIC_BASE_SEVERITY.get(coarse, 0.50)
    atk_density = min(1.0, n_attack / max(1, n_host))
    vol_scale = min(1.0, float(np.log1p(n_attack) / 5.0))
    return min(1.0, max(0.20, base_sev + 0.04 * atk_density + 0.04 * vol_scale))


class _LabelView:
    """The two record fields _window_attack_label reads, for the columnar path."""
    __slots__ = ("coarse_category", "attck_technique_ids")

    def __init__(self, coarse_category, attck_technique_ids):
        self.coarse_category = coarse_category
        self.attck_technique_ids = attck_technique_ids


# ---------------------------------------------------------------------------
# Heuristic label OVERRIDE -- off by default
# ---------------------------------------------------------------------------
#
# Two blocks in extract_trajectories could set `is_attack = True`, rewrite
# `coarse_category`, prepend a technique id and raise `risk_score` AFTER the
# dataset's own label had been read:
#
#   auth:       is_brute_force_flag == 1.0  or  auth_failed_count >= 5
#   behaviour:  is_shell_detected  == 1.0  or  port_mismatch_count >= 2
#
# They are not a small correction. Measured at full density, replaying the
# exact label path of extract_trajectories on real captures with the
# contract's 2 s window:
#
#   CIC-2017 Monday  (train, 529,601 flows, ZERO attack rows in the corpus)
#       203,760 host-windows, 28,748 (14.11%) relabelled attack/Execution,
#       every one of them a fabricated positive.
#   CIC-2017 Wednesday (the held-out TEST split, 692,373 flows)
#       ground truth 178 attack host-windows (0.19%); after the override
#       9,659 (10.11%). 9,481 of 9,659 -- 98.2% of the split's positives --
#       are the heuristic, not the data.
#   CIC-2018 wed_29 (val, stride 4)   11.93% -> 17.45%
#   CTU-13 scenario 11 (val, stride 4) 0.57% ->  9.47%
#
# The dominant driver is a bug in the detector itself: fingerprint_flow marks
# `is_port_mismatch = dst_port not in STANDARD_WEB_PORTS` for its C2Beaconing
# profile, and that profile is any flow with 1-3 packets each way, <=1200
# bytes and a mean packet under 180 B -- i.e. essentially every DNS, NTP or
# short service exchange. On Monday 166,598 of the mismatches came from that
# one profile.
#
# A detector that labels 14% of a capture with no attacks in it is a source of
# systematic label noise, and any metric computed over these labels is partly
# measuring the heuristic rather than the model. The override is therefore
# OFF unless explicitly requested; the dataset's own label stands.
#
# The auth block is additionally dead in every offline path: no caller in this
# repository ever supplies auth_events, so `all_auth_events` is always empty.
_LABEL_OVERRIDE_ENV = "CYBERWORLD_HEURISTIC_LABEL_OVERRIDE"


def heuristic_label_override_enabled() -> bool:
    """True when the auth/behavioural heuristics may overwrite a dataset label."""
    return os.environ.get(_LABEL_OVERRIDE_ENV, "") in ("1", "true", "True", "yes")
@dataclass(slots=True)
class HostWindowSnapshot:
    """Per-host state snapshot for a single time window t.

    slots=True and the absence of the old auth_metrics/behavioral_metrics dicts
    are load-bearing, not style: a full-corpus run holds ~25M of these at once,
    where the per-instance __dict__ plus two populated metric dicts cost ~305
    bytes each (~7.6 GB measured). Those two dicts were written here but never
    read by any consumer -- they are consumed locally in extract_trajectories to
    adjust risk_score before the snapshot is built, and sequence_dataset /
    train_branch_b / train_cwa_decoder only ever touch embedding, temporal_attrs,
    is_attack, coarse_category, technique_ids, window_idx, window_end and
    risk_score. Adding a field here is not free; check the memory budget first.
    """
    host_ip: str
    host_id: int
    window_idx: int
    window_start: float
    window_end: float
    embedding: np.ndarray  # H_t[v] in R^d (d=12 from TGNE-TA)
    temporal_attrs: np.ndarray  # 15 temporal scalar attributes
    is_attack: bool
    coarse_category: str
    technique_ids: List[str]
    risk_score: float  # Ground truth compromise/risk score in [0, 1]


class BoundedHostSlidingBuffer:
    """
    Bounded sliding window buffer for per-host state snapshots.
    
    Prevents Python heap memory bloat and garbage collection latency spikes
    by enforcing strict O(active_hosts * max_windows) memory bounds with O(1) eviction
    and idle endpoint pruning.
    """

    def __init__(self, max_windows_per_host: int = 16, max_idle_ttl_sec: float = 3600.0):
        self.max_windows = max_windows_per_host
        self.max_idle_ttl_sec = max_idle_ttl_sec
        self._buffers: Dict[str, collections.deque] = {}
        self._last_active_time: Dict[str, float] = {}

    def append(self, snapshot: HostWindowSnapshot):
        """Appends a snapshot, automatically evicting oldest window beyond max_windows in O(1)."""
        hip = snapshot.host_ip
        if hip not in self._buffers:
            self._buffers[hip] = collections.deque(maxlen=self.max_windows)
        self._buffers[hip].append(snapshot)
        self._last_active_time[hip] = max(self._last_active_time.get(hip, 0.0), snapshot.window_end)

    def extend_host(self, host_ip: str, snapshots: List[HostWindowSnapshot]):
        """Appends multiple snapshots for a host."""
        for s in snapshots:
            self.append(s)

    def get_host_trajectory(self, host_ip: str) -> List[HostWindowSnapshot]:
        """Retrieves current sliding window history for a host."""
        return list(self._buffers.get(host_ip, []))

    def get_all_trajectories(self) -> Dict[str, List[HostWindowSnapshot]]:
        """Returns snapshot lists for all currently tracked hosts."""
        return {hip: list(dq) for hip, dq in self._buffers.items()}

    def prune_stale_hosts(self, current_time: float, max_idle_sec: Optional[float] = None) -> int:
        """Evicts endpoints that have been idle past max_idle_sec."""
        ttl = max_idle_sec or self.max_idle_ttl_sec
        stale_ips = [
            hip for hip, last_ts in self._last_active_time.items()
            if (current_time - last_ts) > ttl
        ]
        for hip in stale_ips:
            self._buffers.pop(hip, None)
            self._last_active_time.pop(hip, None)
        return len(stale_ips)

    def get_memory_stats(self, current_time: Optional[float] = None) -> Dict[str, Any]:
        """Reports active buffer memory utilization and capacity metrics."""
        total_snaps = sum(len(dq) for dq in self._buffers.values())
        tracked_hosts = len(self._buffers)
        # Approximate footprint: ~320 bytes per HostWindowSnapshot numpy array reference
        est_kb = (total_snaps * 0.32) + (tracked_hosts * 0.1)
        active_10m = 0
        if current_time is not None:
            active_10m = sum(1 for ts in self._last_active_time.values() if (current_time - ts) <= 600.0)

        return {
            "tracked_hosts": tracked_hosts,
            "total_snapshots": total_snaps,
            "max_windows_per_host": self.max_windows,
            "estimated_memory_kb": round(est_kb, 2),
            "active_hosts_last_10m": active_10m,
        }

    def __len__(self) -> int:
        return len(self._buffers)

    def __contains__(self, host_ip: str) -> bool:
        return host_ip in self._buffers


class _WindowedNeighborFinder:
    """Restricts a NeighborFinder to interactions at or after `lower_bound`.

    The TGNE graph for a window is that window's flows. Long-range history
    reaches the embedding through the TGN memory (BiTA), not through the
    neighbour lookup. Without this bound, offline extraction built one graph
    per data chunk and each embedding took a host's 10 most recent
    interactions from ANY earlier window, while live serving only ever had the
    current window's flows -- the same model saw different graphs in training
    and in serving. Padding id 0 is what the attention layer masks.
    """

    def __init__(self, inner):
        self.inner = inner
        self.lower_bound = -np.inf

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
        nbrs, eidx, times = self.inner.get_temporal_neighbor(
            source_nodes, timestamps, n_neighbors=n_neighbors)
        stale = times < self.lower_bound
        if stale.any():
            nbrs = nbrs.copy(); eidx = eidx.copy(); times = times.copy()
            nbrs[stale] = 0
            eidx[stale] = 0
            times[stale] = 0.0
        return nbrs, eidx, times

    def __getattr__(self, name):
        return getattr(self.inner, name)


class HostTrajectoryExtractor:
    """Extracts per-host embedding and temporal attribute trajectories across time windows.

    Per window: build the window's interaction graph, update TGN memory (BiTA)
    from earlier windows, read the 12-D host embedding, then queue the window's
    own interactions for the next memory update.
    """

    def __init__(
        self,
        tgne_ta_model,
        window_size_sec: float = 60.0,
        n_temporal_attrs: int = 15,
        auth_events: Optional[List[AuthEventRecord]] = None,
        spill_dir: Optional[str] = None,
        heuristic_label_augmentation: bool = False,
        attack_role: str = "either",
        include_packet_features: bool = False,
        persist_memory: bool = False,
        n_neighbors: Optional[int] = None,
    ):
        """
        `n_neighbors` -- how many temporal neighbours the graph-attention
        embedding reads per host. None (the default) takes the value the
        encoder was trained with, which build_or_load_tgne_ta attaches from its
        config; override only to study the effect, since the attention weights
        were fitted to the trained count.

        Each host's 12-D latent sees at most that many of its most recent flows
        in the window; the 15 attributes see all of them. A host with more
        flows than that -- a busy server, or one an attacker floods with benign
        traffic after an attack flow -- has the rest left out of its latent
        entirely. `neighbor_exposure_report()` counts how often that happens and
        how often it hides every attack flow of an attack window.

        `persist_memory` -- keep the encoder's TGN memory and host-id map across
        calls. Live serving sets it: each call is one 2 s window of one
        continuous session. Offline extraction leaves it off: each call is a
        separate capture and starts from empty memory, except the continuation
        chunks of chunked extraction (`emit_after` set), which carry on.

        `heuristic_label_augmentation` -- OFF by default, and the default changed.
        CYBERWORLD_HEURISTIC_LABEL_OVERRIDE=1 turns it on as well (either switch).

        Two blocks below used to overwrite `is_attack` from heuristics rather
        than from the corpus label:

          * auth: `auth_failed_count >= 5` -> CredentialAccess / T1110
          * behavioural: `is_shell_detected` or `port_mismatch_count >= 2`
            -> Execution / T1059

        The second is the damaging one. `BehavioralFlowFingerprinter` classifies
        any flow with `tot_bytes <= 1200`, 1-3 packets each way and mean packet
        size < 180 as C2 beaconing, and marks it a port mismatch whenever the
        destination port is not 80/443/8000/8080/8443. Two ordinary DNS lookups
        in one window satisfy both, so the host was labelled "under attack".
        Measured downstream: `results/v4_benchmark.json` reports a
        `current_attack_rate` of 0.677 and a test-split base rate of 0.9997 on
        corpora whose published attack fraction runs 0.03%-57%.

        That is fatal in two separate ways. The labels stop being ground truth,
        so no accuracy number measures detection; and the heuristic reads the
        same flow statistics the model is given as input, so the model is graded
        on reproducing a function of its own features.

        The fingerprinter is still useful -- as a FEATURE. Leave this False and
        feed `compute_host_behavioral_metrics` into the attribute vector if you
        want the signal; do not let it write the target.

        `attack_role` -- which endpoint of an attack flow earns the label; see
        ATTACK_ROLES. Default "either" preserves existing behaviour.

        `include_packet_features` -- append the 30 packet-level attributes
        `telemetry/packet/pcap_engine.py` computes, widening the attribute
        vector from 15 to 45 and the model input from 27-D to 57-D. OFF by
        default because it changes the contract and invalidates every shipped
        checkpoint; the features reach this class through
        `record.metadata["packet_features"]`, which only the PCAP path can
        populate. See host_attributes.PACKET_ATTRIBUTES for what is in it and
        why it matters.
        """
        if attack_role not in ATTACK_ROLES:
            raise ValueError(
                f"attack_role must be one of {ATTACK_ROLES}, got {attack_role!r}"
            )
        self.tgn = tgne_ta_model
        self.window_size_sec = window_size_sec
        self.n_temporal_attrs = n_temporal_attrs
        # When set, the bulk (N, 27) feature block is written here and mapped
        # back read-only, so a full-density corpus does not have to fit in RAM.
        self.spill_dir = spill_dir
        self.heuristic_label_augmentation = bool(heuristic_label_augmentation)
        self.attack_role = attack_role
        self.include_packet_features = bool(include_packet_features)
        if self.include_packet_features:
            # n_temporal_attrs is the contract this extractor emits; widen it
            # here rather than letting a 45-wide vector surprise a consumer
            # that asked for 15.
            expected = HOST_ATTR_DIM + PACKET_ATTR_DIM
            if n_temporal_attrs == HOST_ATTR_DIM:
                self.n_temporal_attrs = EXTENDED_HOST_ATTR_DIM
            elif n_temporal_attrs != expected:
                raise ValueError(
                    f"include_packet_features needs n_temporal_attrs="
                    f"{expected}, got {n_temporal_attrs}"
                )
        self.auth_events: List[AuthEventRecord] = list(auth_events) if auth_events else []
        self.persist_memory = bool(persist_memory)
        self.n_neighbors = int(n_neighbors if n_neighbors is not None
                               else getattr(tgne_ta_model, "serving_n_neighbors", 10))
        self.neighbor_uniform = bool(getattr(tgne_ta_model, "serving_neighbor_uniform", False))
        self._exposure = dict.fromkeys(
            ("host_windows", "truncated_host_windows", "flows", "flows_outside_latent",
             "attack_host_windows", "attack_host_windows_all_attack_flows_outside_latent"), 0)
        # Host-id map that stays fixed while memory is carried, so memory row k
        # keeps meaning the same host. Rebuilt whenever memory is reset.
        self._event_adapter: Optional[FlowToTemporalEventAdapter] = None
        # Timestamp of the newest interaction already queued for memory.
        self._ingested_until = -np.inf

    @property
    def uses_memory(self) -> bool:
        return bool(getattr(self.tgn, "use_memory", False))

    def reset_memory_state(self) -> None:
        """Forget the encoder's memory and host ids (new capture or session)."""
        if hasattr(self.tgn, "reset_state"):
            self.tgn.reset_state()
        self._event_adapter = None
        self._ingested_until = -np.inf

    def _adapter_for_call(self, continuing: bool) -> FlowToTemporalEventAdapter:
        if not self.uses_memory:
            return FlowToTemporalEventAdapter(window_size_sec=self.window_size_sec)
        if not continuing or self._event_adapter is None:
            self.reset_memory_state()
            self._event_adapter = FlowToTemporalEventAdapter(window_size_sec=self.window_size_sec)
        return self._event_adapter

    def add_auth_events(self, events: List[AuthEventRecord]):
        """Ingests additional host authentication security events."""
        self.auth_events.extend(events)

    def _account_exposure(self, host_recs, atk_recs) -> None:
        """Count flows the host's latent cannot see (most-recent sampling only).

        The windowed neighbour finder hands the encoder the host's last
        `n_neighbors` interactions before the window end, ordered by start
        time -- the same order as `host_recs`. Anything earlier is absent from
        this window's latent, though still counted by the 15 attributes. With
        use_memory on (not the shipped config) every interaction is also queued
        for the host's memory, so an evicted flow reaches a LATER window's
        latent that way; this counts the current window only.
        """
        if self.neighbor_uniform:
            return  # uniform sampling draws from all of them; no fixed cut-off
        e = self._exposure
        n = len(host_recs)
        hidden = max(0, n - self.n_neighbors)
        e["host_windows"] += 1
        e["flows"] += n
        if hidden:
            e["truncated_host_windows"] += 1
            e["flows_outside_latent"] += hidden
        if atk_recs:
            e["attack_host_windows"] += 1
            if hidden:
                visible = {id(r) for r in host_recs[-self.n_neighbors:]}
                if not any(id(r) in visible for r in atk_recs):
                    e["attack_host_windows_all_attack_flows_outside_latent"] += 1

    def neighbor_exposure_report(self, reset: bool = False) -> Dict[str, Any]:
        """How often the neighbour cut-off hid flows -- and attacks -- from the latent.

        `attack_hidden_from_latent_rate` is the share of attack host-windows in
        which EVERY attack flow fell outside the latent's view, so only the
        aggregate attributes could carry the attack. On benign traffic this is
        a property of the data; under adversarial flooding it is the evasion
        path, and it takes only `n_neighbors` benign flows after the attack.

        `reset=True` zeroes the counters after reading, so one extractor can
        report each split separately.
        """
        e = dict(self._exposure)
        if reset:
            self._exposure = dict.fromkeys(self._exposure, 0)
        nan = float("nan")
        e["n_neighbors"] = self.n_neighbors
        e["sampling"] = "uniform" if self.neighbor_uniform else "most_recent"
        e["counted"] = not self.neighbor_uniform
        e["truncated_rate"] = (e["truncated_host_windows"] / e["host_windows"]
                               if e["host_windows"] else nan)
        e["flows_outside_latent_rate"] = (e["flows_outside_latent"] / e["flows"]
                                          if e["flows"] else nan)
        e["attack_hidden_from_latent_rate"] = (
            e["attack_host_windows_all_attack_flows_outside_latent"] / e["attack_host_windows"]
            if e["attack_host_windows"] else nan)
        return e

    def compute_host_temporal_attributes(
        self,
        host_ip: str,
        window_records: List[UnifiedFlowRecord],
        window_duration: float,
        packet_features: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Computes 15 per-host temporal scalar attributes for window t:
        0: flow_count
        1: log1p(fwd_bytes)
        2: log1p(bwd_bytes)
        3: log1p(total_bytes)
        4: log1p(fwd_packets)
        5: log1p(bwd_packets)
        6: log1p(total_packets)
        7: unique_peers_count
        8: unique_dst_ports_count
        9: tcp_ratio
        10: udp_ratio
        11: avg_duration
        12: byte_rate (normalized)
        13: packet_rate (normalized)
        14: active_connection_density
        """
        attrs = np.zeros(self.n_temporal_attrs, dtype=np.float32)
        if self.include_packet_features:
            attrs[HOST_ATTR_DIM:] = normalize_packet_features(
                packet_features if packet_features is not None
                else _packet_features_of(window_records)
            )
        n = len(window_records)
        if n == 0:
            return attrs

        fwd_b = sum(r.fwd_bytes for r in window_records)
        bwd_b = sum(r.bwd_bytes for r in window_records)
        tot_b = fwd_b + bwd_b
        fwd_p = sum(r.fwd_packets for r in window_records)
        bwd_p = sum(r.bwd_packets for r in window_records)
        tot_p = fwd_p + bwd_p

        peers = set()
        ports = set()
        tcp_count = 0
        udp_count = 0
        tot_dur = 0.0

        for r in window_records:
            peer = r.dst_ip if r.src_ip == host_ip else r.src_ip
            peers.add(peer)
            ports.add(r.dst_port)
            if r.protocol == 6:
                tcp_count += 1
            elif r.protocol == 17:
                udp_count += 1
            tot_dur += r.duration

        # Divisors come from host_attributes so the values and the documented
        # normalisation cannot drift apart. PEER_COUNT_LOG_SCALE and
        # PORT_COUNT_LOG_SCALE were raised from 5.0, which saturated both
        # attributes at 147 -- flattening exactly the fan-out and port-sweep
        # range that distinguishes a scan from ordinary traffic.
        dur = max(1.0, window_duration)
        attrs[0] = min(1.0, np.log1p(n) / COUNT_LOG_SCALE)
        attrs[1] = min(1.0, np.log1p(fwd_b) / BYTE_LOG_SCALE)
        attrs[2] = min(1.0, np.log1p(bwd_b) / BYTE_LOG_SCALE)
        attrs[3] = min(1.0, np.log1p(tot_b) / BYTE_LOG_SCALE)
        attrs[4] = min(1.0, np.log1p(fwd_p) / COUNT_LOG_SCALE)
        attrs[5] = min(1.0, np.log1p(bwd_p) / COUNT_LOG_SCALE)
        attrs[6] = min(1.0, np.log1p(tot_p) / COUNT_LOG_SCALE)
        attrs[7] = min(1.0, np.log1p(len(peers)) / PEER_COUNT_LOG_SCALE)
        attrs[8] = min(1.0, np.log1p(len(ports)) / PORT_COUNT_LOG_SCALE)
        attrs[9] = float(tcp_count) / n
        attrs[10] = float(udp_count) / n
        attrs[11] = min(1.0, (tot_dur / n) / DURATION_SCALE_SECONDS)
        attrs[12] = min(1.0, np.log1p(tot_b / dur) / BYTE_RATE_LOG_SCALE)
        attrs[13] = min(1.0, np.log1p(tot_p / dur) / COUNT_LOG_SCALE)
        attrs[14] = min(1.0, float(len(peers)) / max(1, n))

        return attrs

    def extract_trajectories(
        self,
        records: List[UnifiedFlowRecord],
        adapter: Optional[FlowToTemporalEventAdapter] = None,
        auth_events: Optional[List[AuthEventRecord]] = None,
        sliding_buffer: Optional["BoundedHostSlidingBuffer"] = None,
        builder: Optional[TrajectoryStoreBuilder] = None,
        emit_after: Optional[float] = None,
        window_idx_base: int = 0,
    ) -> "TrajectoryStore | Dict[str, List[HostWindowSnapshot]]":
        """
        Groups flows into `window_size_sec` windows, extracts TGNE-TA embeddings
        H_t, correlates host authentication logs, and builds per-host timelines.
        """
        continuing = self.persist_memory or emit_after is not None
        adapter = adapter or self._adapter_for_call(continuing)
        event_stream = adapter.process_records(records, sort_by_time=True)
        windowed_nf: Optional[_WindowedNeighborFinder] = None

        all_auth_events: List[AuthEventRecord] = self.auth_events + (list(auth_events) if auth_events else [])
        # Read once, not once per host-window.
        label_override = self.heuristic_label_augmentation or heuristic_label_override_enabled()

        if hasattr(self.tgn, "embedding_module") and len(event_stream.sources) > 0:
            from utils.utils import NeighborFinder
            n_nodes = max(event_stream.n_nodes + 10, getattr(self.tgn, "n_nodes", 0))
            adj_list = [[] for _ in range(n_nodes)]
            for i in range(len(event_stream.sources)):
                u = int(event_stream.sources[i])
                v = int(event_stream.destinations[i])
                t = float(event_stream.timestamps[i])
                adj_list[u].append((v, i, t))
                adj_list[v].append((u, i, t))
            windowed_nf = _WindowedNeighborFinder(NeighborFinder(adj_list, uniform=self.neighbor_uniform))
            self.tgn.neighbor_finder = windowed_nf
            self.tgn.embedding_module.neighbor_finder = windowed_nf
            edge_feats_t = torch.from_numpy(event_stream.edge_features).float().to(self.tgn.device)
            self.tgn.edge_raw_features = edge_feats_t
            self.tgn.embedding_module.edge_features = edge_feats_t
            # Intrinsic IP node features, matching TRAINING exactly.
            #
            # This was torch.zeros(...). The encoder is trained with 12-D
            # features derived from each host's address (see
            # data_unification/ip_features.py), and handing it zeros at
            # inference is a train/serve mismatch that silently destroys the
            # inductive capability those features exist to provide -- the
            # difference between inductive AUC 0.83 and 0.50.
            from data_unification.ip_features import build_node_feature_matrix
            node_feats_t = torch.from_numpy(
                build_node_feature_matrix(event_stream.ip_to_id, n_nodes=n_nodes)
            ).float().to(self.tgn.device)
            self.tgn.node_raw_features = node_feats_t
            self.tgn.embedding_module.node_features = node_feats_t
            self.tgn.n_nodes = n_nodes
            if hasattr(self.tgn, "ensure_capacity"):
                self.tgn.ensure_capacity(n_nodes)

        # Map window boundaries to lists of records
        # `builder` lets a caller accumulate across several chunked calls, so a
        # full-density corpus can be processed a slice at a time while the
        # records for each slice are freed. `emit_after` drops snapshots from a
        # chunk's warm-up overlap -- those windows were already emitted by the
        # previous chunk, and the overlap exists only so TGNE sees the same
        # neighbour history it would have seen processing everything at once.
        owns_builder = builder is None
        if builder is None:
            builder = TrajectoryStoreBuilder(
                spill_dir=self.spill_dir,
                feat_dim=EMB_DIM + self.n_temporal_attrs,
            )
        sorted_records = sorted(records, key=lambda r: r.start_time)

        for win_idx, (win_start, win_end, s_idx, e_idx) in enumerate(event_stream.window_boundaries):
            if s_idx >= e_idx:
                continue

            win_recs = sorted_records[s_idx:e_idx]
            # Identify active hosts in this window
            active_ips = set()
            for r in win_recs:
                active_ips.add(r.src_ip)
                active_ips.add(r.dst_ip)

            # Also check if any hosts had auth events in this window
            for aev in all_auth_events:
                if win_start <= aev.timestamp <= win_end:
                    active_ips.add(aev.host_ip)

            active_ips_sorted = sorted(list(active_ips))
            active_host_ids = np.array([adapter.ip_to_id.get(ip, 0) for ip in active_ips_sorted], dtype=int)

            # 1. Memory <- BiTA(messages from EARLIER windows) for this
            #    window's hosts (TGN memory_update_at_start order).
            if self.uses_memory:
                self.tgn.update_memory_for(active_host_ids)

            # 2. TGNE-TA embeddings H_t over THIS window's interaction graph.
            if windowed_nf is not None:
                windowed_nf.lower_bound = win_start
            with torch.no_grad():
                H_t = self.tgn.get_host_embeddings(
                    active_host_ids, timestamp=win_end, n_neighbors=self.n_neighbors
                ).cpu().numpy()

            # 3. Queue this window's interactions; they reach memory next time
            #    one of their hosts is active, never this window's own embedding.
            #    Only interactions newer than anything already queued: a
            #    continuation chunk re-reads the previous chunk's tail (and
            #    re-windows it on its own grid), and those flows must reach
            #    memory exactly once.
            if self.uses_memory:
                ts_w = event_stream.timestamps[s_idx:e_idx]
                new = ts_w > self._ingested_until
                if new.any():
                    self.tgn.store_interactions(
                        event_stream.sources[s_idx:e_idx][new],
                        event_stream.destinations[s_idx:e_idx][new],
                        ts_w[new],
                        event_stream.edge_idxs[s_idx:e_idx][new],
                    )
                    self._ingested_until = float(ts_w[new].max())

            # For each active host, compute temporal attributes, auth indicators & labels
            for idx, ip in enumerate(active_ips_sorted):
                host_recs = [r for r in win_recs if r.src_ip == ip or r.dst_ip == ip]
                attrs = self.compute_host_temporal_attributes(
                    ip, host_recs, self.window_size_sec
                )

                # Attack flows this host took part in, in the role the caller
                # asked for. `attack_role` used to be implicit: host_recs
                # matched either endpoint, so an attacker and its victim -- and
                # any benign server the attacker merely touched -- all carried
                # the same positive label.
                if self.attack_role == "target":
                    atk_recs = [r for r in host_recs if r.is_attack and r.dst_ip == ip]
                elif self.attack_role == "source":
                    atk_recs = [r for r in host_recs if r.is_attack and r.src_ip == ip]
                else:
                    atk_recs = [r for r in host_recs if r.is_attack]

                if windowed_nf is not None:
                    self._account_exposure(host_recs, atk_recs)

                is_atk = len(atk_recs) > 0
                if is_atk:
                    # Most severe category present + UNION of techniques, not
                    # whichever record happened to sort first.
                    coarse, techs = _window_attack_label(atk_recs)
                    risk = _attack_window_risk(coarse, len(atk_recs), len(host_recs))
                else:
                    coarse, techs, risk = "Benign", [], 0.0

                # Heuristic label augmentation. OFF by default -- see the
                # constructor docstring for why these two blocks are the reason
                # the measured attack rate was 0.68 on corpora whose published
                # rate is a few percent.
                if label_override:
                    auth_metrics = {}
                    if all_auth_events:
                        auth_metrics = AuthLogAdapter.extract_window_auth_metrics(
                            all_auth_events, ip, win_start, win_end
                        )
                        if auth_metrics.get("is_brute_force_flag", 0.0) == 1.0 or auth_metrics.get("auth_failed_count", 0.0) >= 5:
                            is_atk = True
                            coarse = "CredentialAccess"
                            if "T1110" not in techs:
                                techs = ["T1110"] + [t for t in techs if t != "T1110"]
                            brute_score = auth_metrics.get("auth_brute_force_score", 0.5)
                            risk = max(risk, min(1.0, 0.75 + 0.25 * brute_score))

                    behavioral_metrics = BehavioralFlowFingerprinter.compute_host_behavioral_metrics(ip, host_recs)
                    if behavioral_metrics.get("is_shell_detected", 0.0) == 1.0 or behavioral_metrics.get("port_mismatch_count", 0.0) >= 2:
                        is_atk = True
                        if coarse == "Benign":
                            coarse = "Execution"
                        if "T1059" not in techs:
                            techs = ["T1059"] + techs
                        risk = max(risk, min(1.0, 0.65 + 0.35 * behavioral_metrics.get("behavioral_threat_score", 0.5)))

                if sliding_buffer is not None:
                    sliding_buffer.append(HostWindowSnapshot(
                        host_ip=ip,
                        host_id=adapter.ip_to_id.get(ip, 0),
                        window_idx=win_idx,
                        window_start=win_start,
                        window_end=win_end,
                        embedding=H_t[idx],
                        temporal_attrs=attrs,
                        is_attack=is_atk,
                        coarse_category=coarse,
                        technique_ids=techs,
                        risk_score=float(risk),
                    ))
                elif emit_after is None or win_start >= emit_after:
                    builder.append(
                        host_ip=ip,
                        host_id=adapter.ip_to_id.get(ip, 0),
                        window_idx=win_idx + window_idx_base,
                        window_start=win_start,
                        window_end=win_end,
                        embedding=H_t[idx],
                        temporal_attrs=attrs,
                        is_attack=is_atk,
                        coarse_category=coarse,
                        technique_ids=techs,
                        risk_score=float(risk),
                    )

        if sliding_buffer is not None:
            return sliding_buffer.get_all_trajectories()
        return builder.finalize() if owns_builder else builder

    # ------------------------------------------------------------------
    # Columnar extraction (data_unification/capture_columns.py)
    # ------------------------------------------------------------------

    def columns_unsupported_reason(self) -> Optional[str]:
        """Why extract_trajectories_columns cannot reproduce extract_trajectories
        under this extractor's settings, or None when it can.

        Every mode it refuses needs something the columns do not carry (the
        records themselves, packet features, auth events), carries state
        across calls, or draws random numbers in a different order (uniform
        neighbour sampling).
        """
        if self.heuristic_label_augmentation or heuristic_label_override_enabled():
            return "the heuristic label override fingerprints whole records"
        if self.auth_events:
            return "auth events are matched against records"
        if self.persist_memory:
            return "persist_memory carries state across calls (live serving)"
        if self.neighbor_uniform:
            return "uniform neighbour sampling draws random numbers per node"
        if self.n_neighbors < 0:
            return "negative n_neighbors"
        return None

    def extract_trajectories_columns(
        self,
        cols,
        builder: Optional[TrajectoryStoreBuilder] = None,
        window_idx_base: int = 0,
    ) -> "TrajectoryStore | TrajectoryStoreBuilder":
        """extract_trajectories(records, builder=..., window_idx_base=...) on a
        capture_columns.CaptureColumns: the same snapshots, bit for bit.

        Only how the same numbers are computed differs:

        * no Python object per record -- ~127 B/record of columns instead of
          ~1 KB/record of UnifiedFlowRecord plus its shared metadata (measured
          998 B/record resident on a CIC-2018 PCAP day);
        * each window's per-host attributes are grouped with numpy instead of
          a list comprehension over the window's records per active host,
          which is O(hosts x records) per window;
        * the neighbour finder is built in CSR form directly, in exactly the
          per-node order the list-of-tuples form produces, instead of one
          Python tuple per edge direction.

        Float summation order is preserved where it matters: a host's total
        duration is accumulated left to right, as the loop did. Integer sums
        are exact either way. tests/test_capture_columns.py checks the stores.
        """
        why = self.columns_unsupported_reason()
        if why is not None:
            raise ValueError(f"columnar extraction unavailable: {why}; use extract_trajectories")

        n = len(cols)
        src_c = np.asarray(cols.src)
        dst_c = np.asarray(cols.dst)
        timestamps = np.asarray(cols.start, dtype=np.float64)
        n_ips = len(cols.ips)
        pkt_table = getattr(cols, "pkt_table", None)
        if self.include_packet_features and pkt_table is None:
            raise ValueError("include_packet_features needs capture columns of format >= 3 "
                             "(the 'pkt' column and pkt_feats.npy); rebuild the column cache")

        # Node ids exactly as FlowToTemporalEventAdapter.get_or_create_node_id
        # assigns them over time-sorted records: src then dst, first seen first.
        id_of_code = np.zeros(max(n_ips, 1), dtype=np.int64)
        if n:
            inter = np.empty(2 * n, dtype=np.int64)
            inter[0::2] = src_c
            inter[1::2] = dst_c
            uniq, first = np.unique(inter, return_index=True)
            seen_order = uniq[np.argsort(first, kind="stable")]
            id_of_code[seen_order] = np.arange(1, seen_order.size + 1, dtype=np.int64)
            del inter, uniq, first
        else:
            seen_order = np.zeros(0, dtype=np.int64)
        ip_to_id = {cols.ips[int(c)]: int(i) for i, c in enumerate(seen_order, start=1)}

        # What _adapter_for_call(continuing=False) leaves behind.
        if self.uses_memory:
            self.reset_memory_state()
            self._event_adapter = FlowToTemporalEventAdapter(
                window_size_sec=self.window_size_sec, initial_ip_map=ip_to_id)

        sources = id_of_code[src_c] if n else np.zeros(0, dtype=np.int64)
        destinations = id_of_code[dst_c] if n else np.zeros(0, dtype=np.int64)
        edge_idxs = np.arange(n, dtype=int)

        windowed_nf: Optional[_WindowedNeighborFinder] = None
        if hasattr(self.tgn, "embedding_module") and n > 0:
            from utils.utils import NeighborFinder
            n_nodes = max(len(ip_to_id) + 1 + 10, getattr(self.tgn, "n_nodes", 0))
            # The record path appends (v, i, t) to adj[u] and then (u, i, t) to
            # adj[v] for i in time order, then stable-sorts each list on t -- a
            # no-op, since t is non-decreasing in i. So a node's entries run in
            # (i, src side first) order: a stable argsort of interleaved owners.
            owner = np.empty(2 * n, dtype=np.int64)
            owner[0::2] = sources
            owner[1::2] = destinations
            order = np.argsort(owner, kind="stable")
            peer = np.empty(2 * n, dtype=np.int32)
            peer[0::2] = destinations
            peer[1::2] = sources
            flat_nbr = peer[order]
            del peer
            flat_eidx = (order >> 1).astype(np.int32)
            flat_ts = timestamps[order >> 1]
            offsets = np.searchsorted(owner[order], np.arange(n_nodes + 1), side="left")
            del owner, order
            windowed_nf = _WindowedNeighborFinder(NeighborFinder(
                None, uniform=False, _csr=(flat_nbr, flat_eidx, flat_ts, offsets)))
            self.tgn.neighbor_finder = windowed_nf
            self.tgn.embedding_module.neighbor_finder = windowed_nf
            edge_feats_t = torch.from_numpy(cols.edge_features()).float().to(self.tgn.device)
            self.tgn.edge_raw_features = edge_feats_t
            self.tgn.embedding_module.edge_features = edge_feats_t
            from data_unification.ip_features import build_node_feature_matrix
            node_feats_t = torch.from_numpy(
                build_node_feature_matrix(ip_to_id, n_nodes=n_nodes)
            ).float().to(self.tgn.device)
            self.tgn.node_raw_features = node_feats_t
            self.tgn.embedding_module.node_features = node_feats_t
            self.tgn.n_nodes = n_nodes
            if hasattr(self.tgn, "ensure_capacity"):
                self.tgn.ensure_capacity(n_nodes)

        owns_builder = builder is None
        if builder is None:
            builder = TrajectoryStoreBuilder(
                spill_dir=self.spill_dir,
                feat_dim=EMB_DIM + self.n_temporal_attrs,
            )

        # A window's hosts are visited in sorted-address order.
        ip_rank = np.zeros(max(n_ips, 1), dtype=np.int64)
        if n_ips:
            ip_rank[np.array(sorted(range(n_ips), key=cols.ips.__getitem__), dtype=np.int64)] = \
                np.arange(n_ips, dtype=np.int64)
        pos_of_code = np.zeros(max(n_ips, 1), dtype=np.int64)
        K = self.n_neighbors
        count_exposure = windowed_nf is not None and not self.neighbor_uniform
        e_x = self._exposure
        role = self.attack_role
        dur_window = max(1.0, self.window_size_sec)
        cat_col = cols.cat
        tech_col = cols.tech

        for win_idx, (win_start, win_end, s_idx, e_idx) in enumerate(
                _window_boundaries(timestamps, self.window_size_sec)):
            if s_idx >= e_idx:
                continue
            m = e_idx - s_idx
            ws = np.asarray(src_c[s_idx:e_idx], dtype=np.int64)
            wd = np.asarray(dst_c[s_idx:e_idx], dtype=np.int64)
            act = np.unique(np.concatenate([ws, wd]))
            act = act[np.argsort(ip_rank[act], kind="stable")]
            H = act.size
            active_host_ids = np.array(id_of_code[act], dtype=int)

            if self.uses_memory:
                self.tgn.update_memory_for(active_host_ids)
            if windowed_nf is not None:
                windowed_nf.lower_bound = win_start
            with torch.no_grad():
                H_t = self.tgn.get_host_embeddings(
                    active_host_ids, timestamp=win_end, n_neighbors=self.n_neighbors
                ).cpu().numpy()
            if self.uses_memory:
                ts_w = timestamps[s_idx:e_idx]
                new = ts_w > self._ingested_until
                if new.any():
                    self.tgn.store_interactions(
                        sources[s_idx:e_idx][new],
                        destinations[s_idx:e_idx][new],
                        ts_w[new],
                        edge_idxs[s_idx:e_idx][new],
                    )
                    self._ingested_until = float(ts_w[new].max())
                if win_idx % _COMPACT_EVERY == 0:
                    _compact_pending_messages(self.tgn)

            # (host, record) incidences: each record once for its src, and once
            # more for its dst unless that is the same host. Sorted by (host
            # position, record index), each host's run IS its host_recs list.
            pos_of_code[act] = np.arange(H, dtype=np.int64)
            two = ws != wd
            jr = np.arange(m, dtype=np.int64)
            h = np.concatenate([pos_of_code[ws], pos_of_code[wd][two]])
            j = np.concatenate([jr, jr[two]])
            as_src = np.zeros(h.size, dtype=bool)
            as_src[:m] = True
            o = np.argsort(h * m + j, kind="stable")
            h, j, as_src = h[o], j[o], as_src[o]
            # `peer = r.dst_ip if r.src_ip == host_ip else r.src_ip`
            peer = np.where(as_src, wd[j], ws[j])
            is_src = as_src                     # r.src_ip == host
            is_dst = ~as_src | ~two[j]          # r.dst_ip == host
            counts = np.bincount(h, minlength=H)
            starts = np.zeros(H, dtype=np.int64)
            np.cumsum(counts[:-1], out=starts[1:])
            g = j + s_idx                       # row in the columns

            fwd_b = np.add.reduceat(np.asarray(cols.fwd_bytes[s_idx:e_idx])[j], starts)
            bwd_b = np.add.reduceat(np.asarray(cols.bwd_bytes[s_idx:e_idx])[j], starts)
            fwd_p = np.add.reduceat(np.asarray(cols.fwd_packets[s_idx:e_idx])[j], starts)
            bwd_p = np.add.reduceat(np.asarray(cols.bwd_packets[s_idx:e_idx])[j], starts)
            tot_b = fwd_b + bwd_b
            tot_p = fwd_p + bwd_p
            proto = np.asarray(cols.protocol[s_idx:e_idx])[j]
            tcp = np.bincount(h, weights=(proto == 6), minlength=H)
            udp = np.bincount(h, weights=(proto == 17), minlength=H)
            n_peers = _distinct_per_group(h, peer, H)
            n_ports = _distinct_per_group(h, np.asarray(cols.dst_port[s_idx:e_idx])[j], H)
            raw_d = (np.asarray(cols.end[s_idx:e_idx])[j]
                     - np.asarray(cols.start[s_idx:e_idx])[j])
            dur_rec = np.where(raw_d > 0.0, raw_d, 0.0)     # record.duration
            tot_dur = _sequential_group_sums(dur_rec, counts, starts)

            nf = counts.astype(np.float64)
            f64 = lambda a: a.astype(np.float64)
            attrs = np.zeros((H, self.n_temporal_attrs), dtype=np.float32)
            attrs[:, 0] = np.minimum(1.0, np.log1p(nf) / COUNT_LOG_SCALE)
            attrs[:, 1] = np.minimum(1.0, np.log1p(f64(fwd_b)) / BYTE_LOG_SCALE)
            attrs[:, 2] = np.minimum(1.0, np.log1p(f64(bwd_b)) / BYTE_LOG_SCALE)
            attrs[:, 3] = np.minimum(1.0, np.log1p(f64(tot_b)) / BYTE_LOG_SCALE)
            attrs[:, 4] = np.minimum(1.0, np.log1p(f64(fwd_p)) / COUNT_LOG_SCALE)
            attrs[:, 5] = np.minimum(1.0, np.log1p(f64(bwd_p)) / COUNT_LOG_SCALE)
            attrs[:, 6] = np.minimum(1.0, np.log1p(f64(tot_p)) / COUNT_LOG_SCALE)
            attrs[:, 7] = np.minimum(1.0, np.log1p(f64(n_peers)) / PEER_COUNT_LOG_SCALE)
            attrs[:, 8] = np.minimum(1.0, np.log1p(f64(n_ports)) / PORT_COUNT_LOG_SCALE)
            attrs[:, 9] = tcp / nf
            attrs[:, 10] = udp / nf
            attrs[:, 11] = np.minimum(1.0, (tot_dur / nf) / DURATION_SCALE_SECONDS)
            attrs[:, 12] = np.minimum(1.0, np.log1p(f64(tot_b) / dur_window) / BYTE_RATE_LOG_SCALE)
            attrs[:, 13] = np.minimum(1.0, np.log1p(f64(tot_p) / dur_window) / COUNT_LOG_SCALE)
            attrs[:, 14] = np.minimum(1.0, f64(n_peers) / nf)
            if self.include_packet_features:
                # The record path's _packet_features_of(host_recs): the FIRST
                # of the host's records in this window (window order) that
                # carries packet features. Groups are in (host, record) order,
                # so that is each host's first such row.
                pk = np.asarray(cols.pkt[s_idx:e_idx])[j]
                has = np.flatnonzero(pk >= 0)
                if has.size:
                    hh, first = np.unique(h[has], return_index=True)
                    attrs[hh, HOST_ATTR_DIM:] = np.asarray(pkt_table)[pk[has[first]]]

            atk_rec = np.asarray(cols.is_attack[s_idx:e_idx])[j]
            if role == "target":
                atk = atk_rec & is_dst
            elif role == "source":
                atk = atk_rec & is_src
            else:
                atk = atk_rec
            n_atk = np.bincount(h, weights=atk, minlength=H)
            is_atk = n_atk > 0

            if count_exposure:
                hidden = np.maximum(0, counts - K)
                e_x["host_windows"] += int(H)
                e_x["flows"] += int(h.size)
                e_x["truncated_host_windows"] += int((hidden > 0).sum())
                e_x["flows_outside_latent"] += int(hidden.sum())
                e_x["attack_host_windows"] += int(is_atk.sum())
                if K > 0:
                    rank_in = np.arange(h.size, dtype=np.int64) - starts[h]
                    visible = rank_in >= (counts[h] - K)
                else:                       # host_recs[-0:] is every record
                    visible = np.ones(h.size, dtype=bool)
                vis_atk = np.bincount(h, weights=atk & visible, minlength=H)
                e_x["attack_host_windows_all_attack_flows_outside_latent"] += int(
                    (is_atk & (hidden > 0) & (vis_atk == 0)).sum())

            coarse_l = ["Benign"] * H
            techs_l: List[List[str]] = [[] for _ in range(H)]
            risk = np.zeros(H, dtype=np.float64)
            if is_atk.any():
                arows = np.flatnonzero(atk)     # grouped by host, record order within
                ah = h[arows]
                ag = g[arows]
                acat = np.asarray(cat_col[ag])
                atech = np.asarray(tech_col[ag])
                bnd = np.flatnonzero(np.r_[True, ah[1:] != ah[:-1], True])
                cats, tls = cols.categories, cols.tech_lists
                for a, b in zip(bnd[:-1].tolist(), bnd[1:].tolist()):
                    hh = int(ah[a])
                    views = [_LabelView(cats[c], tls[t])
                             for c, t in zip(acat[a:b].tolist(), atech[a:b].tolist())]
                    coarse, techs = _window_attack_label(views)
                    coarse_l[hh] = coarse
                    techs_l[hh] = techs
                    risk[hh] = _attack_window_risk(coarse, b - a, int(counts[hh]))

            builder.append_batch(
                host_ips=[cols.ips[c] for c in act.tolist()],
                host_ids=active_host_ids,
                window_idx=win_idx + window_idx_base,
                window_start=win_start,
                window_end=win_end,
                embeddings=H_t,
                temporal_attrs=attrs,
                is_attack=is_atk,
                coarse_categories=coarse_l,
                technique_ids=techs_l,
                risk_scores=risk,
            )

        return builder.finalize() if owns_builder else builder


#: How often (in windows) extract_trajectories_columns compacts pending messages.
_COMPACT_EVERY = 64


def _compact_pending_messages(tgn) -> None:
    """Give each queued TGN memory message its own storage.

    `TGN.get_raw_messages` queues `source_message[i]` and `edge_times[i]`:
    row VIEWS of the whole window's message batch. A host that is never active
    again keeps its last message queued until the capture ends, and that one
    view pins the entire batch it was sliced from. Over a CIC-2018 PCAP day
    nearly every window's batch stays alive -- measured ~0.5 KB per snapshot,
    growing linearly, ~3.5 GB by the end of a 12M-record day.

    `clone()` copies exactly the viewed values, so the aggregator later reads
    bit-identical inputs; only the pinned batches are released. Messages that
    already own their storage (`_base is None`) are left alone, so each is
    copied at most once. bita/ is not ours to change, hence doing it here.
    """
    mem = getattr(tgn, "memory", None)
    msgs = getattr(mem, "messages", None)
    if not msgs:
        return
    for node, lst in msgs.items():
        if not lst:
            continue
        if any((x[0]._base is not None) or (x[1]._base is not None) for x in lst):
            msgs[node] = [
                (x[0].clone() if x[0]._base is not None else x[0],
                 x[1].clone() if x[1]._base is not None else x[1]) + tuple(x[2:])
                for x in lst]


def _window_boundaries(timestamps: np.ndarray, window_size_sec: float):
    """FlowToTemporalEventAdapter.process_records' window grid: same values, same types.

    The same loop, compared on Python floats for speed; the boundaries are
    built from the same numpy float64 anchor, so they are the same numpy
    scalars the record path yields.
    """
    import math
    n = len(timestamps)
    out = []
    if n == 0:
        return out
    ts = timestamps.tolist()
    first_t = timestamps[0]
    W = window_size_sec
    current_win_start = first_t
    bound = float(current_win_start + W)
    start_idx = 0
    for i in range(n):
        if ts[i] >= bound:
            out.append((current_win_start, current_win_start + W, start_idx, i))
            k = math.floor((timestamps[i] - first_t) / window_size_sec)
            current_win_start = first_t + k * window_size_sec
            bound = float(current_win_start + W)
            start_idx = i
    out.append((current_win_start, current_win_start + window_size_sec, start_idx, n))
    return out


def _distinct_per_group(group: np.ndarray, values: np.ndarray, n_groups: int) -> np.ndarray:
    """Number of distinct `values` within each group id 0..n_groups-1."""
    if group.size == 0:
        return np.zeros(n_groups, dtype=np.int64)
    o = np.lexsort((values, group))
    gs, vs = group[o], values[o]
    new = np.empty(gs.size, dtype=bool)
    new[0] = True
    new[1:] = (gs[1:] != gs[:-1]) | (vs[1:] != vs[:-1])
    return np.bincount(gs[new], minlength=n_groups)


#: Groups up to this long are summed in one padded matrix; longer ones in a loop.
_SEQ_SUM_PAD = 64


def _sequential_group_sums(x: np.ndarray, counts: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Per-group sum of non-negative float64 `x`, accumulated left to right from 0.0.

    That is `t = 0.0; for v in group: t += v`, not numpy's pairwise sum, which
    rounds differently. `np.cumsum` along an axis is strictly sequential, and
    trailing zero padding leaves a non-negative running sum unchanged, so the
    last column of a zero-padded row-wise cumsum is the loop's result exactly.
    """
    G = counts.size
    out = np.zeros(G, dtype=np.float64)
    if G == 0:
        return out
    small = counts <= _SEQ_SUM_PAD
    if small.any():
        gi = np.flatnonzero(small)
        cg = counts[gi]
        L = int(cg.max())
        if L > 0:
            mat = np.zeros((gi.size, L), dtype=np.float64)
            row = np.repeat(np.arange(gi.size), cg)
            col = np.arange(row.size) - np.repeat(np.cumsum(cg) - cg, cg)
            mat[row, col] = x[np.repeat(starts[gi], cg) + col]
            out[gi] = np.cumsum(mat, axis=1)[:, -1]
    for gg in np.flatnonzero(~small).tolist():
        t = 0.0
        for v in x[starts[gg]:starts[gg] + counts[gg]].tolist():
            t += v
        out[gg] = t
    return out


def format_neighbor_exposure(report: Dict[str, Any], where: str = "") -> str:
    """One line for a training log; see HostTrajectoryExtractor.neighbor_exposure_report."""
    if not report.get("counted"):
        return f"{where}: neighbour exposure not counted ({report.get('sampling')} sampling)"
    return (
        f"{where}: latent sees the last {report['n_neighbors']} flows per host-window -- "
        f"{report['truncated_rate']:.1%} of {report['host_windows']} host-windows truncated, "
        f"{report['flows_outside_latent_rate']:.1%} of flows outside the latent, "
        f"{report['attack_host_windows_all_attack_flows_outside_latent']} of "
        f"{report['attack_host_windows']} attack host-windows had EVERY attack flow outside it "
        f"({report['attack_hidden_from_latent_rate']:.1%})"
    )
