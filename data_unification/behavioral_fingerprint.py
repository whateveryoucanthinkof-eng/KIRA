"""
Behavioral Flow-Size & Traffic Profile Fingerprinting.

Replaces static port-to-service assumptions with behavioral fingerprinting:
1. Detects interactive shells / reverse shells on arbitrary ports (e.g., port 4444, 8080, 9001).
2. Detects periodic Command & Control (C2) beaconing without fixed port signatures.
3. Detects burst reconnaissance and bulk exfiltration from flow size distributions,
   packet ratios, and directional asymmetry.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any, Optional
import numpy as np

from data_unification.unified_schema import UnifiedFlowRecord


class BehavioralProfile(str, Enum):
    BENIGN_WEB = "BenignWeb"
    INTERACTIVE_SHELL = "InteractiveShell"
    C2_BEACONING = "C2Beaconing"
    BURST_RECON = "BurstRecon"
    BULK_EXFILTRATION = "BulkExfiltration"
    UNKNOWN = "Unknown"


# Common standard service ports for anomaly mismatch detection
STANDARD_WEB_PORTS = {80, 443, 8000, 8080, 8443}
STANDARD_SHELL_PORTS = {22, 23}
STANDARD_DNS_PORTS = {53}


@dataclass
class FlowBehavioralResult:
    """Behavioral fingerprint assessment for a single flow record."""
    profile: str
    is_port_mismatch: bool
    confidence: float
    avg_packet_size: float
    directional_asymmetry: float
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BehavioralFlowFingerprinter:
    """Analyzes flow packet-size distributions, durations, and byte asymmetry."""

    @classmethod
    def fingerprint_flow(cls, record: UnifiedFlowRecord) -> FlowBehavioralResult:
        """
        Classifies behavioral traffic profile based purely on packet & byte statistics,
        independent of the destination port.
        """
        tot_bytes = record.total_bytes
        tot_pkts = record.total_packets
        dur = record.duration
        fwd_b = record.fwd_bytes
        bwd_b = record.bwd_bytes
        fwd_p = record.fwd_packets
        bwd_p = record.bwd_packets
        dst_p = record.dst_port

        avg_pkt_size = (tot_bytes / max(1, tot_pkts))
        avg_fwd_size = (fwd_b / max(1, fwd_p))
        avg_bwd_size = (bwd_b / max(1, bwd_p))
        asym = float((fwd_b - bwd_b) / (tot_bytes + 1e-5))

        # 1. Interactive Shell / Reverse Shell:
        # Differentiates normal web requests from interactive command shells & persistent tunnels
        is_shell = False
        if dst_p not in STANDARD_WEB_PORTS:
            if (dur >= 2.0 and fwd_p >= 4 and bwd_p >= 4 and
                avg_fwd_size <= 250 and avg_bwd_size <= 350 and tot_bytes < 50000):
                is_shell = True
        else:
            # On web ports (80, 443): must exhibit long-lived persistent interactive tunnel
            if (dur >= 10.0 and fwd_p >= 15 and bwd_p >= 15 and
                avg_fwd_size <= 200 and avg_bwd_size <= 250 and tot_bytes < 50000):
                is_shell = True

        if is_shell:
            # Port mismatch if not running on standard SSH/Telnet ports (22, 23)
            mismatch = dst_p not in STANDARD_SHELL_PORTS
            conf = 0.92 if mismatch else 0.85
            return FlowBehavioralResult(
                profile=BehavioralProfile.INTERACTIVE_SHELL.value,
                is_port_mismatch=mismatch,
                confidence=conf,
                avg_packet_size=round(avg_pkt_size, 2),
                directional_asymmetry=round(asym, 4),
                details=f"Interactive keystroke pattern detected on port {dst_p}" + (" [NON-STANDARD PORT]" if mismatch else ""),
            )

        # 2. Burst Reconnaissance:
        # Rapid SYN or probe packet with zero or single response, short duration
        is_recon = (dur <= 0.5 and tot_pkts <= 3 and bwd_p <= 1)
        if is_recon:
            return FlowBehavioralResult(
                profile=BehavioralProfile.BURST_RECON.value,
                is_port_mismatch=False,
                confidence=0.90,
                avg_packet_size=round(avg_pkt_size, 2),
                directional_asymmetry=round(asym, 4),
                details=f"Unidirectional probe burst to port {dst_p}",
            )

        # 3. Bulk Exfiltration:
        # Heavy outward flow, high forward packet size, high asymmetry
        is_exfil = (fwd_b > 30000 and asym > 0.75 and avg_fwd_size > 800)
        if is_exfil:
            mismatch = (dst_p in STANDARD_DNS_PORTS or dst_p in STANDARD_SHELL_PORTS)
            return FlowBehavioralResult(
                profile=BehavioralProfile.BULK_EXFILTRATION.value,
                is_port_mismatch=mismatch,
                confidence=0.95,
                avg_packet_size=round(avg_pkt_size, 2),
                directional_asymmetry=round(asym, 4),
                details=f"High-volume directional outbound transfer ({fwd_b} bytes)",
            )

        # 4. C2 Beaconing:
        # Periodic tiny payloads, balanced handshakes
        is_c2 = (tot_bytes <= 1200 and fwd_p in (1, 2, 3) and bwd_p in (1, 2, 3) and avg_pkt_size < 180)
        if is_c2:
            mismatch = dst_p not in STANDARD_WEB_PORTS
            return FlowBehavioralResult(
                profile=BehavioralProfile.C2_BEACONING.value,
                is_port_mismatch=mismatch,
                confidence=0.80,
                avg_packet_size=round(avg_pkt_size, 2),
                directional_asymmetry=round(asym, 4),
                details=f"Low-volume heartbeat telemetry on port {dst_p}",
            )

        # Default standard web or data transaction
        return FlowBehavioralResult(
            profile=BehavioralProfile.BENIGN_WEB.value if dst_p in STANDARD_WEB_PORTS else BehavioralProfile.UNKNOWN.value,
            is_port_mismatch=False,
            confidence=0.70,
            avg_packet_size=round(avg_pkt_size, 2),
            directional_asymmetry=round(asym, 4),
            details=f"Standard traffic on port {dst_p}",
        )

    @classmethod
    def compute_host_behavioral_metrics(
        cls,
        host_ip: str,
        records: List[UnifiedFlowRecord],
    ) -> Dict[str, float]:
        """
        Computes aggregate behavioral fingerprint scores for a host across sliding window flows.
        """
        if not records:
            return {
                "shell_behavior_count": 0.0,
                "c2_beacon_count": 0.0,
                "port_mismatch_count": 0.0,
                "exfil_flow_count": 0.0,
                "behavioral_threat_score": 0.0,
                "is_shell_detected": 0.0,
            }

        shell_cnt = 0
        c2_cnt = 0
        mismatch_cnt = 0
        exfil_cnt = 0

        for r in records:
            res = cls.fingerprint_flow(r)
            if res.profile == BehavioralProfile.INTERACTIVE_SHELL.value:
                shell_cnt += 1
            elif res.profile == BehavioralProfile.C2_BEACONING.value:
                c2_cnt += 1
            elif res.profile == BehavioralProfile.BULK_EXFILTRATION.value:
                exfil_cnt += 1

            if res.is_port_mismatch:
                mismatch_cnt += 1

        n = len(records)
        threat_score = min(1.0, (shell_cnt * 0.4 + mismatch_cnt * 0.3 + c2_cnt * 0.2 + exfil_cnt * 0.3) / max(1, n * 0.5))

        return {
            "shell_behavior_count": float(shell_cnt),
            "c2_beacon_count": float(c2_cnt),
            "port_mismatch_count": float(mismatch_cnt),
            "exfil_flow_count": float(exfil_cnt),
            "behavioral_threat_score": float(threat_score),
            "is_shell_detected": 1.0 if shell_cnt > 0 else 0.0,
        }
