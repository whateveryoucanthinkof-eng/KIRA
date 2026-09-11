"""
Multimodal Authentication & L7 Log Ingestion Adapter.

Breaks the L4 NetFlow visibility ceiling on credential-access attacks (MITRE ATT&CK T1110)
by normalizing host authentication logs and L7 HTTP authentication telemetry:
1. Windows Security Event Logs: Event ID 4625 (Logon Failure), Event ID 4624 (Logon Success).
2. Linux Syslog / Auth.log: sshd failed password, PAM authentication failure, invalid user.
3. L7 HTTP Proxy / Web Logs: HTTP 401 (Unauthorized) password spray and brute-force bursts.
"""

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import List, Dict, Any, Optional
import datetime
import re
import numpy as np


class AuthEventType(str, Enum):
    FAILED_LOGON = "FailedLogon"
    SUCCESSFUL_LOGON = "SuccessfulLogon"
    PRIVILEGE_ELEVATION = "PrivilegeElevation"
    ACCOUNT_LOCKED = "AccountLocked"


class AuthLogSource(str, Enum):
    WINDOWS_EVENT_4625 = "WindowsEvent4625"
    WINDOWS_EVENT_4624 = "WindowsEvent4624"
    LINUX_AUTH_LOG = "LinuxAuthLog"
    HTTP_PROXY_401 = "HttpProxy401"
    SYNTHETIC = "SyntheticAuth"


@dataclass
class AuthEventRecord:
    """Standardized authentication security event record."""
    timestamp: float  # Normalized UTC epoch seconds
    host_ip: str  # Destination / Target host experiencing the auth event
    src_ip: str = ""  # Client / Attacking host originating the authentication
    user_account: str = ""  # Target username
    auth_event_type: str = AuthEventType.FAILED_LOGON.value
    log_source: str = AuthLogSource.WINDOWS_EVENT_4625.value
    failure_reason: str = ""  # e.g., BadPassword, UnknownUser, HTTP401
    service_name: str = "ssh"  # e.g., ssh, rdp, smb, http_basic_auth
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuthLogAdapter:
    """Normalizes heterogeneous host authentication logs into AuthEventRecord streams."""

    # Regex patterns for standard Linux auth.log / syslog sshd failures
    LINUX_FAILED_PATTERN = re.compile(
        r"(?P<month>[A-Za-z]{3})\s+(?P<day>\d+)\s+(?P<time>\d+:\d+:\d+).*sshd\[\d+\]:\s+"
        r"(Failed password|Invalid user)\s+(for\s+invalid\s+user\s+|for\s+)?(?P<user>\S+)\s+from\s+(?P<src_ip>\d+\.\d+\.\d+\.\d+)"
    )
    LINUX_ACCEPTED_PATTERN = re.compile(
        r"(?P<month>[A-Za-z]{3})\s+(?P<day>\d+)\s+(?P<time>\d+:\d+).*sshd\[\d+\]:\s+"
        r"Accepted\s+\S+\s+for\s+(?P<user>\S+)\s+from\s+(?P<src_ip>\d+\.\d+\.\d+\.\d+)"
    )

    MONTH_MAP = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }

    @classmethod
    def parse_windows_event(cls, record: Dict[str, Any]) -> AuthEventRecord:
        """
        Parses Windows Security Event 4625 (Logon Failure) or 4624 (Logon Success).
        Expected fields: EventID, TimeCreated, TargetUserName, IpAddress, WorkstationName, Status.
        """
        event_id = int(record.get("EventID", 4625))
        is_failure = (event_id == 4625)

        raw_time = record.get("TimeCreated", record.get("timestamp", 0.0))
        if isinstance(raw_time, (int, float)):
            ts = float(raw_time)
        else:
            try:
                dt = datetime.datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
                ts = dt.timestamp()
            except Exception:
                ts = 0.0

        host_ip = str(record.get("Computer", record.get("host_ip", "127.0.0.1")))
        src_ip = str(record.get("IpAddress", record.get("src_ip", "")))
        if src_ip in ("-", "::1", "127.0.0.1", ""):
            src_ip = record.get("WorkstationName", src_ip)

        user = str(record.get("TargetUserName", record.get("user", "Administrator")))
        failure_code = str(record.get("SubStatus", record.get("Status", "0xC000006A")))

        event_type = AuthEventType.FAILED_LOGON.value if is_failure else AuthEventType.SUCCESSFUL_LOGON.value
        source = AuthLogSource.WINDOWS_EVENT_4625.value if is_failure else AuthLogSource.WINDOWS_EVENT_4624.value

        return AuthEventRecord(
            timestamp=ts,
            host_ip=host_ip,
            src_ip=src_ip,
            user_account=user,
            auth_event_type=event_type,
            log_source=source,
            failure_reason=f"Status_{failure_code}" if is_failure else "Success",
            service_name=record.get("LogonProcessName", "User32"),
            metadata={"logon_type": record.get("LogonType", 3)},
        )

    @classmethod
    def parse_linux_auth_line(
        cls,
        line: str,
        host_ip: str = "127.0.0.1",
        reference_year: int = 2024,
    ) -> Optional[AuthEventRecord]:
        """Parses a single line from /var/log/auth.log."""
        m_fail = cls.LINUX_FAILED_PATTERN.search(line)
        if m_fail:
            month = cls.MONTH_MAP.get(m_fail.group("month"), 1)
            day = int(m_fail.group("day"))
            h, m, s = [int(x) for x in m_fail.group("time").split(":")]
            dt = datetime.datetime(reference_year, month, day, h, m, s, tzinfo=datetime.timezone.utc)
            return AuthEventRecord(
                timestamp=dt.timestamp(),
                host_ip=host_ip,
                src_ip=m_fail.group("src_ip"),
                user_account=m_fail.group("user"),
                auth_event_type=AuthEventType.FAILED_LOGON.value,
                log_source=AuthLogSource.LINUX_AUTH_LOG.value,
                failure_reason="InvalidCredentials",
                service_name="sshd",
            )

        m_ok = cls.LINUX_ACCEPTED_PATTERN.search(line)
        if m_ok:
            month = cls.MONTH_MAP.get(m_ok.group("month"), 1)
            day = int(m_ok.group("day"))
            h, m, s = [int(x) for x in m_ok.group("time").split(":")]
            dt = datetime.datetime(reference_year, month, day, h, m, s, tzinfo=datetime.timezone.utc)
            return AuthEventRecord(
                timestamp=dt.timestamp(),
                host_ip=host_ip,
                src_ip=m_ok.group("src_ip"),
                user_account=m_ok.group("user"),
                auth_event_type=AuthEventType.SUCCESSFUL_LOGON.value,
                log_source=AuthLogSource.LINUX_AUTH_LOG.value,
                failure_reason="Success",
                service_name="sshd",
            )

        return None

    @classmethod
    def parse_http_auth_event(cls, record: Dict[str, Any]) -> AuthEventRecord:
        """Parses web application HTTP 401 Unauthorized log entries."""
        status_code = int(record.get("status_code", 401))
        is_fail = (status_code in (401, 403))
        ts = float(record.get("timestamp", 0.0))

        return AuthEventRecord(
            timestamp=ts,
            host_ip=str(record.get("server_ip", record.get("host_ip", "127.0.0.1"))),
            src_ip=str(record.get("client_ip", record.get("src_ip", ""))),
            user_account=str(record.get("user", record.get("attempted_user", "anonymous"))),
            auth_event_type=AuthEventType.FAILED_LOGON.value if is_fail else AuthEventType.SUCCESSFUL_LOGON.value,
            log_source=AuthLogSource.HTTP_PROXY_401.value,
            failure_reason=f"HTTP_{status_code}",
            service_name="http_basic_auth",
            metadata={"uri": record.get("uri", "/login")},
        )

    @staticmethod
    def extract_window_auth_metrics(
        auth_events: List[AuthEventRecord],
        host_ip: str,
        window_start: float,
        window_end: float,
    ) -> Dict[str, float]:
        """
        Extracts aggregated authentication features for a specific host in a sliding window.
        """
        fails = 0
        successes = 0
        users = set()
        src_ips = set()

        for ev in auth_events:
            # Check host relevance and temporal window
            if ev.host_ip == host_ip and (window_start <= ev.timestamp <= window_end):
                users.add(ev.user_account)
                if ev.src_ip:
                    src_ips.add(ev.src_ip)
                if ev.auth_event_type == AuthEventType.FAILED_LOGON.value:
                    fails += 1
                elif ev.auth_event_type == AuthEventType.SUCCESSFUL_LOGON.value:
                    successes += 1

        total_logons = fails + successes
        fail_ratio = (fails / total_logons) if total_logons > 0 else 0.0
        
        # Brute-force score: high fail count, high fail ratio, or multi-user spray
        user_div = min(1.0, len(users) / 5.0)
        fail_vel = min(1.0, fails / 10.0)
        brute_score = 0.5 * fail_vel + 0.3 * fail_ratio + 0.2 * user_div

        return {
            "auth_failed_count": float(fails),
            "auth_success_count": float(successes),
            "auth_fail_ratio": float(fail_ratio),
            "unique_target_users_count": float(len(users)),
            "unique_auth_sources_count": float(len(src_ips)),
            "auth_brute_force_score": float(brute_score),
            "is_brute_force_flag": 1.0 if (fails >= 5 and fail_ratio >= 0.7) else 0.0,
        }
