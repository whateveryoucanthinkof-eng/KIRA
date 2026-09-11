"""
Forecasting and Early-Warning Metrics for World Model Trajectories.

Evaluates the temporal anticipation capabilities of the World Dynamics Transformer:
1. Lead Time: Seconds / windows in advance an attack is predicted before true onset.
2. Early Warning Rate: Fraction of attack campaigns with advance notice >= 1 window.
3. False Alarm Frequency: False Positives per operational hour in benign telemetry.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def compute_forecasting_metrics(
    attack_onsets: List[int],              # Timestamps or indices of true attack start
    forecast_alerts: List[Tuple[int, int]],# List of (alert_time, predicted_horizon_k)
    window_duration_sec: float = 2.0,
    total_hours_benign: float = 1.0,
    false_alert_count: int = 0,
) -> Dict[str, float]:
    """
    Args:
        attack_onsets: list of true attack onset window indices
        forecast_alerts: tuples of (alert_window, predicted_lead_windows)
        window_duration_sec: duration per telemetry window (default 2s)
        total_hours_benign: total benign hours monitored
        false_alert_count: total false alarms raised
    """
    lead_times_sec = []
    early_warned_count = 0

    for onset in attack_onsets:
        # Find alerts that fired prior to this onset predicting an attack at or covering onset
        valid_leads = [
            (alert_time, k)
            for alert_time, k in forecast_alerts
            if alert_time < onset <= alert_time + k
        ]
        if valid_leads:
            early_warned_count += 1
            # Earliest advance notice
            earliest_alert = min(alert_time for alert_time, _ in valid_leads)
            lead_windows = onset - earliest_alert
            lead_times_sec.append(lead_windows * window_duration_sec)

    total_attacks = max(1, len(attack_onsets))
    early_warning_rate = early_warned_count / total_attacks
    mean_lead_time_sec = float(np.mean(lead_times_sec)) if lead_times_sec else 0.0
    median_lead_time_sec = float(np.median(lead_times_sec)) if lead_times_sec else 0.0

    fp_per_hour = false_alert_count / max(0.1, total_hours_benign)

    return {
        "early_warning_rate": early_warning_rate,
        "mean_lead_time_seconds": mean_lead_time_sec,
        "median_lead_time_seconds": median_lead_time_sec,
        "false_positives_per_hour": fp_per_hour,
        "total_campaigns_evaluated": total_attacks,
    }
