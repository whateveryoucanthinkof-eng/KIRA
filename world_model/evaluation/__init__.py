"""Evaluation subpackage for dynamics, detection, forecasting, and calibration metrics."""

from world_model.evaluation.calibration import TemperatureScaler, compute_calibration_metrics
from world_model.evaluation.detection_metrics import compute_attack_stage_metrics, compute_detection_metrics
from world_model.evaluation.dynamics_metrics import compute_dynamics_metrics
from world_model.evaluation.forecasting_metrics import compute_forecasting_metrics
from world_model.evaluation.run_evaluation import run_full_evaluation

__all__ = [
    "compute_dynamics_metrics",
    "compute_detection_metrics",
    "compute_attack_stage_metrics",
    "compute_forecasting_metrics",
    "compute_calibration_metrics",
    "TemperatureScaler",
    "run_full_evaluation",
]
