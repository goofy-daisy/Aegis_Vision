"""Simulation package for AegisVision.

Provides image augmentation and telemetry simulation capabilities.
"""

from simulation.augmentations import (
    apply_all,
    apply_fog,
    apply_gaussian_noise,
    apply_motion_blur,
    apply_pipeline,
    get_augmentation_pipeline,
)
from simulation.telemetry import TelemetrySimulator

__all__ = [
    "TelemetrySimulator",
    "apply_fog",
    "apply_motion_blur",
    "apply_gaussian_noise",
    "apply_all",
    "get_augmentation_pipeline",
    "apply_pipeline",
]
