"""RGB vs Thermal image modality classifier for AegisVision.

Uses statistical features to classify frames without requiring
trained weights. Works immediately on first run.
"""

import cv2
import numpy as np
from typing import Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


def classify_modality(
    frame: np.ndarray,
    confidence_threshold: float = 0.7,
) -> Tuple[str, float]:
    """Classify whether a frame is RGB or Thermal.

    Returns:
        Tuple of (modality, confidence).
        modality is 'RGB', 'THERMAL', or 'AMBIGUOUS'.
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return "AMBIGUOUS", 0.0
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        return "AMBIGUOUS", 0.0

    f = frame.astype(np.float32)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_saturation = float(np.mean(hsv[:, :, 1]))

    r, g, b = f[:, :, 2], f[:, :, 1], f[:, :, 0]
    rg_corr = float(np.corrcoef(r.flatten(), g.flatten())[0, 1])
    rb_corr = float(np.corrcoef(r.flatten(), b.flatten())[0, 1])
    gb_corr = float(np.corrcoef(g.flatten(), b.flatten())[0, 1])
    mean_channel_corr = (rg_corr + rb_corr + gb_corr) / 3.0

    hue_std = float(np.std(hsv[:, :, 0]))

    channel_means = [np.mean(f[:, :, i]) for i in range(3)]
    channel_dominance = max(channel_means) / (sum(channel_means) + 1e-6)

    thermal_score = 0.0

    if mean_channel_corr > 0.95:
        thermal_score += 0.4
    if mean_saturation < 20:
        thermal_score += 0.35
    elif mean_saturation < 40:
        thermal_score += 0.15
    if hue_std < 15:
        thermal_score += 0.15
    if channel_dominance > 0.5:
        thermal_score += 0.1

    thermal_score = min(1.0, thermal_score)
    rgb_score = 1.0 - thermal_score

    if thermal_score >= confidence_threshold:
        return "THERMAL", thermal_score
    elif rgb_score >= confidence_threshold:
        return "RGB", rgb_score
    else:
        if thermal_score > rgb_score:
            return "AMBIGUOUS", thermal_score
        return "AMBIGUOUS", rgb_score


def get_modality_badge_color(modality: str) -> tuple:
    """Get BGR color for modality overlay badge."""
    colors = {
        "RGB": (255, 200, 0),
        "THERMAL": (0, 165, 255),
        "AMBIGUOUS": (128, 128, 128),
    }
    return colors.get(modality, (128, 128, 128))
