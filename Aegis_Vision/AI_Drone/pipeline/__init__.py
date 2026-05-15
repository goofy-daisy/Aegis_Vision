"""Pipeline package for AegisVision.

Provides preprocessing, postprocessing, and full pipeline orchestration.
"""

from pipeline.full_pipeline import AegisPipeline
from pipeline.postprocess import draw_detections, draw_metrics_overlay, draw_threat_legend
from pipeline.preprocess import prepare_for_inference, resize_frame, normalize_frame

__all__ = [
    "AegisPipeline",
    "prepare_for_inference",
    "resize_frame",
    "normalize_frame",
    "draw_detections",
    "draw_metrics_overlay",
    "draw_threat_legend",
]
