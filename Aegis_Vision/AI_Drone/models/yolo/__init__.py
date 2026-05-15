"""YOLO model package for AegisVision.

Provides training and inference capabilities using Ultralytics YOLOv8.
"""

from models.yolo.infer import get_model_info, load_model, run_inference
from models.yolo.train import train_yolo

__all__ = [
    "load_model",
    "run_inference",
    "get_model_info",
    "train_yolo",
]
