"""Utility modules for AegisVision.

This package provides centralized utilities used across the system:
- config_loader: Typed configuration management
- logger: Centralized logging setup
- device_utils: CUDA/CPU detection and device management
- bbox_utils: Bounding box operations and conversions
"""

from utils.bbox_utils import clip_bbox, compute_iou, obb_to_aabb, xywh_to_xyxy, xyxy_to_xywh
from utils.config_loader import Config, load_config, reload_config
from utils.device_utils import get_device, get_device_info
from utils.logger import get_logger

__all__ = [
    "get_logger",
    "load_config",
    "reload_config",
    "Config",
    "get_device",
    "get_device_info",
    "obb_to_aabb",
    "xywh_to_xyxy",
    "xyxy_to_xywh",
    "clip_bbox",
    "compute_iou",
]
