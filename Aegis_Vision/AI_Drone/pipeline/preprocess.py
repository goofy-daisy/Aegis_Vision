"""Frame preprocessing module for AegisVision.

Provides image preprocessing utilities for preparing frames before
inference. Handles resizing, normalization, and format conversion.
"""

from typing import Tuple

import cv2
import numpy as np

from utils.config_loader import Config
from utils.logger import get_logger

logger = get_logger(__name__)


def resize_frame(frame: np.ndarray, target_size: int) -> np.ndarray:
    """Resize frame maintaining aspect ratio with padding to square.
    
    Resizes the frame so that the longest side equals target_size,
    then pads with gray (128, 128, 128) to create a square output.
    
    Args:
        frame: Input BGR image (H, W, 3).
        target_size: Target size for the longest dimension.
        
    Returns:
        Resized and padded square BGR image (target_size, target_size, 3).
        
    Raises:
        ValueError: If frame format is invalid.
    """
    # Validate input
    if frame is None:
        raise ValueError("Input frame is None")
    
    if not isinstance(frame, np.ndarray):
        raise ValueError(f"Frame must be numpy array, got {type(frame)}")
    
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError(f"Frame must be (H, W, 3) BGR array, got shape {frame.shape}")
    
    h, w = frame.shape[:2]
    
    # Calculate scaling factor to fit within target_size
    scale = target_size / max(h, w)
    
    # Compute new dimensions
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize maintaining aspect ratio
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create square canvas with gray padding
    # Gray padding (128) is standard for YOLO to not bias the model
    canvas = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
    
    # Center the resized image on the canvas
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return canvas


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize frame pixel values to [0, 1] range.
    
    Converts uint8 BGR image (0-255) to float32 (0.0-1.0).
    
    Args:
        frame: Input BGR image (H, W, 3) as uint8.
        
    Returns:
        Normalized float32 image (H, W, 3) in range [0, 1].
        
    Raises:
        ValueError: If frame format is invalid.
    """
    # Validate input
    if frame is None:
        raise ValueError("Input frame is None")
    
    if not isinstance(frame, np.ndarray):
        raise ValueError(f"Frame must be numpy array, got {type(frame)}")
    
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError(f"Frame must be (H, W, 3) BGR array, got shape {frame.shape}")
    
    # Convert to float32 and divide by 255
    normalized = frame.astype(np.float32) / 255.0
    
    return normalized


def prepare_for_inference(frame: np.ndarray, cfg: Config) -> np.ndarray:
    """Prepare frame for YOLO inference.
    
    Applies necessary preprocessing before feeding to YOLO model.
    YOLO handles normalization internally, so only resize is needed.
    
    Args:
        frame: Input BGR image (H, W, 3).
        cfg: Configuration with model input size.
        
    Returns:
        Preprocessed frame ready for YOLO inference.
    """
    # Resize to model input size with padding
    resized = resize_frame(frame, cfg.model.input_size)
    
    return resized


def get_resize_scale(original_size: Tuple[int, int], target_size: int) -> float:
    """Calculate the scale factor used during resize.
    
    Useful for converting detection coordinates back to original size.
    
    Args:
        original_size: (height, width) of original image.
        target_size: Target size used for resize.
        
    Returns:
        Scale factor (original_size / new_size).
    """
    h, w = original_size
    scale = max(h, w) / target_size
    return scale


def denormalize_coords(
    coords: np.ndarray,
    original_size: Tuple[int, int],
    target_size: int,
) -> np.ndarray:
    """Convert normalized coordinates back to original image coordinates.
    
    Reverses the resize and padding transformations.
    
    Args:
        coords: Normalized coordinates in [0, 1] range.
        original_size: (height, width) of original image.
        target_size: Target size used for resize.
        
    Returns:
        Coordinates in original image pixel space.
    """
    h, w = original_size
    scale = target_size / max(h, w)
    
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Calculate padding offsets
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    
    # Convert from padded space to resized space
    coords[:, 0] = coords[:, 0] - x_offset
    coords[:, 1] = coords[:, 1] - y_offset
    
    # Scale back to original size
    coords = coords / scale
    
    return coords
