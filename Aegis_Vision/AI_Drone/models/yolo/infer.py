"""YOLO model inference module for AegisVision.

Provides model loading and inference utilities for object detection
in drone threat detection pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from ultralytics import YOLO

from utils.config_loader import Config
from utils.device_utils import get_device
from utils.logger import get_logger

logger = get_logger(__name__)


def load_model(weights_path: Path, device: Optional[torch.device] = None) -> YOLO:
    """Load YOLO model from weights file.
    
    Loads model once and moves to specified device. Model is set to eval mode.
    
    Args:
        weights_path: Path to .pt weights file.
        device: Torch device to load model on. If None, auto-detects.
        
    Returns:
        Loaded YOLO model instance in eval mode.
        
    Raises:
        FileNotFoundError: If weights file does not exist.
        RuntimeError: If model loading fails.
    """
    # Validate weights file exists
    if not weights_path.exists():
        logger.error(f"Model weights not found: {weights_path}")
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    # Determine device
    if device is None:
        device = get_device()
    
    try:
        # Load model
        logger.info(f"Loading YOLO model from {weights_path}")
        model = YOLO(str(weights_path))
        
        # Move to device
        model.to(device)
        
        # Set to eval mode
        model.eval()
        
        logger.info(f"Model loaded successfully on {device}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


def run_inference(
    model: YOLO,
    frame: np.ndarray,
    config: Config,
) -> List[Dict]:
    """Run YOLO inference on a single frame.
    
    Processes a BGR numpy frame and returns filtered detections with
    bounding boxes, class IDs, names, and confidence scores.
    
    Args:
        model: Loaded YOLO model instance.
        frame: Input image as BGR numpy array (H, W, 3).
        config: Configuration object with detection parameters.
        
    Returns:
        List of detection dicts, each containing:
            - bbox: [x1, y1, x2, y2] absolute pixel coords (xyxy format)
            - bbox_xywh: [x, y, w, h] center format for DeepSORT
            - class_id: int, YOLO class index
            - class_name: str, human-readable class name
            - confidence: float, detection confidence (0-1)
        
        Returns empty list if no detections above threshold.
        
    Raises:
        ValueError: If frame format is invalid.
    """
    # Validate input frame
    if frame is None:
        raise ValueError("Input frame is None")
    
    if not isinstance(frame, np.ndarray):
        raise ValueError(f"Frame must be numpy array, got {type(frame)}")
    
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError(f"Frame must be (H, W, 3) BGR array, got shape {frame.shape}")
    
    # Run inference
    try:
        results = model(
            frame,
            verbose=False,
            conf=config.model.confidence_threshold,
            iou=config.model.nms_iou_threshold,
            max_det=config.model.max_detections,
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise RuntimeError(f"YOLO inference failed: {e}")
    
    # Parse results
    detections = []
    
    for result in results:
        if result.boxes is None:
            continue
        
        boxes = result.boxes
        
        # Extract data from boxes
        for i in range(len(boxes)):
            # Get bounding box in xyxy format
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            
            # Convert to xywh format (center-based, for DeepSORT)
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            
            # Get class info
            class_id = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())
            
            # Get class name from model
            if hasattr(result, "names") and class_id in result.names:
                class_name = result.names[class_id]
            else:
                class_name = f"class_{class_id}"
            
            detection = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "bbox_xywh": [float(cx), float(cy), float(w), float(h)],
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
            }
            detections.append(detection)
    
    logger.debug(f"Detected {len(detections)} objects")
    return detections


def get_model_info(model: YOLO) -> Dict:
    """Get information about the loaded model.
    
    Args:
        model: Loaded YOLO model instance.
        
    Returns:
        Dict containing model information:
            - task: Model task type
            - names: Class names dictionary
            - num_classes: Number of classes
            - input_size: Model input size
    """
    info = {
        "task": getattr(model, "task", "unknown"),
        "names": {},
        "num_classes": 0,
        "input_size": 640,
    }
    
    # Extract class names if available
    if hasattr(model, "names"):
        info["names"] = model.names
        info["num_classes"] = len(model.names)
    
    # Try to get input size from model
    if hasattr(model, "model") and hasattr(model.model, "args"):
        args = model.model.args
        if hasattr(args, "imgsz"):
            info["input_size"] = args.imgsz
    
    return info
