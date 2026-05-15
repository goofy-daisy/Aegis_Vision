"""Device detection utilities for AegisVision.

Provides automatic CUDA/CPU detection with fallback mechanisms.
All PyTorch operations should use the device returned by get_device()
to ensure compatibility across different hardware configurations.
"""

import torch
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


def get_device(preferred_device: Optional[str] = None) -> torch.device:
    """Detect and return the appropriate torch device (CUDA or CPU).
    
    Automatically detects GPU availability and falls back to CPU if needed.
    Logs the selected device and GPU name for debugging purposes.
    
    Args:
        preferred_device: If "cuda" or "cpu", use that device explicitly.
                         If "auto" or None, auto-detect based on availability.
                         
    Returns:
        torch.device: The selected device for PyTorch operations.
        
    Raises:
        RuntimeError: If CUDA is requested but not available.
    """
    # Validate input
    if preferred_device is not None and preferred_device not in ["auto", "cuda", "cpu"]:
        logger.warning(f"Invalid device '{preferred_device}', falling back to auto-detection")
        preferred_device = "auto"
    
    # Handle explicit device requests
    if preferred_device == "cuda":
        if not torch.cuda.is_available():
            logger.error("CUDA requested but not available on this system")
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA device: {gpu_name}")
        return device
    
    if preferred_device == "cpu":
        logger.info("Using CPU device (explicitly requested)")
        return torch.device("cpu")
    
    # Auto-detect: try CUDA first, fall back to CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA detected and available. Using GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        logger.info("No CUDA device detected. Using CPU fallback.")
    
    return device


def get_device_info() -> dict:
    """Get detailed information about the available compute devices.
    
    Returns:
        dict containing device information:
        - cuda_available: bool
        - device_count: int (number of CUDA devices)
        - current_device: str (name of current CUDA device or "cpu")
        - device_name: str (detailed GPU name if CUDA, else "CPU")
        - memory_allocated: float (GB, CUDA only)
        - memory_reserved: float (GB, CUDA only)
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": "cpu",
        "device_name": "CPU",
        "memory_allocated_gb": 0.0,
        "memory_reserved_gb": 0.0,
    }
    
    if torch.cuda.is_available():
        info["current_device"] = f"cuda:{torch.cuda.current_device()}"
        info["device_name"] = torch.cuda.get_device_name(0)
        info["memory_allocated_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
        info["memory_reserved_gb"] = torch.cuda.memory_reserved(0) / (1024**3)
    
    return info
