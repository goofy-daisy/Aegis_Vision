"""Image augmentation module for AegisVision.

Provides realistic image augmentations for simulation and training including
fog, motion blur, and gaussian noise. Uses Albumentations for efficient
and reproducible transformations.
"""

from typing import Optional

import albumentations as A
import numpy as np

from utils.config_loader import Config
from utils.logger import get_logger

logger = get_logger(__name__)


def apply_fog(image: np.ndarray, intensity: float) -> np.ndarray:
    """Apply fog effect to image.
    
    Simulates atmospheric fog reducing visibility. Higher intensity
denser fog.
    
    Args:
        image: Input BGR image as numpy array (H, W, 3).
        intensity: Fog intensity in range [0.0, 1.0].
        
    Returns:
        Fog-augmented BGR image.
        
    Raises:
        ValueError: If image format is invalid.
    """
    # Validate input
    if image is None:
        raise ValueError("Input image is None")
    
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Image must be numpy array, got {type(image)}")
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must be (H, W, 3) BGR array, got shape {image.shape}")
    
    # Create fog augmentation
    # RandomFog uses alpha which maps to visibility - higher alpha = less visibility
    try:
        # Albumentations >= 1.4.0 API
        fog_transform = A.RandomFog(
            fog_coef_range=(intensity * 0.5, intensity),
            alpha_coef=0.1,
            p=1.0,
        )
    except TypeError:
        # Albumentations < 1.4.0 API fallback
        fog_transform = A.RandomFog(
            fog_coef_lower=intensity * 0.5,
            fog_coef_upper=intensity,
            alpha_coef=0.1,
            p=1.0,
        )
    
    # Apply transformation
    augmented = fog_transform(image=image)
    return augmented["image"]


def apply_motion_blur(image: np.ndarray, max_kernel: int) -> np.ndarray:
    """Apply motion blur effect to image.
    
    Simulates camera motion during exposure. Kernel size determines
    blur strength.
    
    Args:
        image: Input BGR image as numpy array (H, W, 3).
        max_kernel: Maximum blur kernel size (odd number).
        
    Returns:
        Motion-blurred BGR image.
        
    Raises:
        ValueError: If image format is invalid or kernel size is even.
    """
    # Validate input
    if image is None:
        raise ValueError("Input image is None")
    
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Image must be numpy array, got {type(image)}")
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must be (H, W, 3) BGR array, got shape {image.shape}")
    
    # Ensure kernel size is odd
    if max_kernel % 2 == 0:
        raise ValueError(f"Kernel size must be odd, got {max_kernel}")
    
    # Create motion blur augmentation
    # Use random kernel size up to max_kernel
    blur_transform = A.MotionBlur(
        blur_limit=(3, max_kernel),  # Random kernel between 3 and max_kernel
        p=1.0,                        # Always apply
    )
    
    # Apply transformation
    augmented = blur_transform(image=image)
    return augmented["image"]


def apply_gaussian_noise(image: np.ndarray, std: float) -> np.ndarray:
    """Apply gaussian noise to image.
    
    Simulates sensor noise. Standard deviation controls noise strength.
    
    Args:
        image: Input BGR image as numpy array (H, W, 3).
        std: Standard deviation of gaussian noise.
        
    Returns:
        Noise-augmented BGR image.
        
    Raises:
        ValueError: If image format is invalid.
    """
    # Validate input
    if image is None:
        raise ValueError("Input image is None")
    
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Image must be numpy array, got {type(image)}")
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must be (H, W, 3) BGR array, got shape {image.shape}")
    
    # Create gaussian noise augmentation
    # Convert std to variance for GaussNoise
    var = std ** 2
    
    try:
        # Albumentations >= 1.4.0 uses std_range normalised to 0-1
        std_norm = min(1.0, std / 255.0)
        noise_transform = A.GaussNoise(
            std_range=(std_norm * 0.5, std_norm),
            mean_range=(0, 0),
            p=1.0,
        )
    except TypeError:
        noise_transform = A.GaussNoise(
            var_limit=(var * 0.5, var),
            mean=0,
            per_channel=True,
            p=1.0,
        )
    
    # Apply transformation
    augmented = noise_transform(image=image)
    return augmented["image"]


def apply_all(image: np.ndarray, cfg: Config) -> np.ndarray:
    """Apply all augmentations sequentially.
    
    Applies fog, motion blur, and gaussian noise in sequence using
    configured parameters.
    
    Args:
        image: Input BGR image.
        cfg: Configuration with augmentation parameters.
        
    Returns:
        Augmented BGR image with all effects applied.
    """
    # Apply fog
    image = apply_fog(image, cfg.simulation.fog_intensity)
    
    # Apply motion blur
    image = apply_motion_blur(image, cfg.simulation.blur_kernel_max)
    
    # Apply gaussian noise
    image = apply_gaussian_noise(image, cfg.simulation.noise_std)
    
    return image


def get_augmentation_pipeline(cfg: Config) -> A.Compose:
    """Create a reusable augmentation pipeline for batch processing.

    Returns an Albumentations Compose object that can be reused
    for multiple images during training. Compatible with both
    Albumentations < 1.4.0 and >= 1.4.0 API.

    Args:
        cfg: Configuration with augmentation parameters.

    Returns:
        Albumentations Compose pipeline.
    """
    var = cfg.simulation.noise_std ** 2
    std = cfg.simulation.noise_std

    # Build fog transform with API compatibility
    try:
        fog_aug = A.RandomFog(
            fog_coef_range=(cfg.simulation.fog_intensity * 0.5,
                            cfg.simulation.fog_intensity),
            alpha_coef=0.1,
            p=0.5,
        )
    except TypeError:
        fog_aug = A.RandomFog(
            fog_coef_lower=cfg.simulation.fog_intensity * 0.5,
            fog_coef_upper=cfg.simulation.fog_intensity,
            alpha_coef=0.1,
            p=0.5,
        )

    # Build noise transform with API compatibility
    try:
        std_norm = min(1.0, std / 255.0)
        noise_aug = A.GaussNoise(
            std_range=(std_norm * 0.5, std_norm),
            mean_range=(0, 0),
            p=0.5,
        )
    except TypeError:
        noise_aug = A.GaussNoise(
            var_limit=(var * 0.5, var),
            mean=0,
            per_channel=True,
            p=0.5,
        )

    pipeline = A.Compose([
        fog_aug,
        A.MotionBlur(
            blur_limit=(3, cfg.simulation.blur_kernel_max),
            p=0.5,
        ),
        noise_aug,
    ])

    return pipeline


def apply_pipeline(image: np.ndarray, pipeline: A.Compose) -> np.ndarray:
    """Apply a pre-built augmentation pipeline to an image.
    
    Args:
        image: Input BGR image.
        pipeline: Albumentations Compose pipeline.
        
    Returns:
        Augmented BGR image.
    """
    augmented = pipeline(image=image)
    return augmented["image"]
