"""
Pre-augmentation script for AegisVision training data.

Generates 5 augmented copies of every training image using
fog, blur, noise, and combined effects. Expands training set
from N images to 6N images before YOLO training begins.

Run this BEFORE training:
    python run_augment_data.py

Output goes to data/processed/images/train/ alongside originals.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from utils.config_loader import load_config
from utils.logger import get_logger
from simulation.augmentations import (
    apply_fog, apply_motion_blur, apply_gaussian_noise, apply_all
)

logger = get_logger(__name__)


def augment_training_data(processed_dir: Path, cfg) -> int:
    """Generate augmented versions of all training images.

    Creates 5 variants per image:
    1. Fog only
    2. Motion blur only
    3. Gaussian noise only
    4. Fog + blur combined
    5. All effects combined

    Args:
        processed_dir: Path to data/processed directory.
        cfg: Configuration object.

    Returns:
        Total number of augmented images created.
    """
    train_img_dir = processed_dir / "images" / "train"
    train_lbl_dir = processed_dir / "labels" / "train"

    if not train_img_dir.exists():
        logger.error(f"Training images directory not found: {train_img_dir}")
        return 0

    # Get all original images (exclude already augmented ones)
    original_images = [
        f for f in train_img_dir.glob("*.jpg")
        if "_aug" not in f.stem
    ]
    original_images += [
        f for f in train_img_dir.glob("*.png")
        if "_aug" not in f.stem
    ]

    if not original_images:
        logger.warning("No original training images found to augment")
        return 0

    logger.info(
        f"Augmenting {len(original_images)} training images "
        f"(will create {len(original_images) * 5} augmented copies)"
    )

    created = 0
    fog_intensity = cfg.simulation.fog_intensity
    blur_kernel = cfg.simulation.blur_kernel_max
    # Ensure blur kernel is odd
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    noise_std = cfg.simulation.noise_std
    # Normalise noise std to 0-1 for Albumentations >= 1.4.0 API
    noise_std_normalised = min(1.0, noise_std / 255.0)

    # Normalise noise_std to 0-1 range for Albumentations >= 1.4.0
    # Raw pixel std (e.g. 12.5) divided by 255 gives normalised value
    noise_std_normalised = min(1.0, noise_std / 255.0)

    augmentations = [
        ("fog",       lambda img: apply_fog(img, fog_intensity)),
        ("blur",      lambda img: apply_motion_blur(img, blur_kernel)),
        ("noise",     lambda img: apply_gaussian_noise(img, noise_std_normalised)),
        ("fogblur",   lambda img: apply_motion_blur(
                          apply_fog(img, fog_intensity), blur_kernel)),
        ("all",       lambda img: apply_all(img, cfg)),
    ]

    for img_path in original_images:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Failed to read: {img_path}")
            continue

        # Find corresponding label file
        lbl_path = train_lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            logger.debug(f"No label for {img_path.stem}, skipping")
            continue

        for aug_name, aug_fn in augmentations:
            try:
                aug_img = aug_fn(img)
            except Exception as e:
                logger.warning(
                    f"Augmentation {aug_name} failed for "
                    f"{img_path.stem}: {e}"
                )
                continue

            # Save augmented image
            aug_stem = f"{img_path.stem}_aug_{aug_name}"
            aug_img_path = train_img_dir / (aug_stem + img_path.suffix)
            aug_lbl_path = train_lbl_dir / (aug_stem + ".txt")

            cv2.imwrite(str(aug_img_path), aug_img)

            # Copy label unchanged (augmentations don't change bboxes
            # for fog/noise/blur which are pixel-level only)
            import shutil
            shutil.copy2(lbl_path, aug_lbl_path)

            created += 1

    logger.info(
        f"Augmentation complete: {created} new images created. "
        f"Total training set: {len(original_images) + created} images."
    )
    return created


def main():
    cfg = load_config()

    logger.info("=== AegisVision Data Augmentation ===")
    logger.info(
        "This expands your training set by generating "
        "fog, blur, noise, and combined variants."
    )

    # Augment main processed dataset
    if cfg.paths.processed.exists():
        count = augment_training_data(cfg.paths.processed, cfg)
        logger.info(f"Main dataset: {count} augmented images created")
    else:
        logger.warning(
            f"Processed data not found at {cfg.paths.processed}. "
            "Run python run_prepare_data.py first."
        )

    # Augment thermal dataset if it exists
    thermal_dir = cfg.paths.data_root / "processed_thermal"
    if thermal_dir.exists():
        count = augment_training_data(thermal_dir, cfg)
        logger.info(f"Thermal dataset: {count} augmented images created")

    logger.info(
        "Augmentation done. Now run: "
        "python models/yolo/train.py --epochs 20 --batch 4"
    )


if __name__ == "__main__":
    main()
