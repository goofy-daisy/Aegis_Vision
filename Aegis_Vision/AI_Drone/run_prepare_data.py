"""
Data preparation entry point for AegisVision.

Run this FIRST before training:
    python run_prepare_data.py

This script:
1. Processes VisDrone2019-DET dataset to YOLO format
2. Processes DOTA dataset to YOLO format
3. Processes HIT-UAV thermal dataset to YOLO format
4. Regenerates configs/dataset.yaml with correct absolute paths
"""
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config_loader import load_config
from utils.logger import get_logger
from data.loaders.visdrone_loader import process_dataset as process_visdrone
from data.loaders.dota_loader import process_dataset as process_dota
from data.loaders.dronevehicle_loader import process_split as process_dronevehicle

logger = get_logger(__name__)


def main():
    cfg = load_config()

    logger.info("=== AegisVision Data Preparation ===")

    # Process VisDrone
    if cfg.paths.visdrone_raw.exists():
        logger.info(f"Processing VisDrone from {cfg.paths.visdrone_raw}")
        stats = process_visdrone(
            raw_dir=cfg.paths.visdrone_raw,
            output_dir=cfg.paths.processed,
            val_split=cfg.training.val_split,
        )
        logger.info(f"VisDrone: {stats['train']} train, {stats['val']} val, {stats['skipped']} skipped")
    else:
        logger.warning(f"VisDrone data not found at {cfg.paths.visdrone_raw}. Skipping.")

    # Process DOTA
    if cfg.paths.dota_raw.exists():
        logger.info(f"Processing DOTA from {cfg.paths.dota_raw}")
        stats = process_dota(
            raw_dir=cfg.paths.dota_raw,
            output_dir=cfg.paths.processed,
            val_split=cfg.training.val_split,
        )
        logger.info(f"DOTA: {stats['train']} train, {stats['val']} val, {stats['skipped']} skipped")
    else:
        logger.warning(f"DOTA data not found at {cfg.paths.dota_raw}. Skipping.")

    # Process HIT-UAV Thermal Dataset
    hituav_dir = cfg.paths.data_root / "hituav"
    if hituav_dir.exists():
        logger.info(f"Processing HIT-UAV thermal dataset from {hituav_dir}")
        try:
            stats = process_dronevehicle(
                raw_dir=hituav_dir,
                output_dir=cfg.paths.data_root / "processed_thermal",
                modality="thermal",
                val_split=cfg.training.val_split,
            )
            logger.info(
                f"HIT-UAV: {stats['train']} train, "
                f"{stats['val']} val, {stats['skipped']} skipped"
            )
        except Exception as e:
            logger.warning(f"HIT-UAV processing failed: {e}")
    else:
        logger.warning(
            f"HIT-UAV not found at {hituav_dir}. "
            f"Download from https://github.com/suojiashun/"
            f"HIT-UAV-Infrared-Thermal-Dataset and place images "
            f"in data/hituav/images/ and xml labels in "
            f"data/hituav/labels/. Skipping."
        )

    logger.info("Data preparation complete. dataset.yaml updated.")
    logger.info(f"Run training with: python models/yolo/train.py")


if __name__ == "__main__":
    main()
