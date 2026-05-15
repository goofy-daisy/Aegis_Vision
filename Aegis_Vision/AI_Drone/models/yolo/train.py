"""YOLO model training module for AegisVision.

Handles YOLOv8 training using Ultralytics, with MLflow integration
for experiment tracking and model registry.
"""

import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from ultralytics import YOLO

from mlops.tracker import MLflowTracker
from utils.config_loader import Config, load_config
from utils.device_utils import get_device
from utils.logger import get_logger

logger = get_logger(__name__)


def train_yolo(
    config: Optional[Config] = None,
    dataset_yaml: Optional[Path] = None,
    resume: Optional[bool] = None,
) -> Dict[str, Union[float, str]]:
    """Train YOLOv8 model on aerial drone detection dataset.
    
    Loads pretrained weights, trains on the configured dataset, and saves
    the best model to the configured weights directory. Logs all metrics
    to MLflow for experiment tracking.
    
    Args:
        config: Configuration object. If None, loads from default config.yaml.
        dataset_yaml: Path to dataset YAML file. If None, uses default.
        resume: Whether to resume from existing checkpoint. If None, uses config value.
        
    Returns:
        Dictionary containing final metrics:
            - mAP50: mAP at IoU=0.5
            - mAP50_95: mAP at IoU=0.5:0.95
            - precision: Model precision
            - recall: Model recall
            - best_model_path: Path to saved best model
            
    Raises:
        FileNotFoundError: If dataset YAML or pretrained weights not found.
        RuntimeError: If training fails.
    """
    # Load configuration if not provided
    if config is None:
        config = load_config()
    
    # Determine dataset YAML path
    if dataset_yaml is None:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        dataset_yaml = project_root / "configs" / "dataset.yaml"
    
    if not dataset_yaml.exists():
        logger.error(f"Dataset YAML not found: {dataset_yaml}")
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    
    # Determine device
    device = get_device(config.model.device if config.model.device != "auto" else None)
    
    # Determine whether to resume training
    should_resume = config.training.resume if resume is None else resume
    best_pt_path = config.paths.weights / "best.pt"
    
    # Initialize MLflow tracker
    tracker = MLflowTracker(config)
    
    try:
        # Start MLflow run
        run = tracker.start_run()
        logger.info(f"MLflow run started: {run.info.run_id}")
        
        # Log configuration parameters
        params = {
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "model_size": config.model.yolo_model_size,
            "input_size": config.model.input_size,
            "val_split": config.training.val_split,
            "resume": should_resume,
        }
        # Flatten nested dicts for MLflow
        flat_params = {}
        for key, value in params.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_params[f"{key}.{sub_key}"] = sub_value
            else:
                flat_params[key] = value
        tracker.log_params(flat_params)
        
        # Load model
        if should_resume and best_pt_path.exists():
            logger.info(f"Resuming training from {best_pt_path}")
            model = YOLO(str(best_pt_path))
        else:
            pretrained = config.model.yolo_weights_pretrained
            logger.info(f"Loading pretrained weights: {pretrained}")
            model = YOLO(pretrained)
        
        # Train the model
        logger.info(f"Starting training on {dataset_yaml}")
        logger.info(f"Epochs: {config.training.epochs}, Batch: {config.training.batch_size}, Device: {device}")
        
        results = model.train(
            data=str(dataset_yaml),
            epochs=config.training.epochs,
            batch=config.training.batch_size,
            lr0=config.training.learning_rate,
            imgsz=config.model.input_size,
            device=str(device),
            patience=config.training.patience,
            save_period=config.training.save_period,
            exist_ok=True,
            verbose=True,
            # V3: Data augmentation parameters
            hsv_h=0.015,  # HSV-Hue augmentation
            hsv_s=0.7,    # HSV-Saturation augmentation
            hsv_v=0.4,    # HSV-Value augmentation
            degrees=0.0,  # rotation (keep 0 for aerial)
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            bgr=0.0,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            auto_augment="randaugment",
            erasing=0.4,
            crop_fraction=1.0,
        )
        
        # Extract metrics from results
        metrics = {
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
            "mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0.0)),
            "recall": float(results.results_dict.get("metrics/recall(B)", 0.0)),
        }
        
        # Log metrics to MLflow
        tracker.log_metrics(metrics)
        logger.info(f"Training complete. mAP50: {metrics['mAP50']:.4f}, mAP50-95: {metrics['mAP50_95']:.4f}")
        
        # Find best.pt — Ultralytics creates runs/detect/trainN/weights/best.pt
        # The N suffix increments if the directory already exists
        runs_detect = Path("runs") / "detect"
        best_pt = None
        
        if runs_detect.exists():
            # Find the most recently modified training run
            train_dirs = sorted(
                [d for d in runs_detect.iterdir() if d.is_dir() and d.name.startswith("train")],
                key=lambda d: d.stat().st_mtime,
                reverse=True
            )
            for train_dir in train_dirs:
                candidate = train_dir / "weights" / "best.pt"
                if candidate.exists():
                    best_pt = candidate
                    break
        
        if best_pt is not None:
            weights_dir = config.paths.weights
            weights_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_pt, best_pt_path)
            # Auto-save modality-named copy so pipeline selects
            # correct model automatically during inference
            dataset_str = str(dataset_yaml).lower() if dataset_yaml else ""
            if "thermal" in dataset_str:
                thermal_path = config.paths.weights / "best_thermal.pt"
                shutil.copy2(best_pt, thermal_path)
                logger.info(f"Thermal model auto-saved as {thermal_path}")
            else:
                rgb_path = config.paths.weights / "best_rgb.pt"
                shutil.copy2(best_pt, rgb_path)
                logger.info(f"RGB model auto-saved as {rgb_path}")
            logger.info(f"Best model copied from {best_pt} to {best_pt_path}")
            tracker.log_artifact(best_pt_path, artifact_path="model")
            tracker.log_model(best_pt_path, model_name="aegisvision_yolo")
            metrics["best_model_path"] = str(best_pt_path)
        else:
            logger.error("Could not find best.pt in any Ultralytics run directory")
            metrics["best_model_path"] = ""
        
        # End MLflow run
        tracker.end_run()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        tracker.end_run()
        raise RuntimeError(f"YOLO training failed: {e}")


def main():
    """Command-line entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train AegisVision YOLO model")
    parser.add_argument("--config", type=Path, help="Path to config.yaml")
    parser.add_argument("--dataset", type=Path, help="Path to dataset.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, help="Override epoch count")
    parser.add_argument("--batch", type=int, help="Override batch size")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config) if args.config else load_config()
    
    # Override config with CLI args if provided
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch:
        config.training.batch_size = args.batch
    if args.resume:
        config.training.resume = True
    
    # Run training
    results = train_yolo(config, args.dataset, args.resume)
    
    # Print summary
    print("\nTraining Summary:")
    print(f"  mAP50:        {results['mAP50']:.4f}")
    print(f"  mAP50-95:     {results['mAP50_95']:.4f}")
    print(f"  Precision:    {results['precision']:.4f}")
    print(f"  Recall:       {results['recall']:.4f}")
    print(f"  Best model:   {results['best_model_path']}")


if __name__ == "__main__":
    main()
