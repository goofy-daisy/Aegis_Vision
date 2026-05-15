"""Threat model training script for AegisVision.

Trains a LightGBM model on synthetic threat data as an alternative
scorer to the weighted function. The LightGBM model is secondary to
the primary interpretable weighted scorer.
"""

import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from utils.config_loader import Config, load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_synthetic_training_data(n_samples: int = 1000, config: Optional[Config] = None) -> pd.DataFrame:
    """Generate synthetic training data for threat model.
    
    Creates realistic distributions of proximity, speed, and class
    combinations with corresponding threat labels.
    
    Args:
        n_samples: Number of samples to generate.
        config: Configuration for threat thresholds. Uses defaults if None.
        
    Returns:
        DataFrame with columns: proximity_m, speed_mps, class_name, threat_score
    """
    if config is None:
        proximity_max = 500.0
        speed_max = 50.0
        dangerous_classes = ["car", "van", "truck", "bus"]
    else:
        proximity_max = config.threat.proximity_max_meters
        speed_max = config.threat.speed_max_mps
        dangerous_classes = config.threat.dangerous_classes
    
    np.random.seed(42)
    
    data = []
    all_classes = dangerous_classes + ["pedestrian", "bicycle", "plane", "ship"]
    
    for _ in range(n_samples):
        # Sample proximity with bias toward closer objects being more common
        # Use exponential distribution for realistic proximity
        proximity_m = np.random.exponential(proximity_max / 3)
        proximity_m = min(proximity_m, proximity_max)
        
        # Sample speed with realistic distribution
        # Mix of stationary and moving objects
        if np.random.random() < 0.3:
            speed_mps = 0  # 30% stationary
        else:
            speed_mps = np.random.beta(2, 5) * speed_max  # Beta distribution for moving
        
        # Sample class
        class_name = np.random.choice(all_classes)
        
        # Compute ground truth threat score (for training labels)
        # Closer and faster = more threatening, dangerous classes = more threatening
        proximity_score = 1.0 - (proximity_m / proximity_max)
        speed_score = speed_mps / speed_max
        class_score = 1.0 if class_name in dangerous_classes else 0.5
        
        # Weighted combination using config-driven weights to match ThreatScorer exactly
        if config is not None:
            w_prox = config.threat.weights["proximity"]
            w_speed = config.threat.weights["speed"]
            w_class = config.threat.weights["class_danger"]
        else:
            w_prox, w_speed, w_class = 0.45, 0.35, 0.20
        
        raw_score = w_prox * proximity_score + w_speed * speed_score + w_class * class_score
        
        # Sigmoid normalization
        import math
        threat_score = 1.0 / (1.0 + math.exp(-10.0 * (raw_score - 0.5)))
        
        data.append({
            "proximity_m": proximity_m,
            "speed_mps": speed_mps,
            "class_name": class_name,
            "threat_score": threat_score,
            "proximity_score": proximity_score,
            "speed_score": speed_score,
            "class_score": class_score,
        })
    
    return pd.DataFrame(data)


def train_threat_model(
    config: Optional[Config] = None,
    n_samples: int = 1000,
    output_path: Optional[Path] = None,
) -> Dict:
    """Train LightGBM threat model on synthetic data.
    
    Args:
        config: Configuration object. Uses default if None.
        n_samples: Number of synthetic samples to generate.
        output_path: Path to save trained model. Uses config if None.
        
    Returns:
        Dict containing training results and model path.
        
    Raises:
        RuntimeError: If training fails.
    """
    if config is None:
        config = load_config()
    
    if output_path is None:
        output_path = config.paths.threat_model
    
    logger.info(f"Generating {n_samples} synthetic training samples...")
    df = generate_synthetic_training_data(n_samples, config)
    
    # Prepare features
    # Encode class as categorical
    class_map = {name: i for i, name in enumerate(df["class_name"].unique())}
    df["class_encoded"] = df["class_name"].map(class_map)
    
    # Feature matrix
    X = df[["proximity_m", "speed_mps", "class_encoded"]].values
    y = df["threat_score"].values
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Training parameters
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "num_threads": 4,
    }
    
    logger.info("Training LightGBM model...")
    
    try:
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
        )
        
        # Evaluate
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        mae = np.mean(np.abs(y_val - y_pred))
        
        logger.info(f"Training complete. RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        # Save model
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": model,
            "class_map": class_map,
            "config": {
                "proximity_max": config.threat.proximity_max_meters if config else 500.0,
                "speed_max": config.threat.speed_max_mps if config else 50.0,
            },
            "metrics": {
                "rmse": float(rmse),
                "mae": float(mae),
            },
        }
        
        with open(output_path, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {output_path}")
        
        return {
            "model_path": str(output_path),
            "rmse": float(rmse),
            "mae": float(mae),
            "n_samples": n_samples,
        }
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise RuntimeError(f"Threat model training failed: {e}")


def load_threat_model(model_path: Path) -> Dict:
    """Load a trained threat model from disk.
    
    Args:
        model_path: Path to saved model pickle file.
        
    Returns:
        Dict containing loaded model and metadata.
        
    Raises:
        FileNotFoundError: If model file does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Threat model not found: {model_path}")
    
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    logger.info(f"Loaded threat model from {model_path}")
    return model_data


def main():
    """Command-line entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train AegisVision threat model")
    parser.add_argument("--config", type=Path, help="Path to config.yaml")
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic samples")
    parser.add_argument("--output", type=Path, help="Output path for model")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config) if args.config else load_config()
    
    # Train model
    results = train_threat_model(
        config=config,
        n_samples=args.samples,
        output_path=args.output,
    )
    
    # Print summary
    print("\nTraining Summary:")
    print(f"  Samples:      {results['n_samples']}")
    print(f"  RMSE:         {results['rmse']:.4f}")
    print(f"  MAE:          {results['mae']:.4f}")
    print(f"  Model saved:  {results['model_path']}")


if __name__ == "__main__":
    main()
