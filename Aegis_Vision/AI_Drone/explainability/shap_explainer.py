"""SHAP explainability module for AegisVision.

Provides feature attribution explanations for threat scores using
SHAP (SHapley Additive exPlanations) with KernelExplainer.

Since the threat scorer is a weighted function (not a tree model),
we use KernelExplainer with a custom prediction wrapper.
"""

from typing import Dict, List, Optional

import numpy as np
import shap

from models.threat import ThreatScorer
from utils.config_loader import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class SHAPExplainer:
    """SHAP explainer for threat scoring model.
    
    Generates feature attribution explanations showing how proximity,
    speed, and class danger contribute to each threat score.
    
    Attributes:
        threat_scorer: The ThreatScorer instance being explained.
        explainer: SHAP KernelExplainer instance.
        feature_names: List of feature names for display.
    """
    
    def __init__(self, threat_scorer: ThreatScorer, cfg: Config):
        """Initialize SHAP explainer with threat scorer.
        
        Args:
            threat_scorer: ThreatScorer instance to explain.
            cfg: Configuration object.
        """
        self.threat_scorer = threat_scorer
        self.cfg = cfg
        self.feature_names = ["proximity", "speed", "class_danger"]
        
        # Build background dataset from synthetic samples
        background_data = self._generate_background_data(n_samples=50)
        
        # Initialize KernelExplainer with custom prediction function
        self.explainer = shap.KernelExplainer(
            self._predict_fn,
            background_data,
        )
        
        logger.info("SHAPExplainer initialized with KernelExplainer")
    
    def _generate_background_data(self, n_samples: int = 50) -> np.ndarray:
        """Generate background data for SHAP explainer.
        
        Creates random samples representing the feature distribution.
        
        Args:
            n_samples: Number of background samples.
            
        Returns:
            Background data array (n_samples, 3).
        """
        np.random.seed(42)
        
        # Generate random feature vectors
        # proximity_score, speed_score, class_score - all in [0, 1]
        data = np.random.rand(n_samples, 3)
        
        return data
    
    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        """Prediction wrapper for SHAP explainer.
        
        Converts numpy feature array to threat scores.
        
        Args:
            X: Feature array (n_samples, 3) with columns [proximity_score, speed_score, class_score].
            
        Returns:
            Array of threat scores (n_samples,).
        """
        # Handle single sample (1D array)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        scores = []
        for features in X:
            # Build track-like dict with the features
            track = {
                "proximity_m": features[0] * self.cfg.threat.proximity_max_meters,
                "speed_mps": features[1] * self.cfg.threat.speed_max_mps,
                "class_name": "unknown",  # Class score already incorporated
            }
            
            # Compute threat score directly
            proximity_score = features[0]
            speed_score = features[1]
            class_score = features[2]
            
            # Manual scoring to match ThreatScorer logic
            weights = self.cfg.threat.weights
            raw_score = (
                proximity_score * weights["proximity"] +
                speed_score * weights["speed"] +
                class_score * weights["class_danger"]
            )
            
            # Sigmoid
            import math
            threat_score = 1.0 / (1.0 + math.exp(-10.0 * (raw_score - 0.5)))
            scores.append(threat_score)
        
        return np.array(scores)
    
    def explain(self, tracks: List[Dict]) -> List[Dict]:
        """Generate SHAP explanations for a list of tracks.
        
        Args:
            tracks: List of track dicts with threat scores.
            
        Returns:
            List of explanation dicts with keys:
                - track_id: int
                - shap_values: Dict[str, float] feature contributions
                - base_value: float expected value
                - predicted_threat: float final threat score
        """
        explanations = []
        
        for track in tracks:
            track_id = track.get("track_id", -1)
            
            # Build feature vector from track data
            # [proximity_score, speed_score, class_score]
            proximity_score = track.get("proximity_score", 0.0)
            speed_score = track.get("speed_score", 0.0)
            class_score = track.get("class_score", 0.0)
            
            features = np.array([proximity_score, speed_score, class_score])
            
            # Compute SHAP values
            try:
                shap_values = self.explainer.shap_values(features, nsamples=100)
                base_value = float(self.explainer.expected_value)
            except Exception as e:
                logger.warning(f"SHAP computation failed for track {track_id}: {e}")
                # Fallback to manual attribution
                shap_values = self._fallback_shap(features)
                base_value = 0.5
            
            # Build explanation dict
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Handle list wrapper
            
            explanation = {
                "track_id": track_id,
                "shap_values": {
                    "proximity": float(shap_values[0]),
                    "speed": float(shap_values[1]),
                    "class": float(shap_values[2]),
                },
                "base_value": base_value,
                "predicted_threat": track.get("threat_score", 0.0),
            }
            explanations.append(explanation)
        
        return explanations
    
    def _fallback_shap(self, features: np.ndarray) -> np.ndarray:
        """Compute approximate SHAP values manually as fallback.
        
        When KernelExplainer fails, use a simple linear attribution.
        
        Args:
            features: Feature vector [proximity, speed, class].
            
        Returns:
            Approximate SHAP values.
        """
        weights = self.cfg.threat.weights
        
        # Simple linear attribution based on weights and feature values
        # This is an approximation but provides reasonable explanations
        shap_proximity = features[0] * weights["proximity"] * 0.5
        shap_speed = features[1] * weights["speed"] * 0.5
        shap_class = features[2] * weights["class_danger"] * 0.5
        
        return np.array([shap_proximity, shap_speed, shap_class])
    
    def explain_single(self, track: Dict) -> Optional[Dict]:
        """Generate SHAP explanation for a single track.
        
        Args:
            track: Single track dict.
            
        Returns:
            Explanation dict or None if computation fails.
        """
        explanations = self.explain([track])
        return explanations[0] if explanations else None
