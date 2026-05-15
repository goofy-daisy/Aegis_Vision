"""Threat scoring model for AegisVision.

Provides interpretable threat scoring based on proximity, speed, and object class.
Uses a weighted scoring function with sigmoid normalization for interpretability.
"""

import math
from typing import Dict, List, Optional

from utils.config_loader import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class ThreatScorer:
    """Interpretable threat scoring model for tracked objects.
    
    Computes threat scores based on three weighted factors:
    - Proximity: Closer objects are more threatening
    - Speed: Faster moving objects are more threatening  
    - Class danger: Certain classes (vehicles) are more dangerous
    
    The scoring is deterministic and interpretable, with all weights
    exposed through configuration.
    
    Attributes:
        weights: Dict of feature weights from config.
        danger_map: Mapping of class names to danger multipliers.
        high_threshold: Threshold for HIGH threat level.
        medium_threshold: Threshold for MEDIUM threat level.
    """
    
    def __init__(self, cfg: Config):
        """Initialize threat scorer with configuration.
        
        Args:
            cfg: Configuration object with threat scoring parameters.
        """
        # Load weights from config
        self.weights = cfg.threat.weights
        self.proximity_max = cfg.threat.proximity_max_meters
        self.speed_max = cfg.threat.speed_max_mps
        self.high_threshold = cfg.threat.high_threat_threshold
        self.medium_threshold = cfg.threat.medium_threat_threshold
        
        # Build danger map: dangerous classes get 1.0, others get 0.5
        self.danger_map: Dict[str, float] = {}
        for class_name in cfg.threat.dangerous_classes:
            self.danger_map[class_name] = 1.0
        
        logger.info(f"ThreatScorer initialized: proximity_weight={self.weights['proximity']}, "
                    f"speed_weight={self.weights['speed']}, "
                    f"class_weight={self.weights['class_danger']}")
    
    def score(self, track: Dict) -> Dict:
        """Compute threat score for a single track.
        
        Calculates three normalized sub-scores and combines them with
        configured weights. Applies sigmoid smoothing to the result.
        
        Args:
            track: Track dict with keys:
                - speed_mps: float, object speed in meters per second
                - proximity_m: float, object proximity in meters
                - class_name: str, object class name
                
        Returns:
            Dict containing:
                - threat_score: float, final threat score (0-1)
                - threat_level: str, "HIGH", "MEDIUM", or "LOW"
                - proximity_score: float, normalized proximity component (0-1)
                - speed_score: float, normalized speed component (0-1)
                - class_score: float, class danger score (0-1)
                
        Raises:
            ValueError: If track is missing required fields.
        """
        # Validate required fields
        required_fields = ["speed_mps", "proximity_m", "class_name"]
        for field in required_fields:
            if field not in track:
                raise ValueError(f"Track missing required field: {field}")
        
        # Extract values
        speed_mps = float(track["speed_mps"])
        proximity_m = float(track["proximity_m"])
        class_name = str(track["class_name"])
        
        # Compute normalized proximity score (closer = higher threat)
        # Invert so 0 distance = 1.0 score, max distance = 0.0 score
        proximity_score = 1.0 - (proximity_m / self.proximity_max)
        proximity_score = max(0.0, min(1.0, proximity_score))  # Clip to [0, 1]
        
        # Compute normalized speed score (faster = higher threat)
        speed_score = speed_mps / self.speed_max
        speed_score = max(0.0, min(1.0, speed_score))  # Clip to [0, 1]
        
        # Compute class danger score
        class_score = self.danger_map.get(class_name, 0.5)
        
        # Compute weighted raw score
        raw_score = (
            proximity_score * self.weights["proximity"] +
            speed_score * self.weights["speed"] +
            class_score * self.weights["class_danger"]
        )
        
        # Apply sigmoid to smooth extremes
        # Using sigmoid centered at 0.5 with steepness of 10
        threat_score = 1.0 / (1.0 + math.exp(-6.0 * (raw_score - 0.3)))
        
        # Determine threat level
        if threat_score >= self.high_threshold:
            threat_level = "HIGH"
        elif threat_score >= self.medium_threshold:
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"
        
        return {
            "threat_score": float(threat_score),
            "threat_level": threat_level,
            "proximity_score": float(proximity_score),
            "speed_score": float(speed_score),
            "class_score": float(class_score),
        }

    def classify_behaviour(self, track_history: list) -> str:
        """Classify object behaviour from position history.

        Args:
            track_history: List of (frame_idx, center_x, center_y) tuples.

        Returns:
            Behaviour string: STATIONARY, APPROACHING, RETREATING,
            CIRCLING, or ERRATIC.
        """
        if len(track_history) < 5:
            return "STATIONARY"

        # Use last 20 positions or all available
        history = track_history[-20:]

        # Compute displacement vectors
        positions = [(h[1], h[2]) for h in history]

        # Total path length
        path_length = sum(
            math.sqrt((positions[i][0] - positions[i - 1][0]) ** 2 +
                      (positions[i][1] - positions[i - 1][1]) ** 2)
            for i in range(1, len(positions))
        )

        # Net displacement (start to end)
        net_dx = positions[-1][0] - positions[0][0]
        net_dy = positions[-1][1] - positions[0][1]
        net_displacement = math.sqrt(net_dx ** 2 + net_dy ** 2)

        # Straightness ratio
        straightness = net_displacement / (path_length + 1e-6)

        # Image center (assume 640px)
        img_center_x, img_center_y = 320, 320

        # Distance to center: start vs end
        start_dist = math.sqrt(
            (positions[0][0] - img_center_x) ** 2 +
            (positions[0][1] - img_center_y) ** 2
        )
        end_dist = math.sqrt(
            (positions[-1][0] - img_center_x) ** 2 +
            (positions[-1][1] - img_center_y) ** 2
        )

        # Classification logic
        if path_length < 5:
            return "STATIONARY"
        elif straightness < 0.3 and path_length > 30:
            return "CIRCLING"
        elif straightness < 0.5:
            return "ERRATIC"
        elif end_dist < start_dist - 10:
            return "APPROACHING"
        elif end_dist > start_dist + 10:
            return "RETREATING"
        else:
            return "STATIONARY"

    def get_counterfactual(self, track: dict) -> str:
        """Generate counterfactual explanation for threat score.

        Args:
            track: Track dict with current threat values.

        Returns:
            Human-readable string explaining what would trigger HIGH threat.
        """
        current_score = track.get("threat_score", 0.0)

        if current_score >= self.high_threshold:
            return "Already HIGH threat."

        speed_needed = None
        proximity_needed = None

        # Find speed needed for HIGH
        for test_speed in range(0, int(self.speed_max) + 1, 5):
            test_track = dict(track)
            test_track["speed_mps"] = float(test_speed)
            result = self.score(test_track)
            if result["threat_score"] >= self.high_threshold:
                speed_needed = test_speed
                break

        # Find proximity needed for HIGH
        for test_prox in range(int(self.proximity_max), 0, -10):
            test_track = dict(track)
            test_track["proximity_m"] = float(test_prox)
            result = self.score(test_track)
            if result["threat_score"] >= self.high_threshold:
                proximity_needed = test_prox
                break

        parts = []
        if speed_needed is not None:
            current_speed = track.get("speed_mps", 0)
            delta = speed_needed - current_speed
            if delta > 0:
                parts.append(f"speed increases by {delta:.0f} m/s")

        if proximity_needed is not None:
            current_prox = track.get("proximity_m", self.proximity_max)
            delta = current_prox - proximity_needed
            if delta > 0:
                parts.append(f"proximity decreases by {delta:.0f} m")

        if not parts:
            return "Would not reach HIGH with current class."

        return "Would become HIGH if: " + " OR ".join(parts)

    def score_batch(self, tracks: List[Dict]) -> List[Dict]:
        """Compute threat scores for multiple tracks.
        
        Args:
            tracks: List of track dicts with required fields.
            
        Returns:
            List of score dicts, one per input track.
        """
        return [self.score(track) for track in tracks]
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names used in scoring.
        
        Returns:
            List of feature names for explainability.
        """
        return ["proximity", "speed", "class_danger"]
    
    def get_feature_weights(self) -> Dict[str, float]:
        """Return feature weights for explainability.
        
        Returns:
            Dict mapping feature names to weights.
        """
        return {
            "proximity": self.weights["proximity"],
            "speed": self.weights["speed"],
            "class_danger": self.weights["class_danger"],
        }
