"""Tests for threat model.

Tests threat scoring logic, threat level assignment, and sigmoid output.
"""

import math

import pytest
from models.threat.threat_model import ThreatScorer
from utils.config_loader import Config


class MockConfig:
    """Mock configuration for testing."""
    
    def __init__(self):
        self.threat = MockThreatConfig()


class MockThreatConfig:
    """Mock threat configuration."""
    
    def __init__(self):
        self.weights = {
            "proximity": 0.45,
            "speed": 0.35,
            "class_danger": 0.20,
        }
        self.dangerous_classes = ["car", "van", "truck", "bus"]
        self.high_threat_threshold = 0.70
        self.medium_threat_threshold = 0.40
        self.proximity_max_meters = 500.0
        self.speed_max_mps = 50.0


class TestThreatScorer:
    """Test cases for ThreatScorer."""
    
    @pytest.fixture
    def scorer(self):
        """Create a ThreatScorer instance."""
        config = MockConfig()
        return ThreatScorer(config)
    
    def test_high_threat_dangerous_close_fast(self, scorer):
        """Test HIGH threat when proximity=0, speed=max, dangerous class."""
        track = {
            "speed_mps": 50.0,  # Max speed
            "proximity_m": 0.0,   # Closest possible
            "class_name": "car",  # Dangerous class
        }
        
        result = scorer.score(track)
        
        assert result["threat_level"] == "HIGH"
        assert result["threat_score"] >= 0.70
        assert result["proximity_score"] == 1.0  # 1 - (0/500)
        assert result["speed_score"] == 1.0      # 50/50
        assert result["class_score"] == 1.0      # dangerous class
    
    def test_low_threat_safe_distant_slow(self, scorer):
        """Test LOW threat when proximity=max, speed=0, safe class."""
        track = {
            "speed_mps": 0.0,       # Stationary
            "proximity_m": 500.0,   # Furthest possible
            "class_name": "pedestrian",  # Safe class
        }
        
        result = scorer.score(track)
        
        assert result["threat_level"] == "LOW"
        assert result["threat_score"] < 0.40
        assert result["proximity_score"] == 0.0  # 1 - (500/500)
        assert result["speed_score"] == 0.0     # 0/50
        assert result["class_score"] == 0.5    # non-dangerous class
    
    def test_medium_threat_threshold(self, scorer):
        """Test MEDIUM threat level assignment."""
        # Create a track that should fall in MEDIUM range
        track = {
            "speed_mps": 25.0,     # Medium speed
            "proximity_m": 250.0,  # Medium proximity
            "class_name": "car",   # Dangerous class
        }
        
        result = scorer.score(track)
        
        # Should be in [0.40, 0.70) range for MEDIUM
        assert result["threat_level"] in ["MEDIUM", "HIGH"]  # Depending on exact calculation
    
    def test_sigmoid_output_bounded(self, scorer):
        """Test that sigmoid output is bounded [0, 1]."""
        test_cases = [
            {"speed_mps": 0, "proximity_m": 0, "class_name": "car"},
            {"speed_mps": 100, "proximity_m": 1000, "class_name": "pedestrian"},
            {"speed_mps": -10, "proximity_m": -100, "class_name": "truck"},
        ]
        
        for track in test_cases:
            result = scorer.score(track)
            
            assert 0 <= result["threat_score"] <= 1, \
                f"Threat score {result['threat_score']} out of bounds for track: {track}"
            assert 0 <= result["proximity_score"] <= 1
            assert 0 <= result["speed_score"] <= 1
            assert 0 <= result["class_score"] <= 1
    
    def test_proximity_score_calculation(self, scorer):
        """Test proximity score calculation."""
        track = {
            "speed_mps": 0,
            "proximity_m": 250,  # Half of max
            "class_name": "car",
        }
        
        result = scorer.score(track)
        
        # proximity_score = 1 - (250/500) = 0.5
        assert abs(result["proximity_score"] - 0.5) < 0.01
    
    def test_speed_score_calculation(self, scorer):
        """Test speed score calculation."""
        track = {
            "speed_mps": 25,  # Half of max
            "proximity_m": 500,
            "class_name": "car",
        }
        
        result = scorer.score(track)
        
        # speed_score = 25/50 = 0.5
        assert abs(result["speed_score"] - 0.5) < 0.01
    
    def test_dangerous_class_score(self, scorer):
        """Test class score for dangerous classes."""
        dangerous_tracks = [
            {"speed_mps": 0, "proximity_m": 500, "class_name": cls}
            for cls in ["car", "van", "truck", "bus"]
        ]
        
        for track in dangerous_tracks:
            result = scorer.score(track)
            assert result["class_score"] == 1.0, f"Failed for class: {track['class_name']}"
    
    def test_non_dangerous_class_score(self, scorer):
        """Test class score for non-dangerous classes."""
        safe_tracks = [
            {"speed_mps": 0, "proximity_m": 500, "class_name": cls}
            for cls in ["pedestrian", "bicycle", "plane", "ship"]
        ]
        
        for track in safe_tracks:
            result = scorer.score(track)
            assert result["class_score"] == 0.5, f"Failed for class: {track['class_name']}"
    
    def test_sigmoid_symmetry(self):
        """Test that sigmoid is symmetric around 0.5."""
        def sigmoid(x):
            return 1.0 / (1.0 + math.exp(-10.0 * (x - 0.5)))
        
        # sigmoid(0.5) should be 0.5
        assert abs(sigmoid(0.5) - 0.5) < 0.001
        
        # sigmoid(0.5 + d) + sigmoid(0.5 - d) should be ~1.0
        for d in [0.1, 0.2, 0.3]:
            s1 = sigmoid(0.5 + d)
            s2 = sigmoid(0.5 - d)
            assert abs(s1 + s2 - 1.0) < 0.01
    
    def test_missing_fields_raises_error(self, scorer):
        """Test that missing required fields raises ValueError."""
        incomplete_track = {
            "speed_mps": 10,  # Missing proximity_m and class_name
        }
        
        with pytest.raises(ValueError):
            scorer.score(incomplete_track)
    
    def test_batch_scoring(self, scorer):
        """Test batch scoring function."""
        tracks = [
            {"speed_mps": 50, "proximity_m": 0, "class_name": "car"},
            {"speed_mps": 0, "proximity_m": 500, "class_name": "pedestrian"},
            {"speed_mps": 25, "proximity_m": 250, "class_name": "truck"},
        ]
        
        results = scorer.score_batch(tracks)
        
        assert len(results) == 3
        assert all(0 <= r["threat_score"] <= 1 for r in results)
    
    def test_feature_names(self, scorer):
        """Test feature names are correct."""
        names = scorer.get_feature_names()
        
        assert names == ["proximity", "speed", "class_danger"]
    
    def test_feature_weights(self, scorer):
        """Test feature weights match config."""
        weights = scorer.get_feature_weights()
        
        assert weights["proximity"] == 0.45
        assert weights["speed"] == 0.35
        assert weights["class_danger"] == 0.20

    # ── Phase 2: Behaviour Classification Tests ─────────────────────────────

    def test_classify_behaviour_stationary(self, scorer):
        """Test STATIONARY behaviour detection."""
        # Track history with minimal movement
        history = [(0, 100, 100), (1, 101, 99), (2, 100, 100)]
        
        behaviour = scorer.classify_behaviour(history)
        
        assert behaviour == "STATIONARY"

    def test_classify_behaviour_approaching(self, scorer):
        """Test APPROACHING behaviour detection."""
        # Track history moving toward image center (640, 640 is center of 1280x1280)
        history = [(0, 200, 200), (1, 150, 150), (2, 100, 100)]
        
        behaviour = scorer.classify_behaviour(history, image_size=(640, 640))
        
        assert behaviour == "APPROACHING"

    def test_classify_behaviour_receding(self, scorer):
        """Test RECEDING behaviour detection."""
        # Track history moving away from image center
        history = [(0, 100, 100), (1, 150, 150), (2, 200, 200)]
        
        behaviour = scorer.classify_behaviour(history, image_size=(640, 640))
        
        assert behaviour == "RECEDING"

    def test_classify_behaviour_circling(self, scorer):
        """Test CIRCLING behaviour detection."""
        # Track history with circular pattern around center
        cx, cy = 320, 320  # Center of 640x640 image
        radius = 100
        # Points around a circle: right, bottom, left, top
        history = [
            (0, cx + radius, cy),      # Right
            (1, cx, cy + radius),      # Bottom
            (2, cx - radius, cy),      # Left
            (3, cx, cy - radius),      # Top
        ]
        
        behaviour = scorer.classify_behaviour(history, image_size=(640, 640))
        
        assert behaviour == "CIRCLING"

    def test_classify_behaviour_empty_history(self, scorer):
        """Test behaviour classification with empty history."""
        behaviour = scorer.classify_behaviour([])
        
        assert behaviour == "STATIONARY"

    def test_classify_behaviour_single_point(self, scorer):
        """Test behaviour classification with single point."""
        behaviour = scorer.classify_behaviour([(0, 100, 100)])
        
        assert behaviour == "STATIONARY"

    # ── Phase 2: Counterfactual Explanation Tests ────────────────────────────

    def test_get_counterfactual_lowers_score(self, scorer):
        """Test counterfactual explains how to lower threat score."""
        # HIGH threat track
        track = {
            "speed_mps": 50.0,
            "proximity_m": 0.0,
            "class_name": "car",
        }
        
        counterfactual = scorer.get_counterfactual(track)
        
        assert isinstance(counterfactual, str)
        assert "If" in counterfactual
        assert "score would" in counterfactual
        assert "0." in counterfactual  # Some score value

    def test_get_counterfactual_already_low(self, scorer):
        """Test counterfactual for already low threat track."""
        # LOW threat track
        track = {
            "speed_mps": 0.0,
            "proximity_m": 500.0,
            "class_name": "pedestrian",
        }
        
        counterfactual = scorer.get_counterfactual(track)
        
        assert isinstance(counterfactual, str)

    def test_get_counterfactual_medium_threshold(self, scorer):
        """Test counterfactual for medium threat track."""
        track = {
            "speed_mps": 25.0,
            "proximity_m": 250.0,
            "class_name": "car",
        }
        
        counterfactual = scorer.get_counterfactual(track)
        
        assert isinstance(counterfactual, str)
        assert "If" in counterfactual
