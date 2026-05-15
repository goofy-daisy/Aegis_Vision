"""Tests for AegisVision pipeline.

Tests the complete processing pipeline including frame processing,
threat scoring, and metrics computation.
"""

from collections import deque
from unittest.mock import Mock, patch

import numpy as np
import pytest
from pipeline.full_pipeline import AegisPipeline


class TestPipeline:
    """Test cases for AegisPipeline."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock()
        config.model.input_size = 640
        config.model.confidence_threshold = 0.4
        config.model.nms_iou_threshold = 0.45
        config.model.max_detections = 20
        config.model.device = "cpu"
        config.model.yolo_weights_pretrained = "yolov8n.pt"
        config.model.yolo_model_size = "n"
        config.tracking.max_age = 30
        config.tracking.min_hits = 3
        config.tracking.iou_threshold = 0.3
        config.tracking.max_tracks = 20
        config.threat.weights = {"proximity": 0.45, "speed": 0.35, "class_danger": 0.20}
        config.threat.dangerous_classes = ["car", "van", "truck", "bus"]
        config.threat.high_threat_threshold = 0.70
        config.threat.medium_threat_threshold = 0.40
        config.threat.proximity_max_meters = 500.0
        config.threat.speed_max_mps = 50.0
        config.dashboard.map_center_lat = -37.8136
        config.dashboard.map_center_lon = 144.9631
        config.simulation.telemetry_update_hz = 10
        config.paths.weights = Mock()
        config.paths.weights.__truediv__ = Mock(return_value=Mock(exists=Mock(return_value=False)))
        return config
    
    def test_process_frame_with_synthetic_frame(self, mock_config):
        """Test pipeline processing with synthetic frame."""
        # Create synthetic 640x640 black frame
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Mock the pipeline dependencies
        with patch("pipeline.full_pipeline.load_model") as mock_load_model, \
             patch("pipeline.full_pipeline.DeepSortTracker") as mock_tracker, \
             patch("pipeline.full_pipeline.TelemetrySimulator") as mock_telemetry, \
             patch("pipeline.full_pipeline.ThreatScorer") as mock_threat:
            
            # Setup mocks
            mock_model = Mock()
            mock_load_model.return_value = mock_model
            
            # Mock inference returning empty detections
            mock_model.return_value = []
            
            mock_tracker_instance = Mock()
            mock_tracker_instance.update.return_value = []
            mock_tracker.return_value = mock_tracker_instance
            
            mock_telemetry_instance = Mock()
            mock_telemetry_instance.update.return_value = []
            mock_telemetry.return_value = mock_telemetry_instance
            
            mock_threat_instance = Mock()
            mock_threat_instance.score.return_value = {
                "threat_score": 0.5,
                "threat_level": "MEDIUM",
            }
            mock_threat.return_value = mock_threat_instance
            
            # Create pipeline
            pipeline = AegisPipeline(mock_config)
            
            # Process frame
            result = pipeline.process_frame(frame, frame_idx=0)
            
            # Verify result structure
            assert "frame_idx" in result
            assert "detections" in result
            assert "tracks" in result
            assert "fps" in result
            assert "latency_ms" in result
            assert "annotated_frame" in result
            
            # Frame index should match
            assert result["frame_idx"] == 0
    
    def test_empty_detections_dont_crash(self, mock_config):
        """Test that empty detections don't crash the pipeline."""
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        
        with patch("pipeline.full_pipeline.load_model") as mock_load_model, \
             patch("pipeline.full_pipeline.DeepSortTracker") as mock_tracker, \
             patch("pipeline.full_pipeline.TelemetrySimulator") as mock_telemetry, \
             patch("pipeline.full_pipeline.ThreatScorer") as mock_threat:
            
            mock_load_model.return_value = Mock()
            mock_tracker.return_value = Mock(update=Mock(return_value=[]))
            mock_telemetry.return_value = Mock(update=Mock(return_value=[]))
            mock_threat.return_value = Mock(score=Mock(return_value={}))
            
            pipeline = AegisPipeline(mock_config)
            
            # Should not raise exception
            result = pipeline.process_frame(frame, frame_idx=0)
            
            assert result["detections"] == []
            assert result["tracks"] == []
    
    def test_threat_scores_in_valid_range(self, mock_config):
        """Test that threat scores are in [0, 1] range."""
        from models.threat import ThreatScorer
        
        scorer = ThreatScorer(mock_config)
        
        # Test with various inputs
        test_tracks = [
            {"speed_mps": 0, "proximity_m": 500, "class_name": "pedestrian"},
            {"speed_mps": 50, "proximity_m": 0, "class_name": "car"},
            {"speed_mps": 25, "proximity_m": 250, "class_name": "truck"},
        ]
        
        for track in test_tracks:
            result = scorer.score(track)
            
            assert 0 <= result["threat_score"] <= 1
            assert 0 <= result["proximity_score"] <= 1
            assert 0 <= result["speed_score"] <= 1
            assert 0 <= result["class_score"] <= 1
    
    def test_fps_computation(self):
        """Test FPS computation with mocked frame times."""
        frame_times = deque([0.033, 0.033, 0.033], maxlen=30)  # ~30 FPS
        
        avg_frame_time = sum(frame_times) / len(frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Should be approximately 30 FPS
        assert 29 < fps < 31
    
    def test_process_frame_error_handling(self, mock_config):
        """Test error handling in process_frame."""
        with patch("pipeline.full_pipeline.load_model") as mock_load_model, \
             patch("pipeline.full_pipeline.DeepSortTracker"), \
             patch("pipeline.full_pipeline.TelemetrySimulator"), \
             patch("pipeline.full_pipeline.ThreatScorer"):
            
            mock_load_model.return_value = Mock()
            pipeline = AegisPipeline(mock_config)
            
            # Test with None frame
            result = pipeline.process_frame(None, frame_idx=0)
            assert "error" in result
            
            # Test with invalid frame type
            result = pipeline.process_frame("invalid", frame_idx=0)
            assert "error" in result


class TestPreprocess:
    """Test cases for preprocessing functions."""
    
    def test_resize_frame_maintains_aspect_ratio(self):
        """Test that resize maintains aspect ratio."""
        from pipeline.preprocess import resize_frame
        
        # Create 4:3 aspect ratio frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = resize_frame(frame, target_size=640)
        
        assert resized.shape == (640, 640, 3)
    
    def test_normalize_frame_range(self):
        """Test that normalization produces [0, 1] range."""
        from pipeline.preprocess import normalize_frame
        
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        normalized = normalize_frame(frame)
        
        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0


class TestPostprocess:
    """Test cases for postprocessing functions."""
    
    def test_draw_detections_preserves_shape(self):
        """Test that draw_detections preserves frame shape."""
        from pipeline.postprocess import draw_detections
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks = [
            {
                "track_id": 1,
                "bbox": [100, 100, 200, 200],
                "class_name": "car",
                "confidence": 0.9,
                "threat_level": "HIGH",
                "threat_score": 0.85,
            }
        ]
        
        result = draw_detections(frame, tracks)
        
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype
    
    def test_draw_metrics_overlay_preserves_shape(self):
        """Test that draw_metrics_overlay preserves frame shape."""
        from pipeline.postprocess import draw_metrics_overlay
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = draw_metrics_overlay(frame, fps=30.5, latency_ms=15.2, object_count=5)
        
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype
