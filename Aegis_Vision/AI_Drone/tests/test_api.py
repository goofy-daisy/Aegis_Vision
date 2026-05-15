"""Tests for API endpoints.

Tests health check, frame prediction, and error handling.
"""

import io
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Create TestClient as context manager to trigger lifespan."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Test cases for /health endpoint."""
    
    def test_health_returns_200(self, client):
        """Test that /health returns 200 status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "device" in data
        assert "model_loaded" in data
    
    def test_health_response_structure(self, client):
        """Test that /health response has correct structure."""
        response = client.get("/health")
        data = response.json()
        
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert isinstance(data["model_loaded"], bool)
        assert "version" in data


class TestFramePrediction:
    """Test cases for /predict/frame endpoint."""
    
    def create_test_image(self, format="JPEG"):
        """Create a test image in memory."""
        img = Image.new("RGB", (640, 480), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format=format)
        img_bytes.seek(0)
        return img_bytes
    
    def test_predict_frame_with_jpeg(self, client):
        """Test frame prediction with JPEG upload."""
        img_bytes = self.create_test_image("JPEG")
        
        response = client.post(
            "/predict/frame",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        
        # Should succeed (or return 503 if pipeline not initialized)
        assert response.status_code in [200, 503]
    
    def test_predict_frame_with_png(self, client):
        """Test frame prediction with PNG upload."""
        img_bytes = self.create_test_image("PNG")
        
        response = client.post(
            "/predict/frame",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        
        assert response.status_code in [200, 503]
    
    def test_invalid_file_type_returns_422(self, client):
        """Test that invalid file type returns 422."""
        response = client.post(
            "/predict/frame",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        
        assert response.status_code == 422
    
    def test_empty_file_returns_error(self, client):
        """Test that empty file returns error."""
        response = client.post(
            "/predict/frame",
            files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
        )
        
        # Should return some error status
        assert response.status_code in [400, 422, 500]


class TestVideoPrediction:
    """Test cases for /predict/video endpoint."""
    
    def create_test_video(self):
        """Create a minimal test video in memory."""
        # Create a simple MP4-like header (not valid, but triggers validation)
        return io.BytesIO(b"\x00\x00\x00\x20ftypisom")
    
    def test_predict_video_with_mp4(self, client):
        """Test video prediction with MP4 upload."""
        video_bytes = self.create_test_video()
        
        response = client.post(
            "/predict/video",
            files={"file": ("test.mp4", video_bytes, "video/mp4")},
        )
        
        # Should succeed or return 503 if pipeline not ready
        assert response.status_code in [200, 503]
    
    def test_invalid_video_type_returns_422(self, client):
        """Test that invalid video type returns 422."""
        response = client.post(
            "/predict/video",
            files={"file": ("test.txt", io.BytesIO(b"not a video"), "text/plain")},
        )
        
        assert response.status_code == 422


class TestErrorHandling:
    """Test cases for API error handling."""
    
    def test_404_error(self, client):
        """Test 404 for non-existent endpoint."""
        response = client.get("/nonexistent")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test 405 for wrong HTTP method."""
        response = client.post("/health")
        
        assert response.status_code == 405


class TestValidation:
    """Test input validation."""
    
    def test_frame_result_schema(self):
        """Test that FrameResult schema is valid."""
        from api.schemas import FrameResult, DetectionResult
        
        detection = DetectionResult(
            track_id=1,
            bbox=[100, 100, 200, 200],
            class_name="car",
            confidence=0.9,
            threat_score=0.85,
            threat_level="HIGH",
        )
        
        result = FrameResult(
            frame_idx=0,
            detections=[detection],
            fps=30.5,
            latency_ms=15.2,
        )
        
        assert result.frame_idx == 0
        assert len(result.detections) == 1
        assert result.fps == 30.5
    
    def test_video_processing_request_schema(self):
        """Test VideoProcessingRequest schema."""
        from api.schemas import VideoProcessingRequest
        
        request = VideoProcessingRequest(explain=True)
        
        assert request.explain is True
