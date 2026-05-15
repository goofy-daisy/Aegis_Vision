"""API package for AegisVision.

Provides REST API and WebSocket endpoints for threat detection.
"""

from api.main import app
from api.schemas import (
    DetectionResult,
    FrameResult,
    HealthResponse,
    VideoProcessingRequest,
    VideoProcessingResponse,
)

__all__ = [
    "app",
    "DetectionResult",
    "FrameResult",
    "HealthResponse",
    "VideoProcessingRequest",
    "VideoProcessingResponse",
]
