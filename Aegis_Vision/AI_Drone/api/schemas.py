"""Pydantic schemas for AegisVision API.

Defines request and response models for all API endpoints.
All schemas include type hints and validation.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    """Single detection result with threat information."""
    track_id: int = Field(..., description="Consistent track identifier")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    class_name: str = Field(..., description="Object class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    threat_score: float = Field(..., ge=0.0, le=1.0, description="Threat score")
    threat_level: str = Field(..., description="Threat level: HIGH, MEDIUM, or LOW")
    speed_mps: Optional[float] = Field(None, description="Speed in meters per second")
    proximity_m: Optional[float] = Field(None, description="Proximity in meters")
    lat: Optional[float] = Field(None, description="Latitude coordinate")
    lon: Optional[float] = Field(None, description="Longitude coordinate")


class FrameResult(BaseModel):
    """Processing result for a single frame."""
    frame_idx: int = Field(..., description="Frame index")
    detections: List[DetectionResult] = Field(default_factory=list, description="Detected objects")
    fps: float = Field(..., description="Current FPS")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Processing timestamp")


class VideoProcessingRequest(BaseModel):
    """Request model for video processing."""
    explain: bool = Field(default=False, description="Generate SHAP explanations")


class VideoProcessingResponse(BaseModel):
    """Response model for video processing."""
    total_frames: int = Field(..., description="Total frames processed")
    avg_fps: float = Field(..., description="Average FPS")
    avg_latency: float = Field(..., description="Average latency in milliseconds")
    message: str = Field(..., description="Status message")
    results: List[FrameResult] = Field(default_factory=list, description="Per-frame results")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="System status")
    device: str = Field(..., description="Compute device (cuda/cpu)")
    model_loaded: bool = Field(..., description="Whether YOLO model is loaded")
    version: str = Field(default="1.0.0", description="API version")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class WebSocketMessage(BaseModel):
    """WebSocket message wrapper."""
    type: str = Field(..., description="Message type: frame, result, error")
    data: Optional[Dict] = Field(None, description="Message payload")
    timestamp: datetime = Field(default_factory=datetime.now)
