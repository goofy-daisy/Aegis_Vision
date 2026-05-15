"""FastAPI application for AegisVision.

Provides REST and WebSocket endpoints for drone threat detection:
- Health check
- Frame prediction (image upload)
- Video prediction (video upload)
- Real-time streaming (WebSocket)
- Session history retrieval

Authentication: X-API-Key header required for all endpoints.
"""

import base64
import io
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, Security, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from api.middleware import (
    global_exception_handler,
    setup_cors_middleware,
    validation_exception_handler,
)
from api.schemas import (
    DetectionResult,
    ErrorResponse,
    FrameResult,
    HealthResponse,
    VideoProcessingRequest,
    VideoProcessingResponse,
)
from database.results_db import get_all_sessions, get_session_detections, get_session_events
from pipeline import AegisPipeline
from utils.config_loader import Config, load_config
from utils.device_utils import get_device, get_device_info
from utils.logger import get_logger

logger = get_logger(__name__)

# Global pipeline instance (initialized in lifespan)
pipeline: Optional[AegisPipeline] = None
config: Optional[Config] = None

# API Key authentication
API_KEY = "aegisvision-demo-key-2024"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify the API key from header."""
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager.
    
    Initializes the pipeline on startup and handles cleanup.
    """
    global pipeline, config
    
    # Startup
    logger.info("Starting AegisVision API...")
    config = load_config()
    pipeline = AegisPipeline(config)
    logger.info("AegisVision API ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AegisVision API...")
    pipeline = None


app = FastAPI(
    title="AegisVision API",
    description="AI-powered drone threat detection system. "
                "All endpoints require X-API-Key header (use 'aegisvision-demo-key-2024').",
    version="1.1.0",
    lifespan=lifespan,
)

# Setup middleware
setup_cors_middleware(app)

# Register exception handlers
app.add_exception_handler(Exception, global_exception_handler)
from fastapi.exceptions import RequestValidationError
app.add_exception_handler(RequestValidationError, validation_exception_handler)


@app.get("/health", response_model=HealthResponse)
async def health_check(api_key: str = Security(verify_api_key)) -> HealthResponse:
    """Health check endpoint.
    
    Returns system status, device info, and model loading status.
    Requires X-API-Key header.
    """
    device_info = get_device_info()
    
    return HealthResponse(
        status="healthy",
        device=device_info["device_name"],
        model_loaded=pipeline is not None,
        version="1.0.0",
    )


@app.post("/predict/frame", response_model=FrameResult)
async def predict_frame(
    file: UploadFile = File(...),
    explain: bool = False,
    api_key: str = Security(verify_api_key),
) -> FrameResult:
    """Process a single image frame.
    
    Args:
        file: Uploaded image file (JPEG/PNG).
        explain: Whether to generate SHAP explanations.
        
    Returns:
        Frame processing results with detections and threat scores.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/jpg", "image/png"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )
    
    # Validate file size
    max_size_mb = config.api.max_upload_mb if config else 100
    contents = await file.read()
    if len(contents) > max_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {max_size_mb}MB"
        )
    
    try:
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Process frame
        result = pipeline.process_frame(frame, frame_idx=0, explain=explain)
        
        # Convert tracks to DetectionResult schema
        detections = []
        for track in result["tracks"]:
            detections.append(DetectionResult(
                track_id=track["track_id"],
                bbox=track["bbox"],
                class_name=track["class_name"],
                confidence=track["confidence"],
                threat_score=track["threat_score"],
                threat_level=track["threat_level"],
                speed_mps=track.get("speed_mps"),
                proximity_m=track.get("proximity_m"),
                lat=track.get("lat"),
                lon=track.get("lon"),
            ))
        
        return FrameResult(
            frame_idx=result["frame_idx"],
            detections=detections,
            fps=result["fps"],
            latency_ms=result["latency_ms"],
        )
        
    except Exception as e:
        logger.error(f"Frame prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/predict/video", response_model=VideoProcessingResponse)
async def predict_video(
    file: UploadFile = File(...),
    request: VideoProcessingRequest = None,
    api_key: str = Security(verify_api_key),
) -> VideoProcessingResponse:
    """Process a video file.
    
    Args:
        file: Uploaded video file (MP4/AVI).
        request: Processing options.
        
    Returns:
        Video processing results with per-frame detections.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Validate file type
    allowed_types = ["video/mp4", "video/avi", "video/x-msvideo"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )
    
    # Validate file size
    max_size_mb = config.api.max_upload_mb if config else 100
    contents = await file.read()
    if len(contents) > max_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {max_size_mb}MB"
        )
    
    # Save to temp file
    try:
        suffix = Path(file.filename).suffix if file.filename else ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)
        
        # Process video
        explain = request.explain if request else False
        result = pipeline.process_video(tmp_path, explain=explain)
        
        # Cleanup temp file
        tmp_path.unlink(missing_ok=True)
        
        return VideoProcessingResponse(
            total_frames=result["total_frames"],
            avg_fps=result["avg_fps"],
            avg_latency=result["avg_latency"],
            message="Video processed successfully",
            results=[],  # Could include full results if needed
        )
        
    except Exception as e:
        logger.error(f"Video prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming.
    
    Client sends base64-encoded JPEG frames.
    Server returns JSON processing results.
    Rate-limited by config.api.websocket_fps.
    """
    if pipeline is None:
        await websocket.close(code=1011, reason="Pipeline not initialized")
        return
    
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    frame_idx = 0
    min_interval = 1.0 / config.api.websocket_fps if config else 1.0 / 15
    last_process_time = 0
    
    try:
        while True:
            # Receive message
            message = await websocket.receive_text()
            
            # Rate limiting
            current_time = time.time()
            if current_time - last_process_time < min_interval:
                continue
            last_process_time = current_time
            
            try:
                # Decode base64 image
                image_data = base64.b64decode(message)
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Failed to decode frame",
                    })
                    continue
                
                # Process frame
                result = pipeline.process_frame(frame, frame_idx)
                frame_idx += 1
                
                # Convert to response format
                detections = []
                for track in result["tracks"]:
                    detections.append({
                        "track_id": track["track_id"],
                        "bbox": track["bbox"],
                        "class_name": track["class_name"],
                        "threat_score": track["threat_score"],
                        "threat_level": track["threat_level"],
                    })
                
                # Send result
                await websocket.send_json({
                    "type": "result",
                    "frame_idx": result["frame_idx"],
                    "detections": detections,
                    "fps": result["fps"],
                    "latency_ms": result["latency_ms"],
                })
                
            except Exception as e:
                logger.error(f"WebSocket processing error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))


# ── Session History Endpoints ─────────────────────────────────────────────


@app.get("/sessions")
async def get_sessions(api_key: str = Security(verify_api_key)):
    """Get all processing sessions.

    Returns a list of all sessions with summary statistics.
    Requires X-API-Key header.
    """
    try:
        sessions = get_all_sessions()
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Failed to retrieve sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/detections")
async def get_session_detections_endpoint(
    session_id: str,
    api_key: str = Security(verify_api_key),
):
    """Get all detections for a specific session.

    Args:
        session_id: The unique session identifier.

    Returns:
        List of all detections for the session.
    """
    try:
        detections = get_session_detections(session_id)
        return {"session_id": session_id, "detections": detections}
    except Exception as e:
        logger.error(f"Failed to retrieve detections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/events")
async def get_session_events_endpoint(
    session_id: str,
    api_key: str = Security(verify_api_key),
):
    """Get all events for a specific session.

    Args:
        session_id: The unique session identifier.

    Returns:
        List of all events (swarm, zone violations, HIGH threats) for the session.
    """
    try:
        events = get_session_events(session_id)
        return {"session_id": session_id, "events": events}
    except Exception as e:
        logger.error(f"Failed to retrieve events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Import for running standalone
if __name__ == "__main__":
    import uvicorn
    
    cfg = load_config()
    uvicorn.run(
        "api.main:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=False,
    )
