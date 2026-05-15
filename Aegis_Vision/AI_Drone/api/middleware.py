"""API middleware for AegisVision.

Provides CORS, request timing, and global exception handling middleware
for the FastAPI application.
"""

import time
from datetime import datetime
from typing import Callable

from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from utils.logger import get_logger

logger = get_logger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware to log request timing information."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log timing.
        
        Args:
            request: Incoming request.
            call_next: Next middleware/handler in chain.
            
        Returns:
            Response from next handler.
        """
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log request details
        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} - {duration_ms:.2f}ms"
        )
        
        # Add timing header
        response.headers["X-Request-Duration-Ms"] = str(int(duration_ms))
        
        return response


def setup_cors_middleware(app):
    """Configure CORS middleware for the application.
    
    Allows all origins for development. Restrict in production.
    
    Args:
        app: FastAPI application instance.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins in development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS middleware configured")


async def global_exception_handler(request: Request, exc: Exception) -> Response:
    """Global exception handler for unhandled exceptions.
    
    Returns structured JSON error response.
    
    Args:
        request: Request that caused exception.
        exc: The unhandled exception.
        
    Returns:
        JSON error response.
    """
    logger.error(f"Unhandled exception in {request.url.path}: {exc}")
    
    from fastapi.responses import JSONResponse
    
    error_response = {
        "error": type(exc).__name__,
        "detail": str(exc),
        "timestamp": datetime.now().isoformat(),
    }
    
    return JSONResponse(
        status_code=500,
        content=error_response,
    )


async def validation_exception_handler(request: Request, exc) -> Response:
    """Handle Pydantic validation errors.
    
    Args:
        request: Request that caused validation error.
        exc: Validation exception.
        
    Returns:
        JSON error response with validation details.
    """
    from fastapi.responses import JSONResponse
    from fastapi.exceptions import RequestValidationError
    
    logger.warning(f"Validation error in {request.url.path}: {exc}")
    
    if isinstance(exc, RequestValidationError):
        errors = []
        for error in exc.errors():
            errors.append({
                "field": error.get("loc", []),
                "message": error.get("msg", ""),
                "type": error.get("type", ""),
            })
        
        return JSONResponse(
            status_code=422,
            content={
                "error": "ValidationError",
                "detail": errors,
                "timestamp": datetime.now().isoformat(),
            },
        )
    
    # Generic error fallback
    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
        },
    )
