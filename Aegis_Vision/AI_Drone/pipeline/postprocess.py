"""Frame postprocessing module for AegisVision.

Provides visualization utilities for drawing detections, tracks, threat
scores, and system metrics overlays on video frames.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


# Color constants in BGR format (OpenCV default)
THREAT_COLORS = {
    "HIGH": (0, 0, 255),       # Red
    "MEDIUM": (0, 165, 255),   # Orange
    "LOW": (0, 255, 0),        # Green
}

# Text rendering constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
LINE_TYPE = cv2.LINE_AA


def draw_detections(
    frame: np.ndarray,
    tracks_with_threats: List[Dict],
) -> np.ndarray:
    """Draw bounding boxes, IDs, and threat information for all tracks.
    
    Draws color-coded boxes based on threat level with anti-aliased text.
    
    Args:
        frame: Input BGR image to draw on.
        tracks_with_threats: List of track dicts with threat information.
            Each dict must contain:
            - track_id: int
            - bbox: [x1, y1, x2, y2]
            - class_name: str
            - confidence: float
            - threat_level: "HIGH", "MEDIUM", or "LOW"
            - threat_score: float
            
    Returns:
        Annotated BGR image.
        
    Raises:
        ValueError: If frame format is invalid.
    """
    # Validate input
    if frame is None:
        raise ValueError("Input frame is None")
    
    if not isinstance(frame, np.ndarray):
        raise ValueError(f"Frame must be numpy array, got {type(frame)}")
    
    # Make a copy to avoid modifying original
    annotated = frame.copy()
    
    for track in tracks_with_threats:
        # Extract track information
        track_id = track.get("track_id", -1)
        bbox = track.get("bbox", [0, 0, 0, 0])
        class_name = track.get("class_name", "unknown")
        confidence = track.get("confidence", 0.0)
        threat_level = track.get("threat_level", "LOW")
        threat_score = track.get("threat_score", 0.0)
        
        # Ensure bbox values are integers
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Get color based on threat level
        color = THREAT_COLORS.get(threat_level, THREAT_COLORS["LOW"])
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, LINE_TYPE)
        
        # Prepare text labels
        # Line 1: ID and class
        label1 = f"ID:{track_id} {class_name}"
        
        # Line 2: threat score and confidence
        label2 = f"{threat_level} {threat_score:.2f} ({confidence:.2f})"
        
        # Calculate text size for background rectangles
        (tw1, th1), _ = cv2.getTextSize(label1, FONT, FONT_SCALE, FONT_THICKNESS)
        (tw2, th2), _ = cv2.getTextSize(label2, FONT, FONT_SCALE, FONT_THICKNESS)
        
        # Draw background rectangles for text readability
        padding = 2
        # Background for line 1
        cv2.rectangle(
            annotated,
            (x1, y1 - th1 - 2 * padding),
            (x1 + tw1 + 2 * padding, y1),
            color,
            -1,  # Filled
        )
        
        # Background for line 2 (positioned below line 1)
        cv2.rectangle(
            annotated,
            (x1, y1 - th1 - th2 - 4 * padding),
            (x1 + tw2 + 2 * padding, y1 - th1 - 2 * padding),
            color,
            -1,  # Filled
        )
        
        # Draw text (white on colored background)
        text_color = (255, 255, 255)
        cv2.putText(
            annotated,
            label1,
            (x1 + padding, y1 - padding),
            FONT,
            FONT_SCALE,
            text_color,
            FONT_THICKNESS,
            LINE_TYPE,
        )
        
        cv2.putText(
            annotated,
            label2,
            (x1 + padding, y1 - th1 - 2 * padding - padding),
            FONT,
            FONT_SCALE,
            text_color,
            FONT_THICKNESS,
            LINE_TYPE,
        )
    
    return annotated


def draw_metrics_overlay(
    frame: np.ndarray,
    fps: float,
    latency_ms: float,
    object_count: int,
) -> np.ndarray:
    """Draw system metrics overlay in top-left corner.
    
    Displays FPS, latency, and object count as a HUD-style overlay.
    
    Args:
        frame: Input BGR image.
        fps: Current frames per second.
        latency_ms: Processing latency in milliseconds.
        object_count: Number of detected/tracked objects.
        
    Returns:
        Frame with metrics overlay drawn.
        
    Raises:
        ValueError: If frame format is invalid.
    """
    # Validate input
    if frame is None:
        raise ValueError("Input frame is None")
    
    annotated = frame.copy()
    
    # Format metrics text
    lines = [
        f"FPS: {fps:.1f}",
        f"Latency: {latency_ms:.1f}ms",
        f"Objects: {object_count}",
    ]
    
    # Position in top-left with padding
    x_offset = 10
    y_offset = 30
    line_spacing = 25
    
    # Calculate background size
    max_width = 0
    total_height = len(lines) * line_spacing
    
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, FONT, FONT_SCALE + 0.1, FONT_THICKNESS + 1)
        max_width = max(max_width, tw)
    
    # Draw semi-transparent background
    bg_padding = 10
    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (x_offset - bg_padding, y_offset - 20),
        (x_offset + max_width + bg_padding, y_offset + total_height - 10),
        (0, 0, 0),
        -1,
    )
    
    # Blend overlay for transparency
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
    
    # Draw text
    text_color = (255, 255, 255)
    for i, line in enumerate(lines):
        y = y_offset + i * line_spacing
        cv2.putText(
            annotated,
            line,
            (x_offset, y),
            FONT,
            FONT_SCALE + 0.1,
            text_color,
            FONT_THICKNESS + 1,
            LINE_TYPE,
        )
    
    return annotated


def draw_threat_legend(
    frame: np.ndarray,
    position: Tuple[int, int] = (10, 10),
) -> np.ndarray:
    """Draw a legend explaining threat level colors.
    
    Args:
        frame: Input BGR image.
        position: (x, y) position for legend (bottom-right of legend box).
        
    Returns:
        Frame with legend drawn.
    """
    annotated = frame.copy()
    
    x, y = position
    box_size = 20
    spacing = 25
    
    levels = ["HIGH", "MEDIUM", "LOW"]
    
    for i, level in enumerate(levels):
        color = THREAT_COLORS[level]
        box_y = y + i * spacing
        
        # Draw color box
        cv2.rectangle(
            annotated,
            (x, box_y),
            (x + box_size, box_y + box_size),
            color,
            -1,
        )
        
        # Draw label
        cv2.putText(
            annotated,
            level,
            (x + box_size + 5, box_y + box_size - 3),
            FONT,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS,
            LINE_TYPE,
        )
    
    return annotated
