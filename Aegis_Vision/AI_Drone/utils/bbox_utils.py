"""Bounding box utility functions for AegisVision.

Provides shared geometric operations for bounding boxes including
format conversions, clipping, and IoU calculations. All functions
operate on list-based bbox representations for consistency across
the codebase.
"""

from typing import List, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


def obb_to_aabb(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Convert oriented bounding box (OBB) to axis-aligned bounding box (AABB).
    
    OBB format: 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    AABB format: (xmin, ymin, xmax, ymax)
    
    Args:
        points: List of 4 (x, y) tuples representing OBB corners.
        
    Returns:
        Tuple of (xmin, ymin, xmax, ymax) as AABB coordinates.
        
    Raises:
        ValueError: If points does not contain exactly 4 coordinates.
    """
    # Validate input has exactly 4 points
    if len(points) != 4:
        raise ValueError(f"OBB must have exactly 4 points, got {len(points)}")
    
    # Extract x and y coordinates
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    # Compute min/max for axis-aligned bounds
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    return (xmin, ymin, xmax, ymax)


def xywh_to_xyxy(bbox: List[float]) -> List[float]:
    """Convert bounding box from xywh format to xyxy format.
    
    xywh format: [x_center, y_center, width, height]
    xyxy format: [x1, y1, x2, y2] (top-left and bottom-right corners)
    
    Args:
        bbox: List of 4 floats [cx, cy, w, h] in any coordinate space.
        
    Returns:
        List of 4 floats [x1, y1, x2, y2] in same coordinate space.
        
    Raises:
        ValueError: If bbox does not have exactly 4 elements.
    """
    if len(bbox) != 4:
        raise ValueError(f"BBox must have exactly 4 elements, got {len(bbox)}")
    
    cx, cy, w, h = bbox
    
    # Calculate corner coordinates
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    
    return [x1, y1, x2, y2]


def xyxy_to_xywh(bbox: List[float]) -> List[float]:
    """Convert bounding box from xyxy format to xywh format.
    
    xyxy format: [x1, y1, x2, y2] (top-left and bottom-right corners)
    xywh format: [x_center, y_center, width, height]
    
    Args:
        bbox: List of 4 floats [x1, y1, x2, y2] in any coordinate space.
        
    Returns:
        List of 4 floats [cx, cy, w, h] in same coordinate space.
        
    Raises:
        ValueError: If bbox does not have exactly 4 elements.
    """
    if len(bbox) != 4:
        raise ValueError(f"BBox must have exactly 4 elements, got {len(bbox)}")
    
    x1, y1, x2, y2 = bbox
    
    # Calculate center and dimensions
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    
    return [cx, cy, w, h]


def clip_bbox(bbox: List[float], img_w: int, img_h: int, min_area: int = 16) -> List[float]:
    """Clip bounding box to image boundaries and validate minimum area.
    
    Args:
        bbox: List of 4 floats [x1, y1, x2, y2] in pixel coordinates.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        min_area: Minimum acceptable box area in pixels squared (default 16).
        
    Returns:
        Clipped bbox [x1, y1, x2, y2], or empty list if area < min_area.
        
    Raises:
        ValueError: If bbox does not have exactly 4 elements.
    """
    if len(bbox) != 4:
        raise ValueError(f"BBox must have exactly 4 elements, got {len(bbox)}")
    
    x1, y1, x2, y2 = bbox
    
    # Clip to image boundaries
    x1 = max(0.0, min(float(x1), float(img_w)))
    y1 = max(0.0, min(float(y1), float(img_h)))
    x2 = max(0.0, min(float(x2), float(img_w)))
    y2 = max(0.0, min(float(y2), float(img_h)))
    
    # Check area after clipping
    area = (x2 - x1) * (y2 - y1)
    if area < min_area:
        logger.debug(f"Box rejected: area {area} < minimum {min_area}")
        return []
    
    return [x1, y1, x2, y2]


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.
    
    Both boxes must be in xyxy format [x1, y1, x2, y2].
    
    Args:
        box1: First bounding box [x1, y1, x2, y2].
        box2: Second bounding box [x1, y1, x2, y2].
        
    Returns:
        IoU value in range [0.0, 1.0]. Returns 0.0 if no overlap.
        
    Raises:
        ValueError: If either box does not have exactly 4 elements.
    """
    # Validate inputs
    if len(box1) != 4:
        raise ValueError(f"Box1 must have exactly 4 elements, got {len(box1)}")
    if len(box2) != 4:
        raise ValueError(f"Box2 must have exactly 4 elements, got {len(box2)}")
    
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Compute intersection coordinates
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    # Compute intersection area
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # If no intersection, return 0 immediately
    if inter_area == 0:
        return 0.0
    
    # Compute box areas
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Compute union area
    union_area = box1_area + box2_area - inter_area
    
    # Handle degenerate case
    if union_area == 0:
        return 0.0
    
    # Calculate IoU
    iou = inter_area / union_area
    
    # Clamp to valid range (handle floating point errors)
    return max(0.0, min(1.0, iou))
