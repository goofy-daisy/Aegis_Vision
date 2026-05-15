"""DeepSORT tracking module for AegisVision.

Provides multi-object tracking capabilities using the deep_sort_realtime
library with appearance-based Re-ID for consistent tracking across frames.
"""

from typing import Dict, List, Optional

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from utils.config_loader import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class DeepSortTracker:
    """DeepSORT tracker wrapper for consistent multi-object tracking.
    
    Maintains track identities across frames using motion and appearance cues.
    Configured via the master config for tracking parameters.
    
    Attributes:
        tracker: Underlying DeepSort tracker instance.
        max_tracks: Maximum number of tracks to maintain.
    """
    
    def __init__(self, cfg: Config):
        """Initialize DeepSORT tracker with configuration.
        
        Args:
            cfg: Configuration object with tracking parameters.
        """
        # Store configuration
        self.cfg = cfg
        self.max_tracks = cfg.tracking.max_tracks
        
        # Determine embedder GPU setting based on device
        embedder_gpu = cfg.model.device == "cuda"
        if cfg.model.device == "auto":
            # Check if CUDA is available
            try:
                import torch
                embedder_gpu = torch.cuda.is_available()
            except ImportError:
                embedder_gpu = False
        
        # Store embedder_gpu for use in reset()
        self.embedder_gpu = embedder_gpu
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(
            max_age=cfg.tracking.max_age,
            n_init=cfg.tracking.min_hits,
            max_iou_distance=cfg.tracking.iou_threshold,
            embedder="mobilenet",  # Lightweight Re-ID model
            half=False,            # Use full precision for CPU compatibility
            embedder_gpu=embedder_gpu,
        )
        
        logger.info(f"DeepSORT initialized: max_age={cfg.tracking.max_age}, "
                    f"min_hits={cfg.tracking.min_hits}, embedder_gpu={embedder_gpu}")
    
    def update(
        self,
        detections: List[Dict],
        frame: np.ndarray,
    ) -> List[Dict]:
        """Update tracker with new detections and return confirmed tracks.
        
        Converts YOLO detections to DeepSORT format, runs tracking update,
        and returns only confirmed tracks with consistent IDs.
        
        Args:
            detections: List of YOLO detection dicts with keys:
                - bbox: [x1, y1, x2, y2] absolute pixel coords
                - bbox_xywh: [cx, cy, w, h] center format
                - class_name: str, object class name
                - confidence: float, detection confidence
            frame: Raw BGR frame numpy array for appearance features.
            
        Returns:
            List of track dicts with keys:
                - track_id: int, consistent track identifier
                - bbox: [x1, y1, x2, y2] current bounding box
                - class_name: str, object class
                - confidence: float, detection confidence
                
            Returns empty list if no confirmed tracks.
            
        Raises:
            ValueError: If frame format is invalid.
        """
        # Validate input frame
        if frame is None:
            raise ValueError("Input frame is None")
        
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Frame must be numpy array, got {type(frame)}")
        
        # Convert YOLO detections to DeepSORT format
        # DeepSORT expects: ([left, top, w, h], confidence, class_name)
        deepsort_detections = []
        
        for det in detections:
            # Extract xywh format (left, top, width, height)
            cx, cy, w, h = det["bbox_xywh"]
            left = cx - w / 2
            top = cy - h / 2
            
            # Create DeepSORT detection tuple
            deepsort_det = ([left, top, w, h], det["confidence"], det["class_name"])
            deepsort_detections.append(deepsort_det)
        
        # Update tracker (must be called even with empty detections to age tracks)
        try:
            tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
        except Exception as e:
            logger.error(f"DeepSORT update failed: {e}")
            raise RuntimeError(f"Tracking update failed: {e}")
        
        # Filter to confirmed tracks and format output
        confirmed_tracks = []
        track_count = 0
        
        for track in tracks:
            # Only return confirmed tracks (have sufficient hits)
            if not track.is_confirmed():
                continue
            
            # Respect max_tracks limit
            track_count += 1
            if track_count > self.max_tracks:
                break
            
            # Get track bounding box
            left, top, w, h = track.to_ltwh()
            x1, y1, x2, y2 = left, top, left + w, top + h
            
            # Get track details
            track_id = track.track_id
            class_name = track.det_class if track.det_class else "unknown"
            
            # Get confidence (may be None for tracks without recent detection)
            confidence = getattr(track, "score", 0.0)
            if confidence is None:
                confidence = 0.0
            
            track_dict = {
                "track_id": int(track_id),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "class_name": str(class_name),
                "confidence": float(confidence),
            }
            confirmed_tracks.append(track_dict)
        
        logger.debug(f"Tracking: {len(detections)} detections → {len(confirmed_tracks)} confirmed tracks")
        
        return confirmed_tracks
    
    def reset(self) -> None:
        """Reset tracker state, clearing all existing tracks.
        
        Use this when switching to a new video or scene.
        """
        self.tracker = DeepSort(
            max_age=self.cfg.tracking.max_age,
            n_init=self.cfg.tracking.min_hits,
            max_iou_distance=self.cfg.tracking.iou_threshold,
            embedder="mobilenet",
            half=False,
            embedder_gpu=self.embedder_gpu,  # Use stored value
        )
        logger.info("DeepSORT tracker reset")
    
    def get_tracker_stats(self) -> Dict:
        """Get statistics about the current tracker state.
        
        Returns:
            Dict containing:
                - n_tracks: Number of active tracks
                - n_confirmed: Number of confirmed tracks
                - n_tentative: Number of tentative (unconfirmed) tracks
                - n_deleted: Number of recently deleted tracks
        """
        stats = {
            "n_tracks": 0,
            "n_confirmed": 0,
            "n_tentative": 0,
            "n_deleted": 0,
        }
        
        # Access internal tracker state if available
        if hasattr(self.tracker, "tracker") and hasattr(self.tracker.tracker, "tracks"):
            tracks = self.tracker.tracker.tracks
            stats["n_tracks"] = len(tracks)
            stats["n_confirmed"] = sum(1 for t in tracks if t.is_confirmed())
            stats["n_tentative"] = sum(1 for t in tracks if t.is_tentative())
            stats["n_deleted"] = sum(1 for t in tracks if t.is_deleted())
        
        return stats
