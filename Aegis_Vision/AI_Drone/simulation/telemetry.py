"""Telemetry simulation module for AegisVision.

Provides synthetic telemetry data generation for tracked objects including
speed, direction, proximity, and mock GPS coordinates. Maintains per-track
history to compute realistic kinematic values.
"""

import math
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from utils.config_loader import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class TelemetrySimulator:
    """Synthetic telemetry generator for tracked objects.
    
    Maintains per-track position history to compute speed, direction,
    and proximity values. Generates mock GPS coordinates based on
    configured map center.
    
    Attributes:
        track_history: Dict mapping track_id to deque of (frame_idx, center_x, center_y).
        scale_factor: Pixel-to-meter conversion factor (assumed).
        cfg: Configuration object with telemetry parameters.
    """
    
    def __init__(self, cfg: Config):
        """Initialize telemetry simulator with configuration.
        
        Args:
            cfg: Configuration object with telemetry and map parameters.
        """
        self.cfg = cfg
        # Store up to 30 frames of history per track for velocity calculation
        self.track_history: Dict[int, deque] = {}
        
        # Scale factor: assume 100m scene width at typical drone altitude
        # Use configured input_size so this is consistent with pipeline
        self.scale_factor = 100.0 / cfg.model.input_size  # meters per pixel
        
        # Map center for mock GPS generation
        self.map_center_lat = cfg.dashboard.map_center_lat
        self.map_center_lon = cfg.dashboard.map_center_lon
        
        # Earth radius in meters (for coordinate offset calculation)
        self.earth_radius = 6371000
        
        logger.info(f"TelemetrySimulator initialized: scale_factor={self.scale_factor:.4f}m/px, "
                    f"map_center=({self.map_center_lat}, {self.map_center_lon})")
    
    def update(self, tracks: List[Dict], frame_idx: int) -> List[Dict]:
        """Update telemetry for all tracked objects.
        
        Computes speed, direction, proximity, and GPS coordinates based on
        current position and historical data.
        
        Args:
            tracks: List of track dicts with keys:
                - track_id: int, unique track identifier
                - bbox: [x1, y1, x2, y2] bounding box in pixels
            frame_idx: Current frame index for temporal reference.
            
        Returns:
            List of track dicts with added telemetry fields:
                - speed_mps: float, speed in meters per second
                - direction_deg: float, movement direction (0-360)
                - proximity_m: float, estimated proximity in meters
                - lat: float, mock latitude
                - lon: float, mock longitude
        """
        updated_tracks = []
        
        for track in tracks:
            track_id = track["track_id"]
            bbox = track["bbox"]
            
            # Compute bbox center
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Initialize history for new tracks
            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=30)
            
            # Store position history
            self.track_history[track_id].append((frame_idx, center_x, center_y))
            
            # Compute speed and direction from history
            speed_mps = 0.0
            direction_deg = 0.0
            
            history = self.track_history[track_id]
            if len(history) >= 2:
                # Get current and previous positions
                curr_frame, curr_x, curr_y = history[-1]
                prev_frame, prev_x, prev_y = history[0]  # Use oldest in window for stability
                
                # Calculate pixel displacement
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                pixel_distance = math.sqrt(dx * dx + dy * dy)
                
                # Convert to meters
                meter_distance = pixel_distance * self.scale_factor
                
                # Calculate time delta (assume 30 FPS if no telemetry_update_hz)
                fps = self.cfg.simulation.telemetry_update_hz
                time_delta = (curr_frame - prev_frame) / fps
                
                if time_delta > 0:
                    speed_mps = meter_distance / time_delta
                    # Cap at maximum speed
                    speed_mps = min(speed_mps, self.cfg.threat.speed_max_mps)
                
                # Calculate direction (0-360 degrees, 0 = North/Up, 90 = East/Right)
                if dx != 0 or dy != 0:
                    direction_rad = math.atan2(dx, -dy)  # Negative dy because Y increases downward
                    direction_deg = math.degrees(direction_rad)
                    direction_deg = (direction_deg + 360) % 360
            
            # Compute proximity from bbox area
            # Larger bbox area = closer object (inverse relationship)
            bbox_area = (x2 - x1) * (y2 - y1)
            # Use configured input size instead of hardcoded 640
            input_size = self.cfg.model.input_size
            image_area = input_size * input_size
            
            # Normalize area to 0-1 range and invert
            normalized_area = min(bbox_area / (image_area * 0.5), 1.0)  # Cap at 50% of image
            proximity_m = self.cfg.threat.proximity_max_meters * (1.0 - normalized_area)
            
            # Generate mock GPS coordinates
            # Convert pixel offset to lat/lon (simplified flat-earth approximation)
            # Offset from image center (using configured input size)
            offset_x = center_x - (input_size / 2)  # pixels from center
            offset_y = center_y - (input_size / 2)  # pixels from center
            
            # Convert to meters
            meter_x = offset_x * self.scale_factor
            meter_y = offset_y * self.scale_factor
            
            # Convert to degrees (approximate at given latitude)
            lat_rad = math.radians(self.map_center_lat)
            meters_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad)
            meters_per_deg_lon = 111412.84 * math.cos(lat_rad)
            
            lat = self.map_center_lat + (meter_y / meters_per_deg_lat)
            lon = self.map_center_lon + (meter_x / meters_per_deg_lon)
            
            # Add telemetry to track dict
            track_with_telemetry = dict(track)  # Copy to avoid modifying original
            track_with_telemetry.update({
                "speed_mps": float(speed_mps),
                "direction_deg": float(direction_deg),
                "proximity_m": float(proximity_m),
                "lat": float(lat),
                "lon": float(lon),
            })
            updated_tracks.append(track_with_telemetry)
        
        # Clean up old track histories
        active_ids = {t["track_id"] for t in tracks}
        stale_ids = set(self.track_history.keys()) - active_ids
        for stale_id in stale_ids:
            del self.track_history[stale_id]
        
        logger.debug(f"Telemetry updated for {len(updated_tracks)} tracks")
        return updated_tracks
    
    def reset(self) -> None:
        """Reset all track history. Use when switching videos."""
        self.track_history.clear()
        logger.info("Telemetry simulator reset")
    
    def get_track_history(self, track_id: int) -> Optional[List[tuple]]:
        """Get position history for a specific track.
        
        Args:
            track_id: Track identifier.
            
        Returns:
            List of (frame_idx, center_x, center_y) tuples or None if track not found.
        """
        if track_id not in self.track_history:
            return None
        return list(self.track_history[track_id])

    def check_zone_violation(
        self,
        lat: float,
        lon: float,
        zones: list
    ) -> bool:
        """Check if a GPS coordinate is inside any protected zone.

        Args:
            lat: Latitude of object.
            lon: Longitude of object.
            zones: List of zone dicts with keys: name, center_lat,
                   center_lon, radius_m.

        Returns:
            True if inside any zone, False otherwise.
        """
        for zone in zones:
            # Haversine distance in meters
            R = 6371000
            lat1 = math.radians(lat)
            lat2 = math.radians(zone["center_lat"])
            dlat = math.radians(zone["center_lat"] - lat)
            dlon = math.radians(zone["center_lon"] - lon)

            a = (math.sin(dlat / 2) ** 2 +
                 math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = R * c

            if distance <= zone["radius_m"]:
                return True
        return False

    def detect_swarm(self, tracks: list, threshold: int = 3) -> bool:
        """Detect if multiple objects are converging — swarm behaviour.

        Args:
            tracks: List of track dicts with bbox field.
            threshold: Minimum objects to count as swarm.

        Returns:
            True if swarm detected, False otherwise.
        """
        if len(tracks) < threshold:
            return False

        # Get all centers
        centers = []
        for track in tracks:
            bbox = track.get("bbox", [0, 0, 0, 0])
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            centers.append((cx, cy))

        # Check if threshold objects are within 200px of each other
        cluster_count = 0
        for i, c1 in enumerate(centers):
            nearby = 0
            for j, c2 in enumerate(centers):
                if i == j:
                    continue
                dist = math.sqrt(
                    (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2
                )
                if dist < 200:
                    nearby += 1
            if nearby >= threshold - 1:
                cluster_count += 1

        return cluster_count >= threshold
