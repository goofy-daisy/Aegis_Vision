"""Intercept prediction module for AegisVision.

Projects tracked object trajectories forward in time using linear
kinematic extrapolation. Predicts zone entry and time-to-intercept.
"""

import math
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


def predict_intercept(
    track: Dict,
    track_history: List[tuple],
    protected_zones: List[Dict],
    scale_factor: float,
    map_center_lat: float,
    map_center_lon: float,
    horizon_seconds: int = 30,
    min_history_frames: int = 10,
    fps: float = 10.0,
) -> Dict:
    """Predict whether a tracked object will enter a protected zone.

    Returns:
        Dict with intercept_predicted, time_to_intercept_s,
        intercept_zone, projected_lat, projected_lon, confidence.
    """
    result = {
        "intercept_predicted": False,
        "time_to_intercept_s": None,
        "intercept_zone": None,
        "projected_lat": track.get("lat", 0.0),
        "projected_lon": track.get("lon", 0.0),
        "confidence": 0.0,
    }

    if not track_history or len(track_history) < min_history_frames:
        return result

    history = track_history[-min_history_frames:]
    positions = [(h[1], h[2]) for h in history]

    dx_total = positions[-1][0] - positions[0][0]
    dy_total = positions[-1][1] - positions[0][1]
    n_frames = len(positions) - 1

    if n_frames == 0:
        return result

    vx = dx_total / n_frames
    vy = dy_total / n_frames

    path_length = sum(
        math.sqrt((positions[i][0] - positions[i-1][0])**2 +
                  (positions[i][1] - positions[i-1][1])**2)
        for i in range(1, len(positions))
    )
    net_displacement = math.sqrt(dx_total**2 + dy_total**2)
    straightness = net_displacement / (path_length + 1e-6)
    result["confidence"] = float(min(1.0, straightness))

    if straightness < 0.3:
        return result

    curr_x = positions[-1][0]
    curr_y = positions[-1][1]

    lat_rad = math.radians(map_center_lat)
    meters_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad)
    meters_per_deg_lon = 111412.84 * math.cos(lat_rad)

    for t in range(1, horizon_seconds + 1):
        proj_x = curr_x + vx * fps * t
        proj_y = curr_y + vy * fps * t

        offset_x = proj_x - 320
        offset_y = proj_y - 320
        meter_x = offset_x * scale_factor
        meter_y = offset_y * scale_factor

        proj_lat = map_center_lat + (meter_y / meters_per_deg_lat)
        proj_lon = map_center_lon + (meter_x / meters_per_deg_lon)

        for zone in protected_zones:
            R = 6371000
            lat1 = math.radians(proj_lat)
            lat2 = math.radians(zone["center_lat"])
            dlat = math.radians(zone["center_lat"] - proj_lat)
            dlon = math.radians(zone["center_lon"] - proj_lon)
            a = (math.sin(dlat/2)**2 +
                 math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = R * c

            if distance <= zone["radius_m"]:
                result["intercept_predicted"] = True
                result["time_to_intercept_s"] = float(t)
                result["intercept_zone"] = zone["name"]
                result["projected_lat"] = float(proj_lat)
                result["projected_lon"] = float(proj_lon)
                return result

    proj_x_f = curr_x + vx * fps * horizon_seconds
    proj_y_f = curr_y + vy * fps * horizon_seconds
    result["projected_lat"] = float(
        map_center_lat + ((proj_y_f - 320) * scale_factor / meters_per_deg_lat)
    )
    result["projected_lon"] = float(
        map_center_lon + ((proj_x_f - 320) * scale_factor / meters_per_deg_lon)
    )
    return result
