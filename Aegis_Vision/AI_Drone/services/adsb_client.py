"""ADS-B client for AegisVision using OpenSky Network.

Free, no API key required. Legal for research use.
See: https://opensky-network.org/apidoc/rest.html
"""

import math
import threading
import time
from typing import Dict, List

import requests

from utils.logger import get_logger

logger = get_logger(__name__)

_cache: Dict = {
    "aircraft": [],
    "last_update": 0.0,
    "status": "OFFLINE",
}
_lock = threading.Lock()


def query_opensky(
    lat_min: float, lon_min: float,
    lat_max: float, lon_max: float,
    timeout_seconds: float = 3.0,
) -> List[Dict]:
    """Query OpenSky Network for aircraft in a bounding box."""
    url = "https://opensky-network.org/api/states/all"
    params = {
        "lamin": lat_min, "lomin": lon_min,
        "lamax": lat_max, "lomax": lon_max,
    }
    try:
        response = requests.get(url, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.json()
        aircraft = []
        for state in (data.get("states", []) or []):
            if state is None or len(state) < 9:
                continue
            if state[5] is None or state[6] is None:
                continue
            aircraft.append({
                "icao24": str(state[0]) if state[0] else "unknown",
                "callsign": str(state[1]).strip() if state[1] else "unknown",
                "lat": float(state[6]),
                "lon": float(state[5]),
                "altitude_m": float(state[7]) if state[7] else 0.0,
                "velocity_mps": float(state[9]) if state[9] else 0.0,
                "heading_deg": float(state[10]) if state[10] else 0.0,
                "on_ground": bool(state[8]) if state[8] is not None else False,
            })
        logger.debug(f"OpenSky returned {len(aircraft)} aircraft")
        return aircraft
    except requests.exceptions.Timeout:
        logger.warning("OpenSky query timed out")
        return []
    except requests.exceptions.ConnectionError:
        logger.warning("OpenSky connection failed")
        return []
    except Exception as e:
        logger.warning(f"OpenSky query failed: {e}")
        return []


def update_cache(
    center_lat: float,
    center_lon: float,
    bbox_degrees: float = 0.1,
    query_interval_seconds: float = 5.0,
) -> None:
    """Update aircraft cache if interval has elapsed. Called from background thread."""
    global _cache
    with _lock:
        elapsed = time.time() - _cache["last_update"]
        if elapsed < query_interval_seconds:
            return

    aircraft = query_opensky(
        lat_min=center_lat - bbox_degrees,
        lon_min=center_lon - bbox_degrees,
        lat_max=center_lat + bbox_degrees,
        lon_max=center_lon + bbox_degrees,
    )

    with _lock:
        _cache["aircraft"] = aircraft
        _cache["last_update"] = time.time()
        _cache["status"] = "LIVE" if aircraft else "OFFLINE"

    logger.debug(
        f"ADS-B cache updated: {len(aircraft)} aircraft, "
        f"status={_cache['status']}"
    )


def get_cached_aircraft() -> List[Dict]:
    """Get cached aircraft list thread-safely."""
    with _lock:
        return list(_cache["aircraft"])


def get_status() -> str:
    """Get current ADS-B status string."""
    with _lock:
        return _cache["status"]


def cross_reference(
    track_lat: float,
    track_lon: float,
    match_radius_meters: float = 500.0,
) -> Dict:
    """Cross-reference a detected object against ADS-B data.

    Returns dict with status ('MATCHED', 'UNREGISTERED', 'NO_DATA'),
    callsign, icao24, distance_m.
    """
    aircraft = get_cached_aircraft()

    if not aircraft:
        return {
            "status": "NO_DATA",
            "callsign": None,
            "icao24": None,
            "distance_m": None,
        }

    best_match = None
    best_distance = float("inf")

    for ac in aircraft:
        if ac.get("on_ground", False):
            continue
        ac_lat = ac.get("lat", 0.0)
        ac_lon = ac.get("lon", 0.0)
        R = 6371000
        lat1 = math.radians(track_lat)
        lat2 = math.radians(ac_lat)
        dlat = math.radians(ac_lat - track_lat)
        dlon = math.radians(ac_lon - track_lon)
        a = (math.sin(dlat/2)**2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        if distance < best_distance:
            best_distance = distance
            best_match = ac

    if best_match and best_distance <= match_radius_meters:
        return {
            "status": "MATCHED",
            "callsign": best_match.get("callsign", "unknown"),
            "icao24": best_match.get("icao24", "unknown"),
            "distance_m": round(best_distance, 1),
        }

    return {
        "status": "UNREGISTERED",
        "callsign": None,
        "icao24": None,
        "distance_m": None,
    }
