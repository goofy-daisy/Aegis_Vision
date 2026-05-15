"""Threat signature matcher for AegisVision.

Matches detected tracks against known threat signatures defined
in configs/signatures.yaml.
"""

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from utils.logger import get_logger

logger = get_logger(__name__)

_signatures: Optional[List[Dict]] = None


def _load_signatures() -> List[Dict]:
    global _signatures
    if _signatures is not None:
        return _signatures
    sig_path = (
        Path(__file__).resolve().parent.parent.parent
        / "configs" / "signatures.yaml"
    )
    if not sig_path.exists():
        logger.warning(f"Signatures file not found: {sig_path}")
        _signatures = []
        return _signatures
    try:
        with open(sig_path, "r") as f:
            data = yaml.safe_load(f)
        _signatures = data.get("signatures", [])
        logger.info(f"Loaded {len(_signatures)} threat signatures")
    except Exception as e:
        logger.error(f"Failed to load signatures: {e}")
        _signatures = []
    return _signatures


def match_signature(track: Dict) -> Dict:
    """Match a track against the signature library.

    Returns dict with signature_name and signature_confidence.
    """
    signatures = _load_signatures()
    if not signatures:
        return {"signature_name": "Unknown", "signature_confidence": 0.0}

    class_name = track.get("class_name", "")
    bbox = track.get("bbox", [0, 0, 100, 100])
    speed_mps = track.get("speed_mps", 0.0)
    behaviour = track.get("behaviour", "STATIONARY")

    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)

    best_name = "Unknown"
    best_score = 0.0

    for sig in signatures:
        score = 0.0
        checks = 0

        sig_classes = sig.get("classes", [])
        if sig_classes:
            checks += 1
            if class_name in sig_classes:
                score += 1.0

        area_min = sig.get("bbox_area_min", 0)
        area_max = sig.get("bbox_area_max", float("inf"))
        checks += 1
        if area_min <= area <= area_max:
            score += 1.0
        else:
            ratio = (area / (area_min + 1e-6) if area < area_min
                     else area_max / (area + 1e-6))
            score += max(0.0, ratio * 0.5)

        speed_min = sig.get("speed_min_mps", 0)
        speed_max = sig.get("speed_max_mps", float("inf"))
        checks += 1
        score += 1.0 if speed_min <= speed_mps <= speed_max else 0.2

        sig_behaviours = sig.get("behaviour_patterns", [])
        if sig_behaviours:
            checks += 1
            if behaviour in sig_behaviours:
                score += 1.0

        if checks > 0:
            normalised = (score / checks) * sig.get("confidence_weight", 1.0)
            if normalised > best_score:
                best_score = normalised
                best_name = sig["name"]

    return {
        "signature_name": best_name,
        "signature_confidence": round(float(best_score), 3),
    }
