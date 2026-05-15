"""Threat model package for AegisVision.

Provides threat scoring capabilities for tracked objects.
"""

from models.threat.threat_model import ThreatScorer
from models.threat.threat_train import load_threat_model, train_threat_model

__all__ = [
    "ThreatScorer",
    "train_threat_model",
    "load_threat_model",
]
