"""Full processing pipeline for AegisVision.

Orchestrates the complete detection and tracking pipeline:
frame → preprocess → YOLO → DeepSORT → telemetry → threat → postprocess.

Provides both single-frame and full-video processing capabilities with
comprehensive error handling and performance metrics.
"""

import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from explainability.shap_explainer import SHAPExplainer
from models.threat import ThreatScorer
from models.tracking import DeepSortTracker
from models.yolo import load_model, run_inference
from pipeline.postprocess import draw_detections, draw_metrics_overlay
from pipeline.preprocess import prepare_for_inference
from simulation.telemetry import TelemetrySimulator
from utils.config_loader import Config, load_config
from utils.device_utils import get_device
from utils.logger import get_logger

logger = get_logger(__name__)

import threading

try:
    from services.adsb_client import (
        update_cache as adsb_update_cache,
        cross_reference as adsb_cross_reference,
        get_status as adsb_get_status,
    )
    _ADSB_AVAILABLE = True
except ImportError:
    _ADSB_AVAILABLE = False

try:
    from models.classifier.channel_classifier import classify_modality
    _CLASSIFIER_AVAILABLE = True
except ImportError:
    _CLASSIFIER_AVAILABLE = False
    def classify_modality(frame, confidence_threshold=0.7):
        return "RGB", 1.0

try:
    from models.prediction.intercept_predictor import predict_intercept
    _PREDICTION_AVAILABLE = True
except ImportError:
    _PREDICTION_AVAILABLE = False

try:
    from models.threat.signature_matcher import match_signature
    _SIGNATURE_AVAILABLE = True
except ImportError:
    _SIGNATURE_AVAILABLE = False
    def match_signature(track):
        return {"signature_name": "Unknown", "signature_confidence": 0.0}


class AegisPipeline:
    """Complete AegisVision processing pipeline.
    
    Integrates YOLO detection, DeepSORT tracking, telemetry simulation,
    threat scoring, and visualization into a unified processing pipeline.
    
    Attributes:
        cfg: Configuration object.
        yolo_model: Loaded YOLO model.
        tracker: DeepSORT tracker instance.
        telemetry: Telemetry simulator.
        threat_scorer: Threat scoring model.
        shap_explainer: SHAP explainability (lazy-loaded).
        frame_times: Deque for FPS calculation.
    """
    
    def __init__(self, cfg: Optional[Config] = None):
        """Initialize the complete processing pipeline.
        
        Args:
            cfg: Configuration object. Loads default if None.
        """
        # Load configuration
        self.cfg = cfg if cfg else load_config()
        
        # Initialize all pipeline components
        logger.info("Initializing AegisPipeline...")
        
        # Determine device
        device = get_device(self.cfg.model.device if self.cfg.model.device != "auto" else None)
        
        # Load YOLO model
        weights_path = self.cfg.paths.weights / "best.pt"
        if not weights_path.exists():
            logger.warning(f"Model not found at {weights_path}, using pretrained")
            weights_path = None  # Will use default pretrained
        
        self.yolo_model = load_model(weights_path or Path("yolov8n.pt"), device)
        
        # Initialize tracker
        self.tracker = DeepSortTracker(self.cfg)
        
        # Initialize telemetry simulator
        self.telemetry = TelemetrySimulator(self.cfg)
        
        # Initialize threat scorer
        self.threat_scorer = ThreatScorer(self.cfg)
        
        # SHAP explainer (lazy-loaded)
        self.shap_explainer: Optional[SHAPExplainer] = None
        
        # Frame timing for FPS calculation
        self.frame_times: deque = deque(maxlen=30)
        self._protected_zones = []
        self._swarm_detected = False

        # V3 state
        self._modality = "RGB"
        self._modality_confidence = 1.0
        self._adsb_thread = None

        # V3 — Load thermal model if weights exist and enabled
        thermal_cfg = getattr(self.cfg, 'thermal_model', None)
        thermal_weights_str = getattr(
            thermal_cfg, 'weights',
            'models/yolo/weights/best_thermal.pt'
        ) if thermal_cfg else 'models/yolo/weights/best_thermal.pt'
        thermal_enabled = getattr(
            thermal_cfg, 'enabled', False
        ) if thermal_cfg else False
        thermal_weights_path = Path(thermal_weights_str)
        if thermal_enabled and thermal_weights_path.exists():
            try:
                self.yolo_model_thermal = load_model(
                    thermal_weights_path, device
                )
                logger.info(
                    f"Thermal YOLO model loaded from {thermal_weights_path}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load thermal model: {e}. "
                    f"Will use RGB model for all frames."
                )
                self.yolo_model_thermal = None
        else:
            self.yolo_model_thermal = None
        
        logger.info("AegisPipeline initialized successfully")
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        explain: bool = False,
    ) -> Dict:
        """Process a single frame through the complete pipeline.
        
        Args:
            frame: Input BGR image.
            frame_idx: Frame index for timing reference.
            explain: Whether to generate SHAP explanations.
            
        Returns:
            Dict containing:
                - frame_idx: int
                - detections: List[dict] raw YOLO output
                - tracks: List[dict] tracked + scored objects
                - fps: float current FPS
                - latency_ms: float processing time
                - annotated_frame: np.ndarray visualization
                - explanations: Optional[List[dict]] SHAP explanations
        """
        start_time = time.time()
        
        try:
            # Validate input
            if frame is None:
                raise ValueError("Input frame is None")
            
            if not isinstance(frame, np.ndarray) or len(frame.shape) != 3:
                raise ValueError(f"Invalid frame format: {type(frame)}, shape {getattr(frame, 'shape', None)}")
            
            # V3.1 — Detect image modality
            modality, mod_confidence = classify_modality(frame)
            self._modality = modality
            self._modality_confidence = mod_confidence

            # V3.2 — Select correct YOLO model based on modality
            # Use thermal model if available and frame is THERMAL
            active_model = self.yolo_model
            if (modality == "THERMAL" and
                    self.yolo_model_thermal is not None):
                active_model = self.yolo_model_thermal
                logger.debug("Using thermal YOLO model for this frame")
            elif modality == "AMBIGUOUS" and self.yolo_model_thermal is not None:
                # Run both models and merge detections for ambiguous frames
                active_model = self.yolo_model  # primary, thermal merged below
                logger.debug("Ambiguous modality — will merge both models")

            # V3.3 — Trigger ADS-B background update
            adsb_cfg = getattr(self.cfg, 'adsb', None)
            if _ADSB_AVAILABLE and getattr(adsb_cfg, 'enabled', False):
                if (self._adsb_thread is None or
                        not self._adsb_thread.is_alive()):
                    self._adsb_thread = threading.Thread(
                        target=adsb_update_cache,
                        args=(
                            self.cfg.dashboard.map_center_lat,
                            self.cfg.dashboard.map_center_lon,
                            getattr(adsb_cfg, 'bbox_degrees', 0.1),
                            getattr(
                                adsb_cfg, 'query_interval_seconds', 5.0
                            ),
                        ),
                        daemon=True,
                    )
                    self._adsb_thread.start()

            # 1. Preprocess (resize)
            processed = prepare_for_inference(frame, self.cfg)
            orig_h, orig_w = frame.shape[:2]
            
            # 2. YOLO inference on resized frame - detections are in resized coordinate space
            # Run inference using selected model
            detections = run_inference(active_model, processed, self.cfg)

            # For AMBIGUOUS modality, merge thermal model detections too
            if (modality == "AMBIGUOUS" and
                    self.yolo_model_thermal is not None):
                thermal_detections = run_inference(
                    self.yolo_model_thermal, processed, self.cfg
                )
                # Simple merge — append thermal detections
                # DeepSORT NMS will handle duplicates
                detections.extend(thermal_detections)
            
            # 2.5. Scale detection coordinates back to original frame space
            # resize_frame uses letterbox: scale = target_size / max(h, w)
            scale = self.cfg.model.input_size / max(orig_h, orig_w)
            pad_x = (self.cfg.model.input_size - int(orig_w * scale)) // 2
            pad_y = (self.cfg.model.input_size - int(orig_h * scale)) // 2
            
            for det in detections:
                # Scale xyxy bbox back to original coords
                x1, y1, x2, y2 = det["bbox"]
                det["bbox"] = [
                    (x1 - pad_x) / scale,
                    (y1 - pad_y) / scale,
                    (x2 - pad_x) / scale,
                    (y2 - pad_y) / scale,
                ]
                # Scale xywh (center format) back too - for DeepSORT
                cx, cy, w, h = det["bbox_xywh"]
                det["bbox_xywh"] = [
                    (cx - pad_x) / scale,
                    (cy - pad_y) / scale,
                    w / scale,
                    h / scale,
                ]
            
            # 3. DeepSORT tracking - pass ORIGINAL frame for Re-ID crops
            tracks = self.tracker.update(detections, frame)
            
            # 4. Telemetry update
            tracks = self.telemetry.update(tracks, frame_idx)
            
            # 5. Threat scoring with behaviour and zone analysis
            # Define protected zones (centred on map_center with offsets)
            protected_zones = [
                {
                    "name": "Zone Alpha",
                    "center_lat": self.cfg.dashboard.map_center_lat + 0.05,
                    "center_lon": self.cfg.dashboard.map_center_lon + 0.05,
                    "radius_m": 80,
                },
                {
                    "name": "Zone Bravo",
                    "center_lat": self.cfg.dashboard.map_center_lat - 0.05,
                    "center_lon": self.cfg.dashboard.map_center_lon - 0.05,
                    "radius_m": 80,
                },
            ]

            for track in tracks:
                threat_result = self.threat_scorer.score(track)
                track.update(threat_result)

                # Behaviour classification from telemetry history
                track_id = track["track_id"]
                history = self.telemetry.get_track_history(track_id)
                if history:
                    behaviour = self.threat_scorer.classify_behaviour(history)
                else:
                    behaviour = "STATIONARY"
                track["behaviour"] = behaviour

                # Boost threat score for approaching/circling behaviour
                if behaviour in ["APPROACHING", "CIRCLING"]:
                    boosted = min(1.0, track["threat_score"] + 0.2)
                    track["threat_score"] = boosted
                    # Re-classify threat level after boost
                    if boosted >= self.cfg.threat.high_threat_threshold:
                        track["threat_level"] = "HIGH"
                    elif boosted >= self.cfg.threat.medium_threat_threshold:
                        track["threat_level"] = "MEDIUM"

                # Zone violation check
                lat = track.get("lat", 0.0)
                lon = track.get("lon", 0.0)
                in_zone = self.telemetry.check_zone_violation(
                    lat, lon, protected_zones
                )
                track["zone_violation"] = in_zone
                track["zone_name"] = ""
                if in_zone:
                    track["threat_score"] = 1.0
                    track["threat_level"] = "HIGH"
                    for zone in protected_zones:
                        if self.telemetry.check_zone_violation(
                            lat, lon, [zone]
                        ):
                            track["zone_name"] = zone["name"]
                            break

                # Counterfactual explanation
                track["counterfactual"] = self.threat_scorer.get_counterfactual(
                    track
                )

            # Swarm detection
            swarm_detected = self.telemetry.detect_swarm(tracks)

            # Store protected zones for dashboard rendering
            self._protected_zones = protected_zones
            self._swarm_detected = swarm_detected

            # V3.4 — Signature matching, intercept prediction, ADS-B
            pred_cfg = getattr(self.cfg, 'prediction', None)
            for track in tracks:

                # Signature matching
                sig_result = match_signature(track)
                track["signature_name"] = sig_result["signature_name"]
                track["signature_confidence"] = (
                    sig_result["signature_confidence"]
                )

                # Intercept prediction
                if _PREDICTION_AVAILABLE:
                    track_id = track["track_id"]
                    history = self.telemetry.get_track_history(track_id)
                    min_hist = getattr(
                        pred_cfg, 'min_history_frames', 10
                    ) if pred_cfg else 10
                    if history and len(history) >= min_hist:
                        intercept = predict_intercept(
                            track=track,
                            track_history=history,
                            protected_zones=protected_zones,
                            scale_factor=self.telemetry.scale_factor,
                            map_center_lat=(
                                self.cfg.dashboard.map_center_lat
                            ),
                            map_center_lon=(
                                self.cfg.dashboard.map_center_lon
                            ),
                            horizon_seconds=getattr(
                                pred_cfg, 'horizon_seconds', 30
                            ) if pred_cfg else 30,
                            min_history_frames=min_hist,
                            fps=float(
                                self.cfg.simulation.telemetry_update_hz
                            ),
                        )
                        track["intercept_predicted"] = (
                            intercept["intercept_predicted"]
                        )
                        track["time_to_intercept_s"] = (
                            intercept["time_to_intercept_s"]
                        )
                        track["intercept_zone"] = (
                            intercept["intercept_zone"]
                        )
                        track["projected_lat"] = (
                            intercept["projected_lat"]
                        )
                        track["projected_lon"] = (
                            intercept["projected_lon"]
                        )
                        track["prediction_confidence"] = (
                            intercept["confidence"]
                        )

                        warn_thresh = getattr(
                            pred_cfg, 'warning_threshold_seconds', 20
                        ) if pred_cfg else 20
                        if (intercept["intercept_predicted"] and
                                intercept["time_to_intercept_s"]
                                is not None and
                                intercept["time_to_intercept_s"]
                                <= warn_thresh):
                            track["threat_score"] = max(
                                track["threat_score"], 0.75
                            )
                            track["threat_level"] = "HIGH"
                    else:
                        track["intercept_predicted"] = False
                        track["time_to_intercept_s"] = None
                        track["intercept_zone"] = None
                        track["projected_lat"] = track.get("lat", 0.0)
                        track["projected_lon"] = track.get("lon", 0.0)
                        track["prediction_confidence"] = 0.0
                else:
                    track["intercept_predicted"] = False
                    track["time_to_intercept_s"] = None
                    track["intercept_zone"] = None
                    track["projected_lat"] = track.get("lat", 0.0)
                    track["projected_lon"] = track.get("lon", 0.0)
                    track["prediction_confidence"] = 0.0

                # ADS-B cross-reference
                if _ADSB_AVAILABLE and getattr(
                    adsb_cfg, 'enabled', False
                ):
                    lat = track.get("lat", 0.0)
                    lon = track.get("lon", 0.0)
                    adsb_result = adsb_cross_reference(
                        lat, lon,
                        getattr(
                            adsb_cfg, 'match_radius_meters', 500.0
                        ),
                    )
                    track["adsb_status"] = adsb_result["status"]
                    track["adsb_callsign"] = adsb_result.get("callsign")
                    if adsb_result["status"] == "UNREGISTERED":
                        if track["threat_level"] == "LOW":
                            track["threat_level"] = "MEDIUM"
                            track["threat_score"] = max(
                                track["threat_score"], 0.45
                            )
                else:
                    track["adsb_status"] = "NO_DATA"
                    track["adsb_callsign"] = None
            
            # 6. Postprocess - draw annotations
            annotated = draw_detections(frame, tracks)
            
            # 7. Compute timing metrics
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            self.frame_times.append(end_time - start_time)
            
            # Calculate FPS from frame times
            if len(self.frame_times) > 0:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
            else:
                fps = 0.0
            
            # Add metrics overlay
            annotated = draw_metrics_overlay(annotated, fps, latency_ms, len(tracks))
            
            # 8. SHAP explanations (if requested)
            explanations = None
            if explain and tracks:
                if self.shap_explainer is None:
                    self.shap_explainer = SHAPExplainer(self.threat_scorer, self.cfg)
                explanations = self.shap_explainer.explain(tracks)
            
            return {
                "frame_idx": frame_idx,
                "detections": detections,
                "tracks": tracks,
                "fps": fps,
                "latency_ms": latency_ms,
                "annotated_frame": annotated,
                "explanations": explanations,
                "swarm_detected": swarm_detected,
                "protected_zones": protected_zones,
                "modality": self._modality,
                "modality_confidence": self._modality_confidence,
            }
            
        except Exception as e:
            logger.error(f"Frame processing failed at frame {frame_idx}: {e}")
            
            # Return partial result with empty tracks on error
            return {
                "frame_idx": frame_idx,
                "detections": [],
                "tracks": [],
                "fps": 0.0,
                "latency_ms": 0.0,
                "annotated_frame": frame,
                "explanations": None,
                "error": str(e),
                "swarm_detected": False,
                "protected_zones": [],
                "modality": "RGB",
                "modality_confidence": 0.0,
            }
    
    def process_video(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        explain: bool = False,
    ) -> Dict:
        """Process a complete video through the pipeline.
        
        Args:
            video_path: Path to input video file.
            output_path: Optional path to save annotated video.
            explain: Whether to generate SHAP explanations.
            
        Returns:
            Dict containing:
                - total_frames: int
                - avg_fps: float
                - avg_latency: float
                - output_path: Optional[Path]
                
        Raises:
            ValueError: If video cannot be opened.
        """
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_path}, {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Process frames
        frame_idx = 0
        all_latencies = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result = self.process_frame(frame, frame_idx, explain)
            all_latencies.append(result["latency_ms"])
            
            # Write output if requested
            if writer:
                writer.write(result["annotated_frame"])
            
            frame_idx += 1
            
            # Log progress every 100 frames
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        
        # Compute summary statistics
        avg_fps = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
        avg_fps = 1.0 / avg_fps if avg_fps > 0 else 0
        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
        
        logger.info(f"Video processing complete: {frame_idx} frames, {avg_fps:.1f} avg FPS")
        
        return {
            "total_frames": frame_idx,
            "avg_fps": avg_fps,
            "avg_latency": avg_latency,
            "output_path": output_path,
        }
    
    def reset(self) -> None:
        """Reset all pipeline state. Use when switching videos."""
        self.tracker.reset()
        self.telemetry.reset()
        self.frame_times.clear()
        logger.info("Pipeline state reset")
