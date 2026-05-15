# AegisVision 🛡️ — Version 3

**AI-powered multi-domain drone threat detection system with dual-modality support, ADS-B cross-referencing, predictive intercept analysis, and military-format reporting.**

AegisVision is a production-grade system that detects, tracks, and assesses threats from aerial imagery using deep learning and computer vision. Version 3 introduces dual-modality detection (RGB/Thermal), ADS-B cross-referencing with the OpenSky Network, intercept prediction, threat signature matching, STANAG-style military reporting, and enhanced data augmentation.

---

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Video Feed │───▶│   Modality  │───▶│  YOLOv8     │───▶│  DeepSORT   │
│(RGB/Thermal)│    │  Classifier │    │  Detection  │    │  Tracking   │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                  │
                        ┌─────────────┐                          │
                        │  Dashboard  │◀───────────────────────────┘
                        │ (Streamlit) │    Telemetry + ADS-B + Signatures
                        └──────┬──────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
         ┌────┴────┐    ┌─────┴──────┐   ┌────┴────┐
         │  SHAP   │    │ Intercept  │   │ STANAG  │
         │ Explain │    │ Prediction │   │  PDF    │
         └─────────┘    └────────────┘   └─────────┘
```

**Pipeline:** Frame → Modality Classification → YOLO Detection → DeepSORT Tracking → Telemetry + ADS-B Cross-Reference + Signature Matching → Intercept Prediction → Threat Scoring → SHAP Explainability → STANAG Report → Visualization

---

## Version 3 Features

| Feature | Description |
|---------|-------------|
| �️ **Dual Modality Detection** | Automatic RGB vs Thermal image classification |
| �️ **ADS-B Cross-Referencing** | Query OpenSky Network to match tracks with registered aircraft |
| 🎯 **Intercept Prediction** | Linear kinematic extrapolation predicts zone entry up to 30s ahead |
| � **Threat Signature Matching** | Match tracks against configurable signature library |
| � **STANAG-Style Reports** | Military-format PDF incident reports with chain of custody |
| � **Data Augmentation** | Training augmentation with fog, blur, noise, and combined effects |
| 🚁 **DroneVehicle Dataset Support** | Native support for RGB + Thermal DroneVehicle dataset |
| � **Enhanced Training** | Built-in augmentation parameters in YOLO training |

**Plus all Phase 2 features:** Behaviour classification, zone monitoring, swarm detection, SHAP explainability, SQLite persistence, API auth, map trajectories.

---

## Quick Start

### 1. Clone and Setup

```bash
cd path\to\AI_Drone          # Windows
cd /path/to/AI_Drone        # Linux/Mac

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate    # Linux/Mac

pip install -r requirements.txt
```

### 2. Configure

Edit `configs/config.yaml` with your paths:

```yaml
paths:
  data_root: "data"
  visdrone_raw: "data/visdrone"
  dota_raw: "data/dota"
  weights: "models/yolo/weights"

model:
  yolo_model_size: "n"  # n/s/m/l
  confidence_threshold: 0.40
  device: "auto"        # auto/cuda/cpu

dashboard:
  map_center_lat: 39.9042   # Beijing latitude
  map_center_lon: 116.4074  # Beijing longitude
```

### 3. Prepare Data

**VisDrone**: Download VisDrone2019-DET-train, extract to `data/visdrone/`:

    data/visdrone/
    ├── images/
    └── annotations/

**DOTA**: Download DOTA-v1.0 train, extract to `data/dota/`:

    data/dota/
    ├── images/
    └── labelTxt/

### 4. Process Datasets

```bash
python run_prepare_data.py
```

### 5. Train YOLO

```bash
python models/yolo/train.py --epochs 50 --batch 8
```

### 6. Augment Training Data (V3)

Generate augmented training images with fog, blur, noise, and combined effects:

```bash
# Fog augmentation
python run_augment_data.py data/processed/train/images --effect fog --intensity 0.4

# Blur augmentation
python run_augment_data.py data/processed/train/images --effect blur --intensity 0.3

# Noise augmentation
python run_augment_data.py data/processed/train/images --effect noise --intensity 0.05

# Combined effects (multiple augmentations)
python run_augment_data.py data/processed/train/images --effect all --intensity 0.5
```

### 7. Run API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**API Key Authentication:** All endpoints require header `X-API-Key: aegisvision-demo-key-2024`

Endpoints:
- `GET /health` - Health check
- `POST /predict/frame` - Process single image
- `POST /predict/video` - Process video file
- `WebSocket /stream` - Real-time streaming
- `GET /sessions` - List all sessions
- `GET /sessions/{id}/detections` - Get session detections
- `GET /sessions/{id}/events` - Get session events

### 8. Run Dashboard

```bash
streamlit run dashboard/app.py
```

Access at http://localhost:8501

**Dashboard Features:**
- Real-time threat gauges (LOW/MEDIUM/HIGH counts)
- **Modality badge** showing RGB/Thermal detection with confidence
- Object table with ADS-B status, signature match, and intercept warnings
- **Intercept projection lines** on map showing predicted paths
- SHAP bar charts showing feature contributions
- Map with protected zone overlays, trajectory trails, and ADS-B callsign tooltips
- CSV/PDF export buttons with STANAG-style military reports
- Video export toggle (annotated video download)
- Session-based persistence to SQLite

---

## Version 3 Usage Examples

### Modality Detection (RGB vs Thermal)

```python
from models.classifier.channel_classifier import ChannelClassifier

classifier = ChannelClassifier()
modality, confidence = classifier.predict(image)
print(f"Detected modality: {modality} ({confidence:.1%} confidence)")
```

### ADS-B Cross-Referencing

```python
from services.adsb_client import ADSBClient

# Query aircraft in 1km radius around Beijing airport
client = ADSBClient(lat=39.9042, lon=116.4074, radius_km=1.0)
aircraft = client.get_aircraft()

for ac in aircraft:
    print(f"Callsign: {ac.callsign}, Alt: {ac.altitude}m")
```

### Intercept Prediction

```python
from models.prediction.intercept_predictor import InterceptPredictor
from shapely.geometry import Polygon

# Define protected zone
zone = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
predictor = InterceptPredictor([zone], horizon_seconds=30)

# Track positions over time
positions = [(10, 10), (20, 20), (30, 30)]  # (x, y) pixels
result = predictor.predict_intercept(track_id=1, positions=positions)

if result:
    print(f"Intercept in {result.time_to_intercept_s:.0f}s to zone: {result.zone_name}")
```

### Threat Signature Matching

```python
from models.threat.signature_matcher import SignatureMatcher

matcher = SignatureMatcher("configs/signatures.yaml")
signature, confidence = matcher.match_track(
    speed_ms=45, altitude_m=120, is_approaching=True
)
print(f"Matched signature: {signature} ({confidence:.1%} confidence)")
```

### Threat Scoring with Behaviour

```python
from models.threat.threat_model import ThreatScorer
from utils.config_loader import load_config

cfg = load_config()
scorer = ThreatScorer(cfg)

# Score a track
track = {"speed_mps": 15.0, "proximity_m": 50.0, "class_name": "drone"}
result = scorer.score(track)
print(f"Threat: {result['threat_score']:.2f} ({result['threat_level']})")

# Classify behaviour from track history
history = [(0, 100, 100), (1, 105, 95), (2, 110, 90)]  # (frame, x, y)
behaviour = scorer.classify_behaviour(history)
print(f"Behaviour: {behaviour}")  # APPROACHING

# Get counterfactual explanation
cf = scorer.get_counterfactual(track)
print(cf)  # "If speed were 5.0 m/s lower, score would drop to 0.35 (LOW)"
```

### Zone Violation Detection

```python
from simulation.telemetry import TelemetrySimulator
from utils.config_loader import load_config

cfg = load_config()
telemetry = TelemetrySimulator(cfg)

# Define protected zones
zones = [
    {"name": "Zone Alpha", "center_lat": 39.9042, "center_lon": 116.4074, "radius_m": 150},
    {"name": "Zone Bravo", "center_lat": 39.9032, "center_lon": 116.4054, "radius_m": 100},
]

# Check violation
in_zone = telemetry.check_zone_violation(39.9043, 116.4075, zones)
print(f"Zone violation: {in_zone}")  # True
```

### Swarm Detection

```python
# In pipeline processing, swarm detection is automatic
tracks = [...]  # Multiple tracked objects
swarm_detected = telemetry.detect_swarm(tracks)
print(f"Swarm detected: {swarm_detected}")  # True if ≥5 objects moving similarly
```

### Database Operations

```python
from database.results_db import (
    create_session, log_detection, close_session,
    get_all_sessions, get_session_detections, get_session_events
)

# Create session
session_id = "abc123"
create_session(session_id, "video.mp4")

# Log detections (called automatically by pipeline)
log_detection(session_id, frame_idx=0, track=track_data, fps=30.0, latency_ms=50.0)

# Close with stats
stats = {"total_frames": 1000, "high_threats": 5, "avg_fps": 25.0}
close_session(session_id, stats)

# Query history
sessions = get_all_sessions()
detections = get_session_detections(session_id)
events = get_session_events(session_id)
```

### PDF Report Generation

```python
from reports.pdf_report import generate_pdf_report
from database.results_db import get_session_detections, get_session_events

detections = get_session_detections(session_id)
events = get_session_events(session_id)
stats = {"total_frames": 1000, "high_threats": 5, "avg_fps": 25.0}

output_path = generate_pdf_report(
    session_id=session_id,
    video_filename="drone_footage.mp4",
    detections=detections,
    events=events,
    stats=stats
)
print(f"Report saved: {output_path}")
```

### ONNX Benchmark

```python
from deployment.optimize_onnx import benchmark_onnx
from pathlib import Path

results = benchmark_onnx(
    onnx_path=Path("models/yolo/weights/best.onnx"),
    num_runs=100
)
print(f"Avg latency: {results['avg_latency_ms']}ms")
print(f"Throughput: {results['throughput_fps']} FPS")
```

---

## Project Structure

```
AI_Drone/
├── configs/              # Configuration files
│   ├── config.yaml       # Master configuration
│   └── dataset.yaml      # YOLO dataset config
├── data/
│   ├── loaders/          # VisDrone & DOTA loaders
│   ├── processed/        # YOLO-formatted output
│   ├── visdrone/         # Raw VisDrone data
│   └── dota/             # Raw DOTA data
├── database/             # SQLite persistence (NEW)
│   ├── __init__.py
│   └── results_db.py
├── models/
│   ├── yolo/             # Training & inference
│   ├── tracking/         # DeepSORT tracker
│   └── threat/           # Threat scoring + behaviour
├── pipeline/             # Full processing pipeline
├── simulation/           # Telemetry + augmentations + zones
├── explainability/       # SHAP explanations
├── api/                  # FastAPI + auth + session endpoints
├── dashboard/            # Streamlit with Phase 2 features
├── deployment/           # Docker & ONNX export + benchmark
├── reports/              # PDF report generation (NEW)
│   ├── __init__.py
│   └── pdf_report.py
├── mlops/                # MLflow tracker
├── utils/                # Config, logging, bbox math
└── tests/                # Test suite
```

---

## Configuration Reference

| Section | Key | Description |
|---------|-----|-------------|
| `model` | `yolo_model_size` | Model size: n/s/m/l |
| `model` | `confidence_threshold` | Detection threshold |
| `training` | `epochs` | Training epochs |
| `training` | `batch_size` | Batch size |
| `tracking` | `max_age` | Frames before track deletion |
| `tracking` | `min_hits` | Frames to confirm track |
| `threat` | `weights` | Threat scoring weights |
| `threat` | `high_threat_threshold` | HIGH threshold (default 0.7) |
| `threat` | `medium_threat_threshold` | MEDIUM threshold (default 0.4) |
| `api` | `port` | API server port |
| `dashboard` | `map_center_lat` | Map center latitude |
| `dashboard` | `map_center_lon` | Map center longitude |

---

## Threat Model Weights

Default threat scoring weights (in `config.yaml`):

```yaml
threat:
  weights:
    proximity: 0.6    # Closer objects = higher threat
    speed: 0.2        # Faster objects = higher threat
    class_danger: 0.2 # Class-based base threat
  high_threat_threshold: 0.7
  medium_threat_threshold: 0.4
```

Behaviour boost: APPROACHING/CIRCLING objects get +0.2 threat score boost.

---

## API Authentication

All API endpoints require the `X-API-Key` header:

```bash
curl -H "X-API-Key: aegisvision-demo-key-2024" \
     http://localhost:8000/health
```

Session history endpoints:
```bash
# List all sessions
curl -H "X-API-Key: aegisvision-demo-key-2024" \
     http://localhost:8000/sessions

# Get session detections
curl -H "X-API-Key: aegisvision-demo-key-2024" \
     http://localhost:8000/sessions/abc123/detections

# Get session events
curl -H "X-API-Key: aegisvision-demo-key-2024" \
     http://localhost:8000/sessions/abc123/events
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_threat_model.py -v
pytest tests/test_loaders.py -v
pytest tests/test_api.py -v
```

Test coverage includes:
- Threat model scoring and behaviour classification
- BBox utility functions
- Data loaders (VisDrone/DOTA)
- API endpoints with authentication
- Pipeline integration

---

## Docker Deployment

```bash
cd deployment
docker-compose up --build
```

Services:
- API: http://localhost:8000
- Dashboard: http://localhost:8501
(Yes it's local only for now)

---

## Class Mapping

**VisDrone (6 classes)**: pedestrian, bicycle, car, van, truck, bus

**DOTA (6 classes added)**: plane, ship, storage-tank, harbor, bridge, helicopter

**Total: 12 classes**

---

## Citation

```bibtex
@software{aegisvision2024,
  title={AegisVision: AI Drone Threat Detection System},
  author={Kushagra},
  year={2026}
}
```

---
