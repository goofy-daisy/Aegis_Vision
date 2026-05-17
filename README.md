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
                        ┌─────────────┐                         │
                        │  Dashboard  │◀────────────────────────┘
                        │ (Streamlit) │    Telemetry + ADS-B + Signatures
                        └──────┬──────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
         ┌────┴────┐    ┌────-─┴──────┐  ┌────--┴────┐
         │  SHAP   │    │ Intercept   │  │ STANAG    │
         │ Explain │    │ Prediction  │  │  PDF      │
         └─────────┘    └─────-───────┘  └─────────--┘
```

**Pipeline:** Frame → Modality Classification → YOLO Detection → DeepSORT Tracking → Telemetry + ADS-B Cross-Reference + Signature Matching → Intercept Prediction → Threat Scoring → SHAP Explainability → STANAG Report → Visualization

---

## Version 3 Features

| Feature | Description |
|---------|-------------|
| 🎭️ **Dual Modality Detection** | Automatic RGB vs Thermal image classification |
| 🛰️ **ADS-B Cross-Referencing** | Query OpenSky Network to match tracks with registered aircraft |
| 🎯 **Intercept Prediction** | Linear kinematic extrapolation predicts zone entry up to 30s ahead |
| 🔍 **Threat Signature Matching** | Match tracks against configurable signature library |
| 📋 **STANAG-Style Reports** | Military-format PDF incident reports with chain of custody |
| 🌫️ **Data Augmentation** | Training augmentation with fog, blur, noise, and combined effects |
| 🚁 **DroneVehicle Dataset Support** | Native support for RGB + Thermal DroneVehicle dataset |
| 📈 **Enhanced Training** | Built-in augmentation parameters in YOLO training |

**Plus all Phase 2 features:** Behaviour classification, zone monitoring, swarm detection, SHAP explainability, SQLite persistence, API auth, map trajectories.

Link with all the database and trial test results: https://drive.google.com/file/d/1orIUBe6Z986d_i6Lcf8jxDLZWftDyCme/view?usp=drive_link

---

## Prerequisites

This guide assumes you already have the following installed:
- **Python 3.11** (`python3.11 --version` should work)
- **Git** (`git --version` should work)
- **VS Code** or any code editor

> **Mac (Apple Silicon M-series) users:** Follow the Mac-specific notes throughout this guide. The project was originally built on Windows — a few extra steps are needed.

---

## Quick Start

### 1. Clone the Repository

```bash
mkdir -p /your/desired/path
cd /your/desired/path
git clone https://github.com/goofy-daisy/Aegis_Vision.git
cd Aegis_Vision/AI_Drone
```

> **Note:** The repo has a nested folder structure. The actual project code lives inside `Aegis_Vision/AI_Drone/`. All commands from here onwards run from inside that `AI_Drone` folder.

Open in VS Code:
```bash
code .
```

---

### 2. Create Virtual Environment

```bash
python3.11 -m venv venv
```

Activate it:
```bash
# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Your terminal prompt should now show `(venv)` — this means you're inside the isolated environment. **Always activate this before working on the project.**

Upgrade pip:
```bash
pip install --upgrade pip
```

---

### 3. Install Dependencies

> **Mac (Apple Silicon) — Install PyTorch separately FIRST before requirements.txt:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
Wait for this to finish completely before moving on.

Now install all remaining dependencies:
```bash
pip install -r requirements.txt
```
This takes 5–10 minutes. Let it run fully.

> **Mac (Apple Silicon) — Run these after requirements.txt to fix known M-series issues:**
```bash
brew install libomp
pip install opencv-python-headless
pip install lightgbm
pip install deep-sort-realtime
pip install shapely
```

---

### 4. Configure

Edit `configs/config.yaml`:

```bash
# Open in VS Code
code configs/config.yaml
```

Update the following values:

```yaml
paths:
  data_root: "data"
  visdrone_raw: "data/visdrone"
  dota_raw: "data/dota"
  weights: "models/yolo/weights"

model:
  yolo_model_size: "n"  # n/s/m/l
  confidence_threshold: 0.40
  device: "cpu"         # Use "cpu" on Mac Apple Silicon; "cuda" if you have an NVIDIA GPU

dashboard:
  map_center_lat: 28.6139   # Change to your city's coordinates
  map_center_lon: 77.2090
```

---

### 5. Initialize the Database

> **Note:** The pre-trained `best.pt` weights file is already included in the repo at `models/yolo/weights/best.pt` — no download needed.

```bash
python3 -c "from database.results_db import create_session; print('DB OK')"
```

Should print `DB OK`. This sets up the SQLite database for session persistence.

---

### 6. Run the API

Open **Terminal Tab 1** and run:

```bash
# Mac/Linux
source venv/bin/activate
PYTHONPATH=/full/path/to/your/AI_Drone uvicorn api.main:app --host 0.0.0.0 --port 8000

# Windows
venv\Scripts\activate
set PYTHONPATH=C:\full\path\to\AI_Drone
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

> **Important for Mac:** Replace `/full/path/to/your/AI_Drone` with your actual path to the `AI_Drone` folder. Example:
> ```bash
> PYTHONPATH=/Users/your_username/Desktop/Projects/Aegis_Vision/AI_Drone uvicorn api.main:app --host 0.0.0.0 --port 8000
> ```
> The `AI_Drone` folder is what you `cd` into after cloning — it sits inside the repo root `Aegis_Vision/`.

Leave this terminal running.

**API Key Authentication:** All endpoints require header `X-API-Key: aegisvision-demo-key-2024`

Endpoints:
- `GET /health` — Health check
- `POST /predict/frame` — Process single image
- `POST /predict/video` — Process video file
- `WebSocket /stream` — Real-time streaming
- `GET /sessions` — List all sessions
- `GET /sessions/{id}/detections` — Get session detections
- `GET /sessions/{id}/events` — Get session events

Test the API is running (open a new terminal tab):
```bash
curl -H "X-API-Key: aegisvision-demo-key-2024" http://localhost:8000/health
```
Should return a JSON response.

---

### 7. Run the Dashboard

Open **Terminal Tab 2** (keep Tab 1 running) and run:

```bash
# Mac/Linux
source venv/bin/activate
PYTHONPATH=/full/path/to/your/AI_Drone streamlit run dashboard/app.py

# Windows
venv\Scripts\activate
set PYTHONPATH=C:\full\path\to\AI_Drone
streamlit run dashboard/app.py
```

> **Important for Mac:** The `PYTHONPATH` prefix is required — without it you'll get `ModuleNotFoundError: No module named 'pipeline'`.

Access at **http://localhost:8501**

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

## Every Time You Come Back (Daily Use)

You need two terminal tabs running simultaneously:

```bash
# Tab 1 — API Server
cd /full/path/to/your/AI_Drone
source venv/bin/activate
PYTHONPATH=/full/path/to/your/AI_Drone uvicorn api.main:app --host 0.0.0.0 --port 8000

# Tab 2 — Dashboard
cd /full/path/to/your/AI_Drone
source venv/bin/activate
PYTHONPATH=/full/path/to/your/AI_Drone streamlit run dashboard/app.py
```

Dashboard is available at **http://localhost:8501**

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

client = ADSBClient(lat=39.9042, lon=116.4074, radius_km=1.0)
aircraft = client.get_aircraft()

for ac in aircraft:
    print(f"Callsign: {ac.callsign}, Alt: {ac.altitude}m")
```

### Intercept Prediction

```python
from models.prediction.intercept_predictor import InterceptPredictor
from shapely.geometry import Polygon

zone = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
predictor = InterceptPredictor([zone], horizon_seconds=30)

positions = [(10, 10), (20, 20), (30, 30)]
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

track = {"speed_mps": 15.0, "proximity_m": 50.0, "class_name": "drone"}
result = scorer.score(track)
print(f"Threat: {result['threat_score']:.2f} ({result['threat_level']})")

history = [(0, 100, 100), (1, 105, 95), (2, 110, 90)]
behaviour = scorer.classify_behaviour(history)
print(f"Behaviour: {behaviour}")  # APPROACHING

cf = scorer.get_counterfactual(track)
print(cf)  # "If speed were 5.0 m/s lower, score would drop to 0.35 (LOW)"
```

### Zone Violation Detection

```python
from simulation.telemetry import TelemetrySimulator
from utils.config_loader import load_config

cfg = load_config()
telemetry = TelemetrySimulator(cfg)

zones = [
    {"name": "Zone Alpha", "center_lat": 39.9042, "center_lon": 116.4074, "radius_m": 150},
    {"name": "Zone Bravo", "center_lat": 39.9032, "center_lon": 116.4054, "radius_m": 100},
]

in_zone = telemetry.check_zone_violation(39.9043, 116.4075, zones)
print(f"Zone violation: {in_zone}")  # True
```

### Swarm Detection

```python
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

session_id = "abc123"
create_session(session_id, "video.mp4")

log_detection(session_id, frame_idx=0, track=track_data, fps=30.0, latency_ms=50.0)

stats = {"total_frames": 1000, "high_threats": 5, "avg_fps": 25.0}
close_session(session_id, stats)

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
Aegis_Vision/          ← repo root (cloned from GitHub)
└── AI_Drone/          ← work from here, all commands run inside this folder
    ├── configs/              # Configuration files
    │   ├── config.yaml       # Master configuration
    │   └── dataset.yaml      # YOLO dataset config
    ├── data/
    │   ├── loaders/          # VisDrone & DOTA loaders
    │   ├── processed/        # YOLO-formatted output
    │   ├── visdrone/         # Raw VisDrone data
    │   └── dota/             # Raw DOTA data
    ├── database/             # SQLite persistence
    │   ├── __init__.py
    │   └── results_db.py
    ├── models/
    │   ├── yolo/             # Inference + weights (best.pt included)
    │   ├── tracking/         # DeepSORT tracker
    │   └── threat/           # Threat scoring + behaviour
    ├── pipeline/             # Full processing pipeline
    ├── simulation/           # Telemetry + augmentations + zones
    ├── explainability/       # SHAP explanations
    ├── api/                  # FastAPI + auth + session endpoints
    ├── dashboard/            # Streamlit dashboard
    ├── deployment/           # Docker & ONNX export + benchmark
    ├── reports/              # PDF report generation
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

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'pipeline'` | Add `PYTHONPATH=/full/path/to/AI_Drone` before your streamlit/uvicorn command |
| `ModuleNotFoundError` for any other module | Make sure venv is active, then `pip install <module>` |
| Port 8000 already in use | `lsof -i :8000` then `kill -9 <PID>` |
| MPS / GPU errors on Mac | Set `device: "cpu"` in `configs/config.yaml` |
| OpenCV errors on Mac | `pip install opencv-python-headless` |
| `(venv)` not showing in terminal | Run `source venv/bin/activate` |
| `brew` or `python3.11` not found | Run `source ~/.zprofile` first |

---

## Class Mapping

**VisDrone (6 classes):** pedestrian, bicycle, car, van, truck, bus

**DOTA (6 classes added):** plane, ship, storage-tank, harbor, bridge, helicopter

**Total: 12 classes**

---

## Citation

```bibtex
@software{aegisvision2026,
  title={AegisVision: AI Drone Threat Detection System},
  author={Kushagra},
  year={2026}
}
```

---
