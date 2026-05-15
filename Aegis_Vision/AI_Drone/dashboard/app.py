"""AegisVision Dashboard — Phase 2 Complete Edition.

Real-time threat detection dashboard with:
- Behaviour classification
- Protected zone overlays
- Swarm detection
- Counterfactual explanations
- SQLite persistence
- PDF report generation
- Video export
"""

import base64
import tempfile
import time
import uuid
from collections import deque, Counter
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from PIL import Image

from pipeline import AegisPipeline
from utils.config_loader import load_config
from utils.logger import get_logger
from database.results_db import (
    create_session, log_detection, log_event,
    close_session, get_session_detections, get_session_events,
    get_session_summary
)
from reports.pdf_report import generate_pdf_report

try:
    from services.adsb_client import get_status as adsb_get_status
    _ADSB_AVAILABLE = True
except ImportError:
    _ADSB_AVAILABLE = False
    def adsb_get_status():
        return "OFFLINE"

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ──────────────────────────────────────────────────────────────────────────────
def init_session_state():
    """Initialize all session state keys."""
    if "pipeline" not in st.session_state:
        cfg = load_config()
        st.session_state.pipeline = AegisPipeline(cfg)
    if "frame_history" not in st.session_state:
        st.session_state.frame_history = deque(maxlen=100)
    if "threat_history" not in st.session_state:
        st.session_state.threat_history = deque(maxlen=100)
    if "fps_history" not in st.session_state:
        st.session_state.fps_history = deque(maxlen=100)
    if "running" not in st.session_state:
        st.session_state.running = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "video_filename" not in st.session_state:
        st.session_state.video_filename = "unknown"
    if "results_log" not in st.session_state:
        st.session_state.results_log = []
    if "session_complete" not in st.session_state:
        st.session_state.session_complete = False
    if "final_stats" not in st.session_state:
        st.session_state.final_stats = None
    if "swarm_alert_frame" not in st.session_state:
        st.session_state.swarm_alert_frame = 0
    if "event_log" not in st.session_state:
        st.session_state.event_log = []
    if "track_trajectories" not in st.session_state:
        st.session_state.track_trajectories = {}
    if "video_writer" not in st.session_state:
        st.session_state.video_writer = None
    if "save_video" not in st.session_state:
        st.session_state.save_video = False


# ──────────────────────────────────────────────────────────────────────────────
# FRAME PROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def process_frame_with_logging(frame, frame_idx, explain, session_id, pipeline):
    """Process frame with database logging."""
    result = pipeline.process_frame(frame, frame_idx, explain)

    # Store modality for session display
    if result.get("modality"):
        st.session_state["last_modality"] = result["modality"]

    # Log detections to database
    fps = result.get("fps", 0)
    latency = result.get("latency_ms", 0)

    for track in result.get("tracks", []):
        log_detection(session_id, frame_idx, track, fps, latency)

        # Log HIGH threat events
        if track.get("threat_level") == "HIGH":
            log_event(
                session_id, frame_idx, "HIGH_THREAT",
                f"Track {track.get('track_id')} ({track.get('class_name')}) — "
                f"score {track.get('threat_score', 0):.2f}"
            )

        # Log zone violations
        if track.get("zone_violation"):
            log_event(
                session_id, frame_idx, "ZONE_VIOLATION",
                f"Track {track.get('track_id')} entered {track.get('zone_name', 'protected zone')}"
            )

    # Log swarm events
    if result.get("swarm_detected"):
        log_event(session_id, frame_idx, "SWARM_DETECTED",
                  "Multiple coordinated objects identified")

    # Store in session state for results panel
    for track in result.get("tracks", []):
        st.session_state.results_log.append({
            "frame_idx": frame_idx,
            "track_id": track.get("track_id"),
            "class_name": track.get("class_name"),
            "threat_score": track.get("threat_score"),
            "threat_level": track.get("threat_level"),
            "behaviour": track.get("behaviour"),
            "zone_violation": track.get("zone_violation"),
            "zone_name": track.get("zone_name"),
            "counterfactual": track.get("counterfactual"),
            "fps": fps,
            "latency_ms": latency,
            "adsb_status": track.get("adsb_status", "NO_DATA"),
            "adsb_callsign": track.get("adsb_callsign"),
            "signature_name": track.get("signature_name", "Unknown"),
            "intercept_predicted": track.get("intercept_predicted", False),
            "time_to_intercept_s": track.get("time_to_intercept_s"),
        })

    return result


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR UI
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    """Render sidebar controls."""
    with st.sidebar:
        st.header("⚙️ Controls")

        # Source selection
        source = st.selectbox(
            "Input Source",
            ["Webcam", "Video File", "Image"],
            key="source"
        )

        # Confidence threshold
        confidence = st.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.5, 0.05,
            key="confidence"
        )

        # Threat filter
        threat_filter = st.multiselect(
            "Threat Filter",
            ["LOW", "MEDIUM", "HIGH"],
            default=["LOW", "MEDIUM", "HIGH"],
            key="threat_filter"
        )

        st.divider()
        st.markdown("**🛰️ ADS-B Status**")
        adsb_status = adsb_get_status()
        if adsb_status == "LIVE":
            st.success("🛰️ ADS-B LIVE")
        elif adsb_status == "OFFLINE":
            st.warning("📡 ADS-B OFFLINE")
        else:
            st.info(f"📡 ADS-B: {adsb_status}")

        # Explainability toggle
        explain = st.checkbox(
            "Enable SHAP Explanations",
            value=False,
            key="explain"
        )

        # Save video toggle
        save_video = st.checkbox(
            "Save Annotated Video",
            value=False,
            key="save_video_check"
        )
        st.session_state.save_video = save_video

        st.divider()

        # Start/Stop buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", type="primary", use_container_width=True):
                st.session_state.running = True
                st.session_state.session_complete = False
                st.session_state.results_log = []
                st.session_state.event_log = []
                st.session_state.track_trajectories = {}
                # Clear video state so next upload re-initialises correctly
                for key in ["video_temp_path", "video_cap", "video_writer",
                            "video_total_frames", "frame_idx"]:
                    if key in st.session_state:
                        del st.session_state[key]
                # Generate new session ID BEFORE writer is created
                st.session_state.session_id = uuid.uuid4().hex
                st.session_state.video_out_path = None
        with col2:
            if st.button("⏹️ Stop", use_container_width=True):
                st.session_state.running = False
                st.rerun()

        st.divider()

        # Export buttons (only when session complete)
        if st.session_state.session_complete:
            if st.button("💾 Save Results CSV", use_container_width=True):
                export_csv()
            if st.button("📄 Generate PDF Report", use_container_width=True):
                export_pdf()
        else:
            st.caption("💾 Export buttons available after video processing completes")

        return source, confidence, threat_filter, explain


def export_csv():
    """Export results to CSV."""
    if not st.session_state.results_log:
        st.warning("No data to export")
        return

    df = pd.DataFrame(st.session_state.results_log)
    csv = df.to_csv(index=False)
    st.download_button(
        "Download CSV",
        csv,
        f"aegisvision_{st.session_state.session_id[:8]}.csv",
        "text/csv",
        use_container_width=True
    )


def export_pdf():
    """Generate and offer PDF download."""
    try:
        detections = get_session_detections(st.session_state.session_id)
        events = get_session_events(st.session_state.session_id)
        stats = st.session_state.final_stats or {}

        output_path = generate_pdf_report(
            session_id=st.session_state.session_id,
            video_filename=st.session_state.video_filename,
            detections=detections,
            events=events,
            stats=stats
        )

        with open(output_path, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            "Download PDF Report",
            pdf_bytes,
            f"AegisVision_Report_{st.session_state.session_id[:8]}.pdf",
            "application/pdf",
            use_container_width=True
        )
        st.success(f"PDF generated: {output_path.name}")
    except Exception as e:
        st.error(f"PDF generation failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# VIDEO FEED RENDERING
# ──────────────────────────────────────────────────────────────────────────────
def render_video_feed():
    """Render video feed area with metrics."""
    st.subheader("📹 Live Feed")

    # Video placeholder
    video_placeholder = st.empty()

    # Modality badge placeholder
    modality_badge = st.empty()

    # Metrics row
    cols = st.columns(4)
    fps_metric = cols[0].empty()
    latency_metric = cols[1].empty()
    count_metric = cols[2].empty()
    threat_metric = cols[3].empty()

    return video_placeholder, fps_metric, latency_metric, count_metric, threat_metric, modality_badge


def get_fps_color(fps):
    """Get delta_color for FPS metric."""
    if fps > 5:
        return "normal"  # green
    elif fps >= 2:
        return "off"  # gray/normal
    else:
        return "inverse"  # red


def get_latency_color(latency_ms):
    """Get delta_color for latency metric."""
    if latency_ms < 200:
        return "normal"  # green
    elif latency_ms <= 400:
        return "off"  # gray/normal
    else:
        return "inverse"  # red


# ──────────────────────────────────────────────────────────────────────────────
# THREAT PANEL RENDERING
# ──────────────────────────────────────────────────────────────────────────────
def render_threat_panel():
    """Render threat analysis panel."""
    st.subheader("🎯 Threat Analysis")

    # Threat level gauges
    cols = st.columns(3)
    with cols[0]:
        low_gauge = st.empty()
    with cols[1]:
        med_gauge = st.empty()
    with cols[2]:
        high_gauge = st.empty()

    threat_gauges = {"LOW": low_gauge, "MEDIUM": med_gauge, "HIGH": high_gauge}

    # Object table
    st.markdown("**Tracked Objects**")
    object_table = st.empty()

    # SHAP explanations
    st.markdown("**Explainability (SHAP)**")
    shap_container = st.empty()

    return threat_gauges, object_table, shap_container


def update_gauges(threat_gauges, tracks):
    """Update threat level gauges."""
    counts = Counter(t.get("threat_level", "LOW") for t in tracks)
    threat_gauges["LOW"].metric("🟢 LOW", counts.get("LOW", 0))
    threat_gauges["MEDIUM"].metric("🟡 MEDIUM", counts.get("MEDIUM", 0))
    threat_gauges["HIGH"].metric("� HIGH", counts.get("HIGH", 0))


def update_object_table(object_table, tracks):
    """Update object table with behaviour and zone info."""
    if not tracks:
        object_table.info("No active tracks")
        return

    df = pd.DataFrame([
        {
            "ID": t.get("track_id", "-"),
            "Class": t.get("class_name", "-"),
            "Threat": f"{t.get('threat_score', 0):.2f}",
            "Level": t.get("threat_level", "LOW"),
            "Behaviour": t.get("behaviour", "STATIONARY"),
            "Zone": "YES" if t.get("zone_violation") else "NO",
            "ADS-B": t.get("adsb_status", "NO_DATA"),
            "Signature": t.get("signature_name", "Unknown")[:20],
            "Intercept": (
                f"⚠️ {t.get('time_to_intercept_s', 0):.0f}s"
                if t.get("intercept_predicted") else "—"
            ),
        }
        for t in tracks
    ])
    object_table.dataframe(df, use_container_width=True, hide_index=True)


def update_shap_panel(shap_container, tracks, threat_scorer):
    """Update SHAP panel with bar charts and counterfactuals."""
    if not tracks:
        shap_container.info("No tracks to explain")
        return

    with shap_container.container():
        for track in tracks[:3]:  # Show top 3 tracks
            tid = track.get("track_id", "-")
            col1, col2 = st.columns([1, 2])

            with col1:
                # Feature contribution chart
                weights = threat_scorer.get_feature_weights()
                chart_data = pd.DataFrame({
                    "Feature": ["Proximity", "Speed", "Class"],
                    "Contribution": [
                        track.get("proximity_score", 0) * weights["proximity"],
                        track.get("speed_score", 0) * weights["speed"],
                        track.get("class_score", 0) * weights["class_danger"],
                    ]
                })
                st.bar_chart(chart_data.set_index("Feature"), height=150)

            with col2:
                st.caption(f"Track {tid} — {track.get('class_name', '-')}")
                cf = track.get("counterfactual", "")
                if cf:
                    st.info(cf)


# ──────────────────────────────────────────────────────────────────────────────
# MAP PANEL RENDERING
# ──────────────────────────────────────────────────────────────────────────────
def render_map_panel():
    """Render map panel."""
    st.subheader("🗺️ Map View")
    map_placeholder = st.empty()
    return map_placeholder


def update_map(map_placeholder, tracks, protected_zones=None, trajectories=None):
    """Update map with objects, zones, and trajectories."""
    if not tracks:
        map_placeholder.info("No GPS data available")
        return

    # Build data layers
    layers = []

    # Protected zones as scatterplot circles
    if protected_zones:
        zone_data = []
        for zone in protected_zones:
            zone_data.append({
                "lat": zone["center_lat"],
                "lon": zone["center_lon"],
                "radius": zone["radius_m"],
                "name": zone["name"],
            })
        if zone_data:
            zone_layer = pdk.Layer(
                "ScatterplotLayer",
                data=zone_data,
                get_position=["lon", "lat"],
                get_radius="radius",
                get_fill_color=[255, 0, 0, 100],
                get_line_color=[255, 0, 0, 200],
                line_width_min_pixels=2,
                filled=True,
                stroked=True,
            )
            layers.append(zone_layer)

    # Trajectory trails as line segments
    if trajectories:
        line_data = []
        for tid, positions in trajectories.items():
            if len(positions) >= 2:
                for i in range(len(positions) - 1):
                    line_data.append({
                        "start_lat": positions[i][0],
                        "start_lon": positions[i][1],
                        "end_lat": positions[i+1][0],
                        "end_lon": positions[i+1][1],
                        "track_id": tid,
                    })
        if line_data:
            line_layer = pdk.Layer(
                "LineLayer",
                data=line_data,
                get_source_position=["start_lon", "start_lat"],
                get_target_position=["end_lon", "end_lat"],
                get_color=[100, 100, 255, 150],
                get_width=2,
            )
            layers.append(line_layer)

    # Object positions
    obj_data = []
    for t in tracks:
        lat = t.get("lat", 0)
        lon = t.get("lon", 0)
        if lat == 0 and lon == 0:
            continue

        threat_level = t.get("threat_level", "LOW")
        color = [0, 255, 0] if threat_level == "LOW" else \
                [255, 165, 0] if threat_level == "MEDIUM" else \
                [255, 0, 0]

        obj_data.append({
            "lat": lat,
            "lon": lon,
            "track_id": t.get("track_id", 0),
            "class": t.get("class_name", "-"),
            "callsign": t.get("adsb_callsign") or t.get("adsb_status", ""),
            "r": color[0],
            "g": color[1],
            "b": color[2],
        })

    # Intercept projection lines
    intercept_lines = []
    for t in tracks:
        lat = t.get("lat", 0)
        lon = t.get("lon", 0)
        proj_lat = t.get("projected_lat", lat)
        proj_lon = t.get("projected_lon", lon)
        if (lat != 0 and lon != 0 and
                (abs(proj_lat - lat) > 0.00001 or
                 abs(proj_lon - lon) > 0.00001)):
            intercept_lines.append({
                "start_lat": lat,
                "start_lon": lon,
                "end_lat": proj_lat,
                "end_lon": proj_lon,
                "is_intercept": 1 if t.get("intercept_predicted") else 0,
            })

    if intercept_lines:
        intercept_layer = pdk.Layer(
            "LineLayer",
            data=intercept_lines,
            get_source_position=["start_lon", "start_lat"],
            get_target_position=["end_lon", "end_lat"],
            get_color=[255, 100, 0, 200],
            get_width=3,
        )
        layers.append(intercept_layer)

    if obj_data:
        obj_layer = pdk.Layer(
            "ScatterplotLayer",
            data=obj_data,
            get_position=["lon", "lat"],
            get_fill_color=["r", "g", "b"],
            get_radius=50,
            filled=True,
        )
        layers.append(obj_layer)

        # Center map on first object
        center_lat = obj_data[0]["lat"]
        center_lon = obj_data[0]["lon"]
    else:
        center_lat, center_lon = 0, 0

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=14,
            pitch=0,
        ),
        tooltip={
            "text": "ID:{track_id} {class}\nADS-B: {callsign}"
        },
    )
    map_placeholder.pydeck_chart(deck)


# ──────────────────────────────────────────────────────────────────────────────
# CHARTS RENDERING
# ──────────────────────────────────────────────────────────────────────────────
def render_charts():
    """Render performance charts."""
    st.subheader("� Performance")
    fps_chart = st.empty()
    threat_chart = st.empty()
    return fps_chart, threat_chart


def update_charts(fps_chart, threat_chart, tracks):
    """Update performance charts."""
    # FPS chart
    if st.session_state.fps_history:
        fps_df = pd.DataFrame({
            "Frame": range(len(st.session_state.fps_history)),
            "FPS": list(st.session_state.fps_history),
        })
        fps_chart.line_chart(fps_df.set_index("Frame"), use_container_width=True)

    # Threat distribution chart
    if tracks:
        threat_counts = Counter(t.get("threat_level", "LOW") for t in tracks)
        threat_df = pd.DataFrame({
            "Level": list(threat_counts.keys()),
            "Count": list(threat_counts.values()),
        })
        threat_chart.bar_chart(threat_df.set_index("Level"), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# RESULTS PANEL (POST-PROCESSING)
# ──────────────────────────────────────────────────────────────────────────────
def render_results_panel():
    """Render post-processing results panel after session complete."""
    if not st.session_state.session_complete:
        return

    st.divider()
    st.subheader("📊 Session Results")

    stats = st.session_state.final_stats or {}
    results = st.session_state.results_log

    if not results:
        st.info("No detections recorded in this session")
        return

    # Metrics row
    cols = st.columns(4)
    cols[0].metric("Total Frames", stats.get("total_frames", 0))
    cols[1].metric("Total Detections", len(results))
    cols[2].metric("Unique Objects", len(set(r["track_id"] for r in results)))
    cols[3].metric("Avg FPS", f"{stats.get('avg_fps', 0):.2f}")

    # V3 metrics
    v3_cols = st.columns(3)
    v3_cols[0].metric(
        "🛰️ Unregistered",
        sum(1 for r in results if r.get("adsb_status") == "UNREGISTERED")
    )
    v3_cols[1].metric(
        "🎯 Intercept Warnings",
        sum(1 for r in results if r.get("intercept_predicted"))
    )
    v3_cols[2].metric(
        "🌡️ Modality",
        st.session_state.get("last_modality", "RGB")
    )

    # Threat breakdown
    st.markdown("**Threat Level Breakdown**")
    threat_counts = Counter(r["threat_level"] for r in results)
    threat_df = pd.DataFrame({
        "Level": list(threat_counts.keys()),
        "Count": list(threat_counts.values()),
    })
    st.bar_chart(threat_df.set_index("Level"), use_container_width=True)

    # Class breakdown
    st.markdown("**Detection by Class**")
    class_counts = Counter(r["class_name"] for r in results)
    class_df = pd.DataFrame({
        "Class": list(class_counts.keys()),
        "Count": list(class_counts.values()),
    })
    st.bar_chart(class_df.set_index("Class"), use_container_width=True)

    # Object summary
    st.markdown("**Per-Object Summary**")
    obj_summary = {}
    for r in results:
        tid = r["track_id"]
        if tid not in obj_summary:
            obj_summary[tid] = {
                "class": r["class_name"],
                "max_threat": 0.0,
                "max_level": "LOW",
                "behaviour": r.get("behaviour", "STATIONARY"),
                "frames": 0,
                "zone_violation": False,
            }
        obj_summary[tid]["frames"] += 1
        if r["threat_score"] > obj_summary[tid]["max_threat"]:
            obj_summary[tid]["max_threat"] = r["threat_score"]
            obj_summary[tid]["max_level"] = r["threat_level"]
        if r.get("zone_violation"):
            obj_summary[tid]["zone_violation"] = True

    summary_df = pd.DataFrame([
        {
            "Track ID": tid,
            "Class": info["class"],
            "Max Threat": f"{info['max_threat']:.3f}",
            "Level": info["max_level"],
            "Behaviour": info["behaviour"],
            "Frames": info["frames"],
            "Zone Violation": "YES" if info["zone_violation"] else "NO",
        }
        for tid, info in sorted(obj_summary.items())
    ])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Export buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Results CSV", key="csv_results"):
            export_csv()
    with col2:
        if st.button("📄 Generate PDF Report", key="pdf_results"):
            export_pdf()

    # Annotated video download — shown here because download button
    # cannot survive a st.rerun(), so it lives in the persistent results panel
    out_path_str = st.session_state.get("video_out_path")
    if out_path_str:
        out_path = Path(out_path_str)
        if out_path.exists() and out_path.stat().st_size > 0:
            st.markdown("---")
            with open(out_path, "rb") as f:
                video_bytes = f.read()
            st.download_button(
                "🎥 Download Annotated Video",
                video_bytes,
                f"aegisvision_annotated_{st.session_state.session_id[:8]}.mp4",
                "video/mp4",
                use_container_width=True,
                key="video_download"
            )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    """Main dashboard entry point."""
    st.set_page_config(
        page_title="AegisVision — Drone Threat Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    st.title("🛡️ AegisVision — Drone Threat Detection")

    # Sidebar
    source, confidence, threat_filter, explain = render_sidebar()

    # Main layout
    col1, col2 = st.columns([0.7, 0.3])

    with col1:
        # Video feed
        video_placeholder, fps_metric, latency_metric, count_metric, threat_metric, modality_badge = render_video_feed()

        # Performance charts
        fps_chart, threat_chart = render_charts()

    with col2:
        # Threat panel
        threat_gauges, object_table, shap_container = render_threat_panel()

        # Map
        map_placeholder = render_map_panel()

    # ── SWARM ALERT ──────────────────────────────────────────────────────────
    if st.session_state.running and st.session_state.frame_history:
        latest = st.session_state.frame_history[-1] if st.session_state.frame_history else {}
        if latest.get("swarm_detected"):
            st.error("⚠️ SWARM DETECTED — Multiple coordinated objects identified")

    # ── HIGH THREAT ALERT ─────────────────────────────────────────────────────
    if st.session_state.running and st.session_state.frame_history:
        latest = st.session_state.frame_history[-1] if st.session_state.frame_history else {}
        for track in latest.get("tracks", []):
            if track.get("threat_level") == "HIGH":
                st.error(
                    f"🔴 HIGH THREAT — Track {track.get('track_id')} "
                    f"({track.get('class_name')}) "
                    f"score: {track.get('threat_score', 0):.2f}"
                )

    # ── SOURCE PROCESSING ─────────────────────────────────────────────────────
    if source == "Webcam" and st.session_state.running:
        # Initialize capture if not started
        if "cap" not in st.session_state:
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.frame_idx = 0
            create_session(st.session_state.session_id, "webcam")

        cap = st.session_state.cap
        ret, frame = cap.read()

        if ret:
            result = process_frame_with_logging(
                frame, st.session_state.frame_idx, explain,
                st.session_state.session_id, st.session_state.pipeline
            )
            st.session_state.frame_idx += 1

            # Store in history
            st.session_state.frame_history.append(result)
            st.session_state.fps_history.append(result["fps"])

            # Update displays
            annotated = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated, channels="RGB", use_container_width=True)

            # Modality badge display
            modality = result.get("modality", "RGB")
            mod_conf = result.get("modality_confidence", 1.0)
            badge = "🔵" if modality == "RGB" else "🟠" if modality == "THERMAL" else "⚪"
            modality_badge.markdown(
                f"{badge} **{modality}** ({mod_conf:.0%} confidence)"
            )

            # Metrics with color coding
            fps = result["fps"]
            latency = result["latency_ms"]
            fps_metric.metric("FPS", f"{fps:.1f}", delta_color=get_fps_color(fps))
            latency_metric.metric("Latency", f"{latency:.1f}ms", delta_color=get_latency_color(latency))
            count_metric.metric("Objects", len(result["tracks"]))

            # Count HIGH threats
            high_count = sum(1 for t in result.get("tracks", []) if t.get("threat_level") == "HIGH")
            threat_metric.metric("🔴 HIGH", high_count)

            # Update panels
            update_gauges(threat_gauges, result["tracks"])
            update_object_table(object_table, result["tracks"])
            update_shap_panel(shap_container, result["tracks"], st.session_state.pipeline.threat_scorer)
            update_map(
                map_placeholder, result["tracks"],
                result.get("protected_zones"),
                st.session_state.track_trajectories
            )
            update_charts(fps_chart, threat_chart, result["tracks"])

        # Trigger rerun for next frame
        time.sleep(0.033)
        st.rerun()

    elif source == "Video File":
        video_file = st.file_uploader("Upload video", type=["mp4", "avi"])
        if video_file is not None:
            # Save uploaded file to temp location once
            if "video_temp_path" not in st.session_state:
                suffix = Path(video_file.name).suffix
                st.session_state.video_filename = video_file.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(video_file.read())
                    st.session_state.video_temp_path = Path(tmp.name)
                st.session_state.video_cap = cv2.VideoCapture(str(st.session_state.video_temp_path))
                st.session_state.frame_idx = 0
                total = int(st.session_state.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                st.session_state.video_total_frames = total

                # Initialize video writer if saving
                if st.session_state.save_video:
                    fourcc = cv2.VideoWriter_fourcc(*"avc1")
                    width = int(st.session_state.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(st.session_state.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps_raw = st.session_state.video_cap.get(cv2.CAP_PROP_FPS)
                    fps_write = fps_raw if fps_raw and fps_raw > 0 else 25.0
                    out_path = Path(tempfile.gettempdir()) / f"annotated_{st.session_state.session_id}.mp4"
                    st.session_state.video_writer = cv2.VideoWriter(
                        str(out_path), fourcc, fps_write, (width, height)
                    )
                    # Store path now so download button finds same file later
                    st.session_state.video_out_path = str(out_path)

                # Create session in DB
                create_session(st.session_state.session_id, video_file.name)

            if st.session_state.running:
                cap = st.session_state.video_cap
                ret, frame = cap.read()

                if ret:
                    result = process_frame_with_logging(
                        frame, st.session_state.frame_idx, explain,
                        st.session_state.session_id, st.session_state.pipeline
                    )
                    st.session_state.frame_idx += 1

                    # Store in history
                    st.session_state.frame_history.append(result)
                    st.session_state.fps_history.append(result["fps"])

                    # Update displays
                    annotated = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
                    video_placeholder.image(annotated, channels="RGB", use_container_width=True)

                    # Modality badge display
                    modality = result.get("modality", "RGB")
                    mod_conf = result.get("modality_confidence", 1.0)
                    badge = "🔵" if modality == "RGB" else "🟠" if modality == "THERMAL" else "⚪"
                    modality_badge.markdown(
                        f"{badge} **{modality}** ({mod_conf:.0%} confidence)"
                    )

                    # Write to video file if saving
                    if st.session_state.get("video_writer") is not None:
                        try:
                            st.session_state.video_writer.write(result["annotated_frame"])
                        except Exception as e:
                            logger.warning(f"Video writer failed: {e}")

                    # Metrics
                    fps = result["fps"]
                    latency = result["latency_ms"]
                    fps_metric.metric("FPS", f"{fps:.1f}", delta_color=get_fps_color(fps))
                    latency_metric.metric("Latency", f"{latency:.1f}ms", delta_color=get_latency_color(latency))
                    count_metric.metric("Objects", len(result["tracks"]))

                    high_count = sum(1 for t in result.get("tracks", []) if t.get("threat_level") == "HIGH")
                    threat_metric.metric("🔴 HIGH", high_count)

                    # Update panels
                    update_gauges(threat_gauges, result["tracks"])
                    update_object_table(object_table, result["tracks"])
                    update_shap_panel(shap_container, result["tracks"], st.session_state.pipeline.threat_scorer)

                    # Update trajectories
                    for t in result.get("tracks", []):
                        tid = t.get("track_id")
                        lat, lon = t.get("lat", 0), t.get("lon", 0)
                        if lat != 0 or lon != 0:
                            if tid not in st.session_state.track_trajectories:
                                st.session_state.track_trajectories[tid] = deque(maxlen=20)
                            st.session_state.track_trajectories[tid].append((lat, lon))

                    update_map(
                        map_placeholder, result["tracks"],
                        result.get("protected_zones"),
                        st.session_state.track_trajectories
                    )
                    update_charts(fps_chart, threat_chart, result["tracks"])

                    # Progress bar
                    total = st.session_state.get("video_total_frames", 0)
                    if total > 0:
                        progress = min(st.session_state.frame_idx / total, 1.0)
                        st.progress(progress, text=f"Frame {st.session_state.frame_idx}/{total}")

                    time.sleep(0.033)
                    st.rerun()

                else:
                    # Video ended
                    st.session_state.running = False
                    st.session_state.session_complete = True
                    st.info("Video processing complete.")

                    # Calculate final stats
                    avg_fps = sum(st.session_state.fps_history) / len(st.session_state.fps_history) if st.session_state.fps_history else 0
                    results = st.session_state.results_log
                    stats = {
                        "total_frames": st.session_state.frame_idx,
                        "avg_fps": avg_fps,
                        "avg_latency": sum(r.get("latency_ms", 0) for r in results) / len(results) if results else 0,
                        "total_detections": len(results),
                        "high_threats": sum(1 for r in results if r.get("threat_level") == "HIGH"),
                        "medium_threats": sum(1 for r in results if r.get("threat_level") == "MEDIUM"),
                        "low_threats": sum(1 for r in results if r.get("threat_level") == "LOW"),
                        "swarm_events": sum(1 for r in st.session_state.event_log if r.get("event_type") == "SWARM_DETECTED"),
                        "zone_violations": sum(1 for r in results if r.get("zone_violation")),
                    }
                    st.session_state.final_stats = stats

                    # Close session in database
                    close_session(st.session_state.session_id, stats)

                    # Release video writer
                    if st.session_state.video_writer:
                        st.session_state.video_writer.release()
                        st.session_state.video_writer = None

                    st.rerun()

            elif not st.session_state.running:
                # Cleanup video resources when stopped
                if "video_cap" in st.session_state:
                    st.session_state.video_cap.release()
                    del st.session_state.video_cap
                if "video_temp_path" in st.session_state:
                    try:
                        st.session_state.video_temp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    del st.session_state.video_temp_path

    elif source == "Image":
        image_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            # Process single image
            img = Image.open(image_file)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            result = st.session_state.pipeline.process_frame(frame, 0, explain)

            annotated = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated, channels="RGB", use_container_width=True)

            update_gauges(threat_gauges, result["tracks"])
            update_object_table(object_table, result["tracks"])
            update_shap_panel(shap_container, result["tracks"], st.session_state.pipeline.threat_scorer)
            update_map(map_placeholder, result["tracks"], result.get("protected_zones"))

    elif not st.session_state.running:
        # Release capture when stopped
        if "cap" in st.session_state:
            st.session_state.cap.release()
            del st.session_state.cap

    # ── POST-PROCESSING RESULTS PANEL ─────────────────────────────────────────
    render_results_panel()


if __name__ == "__main__":
    main()
