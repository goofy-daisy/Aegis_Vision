"""STANAG-style PDF report generator for AegisVision v3.

Military-format incident reports with:
- Cover page with metadata
- Situation summary
- Threat intelligence with per-track analysis
- Predictive analysis with intercept warnings
- System performance metrics
- Recommended actions
- Chain of custody
- Annexes with raw data
"""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageBreak, PageTemplate,
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    HRFlowable,
)

from utils.logger import get_logger

logger = get_logger(__name__)


def _draw_footer(canvas, doc, session_id_short):
    """Draw footer on each page."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    footer_text = f"{session_id_short} | Page {doc.page} | UNCLASSIFIED"
    canvas.drawString(2*cm, 1*cm, footer_text)
    canvas.restoreState()


def generate_pdf_report(
    session_id: str,
    video_filename: str,
    detections: List[Dict],
    events: List[Dict],
    stats: Dict,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate a STANAG-style PDF incident report.

    Args:
        session_id: Unique session identifier.
        video_filename: Name of the processed video file.
        detections: List of detection dictionaries.
        events: List of event dictionaries.
        stats: Statistics dictionary from session.
        output_path: Optional path for output PDF.

    Returns:
        Path to the generated PDF file.
    """
    if output_path is None:
        output_path = Path(f"AegisVision_Report_{session_id[:8]}.pdf")

    # Create document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1f1f1f"),
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.HexColor("#2c3e50"),
    )
    normal_style = styles["Normal"]
    normal_style.fontSize = 10

    story = []
    session_id_short = session_id[:8].upper()

    # ── COVER PAGE ─────────────────────────────────────────────────────────────
    story.append(Paragraph("🛡️ AEGISVISION", title_style))
    story.append(Paragraph(
        "Autonomous Multi-Domain Threat Detection System — v3.0",
        ParagraphStyle("Subtitle", parent=normal_style, alignment=TA_CENTER, spaceAfter=20)
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2c3e50")))
    story.append(Spacer(1, 20))

    # Metadata table
    now = datetime.now(timezone.utc)
    dtg = now.strftime("%d%H%MZ %b %Y").upper()

    meta_data = [
        ["Report Number", f"REPORT-{session_id_short}"],
        ["Date-Time Group", dtg],
        ["Originator", "AegisVision Threat Detection System"],
        ["Classification", "UNCLASSIFIED // FOR DEMONSTRATION PURPOSES ONLY"],
        ["Source File", video_filename],
    ]
    meta_table = Table(meta_data, colWidths=[6*cm, 10*cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ecf0f1")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(PageBreak())

    # ── SECTION 1: SITUATION SUMMARY ─────────────────────────────────────────
    story.append(Paragraph("1. SITUATION SUMMARY", heading_style))

    # Count metrics
    high_threats = stats.get("high_threats", 0)
    medium_threats = stats.get("medium_threats", 0)
    low_threats = stats.get("low_threats", 0)
    zone_violations = stats.get("zone_violations", 0)
    unregistered = sum(1 for d in detections if d.get("adsb_status") == "UNREGISTERED")
    intercepts = sum(1 for d in detections if d.get("intercept_predicted"))

    # Overall assessment
    if high_threats > 0 or zone_violations > 0:
        assessment = "HIGH"
        assess_color = colors.red
    elif medium_threats > 0 or unregistered > 0 or intercepts > 0:
        assessment = "MEDIUM"
        assess_color = colors.orange
    else:
        assessment = "LOW"
        assess_color = colors.green

    story.append(Paragraph(
        f"<b>Overall Threat Assessment:</b> ",
        normal_style
    ))
    story.append(Paragraph(
        f'<font color="{assess_color.hexval()}"><b>{assessment}</b></font>',
        ParagraphStyle("Assessment", parent=normal_style, fontSize=16, spaceAfter=12)
    ))

    # Stats table
    stats_data = [
        ["Metric", "Value"],
        ["Total Frames", str(stats.get("total_frames", 0))],
        ["Total Detections", str(len(detections))],
        ["Unique Objects", str(len(set(d.get("track_id") for d in detections)))],
        ["Zone Violations", str(zone_violations)],
        ["Swarm Events", str(stats.get("swarm_events", 0))],
        ["Unregistered ADS-B Objects", str(unregistered)],
        ["Intercept Warnings", str(intercepts)],
    ]
    stats_table = Table(stats_data, colWidths=[8*cm, 8*cm])
    stats_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 20))

    # ── SECTION 2: THREAT INTELLIGENCE ───────────────────────────────────────
    story.append(Paragraph("2. THREAT INTELLIGENCE", heading_style))

    # Group detections by track_id
    tracks = {}
    for d in detections:
        tid = d.get("track_id")
        if tid not in tracks:
            tracks[tid] = []
        tracks[tid].append(d)

    for tid, track_dets in sorted(tracks.items()):
        # Get max values
        max_threat = max(d.get("threat_score", 0) for d in track_dets)
        max_level = max((d.get("threat_level", "LOW") for d in track_dets),
                       key=lambda x: {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(x, 0))
        class_name = track_dets[0].get("class_name", "-")
        behaviour = Counter(d.get("behaviour", "STATIONARY") for d in track_dets).most_common(1)[0][0]
        sig_name = track_dets[0].get("signature_name", "Unknown")
        sig_conf = track_dets[0].get("signature_confidence", 0)
        adsb_status = track_dets[0].get("adsb_status", "NO_DATA")
        callsign = track_dets[0].get("adsb_callsign")
        zone_viol = any(d.get("zone_violation") for d in track_dets)

        # Track subsection
        story.append(Paragraph(f"<b>Track {tid}</b> — {class_name}", heading_style))

        track_data = [
            ["Attribute", "Value"],
            ["Max Threat Score", f"{max_threat:.3f}"],
            ["Threat Level", max_level],
            ["Behaviour", behaviour],
            ["Signature Match", f"{sig_name} ({sig_conf:.0%})"],
            ["ADS-B Status", f"{adsb_status}" + (f" — {callsign}" if callsign else "")],
            ["Frames Tracked", str(len(track_dets))],
            ["Zone Violation", "YES" if zone_viol else "NO"],
        ]
        track_table = Table(track_data, colWidths=[6*cm, 10*cm])
        track_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7f8c8d")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(track_table)

        # ADS-B alerts
        if adsb_status == "UNREGISTERED":
            story.append(Paragraph(
                "⚠ <b>UNREGISTERED AERIAL OBJECT</b> — No ADS-B transponder detected. "
                "Manual identification required.",
                ParagraphStyle("Alert", parent=normal_style, textColor=colors.red, spaceAfter=6)
            ))
        elif adsb_status == "MATCHED" and callsign:
            story.append(Paragraph(
                f"✓ <b>REGISTERED</b> — Callsign: {callsign}",
                ParagraphStyle("Registered", parent=normal_style, textColor=colors.green, spaceAfter=6)
            ))

        # Counterfactual
        cf = track_dets[0].get("counterfactual")
        if cf:
            story.append(Paragraph(f"<i>Insight:</i> {cf}", normal_style))

        story.append(Spacer(1, 10))

    story.append(PageBreak())

    # ── SECTION 3: PREDICTIVE ANALYSIS ───────────────────────────────────────
    story.append(Paragraph("3. PREDICTIVE ANALYSIS", heading_style))

    intercept_detections = [d for d in detections if d.get("intercept_predicted")]
    if intercept_detections:
        story.append(Paragraph(
            "The following tracks show predicted zone entry within the analysis horizon:",
            normal_style
        ))
        story.append(Spacer(1, 10))

        pred_data = [["Track ID", "Zone", "Time-to-Intercept", "Confidence"]]
        for d in intercept_detections:
            pred_data.append([
                str(d.get("track_id", "-")),
                d.get("intercept_zone", "Unknown"),
                f"{d.get('time_to_intercept_s', 0):.0f}s",
                f"{d.get('prediction_confidence', 0):.0%}",
            ])

        pred_table = Table(pred_data, colWidths=[3*cm, 5*cm, 4*cm, 4*cm])
        pred_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e74c3c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(pred_table)
    else:
        story.append(Paragraph(
            f"No intercept threats detected within {stats.get('horizon_seconds', 30)}s prediction horizon.",
            normal_style
        ))

    story.append(Spacer(1, 15))
    story.append(Paragraph(
        "<i>Disclaimer:</i> Predictions based on kinematic extrapolation assuming constant velocity. "
        "Accuracy dependent on sensor calibration and environmental conditions.",
        ParagraphStyle("Disclaimer", parent=normal_style, fontSize=8, textColor=colors.grey)
    ))
    story.append(PageBreak())

    # ── SECTION 4: SYSTEM PERFORMANCE ──────────────────────────────────────────
    story.append(Paragraph("4. SYSTEM PERFORMANCE", heading_style))

    avg_fps = stats.get("avg_fps", 0)
    perf_data = [
        ["Metric", "Value"],
        ["Average FPS", f"{avg_fps:.1f} ({'NOMINAL' if avg_fps > 5 else 'DEGRADED'})"],
        ["Average Latency", f"{stats.get('avg_latency', 0):.1f} ms"],
        ["Total Frames Processed", str(stats.get("total_frames", 0))],
        ["Modality Detected", stats.get("modality", "RGB")],
        ["ADS-B Integration", "ACTIVE" if unregistered > 0 or sum(1 for d in detections if d.get("adsb_status") != "NO_DATA") > 0 else "OFFLINE"],
        ["Model", "YOLOv8n — AegisVision v3.0"],
        ["Dataset", "VisDrone2019-DET + DOTA-v1.0 + DroneVehicle"],
    ]
    perf_table = Table(perf_data, colWidths=[8*cm, 8*cm])
    perf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
    ]))
    story.append(perf_table)
    story.append(PageBreak())

    # ── SECTION 5: RECOMMENDED ACTIONS ─────────────────────────────────────────
    story.append(Paragraph("5. RECOMMENDED ACTIONS", heading_style))

    # Determine action level
    if zone_violations > 0 or stats.get("swarm_events", 0) > 0:
        action_level = "INTERCEPT"
        action_text = (
            "INTERCEPT: Immediate response authorised under current ROE. "
            "All available assets to be directed to affected zone. "
            "Preserve all sensor data for chain of custody."
        )
        action_color = colors.red
    elif high_threats > 0 or intercepts > 0:
        action_level = "ALERT"
        action_text = (
            "ALERT: Notify response team immediately. "
            "Activate countermeasure protocols. "
            "Log all HIGH-threat track IDs for post-incident review."
        )
        action_color = colors.orange
    elif medium_threats > 0 or unregistered > 0:
        action_level = "TRACK"
        action_text = (
            "TRACK: Assign dedicated tracking resource to identified objects. "
            "Prepare response team for possible escalation. "
            "Verify ADS-B cross-reference with secondary source."
        )
        action_color = colors.HexColor("#f39c12")
    else:
        action_level = "MONITOR"
        action_text = (
            "MONITOR: Continue surveillance. No immediate action required. "
            "Review detection logs at next scheduled interval."
        )
        action_color = colors.green

    story.append(Paragraph(
        f'<font color="{action_color.hexval()}"><b>{action_level}</b></font>',
        ParagraphStyle("ActionLevel", parent=normal_style, fontSize=18, spaceAfter=10)
    ))
    story.append(Paragraph(action_text, normal_style))
    story.append(PageBreak())

    # ── SECTION 6: CHAIN OF CUSTODY ───────────────────────────────────────────
    story.append(Paragraph("6. CHAIN OF CUSTODY", heading_style))

    custody_data = [
        ["Field", "Value"],
        ["Session ID", session_id],
        ["Processing Timestamp", now.strftime("%Y-%m-%d %H:%M:%S UTC")],
        ["Model Version", "YOLOv8n — AegisVision v3.0"],
        ["Training Dataset", "VisDrone2019-DET + DOTA-v1.0 + DroneVehicle"],
        ["Pipeline Version", "YOLOv8 → DeepSORT → TelemetrySim → ThreatScorer v3"],
        ["ADS-B Source", "OpenSky Network (opensky-network.org)"],
        ["Classification", "UNCLASSIFIED // FOR DEMONSTRATION PURPOSES ONLY"],
    ]
    custody_table = Table(custody_data, colWidths=[6*cm, 10*cm])
    custody_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ecf0f1")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(custody_table)
    story.append(PageBreak())

    # ── ANNEX A: DETECTION DATA ───────────────────────────────────────────────
    story.append(Paragraph("ANNEX A — DETECTION DATA", heading_style))

    if detections:
        # Limit to 200 rows for PDF size
        det_subset = detections[:200]
        det_data = [["Frame", "Track ID", "Class", "Score", "Level", "Behaviour", "ADS-B", "Signature", "Intercept"]]
        for d in det_subset:
            det_data.append([
                str(d.get("frame_idx", "-")),
                str(d.get("track_id", "-")),
                d.get("class_name", "-"),
                f"{d.get('threat_score', 0):.2f}",
                d.get("threat_level", "LOW"),
                d.get("behaviour", "STATIONARY"),
                d.get("adsb_status", "NO_DATA"),
                d.get("signature_name", "Unknown")[:15],
                f"{d.get('time_to_intercept_s', 0):.0f}s" if d.get("intercept_predicted") else "—",
            ])

        det_table = Table(det_data, colWidths=[1.5*cm, 1.5*cm, 2*cm, 1.5*cm, 1.5*cm, 2*cm, 1.8*cm, 2*cm, 1.5*cm])
        det_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(det_table)

        if len(detections) > 200:
            story.append(Spacer(1, 10))
            story.append(Paragraph(
                f"<i>Note: Showing first 200 of {len(detections)} detections. Full data available in CSV export.</i>",
                ParagraphStyle("Note", parent=normal_style, fontSize=8)
            ))
    else:
        story.append(Paragraph("No detection data recorded.", normal_style))

    story.append(PageBreak())

    # ── ANNEX B: EVENT TIMELINE ───────────────────────────────────────────────
    story.append(Paragraph("ANNEX B — EVENT TIMELINE", heading_style))

    if events:
        event_data = [["Frame", "Timestamp", "Event Type", "Description"]]
        for e in events[:100]:  # Limit to 100 events
            event_data.append([
                str(e.get("frame_idx", "-")),
                e.get("timestamp", "-")[:19],
                e.get("event_type", "-"),
                e.get("description", "-"),
            ])

        event_table = Table(event_data, colWidths=[2*cm, 3.5*cm, 3*cm, 8.5*cm])
        event_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(event_table)
    else:
        story.append(Paragraph("No events recorded in this session.", normal_style))

    # Build PDF
    def _first_page(canvas, doc):
        _draw_footer(canvas, doc, session_id_short)

    def _later_pages(canvas, doc):
        _draw_footer(canvas, doc, session_id_short)

    doc.build(story, onFirstPage=_first_page, onLaterPages=_later_pages)
    logger.info(f"PDF report generated: {output_path}")
    return output_path
