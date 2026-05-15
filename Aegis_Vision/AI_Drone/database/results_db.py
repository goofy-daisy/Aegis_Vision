"""SQLite results database for AegisVision.

Stores all detection events, threat assessments, and session
metadata permanently for historical querying.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "aegisvision.db"


def get_connection() -> sqlite3.Connection:
    """Get database connection, creating DB and tables if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    """Create all tables if they do not exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            video_filename TEXT,
            started_at TEXT,
            ended_at TEXT,
            total_frames INTEGER DEFAULT 0,
            avg_fps REAL DEFAULT 0,
            avg_latency_ms REAL DEFAULT 0,
            total_detections INTEGER DEFAULT 0,
            high_threats INTEGER DEFAULT 0,
            medium_threats INTEGER DEFAULT 0,
            low_threats INTEGER DEFAULT 0,
            swarm_events INTEGER DEFAULT 0,
            zone_violations INTEGER DEFAULT 0
        );
        
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            frame_idx INTEGER,
            track_id INTEGER,
            class_name TEXT,
            threat_score REAL,
            threat_level TEXT,
            behaviour TEXT,
            speed_mps REAL,
            proximity_m REAL,
            lat REAL,
            lon REAL,
            zone_violation INTEGER DEFAULT 0,
            zone_name TEXT,
            fps REAL,
            latency_ms REAL,
            timestamp TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
        
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            frame_idx INTEGER,
            event_type TEXT,
            description TEXT,
            timestamp TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
    """)
    conn.commit()


def create_session(session_id: str, video_filename: str) -> None:
    """Create a new processing session record."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO sessions 
               (session_id, video_filename, started_at) 
               VALUES (?, ?, ?)""",
            (session_id, video_filename, datetime.now().isoformat())
        )
        conn.commit()
        logger.info(f"Session created: {session_id}")
    finally:
        conn.close()


def log_detection(session_id: str, frame_idx: int,
                  track: Dict, fps: float, latency_ms: float) -> None:
    """Log a single detection event to the database."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO detections 
               (session_id, frame_idx, track_id, class_name, 
                threat_score, threat_level, behaviour, speed_mps,
                proximity_m, lat, lon, zone_violation, zone_name,
                fps, latency_ms, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id, frame_idx,
                track.get("track_id", 0),
                track.get("class_name", "unknown"),
                track.get("threat_score", 0.0),
                track.get("threat_level", "LOW"),
                track.get("behaviour", "STATIONARY"),
                track.get("speed_mps", 0.0),
                track.get("proximity_m", 0.0),
                track.get("lat", 0.0),
                track.get("lon", 0.0),
                1 if track.get("zone_violation", False) else 0,
                track.get("zone_name", ""),
                fps, latency_ms,
                datetime.now().isoformat()
            )
        )
        conn.commit()
    finally:
        conn.close()


def log_event(session_id: str, frame_idx: int,
              event_type: str, description: str) -> None:
    """Log a system event (swarm, zone violation, HIGH threat)."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO events 
               (session_id, frame_idx, event_type, description, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, frame_idx, event_type,
             description, datetime.now().isoformat())
        )
        conn.commit()
    finally:
        conn.close()


def close_session(session_id: str, stats: Dict) -> None:
    """Update session record with final statistics."""
    conn = get_connection()
    try:
        conn.execute(
            """UPDATE sessions SET
               ended_at = ?,
               total_frames = ?,
               avg_fps = ?,
               avg_latency_ms = ?,
               total_detections = ?,
               high_threats = ?,
               medium_threats = ?,
               low_threats = ?,
               swarm_events = ?,
               zone_violations = ?
               WHERE session_id = ?""",
            (
                datetime.now().isoformat(),
                stats.get("total_frames", 0),
                stats.get("avg_fps", 0),
                stats.get("avg_latency", 0),
                stats.get("total_detections", 0),
                stats.get("high_threats", 0),
                stats.get("medium_threats", 0),
                stats.get("low_threats", 0),
                stats.get("swarm_events", 0),
                stats.get("zone_violations", 0),
                session_id
            )
        )
        conn.commit()
        logger.info(f"Session closed: {session_id}")
    finally:
        conn.close()


def get_session_summary(session_id: str) -> Optional[Dict]:
    """Get summary for a specific session."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_sessions() -> List[Dict]:
    """Get all sessions ordered by most recent first."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_session_detections(session_id: str) -> List[Dict]:
    """Get all detections for a session."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT * FROM detections 
               WHERE session_id = ? 
               ORDER BY frame_idx""",
            (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_session_events(session_id: str) -> List[Dict]:
    """Get all events for a session."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT * FROM events 
               WHERE session_id = ? 
               ORDER BY frame_idx""",
            (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
