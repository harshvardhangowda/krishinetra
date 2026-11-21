"""
AI-Powered Precision Pest Management - PostgreSQL Telemetry Client
File: deployment/raspberry_pi/db_client.py

Handles telemetry insertion into PostgreSQL database with automatic failover
to local JSONL/CSV logging when database connectivity is unavailable.

Configured exclusively via environment variables:
- PG_HOST (default: 'localhost')
- PG_PORT (default: 5432)
- PG_DATABASE (default: 'krishnetra_pest_db')
- PG_USER (default: 'postgres')
- PG_PASSWORD (default: '')
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Optional psycopg2 import
try:
    import psycopg2
    from psycopg2 import sql
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


class PostgresTelemetryClient:
    """Manages PostgreSQL connection with failover to local storage."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        dbname: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        local_log_dir: Optional[str] = None,
    ) -> None:
        self.host = host or os.getenv("PG_HOST", "localhost")
        self.port = int(port or os.getenv("PG_PORT", "5432"))
        self.dbname = dbname or os.getenv("PG_DATABASE", "krishnetra_pest_db")
        self.user = user or os.getenv("PG_USER", "postgres")
        self.password = password or os.getenv("PG_PASSWORD", "")

        self.conn = None
        self.db_connected = False
        self.local_log_dir = Path(local_log_dir) if local_log_dir else Path("logs")
        self.local_log_dir.mkdir(parents=True, exist_ok=True)

        self._connect()

    def _connect(self) -> None:
        """Attempt connection to PostgreSQL and initialize table if needed."""
        if not PSYCOPG2_AVAILABLE:
            print("[DBClient] Notice: psycopg2 not installed. Operating in Local Fallback Mode.")
            self.db_connected = False
            return

        try:
            print(f"[DBClient] Connecting to PostgreSQL at {self.host}:{self.port}/{self.dbname}...")
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                connect_timeout=3,
            )
            self.conn.autocommit = True
            self._create_table_if_not_exists()
            self.db_connected = True
            print("[DBClient] PostgreSQL connection established successfully.")
        except Exception as e:
            print(f"[DBClient] Warning: PostgreSQL unavailable ({e}). Operating in Local Fallback Mode.")
            self.conn = None
            self.db_connected = False

    def _create_table_if_not_exists(self) -> None:
        """Create pest detection telemetry schema if not present."""
        if not self.conn:
            return
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS pest_detections (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            frame_id BIGINT,
            model_type VARCHAR(64),
            class_id INT,
            class_name VARCHAR(64),
            confidence REAL,
            bbox_x1 REAL,
            bbox_y1 REAL,
            bbox_x2 REAL,
            bbox_y2 REAL,
            decision VARCHAR(64),
            decision_reason TEXT,
            inference_latency_ms REAL,
            node_id VARCHAR(64) DEFAULT 'RPI4_EDGE_01'
        );
        """
        with self.conn.cursor() as cur:
            cur.execute(create_table_sql)

    def log_detection_event(
        self,
        detection_packet: Dict[str, Any],
        decision_result: Dict[str, Any],
        node_id: str = "RPI4_EDGE_01",
    ) -> bool:
        """
        Insert detection records into PostgreSQL. If disconnected, fallback to local logger.
        """
        detections = detection_packet.get("detections", [])
        if not detections:
            return True

        if self.db_connected and self.conn:
            try:
                timestamp = detection_packet.get("timestamp", datetime.utcnow().isoformat() + "Z")
                frame_id = detection_packet.get("frame_id", 0)
                model_type = detection_packet.get("model_type", "PEST_YOLO")
                latency = detection_packet.get("inference_latency_ms", 0.0)
                decision = decision_result.get("decision", "NO_ACTION")
                reason = decision_result.get("reason", "NONE")

                insert_sql = """
                INSERT INTO pest_detections (
                    timestamp, frame_id, model_type, class_id, class_name,
                    confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    decision, decision_reason, inference_latency_ms, node_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """

                with self.conn.cursor() as cur:
                    for det in detections:
                        bbox = det.get("bbox", [0, 0, 0, 0])
                        cur.execute(
                            insert_sql,
                            (
                                timestamp,
                                frame_id,
                                model_type,
                                det.get("class_id", -1),
                                det.get("class_name", "unknown"),
                                float(det.get("confidence", 0.0)),
                                float(bbox[0]),
                                float(bbox[1]),
                                float(bbox[2]),
                                float(bbox[3]),
                                decision,
                                reason,
                                float(latency),
                                node_id,
                            ),
                        )
                return True
            except Exception as e:
                print(f"[DBClient] Database insert failed ({e}). Reverting to local fallback...")
                self.db_connected = False
                if self.conn:
                    try:
                        self.conn.close()
                    except Exception:
                        pass
                self.conn = None

        return False  # Caller should log to local file

    def close(self) -> None:
        """Cleanly close database connection."""
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
            self.db_connected = False
        print("[DBClient] Database client closed cleanly.")
