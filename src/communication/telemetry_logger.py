"""
AI-Powered Precision Pest Management - Communication Telemetry Logger
File: src/communication/telemetry_logger.py

Logs command, ACK, and heartbeat events into PostgreSQL `communication_events` table
with automatic fallback to local JSONL when database is unavailable.
"""

import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


class CommunicationTelemetryLogger:
    """Logs communication protocol events to PostgreSQL or local JSONL fallback."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        dbname: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        local_log_dir: Optional[str] = None,
    ):
        self.host = host or os.getenv("PG_HOST", "localhost")
        self.port = int(port or os.getenv("PG_PORT", "5432"))
        self.dbname = dbname or os.getenv("PG_DATABASE", "krishnetra_pest_db")
        self.user = user or os.getenv("PG_USER", "postgres")
        self.password = password or os.getenv("PG_PASSWORD", "")

        self.conn = None
        self.db_connected = False
        self.local_log_dir = Path(local_log_dir) if local_log_dir else Path("logs")
        self.local_log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_file = self.local_log_dir / "communication_events.jsonl"

        self._connect()

    def _connect(self) -> None:
        """Attempt connection to PostgreSQL and create table schema if present."""
        if not PSYCOPG2_AVAILABLE:
            self.db_connected = False
            return

        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                connect_timeout=2,
            )
            self.conn.autocommit = True
            self._create_table_if_not_exists()
            self.db_connected = True
        except Exception:
            self.conn = None
            self.db_connected = False

    def _create_table_if_not_exists(self) -> None:
        if not self.conn:
            return
        create_sql = """
        CREATE TABLE IF NOT EXISTS communication_events (
            event_id VARCHAR(64) PRIMARY KEY,
            command_id VARCHAR(64),
            timestamp TIMESTAMPTZ NOT NULL,
            source VARCHAR(64) NOT NULL,
            device VARCHAR(64) NOT NULL,
            event_type VARCHAR(64) NOT NULL,
            status VARCHAR(64) NOT NULL,
            pest_class VARCHAR(64),
            confidence REAL,
            decision VARCHAR(64),
            action VARCHAR(64),
            safety_state VARCHAR(32),
            error_message TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """
        with self.conn.cursor() as cur:
            cur.execute(create_sql)

    def log_event(
        self,
        event_type: str,
        status: str,
        command_id: Optional[str] = None,
        source: str = "raspberry_pi",
        device: str = "esp32",
        pest_class: Optional[str] = None,
        confidence: Optional[float] = None,
        decision: Optional[str] = None,
        action: Optional[str] = None,
        safety_state: str = "LOCKED",
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record a communication event to PostgreSQL (or local JSONL if disconnected).
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "event_id": event_id,
            "command_id": command_id or "N/A",
            "timestamp": timestamp,
            "source": source,
            "device": device,
            "event_type": event_type,
            "status": status,
            "pest_class": pest_class or "N/A",
            "confidence": float(confidence) if confidence is not None else 0.0,
            "decision": decision or "NO_ACTION",
            "action": action or "NONE",
            "safety_state": safety_state,
            "error_message": error_message or "None",
        }

        # Try PostgreSQL
        if self.db_connected and self.conn:
            try:
                insert_sql = """
                INSERT INTO communication_events (
                    event_id, command_id, timestamp, source, device,
                    event_type, status, pest_class, confidence, decision,
                    action, safety_state, error_message
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """
                with self.conn.cursor() as cur:
                    cur.execute(
                        insert_sql,
                        (
                            event_id,
                            record["command_id"],
                            timestamp,
                            source,
                            device,
                            event_type,
                            status,
                            record["pest_class"],
                            record["confidence"],
                            record["decision"],
                            record["action"],
                            safety_state,
                            record["error_message"],
                        ),
                    )
                return record
            except Exception:
                self.db_connected = False
                if self.conn:
                    try:
                        self.conn.close()
                    except Exception:
                        pass
                self.conn = None

        # Local fallback JSONL
        with open(self.jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record

    def close(self) -> None:
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
            self.db_connected = False
