"""
AI-Powered Precision Pest Management - Telemetry and Event Logger
File: src/monitoring/logger.py

Records structured detection and decision events into JSONL and CSV formats.
Does NOT store raw image frames by default to prevent storage saturation.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class DetectionEventLogger:
    """Logs structured telemetry for agricultural pest detection analysis."""

    def __init__(
        self,
        log_dir: Optional[str] = None,
        session_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the event logger.

        Args:
            log_dir: Directory where log files are stored (default: 'logs').
            session_name: Optional custom prefix for log files.
        """
        if log_dir is None:
            self.log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
        else:
            self.log_dir = Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)

        datestr = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{session_name}_{datestr}" if session_name else f"detection_events_{datestr}"

        self.jsonl_path = self.log_dir / f"{prefix}.jsonl"
        self.csv_path = self.log_dir / f"{prefix}.csv"

        # Initialize CSV header
        self._init_csv()
        print(f"[Logger] Structured logs will be written to:\n  JSONL: {self.jsonl_path}\n  CSV:   {self.csv_path}")

    def _init_csv(self) -> None:
        """Create CSV file with column headers."""
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "frame_id",
                        "model_type",
                        "class_id",
                        "class_name",
                        "confidence",
                        "bbox_x1",
                        "bbox_y1",
                        "bbox_x2",
                        "bbox_y2",
                        "decision",
                        "decision_reason",
                        "inference_latency_ms",
                    ]
                )

    def log_event(
        self,
        detection_packet: Dict[str, Any],
        decision_result: Dict[str, Any],
    ) -> None:
        """
        Log a detection and decision event to both JSONL and CSV if detections are present.
        """
        detections = detection_packet.get("detections", [])
        if not detections:
            return  # Do not flood log files when nothing is detected

        timestamp = detection_packet.get("timestamp", datetime.utcnow().isoformat() + "Z")
        frame_id = detection_packet.get("frame_id", 0)
        model_type = detection_packet.get("model_type", "UNKNOWN")
        latency = detection_packet.get("inference_latency_ms", 0.0)
        decision = decision_result.get("decision", "NO_ACTION")
        reason = decision_result.get("reason", "NONE")

        # 1. Write JSONL entry
        jsonl_record = {
            "timestamp": timestamp,
            "frame_id": frame_id,
            "model_type": model_type,
            "inference_latency_ms": latency,
            "decision": decision_result,
            "detections": detections,
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as jf:
            jf.write(json.dumps(jsonl_record) + "\n")

        # 2. Write CSV rows (one row per detected bounding box)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            for d in detections:
                bbox = d.get("bbox", [0, 0, 0, 0])
                writer.writerow(
                    [
                        timestamp,
                        frame_id,
                        model_type,
                        d.get("class_id", -1),
                        d.get("class_name", "unknown"),
                        d.get("confidence", 0.0),
                        bbox[0] if len(bbox) > 0 else 0,
                        bbox[1] if len(bbox) > 1 else 0,
                        bbox[2] if len(bbox) > 2 else 0,
                        bbox[3] if len(bbox) > 3 else 0,
                        decision,
                        reason,
                        latency,
                    ]
                )
