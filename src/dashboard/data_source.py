"""
AI-Powered Precision Pest Management - Analytics & Data Access Layer
File: src/dashboard/data_source.py

Provides read-only analytical queries for the Krishnetra Dashboard.
Supports PostgreSQL queries with automatic failover to local JSONL/CSV logs.
100% Parameterized queries with zero database mutation capabilities.
"""

import csv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Optional psycopg2 import
try:
    import psycopg2
    from psycopg2 import sql
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"


class DashboardDataSource:
    """Manages read-only telemetry queries from PostgreSQL or local log files."""

    def __init__(self) -> None:
        self.host = os.getenv("PG_HOST", "localhost")
        self.port = int(os.getenv("PG_PORT", "5432"))
        self.dbname = os.getenv("PG_DATABASE", "krishnetra_pest_db")
        self.user = os.getenv("PG_USER", "postgres")
        self.password = os.getenv("PG_PASSWORD", "")

        self.db_connected = False
        self.conn = None
        self._check_db_connection()

    def _check_db_connection(self) -> None:
        """Attempt read-only connection to PostgreSQL."""
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
            self.conn.set_session(readonly=True)  # Enforce strict read-only session
            self.db_connected = True
        except Exception:
            self.conn = None
            self.db_connected = False

    def get_available_models(self) -> Dict[str, Any]:
        """Check available model checkpoints on disk."""
        pt_path = PROJECT_ROOT / "models" / "pest_yolov8n_best.pt"
        onnx_path = PROJECT_ROOT / "models" / "pest_yolov8n_best.onnx"
        return {
            "pytorch_available": pt_path.exists(),
            "pytorch_path": str(pt_path) if pt_path.exists() else None,
            "onnx_available": onnx_path.exists(),
            "onnx_path": str(onnx_path) if onnx_path.exists() else None,
            "both_available": pt_path.exists() and onnx_path.exists(),
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Return system connection and readiness telemetry."""
        models = self.get_available_models()

        # Dynamically compute AI detector readiness instead of hardcoding
        if models["both_available"]:
            ai_status = "READY (PyTorch & ONNX Models Available)"
        elif models["pytorch_available"]:
            ai_status = "READY (PyTorch pest_yolov8n_best.pt Available)"
        elif models["onnx_available"]:
            ai_status = "READY (ONNX pest_yolov8n_best.onnx Available)"
        else:
            ai_status = "UNAVAILABLE (No Model Weights in models/)"

        return {
            "postgres_connected": self.db_connected,
            "host": f"{self.host}:{self.port}" if self.db_connected else "Offline (Local Logs Active)",
            "database": self.dbname if self.db_connected else "N/A",
            "local_logging": "ACTIVE",
            "ai_detector": ai_status,
            "models_available": models,
            "decision_engine": "NO_ACTION (Safety Interlock Engaged)",
            "actuators": "DISABLED",
        }

    def run_pipeline_benchmark(
        self,
        model_format: str = "PYTORCH",
        source_type: str = "SYNTHETIC",
        source_path: Optional[str] = None,
        max_frames: int = 15,
        warmup_frames: int = 3,
        width: int = 640,
        height: int = 480,
    ) -> Dict[str, Any]:
        """Execute edge AI pipeline profiling and return structured benchmark report dict."""
        from src.detection.detector import PestDetector
        from src.monitoring.video_stream import SafeVideoStream
        from src.monitoring.pipeline_profiler import PipelineProfiler

        fmt = model_format.upper()
        if fmt not in ("ONNX", "PYTORCH", "PT"):
            raise ValueError(f"Unsupported model format: '{model_format}'. Must be 'PYTORCH' or 'ONNX'.")

        models = self.get_available_models()
        if fmt == "ONNX":
            if not models["onnx_available"]:
                raise FileNotFoundError("ONNX model weights not found at models/pest_yolov8n_best.onnx")
            weights = models["onnx_path"]
        else:
            if not models["pytorch_available"]:
                raise FileNotFoundError("PyTorch model weights not found at models/pest_yolov8n_best.pt")
            weights = models["pytorch_path"]

        # Configure video stream safely
        if source_type.upper() == "FILE":
            video_src = source_path or str(PROJECT_ROOT / "tests" / "data" / "sample_pest_clip.mp4")
            stream = SafeVideoStream(source=video_src, loop=True, width=width, height=height, fallback_synthetic=True)
        else:
            stream = SafeVideoStream(source="synthetic", width=width, height=height)

        try:
            detector = PestDetector(weights_path=weights, device="cpu")
            profiler = PipelineProfiler(warmup_frames=warmup_frames, input_width=width, input_height=height)
            report = profiler.profile_pipeline(stream=stream, detector=detector, max_frames=max_frames)
            return report.to_dict()
        finally:
            stream.release()

    def run_model_comparison(
        self,
        source_type: str = "SYNTHETIC",
        source_path: Optional[str] = None,
        num_frames: int = 15,
        warmup_frames: int = 3,
        width: int = 640,
        height: int = 480,
    ) -> Dict[str, Any]:
        """Execute comparative PyTorch vs ONNX benchmark."""
        from src.monitoring.video_stream import SafeVideoStream
        from src.monitoring.pipeline_profiler import PipelineProfiler

        models = self.get_available_models()
        if not models["both_available"]:
            return {
                "error": "Both PyTorch and ONNX models must exist to run comparative benchmark.",
                "pytorch_available": models["pytorch_available"],
                "onnx_available": models["onnx_available"],
            }

        if source_type.upper() == "FILE":
            video_src = source_path or str(PROJECT_ROOT / "tests" / "data" / "sample_pest_clip.mp4")
            stream_factory = lambda: SafeVideoStream(source=video_src, loop=True, width=width, height=height, fallback_synthetic=True)
        else:
            stream_factory = lambda: SafeVideoStream(source="synthetic", width=width, height=height)

        profiler = PipelineProfiler(warmup_frames=warmup_frames, input_width=width, input_height=height)
        comparison = profiler.compare_models(
            pt_path=models["pytorch_path"],
            onnx_path=models["onnx_path"],
            stream_factory=stream_factory,
            num_frames=num_frames,
        )
        return comparison

    def _load_local_telemetry_df(self) -> pd.DataFrame:
        """Parse local CSV log files into a unified pandas DataFrame."""
        if not LOGS_DIR.exists():
            return pd.DataFrame()

        csv_files = sorted(list(LOGS_DIR.glob("*.csv")), reverse=True)
        if not csv_files:
            return pd.DataFrame()

        dfs = []
        for cf in csv_files:
            try:
                df = pd.read_csv(cf)
                dfs.append(df)
            except Exception:
                continue

        if not dfs:
            return pd.DataFrame()

        full_df = pd.concat(dfs, ignore_index=True)
        if "timestamp" in full_df.columns:
            full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], errors="coerce")
            full_df = full_df.dropna(subset=["timestamp"])
            full_df = full_df.sort_values(by="timestamp", ascending=False)
        return full_df

    def fetch_all_records(
        self,
        time_horizon: str = "all",
        pest_filter: Optional[List[str]] = None,
        min_confidence: float = 0.0,
    ) -> pd.DataFrame:
        """
        Fetch filtered records from PostgreSQL or local fallback dataframe.
        """
        now = datetime.utcnow()
        start_time = None

        if time_horizon == "today":
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_horizon == "24h":
            start_time = now - timedelta(hours=24)
        elif time_horizon == "7d":
            start_time = now - timedelta(days=7)
        elif time_horizon == "30d":
            start_time = now - timedelta(days=30)

        # 1. Attempt PostgreSQL Query
        if self.db_connected and self.conn:
            try:
                query = "SELECT * FROM pest_detections WHERE confidence >= %s"
                params: List[Any] = [min_confidence]

                if start_time:
                    query += " AND timestamp >= %s"
                    params.append(start_time)

                if pest_filter and "All" not in pest_filter:
                    query += " AND class_name = ANY(%s)"
                    params.append(pest_filter)

                query += " ORDER BY timestamp DESC LIMIT 5000;"
                df = pd.read_sql_query(query, self.conn, params=tuple(params))
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df
            except Exception:
                self.db_connected = False
                self.conn = None

        # 2. Fallback to Local Log Files
        df = self._load_local_telemetry_df()
        if df.empty:
            return pd.DataFrame()

        # Apply in-memory filters
        if "confidence" in df.columns:
            df = df[df["confidence"] >= min_confidence]

        if start_time and "timestamp" in df.columns:
            df = df[df["timestamp"] >= pd.Timestamp(start_time, tz="UTC" if df["timestamp"].dt.tz else None)]

        if pest_filter and "All" not in pest_filter and "class_name" in df.columns:
            df = df[df["class_name"].isin(pest_filter)]

        return df

    def get_overview_metrics(
        self,
        time_horizon: str = "all",
        pest_filter: Optional[List[str]] = None,
        min_confidence: float = 0.0,
    ) -> Dict[str, Any]:
        """Calculate high-level summary metrics."""
        df = self.fetch_all_records(time_horizon, pest_filter, min_confidence)
        if df.empty:
            return {
                "total_detections": 0,
                "today_detections": 0,
                "unique_pests": 0,
                "avg_confidence": 0.0,
                "avg_latency_ms": 0.0,
            }

        now_utc = pd.Timestamp.utcnow()
        today_start = now_utc.floor("D")

        # Today's count
        if "timestamp" in df.columns:
            ts_col = df["timestamp"]
            if ts_col.dt.tz is None:
                today_count = (ts_col >= today_start.tz_localize(None)).sum()
            else:
                today_count = (ts_col >= today_start).sum()
        else:
            today_count = len(df)

        unique_pests = df["class_name"].nunique() if "class_name" in df.columns else 0
        avg_conf = float(df["confidence"].mean()) if "confidence" in df.columns else 0.0
        avg_lat = float(df["inference_latency_ms"].mean()) if "inference_latency_ms" in df.columns else 0.0

        return {
            "total_detections": int(len(df)),
            "today_detections": int(today_count),
            "unique_pests": int(unique_pests),
            "avg_confidence": round(avg_conf, 4),
            "avg_latency_ms": round(avg_lat, 2),
        }

    def get_pest_statistics_table(
        self,
        time_horizon: str = "all",
        min_confidence: float = 0.0,
    ) -> pd.DataFrame:
        """Generate per-pest aggregation statistics."""
        df = self.fetch_all_records(time_horizon, min_confidence=min_confidence)
        if df.empty or "class_name" not in df.columns:
            return pd.DataFrame(columns=["Pest Class", "Detection Count", "Avg Confidence", "Min Confidence", "Max Confidence"])

        grouped = df.groupby("class_name").agg(
            detection_count=("confidence", "count"),
            avg_confidence=("confidence", "mean"),
            min_confidence=("confidence", "min"),
            max_confidence=("confidence", "max"),
        ).reset_index()

        grouped.columns = ["Pest Class", "Detection Count", "Avg Confidence", "Min Confidence", "Max Confidence"]
        grouped["Avg Confidence"] = grouped["Avg Confidence"].round(4)
        grouped["Min Confidence"] = grouped["Min Confidence"].round(4)
        grouped["Max Confidence"] = grouped["Max Confidence"].round(4)
        return grouped.sort_values(by="Detection Count", ascending=False)
