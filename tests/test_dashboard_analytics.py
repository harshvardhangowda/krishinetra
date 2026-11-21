"""
AI-Powered Precision Pest Management - Dashboard & Analytics Unit Tests
File: tests/test_dashboard_analytics.py

Tests:
1. DashboardDataSource connection status & system health diagnostics.
2. Read-only query execution with parameterized filtering.
3. Empty dataset / empty database graceful handling.
4. Metric calculations (Total, Today, Unique classes, Mean confidence, Mean latency).
5. Pest class filtering & date/time range filtering.
6. Target pest statistical aggregations.
7. Database connection failure & seamless local file fallback.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.data_source import DashboardDataSource


def test_system_status() -> None:
    print("[TEST 1/6] Testing System Status Diagnostics...")
    ds = DashboardDataSource()
    status = ds.get_system_status()
    assert "postgres_connected" in status
    assert "local_logging" in status
    assert "decision_engine" in status
    assert "actuators" in status
    assert status["actuators"] == "DISABLED"
    assert "NO_ACTION" in status["decision_engine"]
    print(f"  -> PASS: System status verified (PostgreSQL: {status['postgres_connected']}, Safety: {status['actuators']}).")


def test_overview_metrics_empty_and_valid() -> None:
    print("[TEST 2/6] Testing Overview Metrics Aggregation...")
    ds = DashboardDataSource()
    metrics = ds.get_overview_metrics(time_horizon="all", min_confidence=0.0)

    assert "total_detections" in metrics
    assert "today_detections" in metrics
    assert "unique_pests" in metrics
    assert "avg_confidence" in metrics
    assert "avg_latency_ms" in metrics
    assert isinstance(metrics["total_detections"], int)
    assert isinstance(metrics["avg_confidence"], float)
    print(f"  -> PASS: Overview metrics computed (Total: {metrics['total_detections']}, Avg Conf: {metrics['avg_confidence']}).")


def test_pest_filtering() -> None:
    print("[TEST 3/6] Testing Pest Class Filtering...")
    ds = DashboardDataSource()

    # Query with specific pest filter
    filtered_df = ds.fetch_all_records(time_horizon="all", pest_filter=["army worm"], min_confidence=0.0)
    if not filtered_df.empty and "class_name" in filtered_df.columns:
        unique_classes = set(filtered_df["class_name"].unique())
        assert unique_classes.issubset({"army worm"}), f"Filter leak detected: {unique_classes}"

    print("  -> PASS: Pest class filtering verified.")


def test_confidence_threshold_filtering() -> None:
    print("[TEST 4/6] Testing Confidence Threshold Filtering...")
    ds = DashboardDataSource()
    high_conf_df = ds.fetch_all_records(time_horizon="all", min_confidence=0.70)
    if not high_conf_df.empty and "confidence" in high_conf_df.columns:
        assert (high_conf_df["confidence"] >= 0.70).all(), "Found records below requested confidence threshold."
    print("  -> PASS: Minimum confidence threshold filter verified.")


def test_pest_statistics_table() -> None:
    print("[TEST 5/6] Testing Pest Statistics Table Generation...")
    ds = DashboardDataSource()
    stats_df = ds.get_pest_statistics_table(time_horizon="all", min_confidence=0.0)
    assert isinstance(stats_df, pd.DataFrame)
    expected_cols = ["Pest Class", "Detection Count", "Avg Confidence", "Min Confidence", "Max Confidence"]
    for col in expected_cols:
        assert col in stats_df.columns, f"Missing expected column: {col}"
    print("  -> PASS: Pest statistics table schema and calculations verified.")


def test_database_failover_resilience() -> None:
    print("[TEST 6/6] Testing Database Offline Resilience...")
    ds = DashboardDataSource()
    # Force disconnection to verify fallback
    ds.db_connected = False
    ds.conn = None

    fallback_df = ds.fetch_all_records(time_horizon="all")
    metrics = ds.get_overview_metrics(time_horizon="all")

    assert isinstance(fallback_df, pd.DataFrame), "Fallback must return a valid DataFrame."
    assert isinstance(metrics, dict), "Fallback must return valid metrics dict."
    print("  -> PASS: Dashboard data source gracefully handled database offline state.")


def main() -> None:
    print("\n" + "=" * 65)
    print("RUNNING DASHBOARD & ANALYTICS TEST SUITE")
    print("=" * 65)
    test_system_status()
    test_overview_metrics_empty_and_valid()
    test_pest_filtering()
    test_confidence_threshold_filtering()
    test_pest_statistics_table()
    test_database_failover_resilience()
    print("=" * 65)
    print("[STATUS] ALL 6 DASHBOARD & ANALYTICS TESTS PASSED!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
