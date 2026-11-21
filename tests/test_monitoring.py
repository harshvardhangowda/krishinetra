"""
AI-Powered Precision Pest Management - Monitoring Pipeline Unit Tests
File: tests/test_monitoring.py

Tests:
1. CameraStream initialization & frame acquisition.
2. RealtimeDetector inference output schema & latency benchmarking.
3. DecisionEngine safety interlock & whitelist evaluation.
4. DetectionEventLogger CSV & JSONL serialization.
5. End-to-end headless monitoring session loop.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decision.decision_engine import DecisionEngine
from src.monitoring.camera import CameraStream
from src.monitoring.logger import DetectionEventLogger
from src.monitoring.realtime_detector import RealtimeDetector


def test_camera_stream() -> None:
    print("[TEST 1/5] Testing CameraStream...")
    # Test with synthetic fallback mode enabled
    cam = CameraStream(source=0, width=640, height=480, fps_target=30, fallback_synthetic=True)
    assert cam.is_opened(), "Camera stream should be opened or in synthetic fallback mode."
    ret, frame = cam.read_frame()
    assert ret is True, "read_frame() must return True."
    assert frame is not None, "Frame cannot be None."
    assert frame.shape == (480, 640, 3), f"Frame shape mismatch: {frame.shape}"
    cam.release()
    print("  -> PASS: CameraStream initializes, reads frames, and releases cleanly.")


def test_realtime_detector() -> None:
    print("[TEST 2/5] Testing RealtimeDetector...")
    detector = RealtimeDetector(weights_path="yolov8n.pt", device="cpu", conf_threshold=0.25)
    assert detector.is_baseline is True, "yolov8n.pt must be identified as baseline model."

    # Test synthetic image
    test_frame = np.full((480, 640, 3), (34, 110, 34), dtype=np.uint8)
    packet = detector.process_frame(test_frame, frame_id=1)

    assert "timestamp" in packet, "Missing 'timestamp' in packet."
    assert "detections" in packet, "Missing 'detections' in packet."
    assert "inference_latency_ms" in packet, "Missing 'inference_latency_ms' in packet."
    assert packet["is_baseline_model"] is True, "Packet should tag baseline model status."
    print(f"  -> PASS: Detector returns valid packet (latency: {packet['inference_latency_ms']}ms).")


def test_decision_engine() -> None:
    print("[TEST 3/5] Testing DecisionEngine safety logic...")
    engine = DecisionEngine(min_confidence_threshold=0.60, hardware_safety_lock=True)

    # Test 1: Baseline model must always produce NO_ACTION
    baseline_packet = {
        "is_baseline_model": True,
        "detections": [{"class_id": 0, "class_name": "person", "confidence": 0.95, "bbox": [10, 10, 50, 50]}],
    }
    decision = engine.evaluate(baseline_packet)
    assert decision["decision"] == "NO_ACTION", "Baseline model must strictly yield NO_ACTION."
    assert decision["actuator_command"] is None, "Baseline model must not issue actuator commands."

    # Test 2: Beneficial insect safeguard
    pest_packet_bee = {
        "is_baseline_model": False,
        "detections": [{"class_id": 99, "class_name": "honey bee", "confidence": 0.90, "bbox": [10, 10, 50, 50]}],
    }
    decision_bee = engine.evaluate(pest_packet_bee)
    assert decision_bee["decision"] == "NO_ACTION", "Beneficial insect must yield NO_ACTION."

    # Test 3: Target pest with hardware safety lock engaged
    pest_packet_target = {
        "is_baseline_model": False,
        "detections": [{"class_id": 0, "class_name": "army worm", "confidence": 0.88, "bbox": [10, 10, 50, 50]}],
    }
    decision_target = engine.evaluate(pest_packet_target)
    assert decision_target["decision"] == "TARGET_OBSERVED_SAFETY_LOCKED", "Safety-locked target must be recognized."
    assert decision_target["actuator_command"] == "NO_ACTION_SAFETY_LOCKED"
    print("  -> PASS: DecisionEngine safety interlocks and whitelist logic verified.")


def test_logger() -> None:
    print("[TEST 4/5] Testing DetectionEventLogger...")
    logger = DetectionEventLogger(session_name="test_session")
    sample_packet = {
        "timestamp": "2026-10-07T13:30:00Z",
        "frame_id": 10,
        "model_type": "BASELINE COCO MODEL",
        "inference_latency_ms": 15.2,
        "detections": [{"class_id": 0, "class_name": "person", "confidence": 0.85, "bbox": [10, 20, 100, 200]}],
    }
    sample_decision = {"decision": "NO_ACTION", "reason": "BASELINE_OBSERVATION"}
    logger.log_event(sample_packet, sample_decision)

    assert logger.jsonl_path.exists(), "JSONL log file was not created."
    assert logger.csv_path.exists(), "CSV log file was not created."
    print("  -> PASS: Structured telemetry written to JSONL and CSV.")


def test_live_monitor_headless_run() -> None:
    print("[TEST 5/5] Testing LiveMonitor headless 5-frame execution...")
    from src.monitoring.live_monitor import run_monitoring_session

    run_monitoring_session(
        source="0",
        weights_path="yolov8n.pt",
        conf_threshold=0.25,
        max_frames=5,
        headless=True,
        device="cpu",
    )
    print("  -> PASS: Live monitor executed 5 frames in headless mode and terminated cleanly.")


def main() -> None:
    print("\n" + "=" * 65)
    print("RUNNING MONITORING PIPELINE TEST SUITE")
    print("=" * 65)
    test_camera_stream()
    test_realtime_detector()
    test_decision_engine()
    test_logger()
    test_live_monitor_headless_run()
    print("=" * 65)
    print("[STATUS] ALL 5 MONITORING TESTS PASSED SUCCESSFULLY!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
