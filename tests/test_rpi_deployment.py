"""
AI-Powered Precision Pest Management - Raspberry Pi 4 Deployment Reliability Test
File: tests/test_rpi_deployment.py

Verifies all deployment requirements for Raspberry Pi 4:
1. RPiCameraStream initialization and frame retrieval.
2. ONNX edge model loading and structured detection output.
3. DecisionEngine safety interlocks (actuators disabled).
4. PostgreSQL client connectivity with graceful local logging failover.
5. End-to-end edge monitoring session with clean shutdown.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment.raspberry_pi.db_client import PostgresTelemetryClient
from deployment.raspberry_pi.rpi_camera import RPiCameraStream
from src.decision.decision_engine import DecisionEngine
from src.monitoring.realtime_detector import RealtimeDetector

ONNX_MODEL = PROJECT_ROOT / "models" / "pest_yolov8n_best.onnx"


def test_rpi_camera_stream() -> None:
    print("[TEST 1/5] Testing RPiCameraStream...")
    cam = RPiCameraStream(source=0, width=640, height=480, fps=30, use_picam2=False)
    assert cam.is_opened(), "RPiCameraStream must be open or in synthetic fallback mode."
    ret, frame = cam.read_frame()
    assert ret is True and frame is not None, "Failed to read valid camera frame."
    assert frame.shape == (480, 640, 3), f"Unexpected frame shape: {frame.shape}"
    cam.release()
    print("  -> PASS: RPiCameraStream initialized and released cleanly.")


def test_onnx_edge_inference() -> None:
    print("[TEST 2/5] Testing ONNX Edge Model Inference...")
    assert ONNX_MODEL.exists(), f"ONNX model missing: {ONNX_MODEL}"
    detector = RealtimeDetector(weights_path=str(ONNX_MODEL), device="cpu", conf_threshold=0.25)
    sample_frame = np.full((480, 640, 3), (34, 110, 34), dtype=np.uint8)
    packet = detector.process_frame(sample_frame, frame_id=1)

    assert "timestamp" in packet
    assert "detections" in packet
    assert "inference_latency_ms" in packet
    print(f"  -> PASS: ONNX model loaded and executed (latency: {packet['inference_latency_ms']}ms).")


def test_decision_engine_safety() -> None:
    print("[TEST 3/5] Testing DecisionEngine Edge Safety Interlock...")
    engine = DecisionEngine(min_confidence_threshold=0.55, hardware_safety_lock=True)
    sample_target = {
        "is_baseline_model": False,
        "detections": [{"class_id": 0, "class_name": "army worm", "confidence": 0.92, "bbox": [10, 10, 50, 50]}],
    }
    decision = engine.evaluate(sample_target)
    assert decision["decision"] == "TARGET_OBSERVED_SAFETY_LOCKED"
    assert decision["actuator_command"] == "NO_ACTION_SAFETY_LOCKED"
    assert decision["safety_interlock_engaged"] is True
    print("  -> PASS: DecisionEngine enforces NO_ACTION default safety state.")


def test_db_client_failover() -> None:
    print("[TEST 4/5] Testing PostgreSQL Client with Automatic Local Failover...")
    # Intentionally test unroutable/dummy port to verify failover mechanism
    client = PostgresTelemetryClient(host="127.0.0.1", port=9999, local_log_dir=str(PROJECT_ROOT / "logs"))
    assert client.db_connected is False, "DB should report disconnected on invalid port."

    packet = {
        "timestamp": "2026-10-07T18:45:00Z",
        "frame_id": 5,
        "model_type": "PEST_ONNX",
        "inference_latency_ms": 42.1,
        "detections": [{"class_id": 1, "class_name": "blister beetle", "confidence": 0.85, "bbox": [20, 20, 80, 80]}],
    }
    decision = {"decision": "TARGET_OBSERVED_SAFETY_LOCKED", "reason": "PEST_DETECTED"}

    # Must return False to signal local logger fallback without raising an exception
    success = client.log_detection_event(packet, decision)
    assert success is False, "log_detection_event should trigger local failover return code."
    client.close()
    print("  -> PASS: PostgreSQL failover handled gracefully with zero process crashes.")


def test_rpi_edge_monitor_run() -> None:
    print("[TEST 5/5] Testing RPi Edge Monitor Headless 5-Frame Session...")
    from deployment.raspberry_pi.rpi_edge_monitor import run_rpi_edge_session

    run_rpi_edge_session(
        model_path=str(ONNX_MODEL),
        source="0",
        use_picam2=False,
        conf_threshold=0.25,
        max_frames=5,
        headless=True,
    )
    print("  -> PASS: RPi edge monitoring session executed 5 frames and closed cleanly.")


def main() -> None:
    print("\n" + "=" * 65)
    print("RUNNING RASPBERRY PI 4 DEPLOYMENT RELIABILITY TEST SUITE")
    print("=" * 65)
    test_rpi_camera_stream()
    test_onnx_edge_inference()
    test_decision_engine_safety()
    test_db_client_failover()
    test_rpi_edge_monitor_run()
    print("=" * 65)
    print("[STATUS] ALL 5 RASPBERRY PI DEPLOYMENT TESTS PASSED!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
