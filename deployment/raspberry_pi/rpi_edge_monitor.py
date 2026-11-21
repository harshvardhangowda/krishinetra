"""
AI-Powered Precision Pest Management - Raspberry Pi 4 Edge Monitoring Application
File: deployment/raspberry_pi/rpi_edge_monitor.py

Lightweight edge execution pipeline for Raspberry Pi 4:
1. Video Capture (Picamera2 / USB Camera / Video)
2. YOLO Inference (ONNX Runtime / Ultralytics ONNX or PT)
3. Decision Engine (Target Pest Whitelist & Safety Lock)
4. Telemetry Logging (PostgreSQL with automatic local JSONL/CSV failover)
5. Performance Benchmarking (FPS, Latency, Memory)

Safety Constraint: ACTUATORS DISABLED (Observation & Telemetry Only).
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment.raspberry_pi.db_client import PostgresTelemetryClient
from deployment.raspberry_pi.rpi_camera import RPiCameraStream
from src.decision.decision_engine import DecisionEngine
from src.monitoring.logger import DetectionEventLogger
from src.monitoring.realtime_detector import RealtimeDetector


def run_rpi_edge_session(
    model_path: str = "models/pest_yolov8n_best.onnx",
    source: str = "0",
    use_picam2: bool = False,
    conf_threshold: float = 0.35,
    max_frames: int = -1,
    headless: bool = True,
) -> None:
    print("=" * 65)
    print("KRISHNETRA RASPBERRY PI 4 EDGE PEST DETECTOR")
    print(f"Model Path  : {model_path}")
    print(f"Camera Mode : {'Picamera2' if use_picam2 else 'OpenCV / USB'}")
    print("SAFETY LOCK : ENGAGED (Actuators Disabled / Telemetry Only)")
    print("=" * 65)

    resolved_model = PROJECT_ROOT / model_path if not Path(model_path).is_absolute() else Path(model_path)
    if not resolved_model.exists():
        # Fallback to PT if ONNX not found
        pt_fallback = resolved_model.with_suffix(".pt")
        if pt_fallback.exists():
            resolved_model = pt_fallback
            print(f"[RPiEdge] Using PyTorch model fallback: {resolved_model}")
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {resolved_model}")

    # Initialize submodules
    camera = RPiCameraStream(source=source, width=640, height=480, fps=30, use_picam2=use_picam2)
    detector = RealtimeDetector(weights_path=str(resolved_model), device="cpu", conf_threshold=conf_threshold)
    decision_engine = DecisionEngine(min_confidence_threshold=0.55, hardware_safety_lock=True)
    db_client = PostgresTelemetryClient(local_log_dir=str(PROJECT_ROOT / "logs"))
    local_logger = DetectionEventLogger(session_name="rpi4_edge_events")

    frame_count = 0
    t_session_start = time.time()
    latencies = []
    fps = 0.0

    print("\n[RPiEdge] Inference loop running. Processing frames...\n")

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                print("[RPiEdge] End of video source or camera failure.")
                break

            frame_count += 1
            t_frame_start = time.perf_counter()

            # 1. Detection
            packet = detector.process_frame(frame, frame_id=frame_count)
            t_infer_end = time.perf_counter()
            lat_ms = (t_infer_end - t_frame_start) * 1000.0
            latencies.append(lat_ms)

            # 2. Decision Engine
            decision = decision_engine.evaluate(packet)

            # 3. Telemetry Dispatch (PostgreSQL -> Local Fallback)
            if packet.get("detections"):
                db_success = db_client.log_detection_event(packet, decision, node_id="RPI4_EDGE_01")
                if not db_success:
                    local_logger.log_event(packet, decision)

            # 4. Moving Average FPS
            instant_fps = 1000.0 / max(0.001, lat_ms)
            fps = 0.9 * fps + 0.1 * instant_fps if fps > 0 else instant_fps

            # 5. Display (if not headless)
            if not headless:
                hud_frame = detector.annotate_frame(
                    frame=frame,
                    detection_packet=packet,
                    fps=fps,
                    status_text=decision["decision"],
                )
                cv2.imshow("Krishnetra - Raspberry Pi Edge Monitor", hud_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[RPiEdge] Exit requested.")
                    break

            if max_frames > 0 and frame_count >= max_frames:
                print(f"[RPiEdge] Processed {max_frames} target frames.")
                break

    except KeyboardInterrupt:
        print("\n[RPiEdge] Interrupted by user.")
    finally:
        total_time = time.time() - t_session_start
        avg_latency = np.mean(latencies) if latencies else 0.0
        avg_fps = frame_count / max(0.001, total_time)

        camera.release()
        db_client.close()
        if not headless:
            cv2.destroyAllWindows()

        print("\n" + "=" * 65)
        print("RPI4 EDGE BENCHMARK SUMMARY")
        print("=" * 65)
        print(f"Total Frames Processed : {frame_count}")
        print(f"Average System FPS     : {avg_fps:.2f} FPS")
        print(f"Average Latency        : {avg_latency:.2f} ms")
        print(f"Telemetry Status       : {'PostgreSQL Active' if db_client.db_connected else 'Local File Fallback (JSONL/CSV)'}")
        print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Raspberry Pi 4 Edge Pest Detector")
    parser.add_argument("--model", type=str, default="models/pest_yolov8n_best.onnx", help="Path to ONNX or PT model")
    parser.add_argument("--source", type=str, default="0", help="Camera index or video file")
    parser.add_argument("--picam2", action="store_true", help="Use native Picamera2")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--max-frames", type=int, default=-1, help="Max frames to process (-1 for continuous)")
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless without GUI display")
    parser.add_argument("--gui", action="store_true", help="Enable OpenCV GUI window")
    args = parser.parse_args()

    run_rpi_edge_session(
        model_path=args.model,
        source=args.source,
        use_picam2=args.picam2,
        conf_threshold=args.conf,
        max_frames=args.max_frames,
        headless=not args.gui,
    )


if __name__ == "__main__":
    main()
