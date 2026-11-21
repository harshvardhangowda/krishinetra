"""
AI-Powered Precision Pest Management - Live Real-Time Monitoring Interface
File: src/monitoring/live_monitor.py

Main entry point for real-time camera observation, YOLO inference, decision
evaluation, telemetry logging, and visual display HUD.

Run with:
  python src/monitoring/live_monitor.py --source 0
  python src/monitoring/live_monitor.py --source path/to/video.mp4
  python src/monitoring/live_monitor.py --headless --max-frames 50
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decision.decision_engine import DecisionEngine
from src.monitoring.camera import CameraStream
from src.monitoring.logger import DetectionEventLogger
from src.monitoring.realtime_detector import RealtimeDetector


def run_monitoring_session(
    source: str = "0",
    weights_path: str = "yolov8n.pt",
    conf_threshold: float = 0.35,
    max_frames: int = -1,
    headless: bool = False,
    device: str = "cpu",
) -> None:
    """Run real-time camera monitoring session."""
    print("=" * 65)
    print("KRISHNETRA PRECISION PEST MANAGEMENT: REAL-TIME MONITORING")
    print("=" * 65)
    print("SAFETY INTERLOCK: ENGAGED (Observation-only / Actuators Disabled)")
    print("=" * 65)

    # Initialize components
    cam_source = int(source) if source.isdigit() else source
    camera = CameraStream(source=cam_source, width=640, height=480, fps_target=30)
    detector = RealtimeDetector(weights_path=weights_path, device=device, conf_threshold=conf_threshold)
    decision_engine = DecisionEngine(min_confidence_threshold=0.60, hardware_safety_lock=True)
    logger = DetectionEventLogger(session_name="live_monitor")

    frame_count = 0
    t_start = time.time()
    fps = 0.0

    print("\n[LiveMonitor] Starting stream loop. Press 'q' in window to exit.\n")

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                print("[LiveMonitor] End of video stream or camera read failure.")
                break

            frame_count += 1
            t_frame_start = time.perf_counter()

            # 1. Inference
            detection_packet = detector.process_frame(frame, frame_id=frame_count)

            # 2. Decision Engine evaluation
            decision_result = decision_engine.evaluate(detection_packet)

            # 3. Log events
            logger.log_event(detection_packet, decision_result)

            # 4. Calculate smoothed FPS
            t_frame_end = time.perf_counter()
            instant_fps = 1.0 / max(0.001, (t_frame_end - t_frame_start))
            fps = 0.9 * fps + 0.1 * instant_fps if fps > 0 else instant_fps

            # 5. Visual HUD and rendering
            if not headless:
                hud_frame = detector.annotate_frame(
                    frame=frame,
                    detection_packet=detection_packet,
                    fps=fps,
                    status_text=decision_result["decision"],
                )
                cv2.imshow("Krishnetra - Precision Pest Monitoring", hud_frame)

                # Check for exit key (q or Esc)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[LiveMonitor] Exit requested by user.")
                    break

            if max_frames > 0 and frame_count >= max_frames:
                print(f"[LiveMonitor] Reached max requested frames ({max_frames}). Stopping.")
                break

    except KeyboardInterrupt:
        print("\n[LiveMonitor] Session interrupted by user (Ctrl+C).")
    finally:
        total_time = time.time() - t_start
        avg_fps = frame_count / max(0.001, total_time)
        camera.release()
        if not headless:
            cv2.destroyAllWindows()

        print("\n" + "=" * 65)
        print("MONITORING SESSION SUMMARY")
        print("=" * 65)
        print(f"Total Frames Processed : {frame_count}")
        print(f"Total Duration (sec)   : {total_time:.2f}s")
        print(f"Average System FPS     : {avg_fps:.2f} FPS")
        print(f"Model In Use           : {detector.model_tag}")
        print(f"Structured Logs Saved  : {logger.jsonl_path.name}")
        print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Krishnetra Real-Time Camera Pest Monitor")
    parser.add_argument("--source", type=str, default="0", help="Camera device index (0) or path to video file")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to YOLO model checkpoint")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device (cpu/cuda)")
    parser.add_argument("--max-frames", type=int, default=-1, help="Max frames to process before exiting (-1 for continuous)")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display window")
    args = parser.parse_args()

    run_monitoring_session(
        source=args.source,
        weights_path=args.weights,
        conf_threshold=args.conf,
        max_frames=args.max_frames,
        headless=args.headless,
        device=args.device,
    )


if __name__ == "__main__":
    main()
