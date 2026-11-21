"""
AI-Powered Precision Pest Management - Real-Time Inference Interface
File: src/monitoring/realtime_detector.py

Performs real-time YOLO object detection on video frames and formats
results into structured event packets for the Decision Engine.

SAFETY NOTICE:
At this milestone, this detector operates with the BASELINE COCO MODEL (yolov8n.pt).
It is explicitly labeled: "BASELINE COCO MODEL - NOT A PEST MODEL".
"""

from datetime import datetime
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


class RealtimeDetector:
    """Real-time YOLO inference wrapper with latency benchmarking and structured output."""

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
    ) -> None:
        """
        Initialize real-time detector.

        Args:
            weights_path: Path to model weights (.pt).
            device: 'cpu', 'cuda', or None for auto.
            conf_threshold: Minimum confidence score to register detection.
            iou_threshold: NMS IOU threshold.
        """
        import torch
        from ultralytics import YOLO

        self.weights_path = str(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Determine model classification mode
        self.is_baseline = "yolov8n.pt" in Path(self.weights_path).name.lower() or "coco" in self.weights_path.lower()
        self.model_tag = "BASELINE COCO MODEL (NON-PEST)" if self.is_baseline else "CUSTOM PEST DETECTOR"

        print(f"[RealtimeDetector] Loading {self.weights_path} ({self.model_tag}) on {self.device}...")
        self.model = YOLO(self.weights_path)
        print("[RealtimeDetector] Ready for live inference.")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Execute YOLO inference on a single video frame.

        Returns:
            Structured dictionary containing timestamp, latency, and detections.
        """
        t0 = time.perf_counter()
        timestamp = datetime.utcnow().isoformat() + "Z"

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        inference_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        detections: List[Dict[str, Any]] = []

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0].item())
                xyxy = [round(float(c), 1) for c in box.xyxy[0].tolist()]

                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(conf, 4),
                        "bbox": xyxy,
                    }
                )

        return {
            "timestamp": timestamp,
            "frame_id": frame_id,
            "model_type": self.model_tag,
            "is_baseline_model": self.is_baseline,
            "inference_latency_ms": inference_latency_ms,
            "detections": detections,
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        detection_packet: Dict[str, Any],
        fps: float = 0.0,
        status_text: str = "MONITORING (OBSERVATION ONLY)",
    ) -> np.ndarray:
        """
        Draw visual overlays, bounding boxes, labels, and safety HUD on the frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Top HUD banner
        hud_bg = np.zeros((70, w, 3), dtype=np.uint8)
        # Alpha blend HUD background
        annotated[0:70, 0:w] = cv2.addWeighted(annotated[0:70, 0:w], 0.3, hud_bg, 0.7, 0)

        # HUD Text
        model_label = f"MODEL: {Path(self.weights_path).name} [{'BASELINE COCO' if self.is_baseline else 'PEST'}]"
        cv2.putText(annotated, model_label, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(annotated, f"FPS: {fps:.1f} | Latency: {detection_packet['inference_latency_ms']}ms", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(annotated, f"STATUS: {status_text}", (w - 380, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1)
        cv2.putText(annotated, "SAFETY: ACTUATORS DISABLED", (w - 380, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)

        # 2. Draw Bounding Boxes
        for det in detection_packet.get("detections", []):
            x1, y1, x2, y2 = map(int, det["bbox"])
            cls_name = det["class_name"]
            conf = det["confidence"]

            # Color selection: Orange for baseline COCO, Red for target pests
            box_color = (0, 165, 255) if self.is_baseline else (0, 0, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(0, y1)), box_color, -1)
            cv2.putText(annotated, label, (x1 + 3, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return annotated
