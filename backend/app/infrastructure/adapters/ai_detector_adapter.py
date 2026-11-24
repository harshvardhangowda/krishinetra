"""
KrishiNetra Backend - Existing YOLOv8 AI Model Inference Adapter
File: backend/app/infrastructure/adapters/ai_detector_adapter.py
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe integration with existing detector
try:
    from src.detection.detector import PestDetector
    DETECTOR_AVAILABLE = True
except ImportError:
    DETECTOR_AVAILABLE = False


class AIDetectorAdapter:
    """Wraps existing trained YOLOv8 PyTorch and ONNX models for API inference."""

    def __init__(self) -> None:
        self.detector = None
        if DETECTOR_AVAILABLE:
            onnx_model = PROJECT_ROOT / "models" / "pest_yolov8n_best.onnx"
            pt_model = PROJECT_ROOT / "models" / "pest_yolov8n_best.pt"
            weights = str(onnx_model if onnx_model.exists() else pt_model)
            if Path(weights).exists():
                try:
                    self.detector = PestDetector(weights_path=weights, device="cpu")
                except Exception as exc:
                    print(f"[AIDetectorAdapter] Note: PestDetector load deferred: {exc}")

    def predict(self, image_input: Any) -> List[Dict[str, Any]]:
        """
        Run inference and return standard detection dictionary.
        Returns list of {class_id, class_name, confidence, bbox_xc, bbox_yc, bbox_w, bbox_h, is_target_pest}
        Truthful fail-closed behavior: raises on failure, never fabricates fake detections.
        """
        if self.detector is None:
            raise RuntimeError("Pest detection model is not loaded. Model weights missing or invalid.")

        try:
            if isinstance(image_input, bytes):
                import cv2
                import numpy as np
                nparr = np.frombuffer(image_input, np.uint8)
                source = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if source is None:
                    raise ValueError("Failed to decode image bytes with OpenCV.")
                height, width = source.shape[:2]
            else:
                source = image_input
                height, width = 640, 640

            results = self.detector.detect(source)
            detections = []
            for det in results:
                bbox = det.get("bbox", [0, 0, 10, 10])
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                w = max(1.0, float(x2 - x1))
                h = max(1.0, float(y2 - y1))
                xc = float(x1) + w / 2.0
                yc = float(y1) + h / 2.0

                # Normalized coordinates (0.0 to 1.0)
                norm_xc = min(1.0, max(0.0, xc / max(1, width)))
                norm_yc = min(1.0, max(0.0, yc / max(1, height)))
                norm_w = min(1.0, max(0.0, w / max(1, width)))
                norm_h = min(1.0, max(0.0, h / max(1, height)))

                detections.append({
                    "class_id": det.get("class_id", 0),
                    "class_name": det.get("class_name", "pest"),
                    "confidence": float(det.get("confidence", 0.0)),
                    "bbox_xc": norm_xc,
                    "bbox_yc": norm_yc,
                    "bbox_w": norm_w,
                    "bbox_h": norm_h,
                    "box_pixels": {
                        "x1": round(float(x1), 2),
                        "y1": round(float(y1), 2),
                        "x2": round(float(x2), 2),
                        "y2": round(float(y2), 2),
                    },
                    "image_width": int(width),
                    "image_height": int(height),
                    "is_target_pest": True,
                })
            return detections
        except Exception as e:
            print(f"[AIDetectorAdapter] Inference failed: {e}")
            raise RuntimeError(f"YOLO inference failed: {str(e)}") from e


ai_detector = AIDetectorAdapter()
