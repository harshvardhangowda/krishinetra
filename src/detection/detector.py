"""
AI-Powered Precision Pest Management - Detection Module
File: src/detection/detector.py

This module encapsulates YOLO-based object detection for agricultural insect
detection. It loads model weights, runs inference with configurable confidence
and IOU thresholds, and formats detections for downstream processing by the
Decision Engine.

Pipeline Stage: Detect -> Identify (precursor to Decision Engine)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


class PestDetector:
    """YOLO-based detector for insect pest identification."""

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        """
        Initialize the YOLO detector.

        Args:
            weights_path: Path to model checkpoint file (.pt) or model name.
            device: Computation device ('cpu', 'cuda', or None for auto-detect).
            conf_threshold: Minimum confidence score to retain detection.
            iou_threshold: Intersection over Union threshold for NMS.
        """
        # Lazy import of ultralytics and torch so module can be imported even during setup
        import torch
        from ultralytics import YOLO

        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[PestDetector] Loading weights: {self.weights_path} on device: {self.device}")
        self.model = YOLO(self.weights_path)
        print("[PestDetector] Model loaded successfully.")

    def detect(
        self,
        source: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
        save_annotated: bool = False,
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on the provided image source.

        Args:
            source: Image filepath or OpenCV/NumPy BGR image array.
            conf_threshold: Optional override for confidence threshold.
            save_annotated: Whether to save visualization of detections.
            output_dir: Directory where annotated output will be saved.

        Returns:
            List of detection dicts:
            [
                {
                    "class_id": int,
                    "class_name": str,
                    "confidence": float,
                    "bbox": [x1, y1, x2, y2],  # pixel coordinates
                },
                ...
            ]
        """
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold

        results = self.model.predict(
            source=source,
            conf=threshold,
            iou=self.iou_threshold,
            device=self.device,
            save=save_annotated,
            project=output_dir if output_dir else "runs/detect",
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0].item())
                xyxy = [round(float(c), 2) for c in box.xyxy[0].tolist()]

                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(conf, 4),
                        "bbox": xyxy,
                    }
                )

        return detections

    def get_class_names(self) -> Dict[int, str]:
        """Return the dictionary of class IDs to class names known by the model."""
        return dict(self.model.names)
