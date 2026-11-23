"""
KrishiNetra Backend - AI Inference Client
File: backend/app/infrastructure/inference_client/yolo_client.py
"""

from typing import List, Dict, Any
from backend.app.infrastructure.adapters.ai_detector_adapter import ai_detector


class InferenceClient:
    """Wraps YOLOv8 detection engine for domain services."""

    def run_detection(self, image_data: Any) -> List[Dict[str, Any]]:
        return ai_detector.predict(image_data)


inference_client = InferenceClient()
