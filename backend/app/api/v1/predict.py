"""
KrishiNetra Backend - Frontend-Compatible Prediction API
File: backend/app/api/v1/predict.py

Exposes POST /api/v1/predict for client web dashboards and external integration.
Strictly separates:
1. Pure neural network outputs (trained YOLOv8 / ONNX inference)
2. Verified agronomic knowledge (configs/target_pests.yaml, pest_acoustic_profile.yaml)
3. Deterministic heuristic estimates (infestation severity and coverage)

Hardware Safety Standard: Strictly read-only. Never commands rovers or acoustic hardware.
"""

import io
import time
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from PIL import Image

from backend.app.core.config import settings
from backend.app.contracts.prediction import (
    PredictionResponse,
    EnrichedDetection,
    ModelPrediction,
    BoundingBoxPixels,
    BoundingBoxNormalized,
    ImageMetadata,
)
from backend.app.domains.auth_service import get_optional_current_user
from backend.app.infrastructure.db.models import User
from backend.app.infrastructure.adapters.ai_detector_adapter import ai_detector
from backend.app.domains.pest_knowledge_service import pest_knowledge_service

router = APIRouter(tags=["AI Pest Prediction"])

CHUNK_SIZE = 64 * 1024  # 64 KB streaming chunks


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Run AI pest inference with agronomic knowledge enrichment",
    description=(
        "Accepts an uploaded image (JPEG, PNG, WebP) up to 10MB, executes real YOLOv8/ONNX "
        "inference, and enriches detections with verified scientific taxonomy, host crop data, "
        "and evidence-based non-chemical IPM recommendations. Strictly read-only with respect to hardware."
    ),
)
async def predict_pest(
    file: UploadFile = File(..., description="Multipart image file (JPEG, PNG, WebP)"),
    confidence_threshold: Optional[float] = Form(
        None,
        ge=0.0,
        le=1.0,
        description="Optional minimum confidence threshold override (0.0 to 1.0)",
    ),
    current_user: Optional[User] = Depends(get_optional_current_user),
) -> PredictionResponse:
    """Execute pest prediction on uploaded image frame."""

    # 1. Enforce stream-based 10 MB upload size limit
    total_bytes = 0
    chunks = []
    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > settings.PREDICTION_MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Uploaded image exceeds maximum allowable limit of {settings.PREDICTION_MAX_UPLOAD_BYTES // (1024 * 1024)}MB.",
            )
        chunks.append(chunk)

    image_bytes = b"".join(chunks)

    if len(image_bytes) < 16:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty or corrupted.",
        )

    # 2. Strict Magic Byte Verification (JPEG, PNG, WebP)
    is_jpeg = image_bytes.startswith(b"\xff\xd8\xff")
    is_png = image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    is_webp = (
        len(image_bytes) >= 12
        and image_bytes[:4] == b"RIFF"
        and image_bytes[8:12] == b"WEBP"
    )

    if not (is_jpeg or is_png or is_webp):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Invalid image format. Supported formats are JPEG, PNG, and WebP.",
        )

    image_format = "JPEG" if is_jpeg else ("PNG" if is_png else "WEBP")

    # 3. Verify decodability and extract physical image dimensions
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            img_width, img_height = pil_img.size
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to decode image bytes. The image payload appears corrupted.",
        ) from exc

    # 4. Truthful AI Model Inference (timed)
    start_time = time.perf_counter()
    try:
        raw_detections = ai_detector.predict(image_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image decoding failed during preprocessing: {str(exc)}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI inference engine failure: {str(exc)}",
        ) from exc
    inference_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    # 5. Filter and Enrich Detections
    enriched_detections: List[EnrichedDetection] = []
    severities: List[str] = []

    # Filter by optional threshold if provided
    active_detections = [
        d for d in raw_detections
        if confidence_threshold is None or d.get("confidence", 0.0) >= confidence_threshold
    ]
    total_active_count = len(active_detections)

    for det in active_detections:
        cls_id = int(det.get("class_id", 0))
        cls_name = str(det.get("class_name", "pest"))
        conf = round(float(det.get("confidence", 0.0)), 4)

        # Pixel coordinates
        pixels_dict = det.get("box_pixels", {})
        x1 = float(pixels_dict.get("x1", det.get("bbox_xc", 0.0) * img_width))
        y1 = float(pixels_dict.get("y1", det.get("bbox_yc", 0.0) * img_height))
        x2 = float(pixels_dict.get("x2", x1 + det.get("bbox_w", 0.0) * img_width))
        y2 = float(pixels_dict.get("y2", y1 + det.get("bbox_h", 0.0) * img_height))

        box_pixels = BoundingBoxPixels(x1=x1, y1=y1, x2=x2, y2=y2)
        box_normalized = BoundingBoxNormalized(
            xc=round(float(det.get("bbox_xc", 0.0)), 4),
            yc=round(float(det.get("bbox_yc", 0.0)), 4),
            w=round(float(det.get("bbox_w", 0.0)), 4),
            h=round(float(det.get("bbox_h", 0.0)), 4),
        )

        model_prediction = ModelPrediction(
            class_id=cls_id,
            class_name=cls_name,
            confidence=conf,
            box_pixels=box_pixels,
            box_normalized=box_normalized,
        )

        # Knowledge Base Metadata
        knowledge = pest_knowledge_service.get_knowledge(cls_id, cls_name)

        # Deterministic Heuristic Severity Calculation
        heuristics = pest_knowledge_service.compute_heuristic_estimates(
            box_pixels={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            image_width=img_width,
            image_height=img_height,
            confidence=conf,
            total_pests=total_active_count,
        )
        severities.append(heuristics.severity)

        enriched_detections.append(
            EnrichedDetection(
                model_prediction=model_prediction,
                knowledge_base=knowledge,
                heuristic_estimates=heuristics,
            )
        )

    overall_severity = pest_knowledge_service.compute_overall_severity(severities)

    # 6. Build response
    mode = "authenticated" if current_user else "demo_unauthenticated"

    return PredictionResponse(
        success=True,
        inference_time_ms=inference_time_ms,
        total_detections=len(enriched_detections),
        overall_severity=overall_severity,
        image_metadata=ImageMetadata(
            width=img_width,
            height=img_height,
            format=image_format,
            file_size_bytes=total_bytes,
        ),
        detections=enriched_detections,
        mode=mode,
        hardware_safeguard="READ_ONLY_ACTIVE",
    )
