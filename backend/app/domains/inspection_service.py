"""
KrishiNetra Backend - Inspection & Evidence Domain Service
File: backend/app/domains/inspection_service.py
"""

import json
from typing import List, Optional, Tuple
from fastapi import HTTPException, status
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.infrastructure.db.models import (
    InspectionSession,
    InspectionFrame,
    Detection,
    HumanReview,
    User,
    UserRole,
    InspectionStatus,
    ReviewStatus,
    utcnow,
)
from backend.app.infrastructure.storage.media_storage import media_storage
from backend.app.infrastructure.adapters.ai_detector_adapter import ai_detector


async def create_inspection_session(
    db: AsyncSession,
    rover_id: str,
    user: User,
    title: str,
    field_location: str,
) -> InspectionSession:
    """Create a new field inspection session for pest monitoring."""
    session = InspectionSession(
        rover_id=rover_id,
        organization_id=user.organization_id,
        operator_id=user.id,
        title=title,
        field_location=field_location,
        status=InspectionStatus.IN_PROGRESS,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def ingest_frame_and_run_detection(
    db: AsyncSession,
    session_id: str,
    user: User,
    frame_index: int,
    image_bytes: bytes,
    timestamp_ms: float = 0.0,
) -> Tuple[InspectionFrame, List[Detection]]:
    """
    Ingests an inspection frame, computes SHA-256 for audit integrity,
    runs the AI detector, and stores immutable detection records.
    """
    # 1. Verify session exists & belongs to org
    stmt = select(InspectionSession).where(
        and_(
            InspectionSession.id == session_id,
            InspectionSession.organization_id == user.organization_id,
        )
    )
    res = await db.execute(stmt)
    inspection_session = res.scalar_one_or_none()
    if not inspection_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Inspection session not found in your organization.",
        )

    # 2. Store frame with hash verification
    storage_path, sha256_hash = media_storage.store_frame(
        session_id=session_id,
        frame_index=frame_index,
        image_bytes=image_bytes,
    )

    frame = InspectionFrame(
        session_id=session_id,
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        storage_path=storage_path,
        file_hash_sha256=sha256_hash,
    )
    db.add(frame)
    await db.flush()

    # 3. Run AI inference
    raw_detections = ai_detector.predict(image_bytes)
    detection_entities = []

    for d in raw_detections:
        det = Detection(
            frame_id=frame.id,
            class_id=d["class_id"],
            class_name=d["class_name"],
            confidence=d["confidence"],
            bbox_xc=d["bbox_xc"],
            bbox_yc=d["bbox_yc"],
            bbox_w=d["bbox_w"],
            bbox_h=d["bbox_h"],
            is_target_pest=d["is_target_pest"],
            raw_prediction_json=json.dumps(d),
        )
        db.add(det)
        detection_entities.append(det)

    # Update session counters
    inspection_session.total_frames += 1
    inspection_session.total_detections += len(detection_entities)

    await db.commit()
    await db.refresh(frame)
    return frame, detection_entities


async def submit_human_review(
    db: AsyncSession,
    frame_id: str,
    user: User,
    review_status: ReviewStatus,
    detection_id: Optional[str] = None,
    corrected_class_id: Optional[int] = None,
    corrected_class_name: Optional[str] = None,
    corrected_bbox: Optional[List[float]] = None,
    notes: Optional[str] = None,
) -> HumanReview:
    """
    Submits a human review record without modifying the raw AI prediction.
    Preserves raw ML data for retraining and false-positive tracking.
    """
    review = HumanReview(
        frame_id=frame_id,
        detection_id=detection_id,
        reviewer_id=user.id,
        review_status=review_status,
        corrected_class_id=corrected_class_id,
        corrected_class_name=corrected_class_name,
        corrected_bbox_json=json.dumps(corrected_bbox) if corrected_bbox else None,
        notes=notes,
        reviewed_at=utcnow(),
    )
    db.add(review)
    await db.commit()
    await db.refresh(review)
    return review
