"""
KrishiNetra Backend - Inspection & Evidence API
File: backend/app/api/v1/inspections.py
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.core.database import get_db
from backend.app.infrastructure.db.models import (
    InspectionSession,
    InspectionFrame,
    Detection,
    User,
    UserRole,
)
from backend.app.contracts.inspection import (
    InspectionSessionCreate,
    InspectionSessionResponse,
    FrameUploadResponse,
    DetectionResponse,
    HumanReviewCreate,
    HumanReviewResponse,
)
from backend.app.domains.auth_service import get_current_user, require_roles
from backend.app.domains.inspection_service import (
    create_inspection_session,
    ingest_frame_and_run_detection,
    submit_human_review,
)

router = APIRouter(prefix="/inspections", tags=["Inspections & AI Evidence"])


@router.post("", response_model=InspectionSessionResponse, status_code=status.HTTP_201_CREATED)
async def start_session(
    body: InspectionSessionCreate,
    current_user: User = Depends(require_roles([UserRole.OWNER, UserRole.OPERATOR])),
    db: AsyncSession = Depends(get_db),
):
    """Start a new field pest inspection session."""
    return await create_inspection_session(
        db=db,
        rover_id=body.rover_id,
        user=current_user,
        title=body.title,
        field_location=body.field_location,
    )


@router.get("", response_model=List[InspectionSessionResponse])
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all inspection sessions in the organization."""
    stmt = (
        select(InspectionSession)
        .where(InspectionSession.organization_id == current_user.organization_id)
        .order_by(InspectionSession.started_at.desc())
    )
    sessions = (await db.execute(stmt)).scalars().all()
    return sessions


@router.post("/{session_id}/frames", response_model=FrameUploadResponse)
async def upload_frame(
    session_id: str,
    frame_index: int = Form(1),
    timestamp_ms: float = Form(0.0),
    file: UploadFile = File(...),
    current_user: User = Depends(require_roles([UserRole.OWNER, UserRole.OPERATOR])),
    db: AsyncSession = Depends(get_db),
):
    """
    Ingests an inspection frame, computes standard SHA-256 integrity hash,
    runs the YOLOv8 model, and stores immutable detection records.
    """
    image_bytes = await file.read()
    frame, detections = await ingest_frame_and_run_detection(
        db=db,
        session_id=session_id,
        user=current_user,
        frame_index=frame_index,
        image_bytes=image_bytes,
        timestamp_ms=timestamp_ms,
    )

    det_responses = [
        DetectionResponse(
            id=d.id,
            class_id=d.class_id,
            class_name=d.class_name,
            confidence=d.confidence,
            bbox_xc=d.bbox_xc,
            bbox_yc=d.bbox_yc,
            bbox_w=d.bbox_w,
            bbox_h=d.bbox_h,
            is_target_pest=d.is_target_pest,
        )
        for d in detections
    ]

    return FrameUploadResponse(
        frame_id=frame.id,
        session_id=frame.session_id,
        frame_index=frame.frame_index,
        file_hash_sha256=frame.file_hash_sha256,
        detections=det_responses,
    )


@router.post("/frames/{frame_id}/reviews", response_model=HumanReviewResponse)
async def create_review(
    frame_id: str,
    body: HumanReviewCreate,
    current_user: User = Depends(require_roles([UserRole.OWNER, UserRole.REVIEWER])),
    db: AsyncSession = Depends(get_db),
):
    """
    Reviewer submits human validation/correction for an inspection frame.
    Non-destructive: Raw AI predictions remain completely preserved.
    """
    return await submit_human_review(
        db=db,
        frame_id=frame_id,
        user=current_user,
        review_status=body.review_status,
        detection_id=body.detection_id,
        corrected_class_id=body.corrected_class_id,
        corrected_class_name=body.corrected_class_name,
        corrected_bbox=body.corrected_bbox,
        notes=body.notes,
    )
