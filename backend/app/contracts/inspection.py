"""
KrishiNetra Backend - Inspection & Evidence Pydantic Contracts
File: backend/app/contracts/inspection.py
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from backend.app.infrastructure.db.models import InspectionStatus, ReviewStatus


class InspectionSessionCreate(BaseModel):
    rover_id: str
    title: str = Field(..., min_length=2, max_length=128)
    field_location: str = "Plot A-Field 1"


class InspectionSessionResponse(BaseModel):
    id: str
    rover_id: str
    organization_id: str
    operator_id: str
    title: str
    field_location: str
    status: InspectionStatus
    started_at: datetime
    ended_at: Optional[datetime]
    total_frames: int
    total_detections: int

    model_config = ConfigDict(from_attributes=True)


class DetectionResponse(BaseModel):
    id: str
    class_id: int
    class_name: str
    confidence: float
    bbox_xc: float
    bbox_yc: float
    bbox_w: float
    bbox_h: float
    is_target_pest: bool

    model_config = ConfigDict(from_attributes=True)


class FrameUploadResponse(BaseModel):
    frame_id: str
    session_id: str
    frame_index: int
    file_hash_sha256: str
    detections: List[DetectionResponse]


class HumanReviewCreate(BaseModel):
    detection_id: Optional[str] = None
    review_status: ReviewStatus = ReviewStatus.CONFIRMED
    corrected_class_id: Optional[int] = None
    corrected_class_name: Optional[str] = None
    corrected_bbox: Optional[List[float]] = None
    notes: Optional[str] = None


class HumanReviewResponse(BaseModel):
    id: str
    frame_id: str
    detection_id: Optional[str]
    reviewer_id: str
    review_status: ReviewStatus
    corrected_class_id: Optional[int]
    corrected_class_name: Optional[str]
    notes: Optional[str]
    reviewed_at: datetime

    model_config = ConfigDict(from_attributes=True)
