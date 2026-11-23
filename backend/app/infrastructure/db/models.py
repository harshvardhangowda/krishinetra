"""
KrishiNetra Backend - Relational Database Entities
File: backend/app/infrastructure/db/models.py
"""

from datetime import datetime, timezone
import uuid
import enum
from typing import Optional, List
from sqlalchemy import (
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    Enum,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.app.infrastructure.db.base import Base


def utcnow() -> datetime:
    """Return current UTC timestamp with timezone."""
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    """Generate RFC 4122 compliant UUID4 string."""
    return str(uuid.uuid4())


class UserRole(str, enum.Enum):
    OWNER = "owner"
    OPERATOR = "operator"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


class RoverStatus(str, enum.Enum):
    IDLE = "idle"
    ACTIVE = "active"
    LEASED = "leased"
    FAULT = "fault"
    DISCONNECTED = "disconnected"


class CommandStatus(str, enum.Enum):
    PENDING = "pending"
    SENT = "sent"
    ACK_RECEIVED = "ack_received"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"


class InspectionStatus(str, enum.Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ReviewStatus(str, enum.Enum):
    UNREVIEWED = "unreviewed"
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"
    REJECTED = "rejected"


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ==============================================================================
# Multi-Tenant Core: Organizations, Users, Sessions, Audit Logs
# ==============================================================================

class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    users: Mapped[List["User"]] = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    rovers: Mapped[List["Rover"]] = relationship("Rover", back_populates="organization", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(128), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    organization: Mapped["Organization"] = relationship("Organization", back_populates="users")
    sessions: Mapped[List["Session"]] = relationship("Session", back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    session_token: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    csrf_token: Mapped[str] = mapped_column(String(128), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    user_agent: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)

    user: Mapped["User"] = relationship("User", back_populates="sessions")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g., AUTH_LOGIN, ROVER_LEASE_ACQUIRE, COMMAND_SENT
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


# ==============================================================================
# Rover Control Authority: Registry, Leases, and Command Lifecycle
# ==============================================================================

class Rover(Base):
    __tablename__ = "rovers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    serial_number: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    hardware_version: Mapped[str] = mapped_column(String(32), default="v1.0-sim")
    firmware_version: Mapped[str] = mapped_column(String(32), default="v1.0.0")
    status: Mapped[RoverStatus] = mapped_column(Enum(RoverStatus), default=RoverStatus.IDLE)
    battery_pct: Mapped[float] = mapped_column(Float, default=100.0)
    last_heartbeat: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    organization: Mapped["Organization"] = relationship("Organization", back_populates="rovers")
    leases: Mapped[List["RoverControlLease"]] = relationship("RoverControlLease", back_populates="rover", cascade="all, delete-orphan")
    commands: Mapped[List["RoverCommand"]] = relationship("RoverCommand", back_populates="rover", cascade="all, delete-orphan")


class RoverControlLease(Base):
    """Exclusive, time-bounded authority granting one user control of a rover."""
    __tablename__ = "rover_control_leases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    rover_id: Mapped[str] = mapped_column(String(36), ForeignKey("rovers.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    acquired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    renewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    rover: Mapped["Rover"] = relationship("Rover", back_populates="leases")


class RoverCommand(Base):
    """Command lifecycle tracking with idempotency and acknowledgement confirmation."""
    __tablename__ = "rover_commands"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    rover_id: Mapped[str] = mapped_column(String(36), ForeignKey("rovers.id", ondelete="CASCADE"), nullable=False, index=True)
    lease_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("rover_control_leases.id", ondelete="SET NULL"), nullable=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    command_type: Mapped[str] = mapped_column(String(64), nullable=False)  # STOP, DRIVE, SERVO, DETER
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[CommandStatus] = mapped_column(Enum(CommandStatus), default=CommandStatus.PENDING, index=True)
    dispatched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error_details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    rover: Mapped["Rover"] = relationship("Rover", back_populates="commands")


# ==============================================================================
# AI Evidence Platform: Inspections, Frames, Detections, and Human Review
# ==============================================================================

class InspectionSession(Base):
    __tablename__ = "inspection_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    rover_id: Mapped[str] = mapped_column(String(36), ForeignKey("rovers.id", ondelete="CASCADE"), nullable=False, index=True)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    operator_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(128), nullable=False)
    field_location: Mapped[str] = mapped_column(String(128), default="Plot A-Field 1")
    status: Mapped[InspectionStatus] = mapped_column(Enum(InspectionStatus), default=InspectionStatus.IN_PROGRESS)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    total_frames: Mapped[int] = mapped_column(Integer, default=0)
    total_detections: Mapped[int] = mapped_column(Integer, default=0)

    frames: Mapped[List["InspectionFrame"]] = relationship("InspectionFrame", back_populates="session", cascade="all, delete-orphan")


class InspectionFrame(Base):
    __tablename__ = "inspection_frames"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("inspection_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_ms: Mapped[float] = mapped_column(Float, nullable=False)
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_hash_sha256: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    width: Mapped[int] = mapped_column(Integer, default=640)
    height: Mapped[int] = mapped_column(Integer, default=640)
    is_quarantined: Mapped[bool] = mapped_column(Boolean, default=False)

    session: Mapped["InspectionSession"] = relationship("InspectionSession", back_populates="frames")
    detections: Mapped[List["Detection"]] = relationship("Detection", back_populates="frame", cascade="all, delete-orphan")
    reviews: Mapped[List["HumanReview"]] = relationship("HumanReview", back_populates="frame", cascade="all, delete-orphan")


class Detection(Base):
    """Raw, immutable model predictions captured during inspection."""
    __tablename__ = "detections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    frame_id: Mapped[str] = mapped_column(String(36), ForeignKey("inspection_frames.id", ondelete="CASCADE"), nullable=False, index=True)
    class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_xc: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_yc: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_w: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_h: Mapped[float] = mapped_column(Float, nullable=False)
    is_target_pest: Mapped[bool] = mapped_column(Boolean, default=True)
    raw_prediction_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    frame: Mapped["InspectionFrame"] = relationship("InspectionFrame", back_populates="detections")


class HumanReview(Base):
    """Audit-proof human review corrections that never overwrite raw AI predictions."""
    __tablename__ = "human_reviews"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    frame_id: Mapped[str] = mapped_column(String(36), ForeignKey("inspection_frames.id", ondelete="CASCADE"), nullable=False, index=True)
    detection_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("detections.id", ondelete="SET NULL"), nullable=True)
    reviewer_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    review_status: Mapped[ReviewStatus] = mapped_column(Enum(ReviewStatus), default=ReviewStatus.CONFIRMED)
    corrected_class_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    corrected_class_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    corrected_bbox_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    frame: Mapped["InspectionFrame"] = relationship("InspectionFrame", back_populates="reviews")


# ==============================================================================
# Model & Dataset Lifecycle and Background Jobs
# ==============================================================================

class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    framework: Mapped[str] = mapped_column(String(32), default="TORCH_PT")  # TORCH_PT, ONNX
    weights_path: Mapped[str] = mapped_column(String(512), nullable=False)
    input_shape: Mapped[str] = mapped_column(String(32), default="[1, 3, 640, 640]")
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class BackgroundJob(Base):
    __tablename__ = "background_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    job_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.QUEUED, index=True)
    progress_pct: Mapped[float] = mapped_column(Float, default=0.0)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    result_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
