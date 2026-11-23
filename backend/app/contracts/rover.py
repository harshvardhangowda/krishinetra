"""
KrishiNetra Backend - Rover & Control Lease Contracts
File: backend/app/contracts/rover.py
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from backend.app.infrastructure.db.models import RoverStatus, CommandStatus


class RoverCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=64)
    serial_number: str = Field(..., min_length=3, max_length=64)
    hardware_version: str = "v1.0-sim"
    firmware_version: str = "v1.0.0"


class RoverResponse(BaseModel):
    id: str
    organization_id: str
    name: str
    serial_number: str
    hardware_version: str
    firmware_version: str
    status: RoverStatus
    battery_pct: float
    last_heartbeat: Optional[datetime]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class LeaseAcquireRequest(BaseModel):
    ttl_seconds: int = Field(default=30, ge=5, le=300)


class LeaseResponse(BaseModel):
    id: str
    rover_id: str
    user_id: str
    acquired_at: datetime
    expires_at: datetime
    renewed_at: datetime
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


class CommandDispatch(BaseModel):
    idempotency_key: str = Field(..., min_length=8, max_length=128)
    command_type: str = Field(..., description="STOP, DRIVE, SERVO, DETER")
    payload: Dict[str, Any] = Field(default_factory=dict)


class CommandResponse(BaseModel):
    id: str
    rover_id: str
    user_id: str
    idempotency_key: str
    command_type: str
    status: CommandStatus
    payload_json: str
    dispatched_at: datetime
    acknowledged_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_details: Optional[str]

    model_config = ConfigDict(from_attributes=True)
