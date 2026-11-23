"""
KrishiNetra Backend - Rover & Control Lease API
File: backend/app/api/v1/rovers.py
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.infrastructure.db.models import Rover, RoverCommand, User, UserRole
from backend.app.contracts.rover import (
    RoverCreate,
    RoverResponse,
    LeaseAcquireRequest,
    LeaseResponse,
    CommandDispatch,
    CommandResponse,
)
from backend.app.domains.auth_service import get_current_user, require_roles
from backend.app.domains.rover_service import acquire_control_lease, dispatch_rover_command

router = APIRouter(prefix="/rovers", tags=["Rovers & Control Leases"])


@router.get("", response_model=List[RoverResponse])
async def list_rovers(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all rovers registered to the user's organization."""
    stmt = select(Rover).where(Rover.organization_id == current_user.organization_id)
    rovers = (await db.execute(stmt)).scalars().all()
    return rovers


@router.post("", response_model=RoverResponse, status_code=status.HTTP_201_CREATED)
async def register_rover(
    body: RoverCreate,
    current_user: User = Depends(require_roles([UserRole.OWNER, UserRole.OPERATOR])),
    db: AsyncSession = Depends(get_db),
):
    """Register a new autonomous rover into the organization fleet."""
    stmt = select(Rover).where(Rover.serial_number == body.serial_number)
    existing = (await db.execute(stmt)).scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Rover with serial {body.serial_number} already registered.",
        )

    rover = Rover(
        organization_id=current_user.organization_id,
        name=body.name,
        serial_number=body.serial_number,
        hardware_version=body.hardware_version,
        firmware_version=body.firmware_version,
    )
    db.add(rover)
    await db.commit()
    await db.refresh(rover)
    return rover


@router.post("/{rover_id}/leases", response_model=LeaseResponse)
async def acquire_lease(
    rover_id: str,
    body: LeaseAcquireRequest,
    current_user: User = Depends(require_roles([UserRole.OWNER, UserRole.OPERATOR])),
    db: AsyncSession = Depends(get_db),
):
    """Acquire exclusive, time-limited control lease for driving the rover."""
    return await acquire_control_lease(
        db=db,
        rover_id=rover_id,
        user=current_user,
        ttl_seconds=body.ttl_seconds,
    )


@router.post("/{rover_id}/commands", response_model=CommandResponse)
async def send_command(
    rover_id: str,
    body: CommandDispatch,
    current_user: User = Depends(require_roles([UserRole.OWNER, UserRole.OPERATOR])),
    db: AsyncSession = Depends(get_db),
):
    """
    Dispatch movement, servoing, or deterrence command.
    Enforces active lease, idempotency, and allows unconditional STOP.
    """
    return await dispatch_rover_command(
        db=db,
        rover_id=rover_id,
        user=current_user,
        idempotency_key=body.idempotency_key,
        command_type=body.command_type,
        payload=body.payload,
    )
