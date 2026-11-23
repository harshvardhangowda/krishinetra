"""
KrishiNetra Backend - Direct Rover Control Routes
File: backend/app/api/routes/control.py
"""

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.domains.auth_service import require_roles
from backend.app.domains.rover_service import dispatch_rover_command
from backend.app.infrastructure.db.models import User, UserRole
from backend.app.contracts.rover import CommandDispatch, CommandResponse

router = APIRouter(prefix="/rovers", tags=["Rover Motion & Actuation Control"])


@router.post("/{rover_id}/commands/arm")
async def arm_rover(
    rover_id: str,
    body: CommandDispatch,
    user: User = Depends(require_roles([UserRole.OWNER, UserRole.OPERATOR])),
    db: AsyncSession = Depends(get_db),
):
    """Arm the rover systems (Simulation mode)."""
    return await dispatch_rover_command(
        db=db,
        rover_id=rover_id,
        user=user,
        idempotency_key=body.idempotency_key,
        command_type="ARM",
        payload=body.payload,
    )


@router.post("/{rover_id}/commands/disarm")
async def disarm_rover(
    rover_id: str,
    body: CommandDispatch,
    user: User = Depends(require_roles([UserRole.OWNER, UserRole.OPERATOR])),
    db: AsyncSession = Depends(get_db),
):
    """Disarm the rover systems."""
    return await dispatch_rover_command(
        db=db,
        rover_id=rover_id,
        user=user,
        idempotency_key=body.idempotency_key,
        command_type="DISARM",
        payload=body.payload,
    )


@router.post("/{rover_id}/commands/stop")
async def emergency_stop(
    rover_id: str,
    body: CommandDispatch,
    user: User = Depends(require_roles([UserRole.OWNER, UserRole.OPERATOR, UserRole.VIEWER])),
    db: AsyncSession = Depends(get_db),
):
    """
    Emergency STOP: Allowed by ALL authenticated roles,
    even without active control lease. Immediately locks all motion.
    """
    return await dispatch_rover_command(
        db=db,
        rover_id=rover_id,
        user=user,
        idempotency_key=body.idempotency_key,
        command_type="STOP",
        payload=body.payload,
    )


@router.post("/{rover_id}/commands/drive")
async def drive_rover(
    rover_id: str,
    body: CommandDispatch,
    user: User = Depends(require_roles([UserRole.OWNER, UserRole.OPERATOR])),
    db: AsyncSession = Depends(get_db),
):
    """
    Send velocity drive command (vx, vy, omega) to Mecanum rover.
    Requires active, non-expired control lease.
    """
    return await dispatch_rover_command(
        db=db,
        rover_id=rover_id,
        user=user,
        idempotency_key=body.idempotency_key,
        command_type="DRIVE",
        payload=body.payload,
    )
