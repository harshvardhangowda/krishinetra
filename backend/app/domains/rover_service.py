"""
KrishiNetra Backend - Rover Control Authority & Lease Management
File: backend/app/domains/rover_service.py
"""

from datetime import datetime, timedelta, timezone
import json
from typing import Optional, List
from fastapi import HTTPException, status
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.infrastructure.db.models import (
    Rover,
    RoverControlLease,
    RoverCommand,
    User,
    UserRole,
    AuditLog,
    RoverStatus,
    CommandStatus,
    utcnow,
)
from backend.app.infrastructure.adapters.rover_command_adapter import command_adapter


async def acquire_control_lease(
    db: AsyncSession,
    rover_id: str,
    user: User,
    ttl_seconds: int = 30,
) -> RoverControlLease:
    """
    Acquire exclusive, time-bounded control lease over a rover.
    Guarantees only ONE operator can drive/steer at any given time.
    """
    # 1. Verify rover exists and belongs to user's organization
    stmt = select(Rover).where(
        and_(Rover.id == rover_id, Rover.organization_id == user.organization_id)
    )
    result = await db.execute(stmt)
    rover = result.scalar_one_or_none()
    if not rover:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rover {rover_id} not found in your organization.",
        )

    now = utcnow()
    expires = now + timedelta(seconds=ttl_seconds)

    # 2. Check for active unexpired lease
    lease_stmt = select(RoverControlLease).where(
        and_(
            RoverControlLease.rover_id == rover_id,
            RoverControlLease.is_active.is_(True),
            RoverControlLease.expires_at > now,
        )
    )
    lease_res = await db.execute(lease_stmt)
    existing_lease = lease_res.scalar_one_or_none()

    if existing_lease:
        if existing_lease.user_id == user.id:
            # Renew existing lease
            existing_lease.expires_at = expires
            existing_lease.renewed_at = now
            await db.commit()
            await db.refresh(existing_lease)
            return existing_lease
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Rover is currently leased to another operator (expires at {existing_lease.expires_at.isoformat()}).",
            )

    # 3. Create fresh lease
    new_lease = RoverControlLease(
        rover_id=rover_id,
        user_id=user.id,
        organization_id=user.organization_id,
        acquired_at=now,
        expires_at=expires,
        renewed_at=now,
        is_active=True,
    )
    rover.status = RoverStatus.LEASED
    db.add(new_lease)

    # Audit log
    audit = AuditLog(
        organization_id=user.organization_id,
        user_id=user.id,
        action="ROVER_LEASE_ACQUIRED",
        entity_type="rover",
        entity_id=rover_id,
        payload_json=json.dumps({"lease_id": new_lease.id, "ttl": ttl_seconds}),
    )
    db.add(audit)

    await db.commit()
    await db.refresh(new_lease)
    return new_lease


async def dispatch_rover_command(
    db: AsyncSession,
    rover_id: str,
    user: User,
    idempotency_key: str,
    command_type: str,
    payload: dict,
) -> RoverCommand:
    """
    Dispatch rover movement or actuation command with strict lease enforcement,
    idempotency protection, and non-bypassable emergency STOP override.
    """
    # 1. Idempotency Check: if already processed, return existing command record
    idem_stmt = select(RoverCommand).where(RoverCommand.idempotency_key == idempotency_key)
    idem_res = await db.execute(idem_stmt)
    existing_cmd = idem_res.scalar_one_or_none()
    if existing_cmd:
        return existing_cmd

    # 2. Check Rover ownership
    r_stmt = select(Rover).where(
        and_(Rover.id == rover_id, Rover.organization_id == user.organization_id)
    )
    r_res = await db.execute(r_stmt)
    rover = r_res.scalar_one_or_none()
    if not rover:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rover not found in your organization.",
        )

    now = utcnow()
    cmd_type_upper = command_type.upper()
    active_lease = None

    # 3. Control Authority Validation:
    # EMERGENCY STOP is universally allowed without requiring an active lease
    if cmd_type_upper != "STOP":
        lease_stmt = select(RoverControlLease).where(
            and_(
                RoverControlLease.rover_id == rover_id,
                RoverControlLease.user_id == user.id,
                RoverControlLease.is_active.is_(True),
                RoverControlLease.expires_at > now,
            )
        )
        l_res = await db.execute(lease_stmt)
        active_lease = l_res.scalar_one_or_none()
        if not active_lease:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Active control lease required to dispatch commands to this rover. Please acquire lease first.",
            )

    # 4. Record Command as PENDING
    cmd = RoverCommand(
        rover_id=rover_id,
        lease_id=active_lease.id if active_lease else None,
        user_id=user.id,
        idempotency_key=idempotency_key,
        command_type=cmd_type_upper,
        payload_json=json.dumps(payload),
        status=CommandStatus.SENT,
        dispatched_at=now,
    )
    db.add(cmd)
    await db.flush()

    # 5. Execute via safe firmware / simulation adapter
    success, message, resp_data = await command_adapter.execute_command(
        command_type=cmd_type_upper,
        payload=payload,
    )

    if success:
        cmd.status = CommandStatus.COMPLETED
        cmd.acknowledged_at = utcnow()
        cmd.completed_at = utcnow()
    else:
        cmd.status = CommandStatus.FAILED
        cmd.error_details = message

    await db.commit()
    await db.refresh(cmd)
    return cmd
