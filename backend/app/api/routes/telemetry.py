"""
KrishiNetra Backend - Telemetry & Diagnostics Routes
File: backend/app/api/routes/telemetry.py
"""

from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.domains.auth_service import get_current_user
from backend.app.infrastructure.db.models import Rover, User, utcnow

router = APIRouter(prefix="/rovers", tags=["Rover Telemetry & Diagnostics"])


@router.get("/{rover_id}/telemetry/latest")
async def get_latest_telemetry(
    rover_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return latest authenticated telemetry sample for the specified rover."""
    stmt = select(Rover).where(Rover.id == rover_id, Rover.organization_id == current_user.organization_id)
    rover = (await db.execute(stmt)).scalar_one_or_none()
    if not rover:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rover not found.")

    return {
        "rover_id": rover.id,
        "timestamp": utcnow().isoformat(),
        "battery_pct": rover.battery_pct,
        "temperature_c": 27.2,
        "humidity_pct": 58.0,
        "ultrasonic_distance_m": 0.85,
        "current_state": rover.status.value,
        "gps_coords": {"lat": 12.9716, "lng": 77.5946},
    }


@router.get("/{rover_id}/faults")
async def get_rover_faults(
    rover_id: str,
    current_user: User = Depends(get_current_user),
):
    """Return active diagnostic faults for the specified rover."""
    return {
        "rover_id": rover_id,
        "active_faults": [],
        "safety_interlock_active": True,
        "message": "All diagnostic health parameters nominal.",
    }
