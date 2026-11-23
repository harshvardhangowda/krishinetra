"""
KrishiNetra Backend - Treatment & Deterrence Workflow API
File: backend/app/api/routes/treatments.py
"""

import uuid
import json
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.domains.auth_service import require_roles
from backend.app.infrastructure.db.models import User, UserRole, Rover, AuditLog

router = APIRouter(prefix="/treatments", tags=["Targeted Pest Treatments & Bioassays"])


class TreatmentRequest(BaseModel):
    rover_id: str
    target_pest: str = Field(..., description="army worm, blister beetle, corn borer")
    treatment_type: str = Field(..., description="SWEPT_PULSE_US, CONTINUOUS_US, LOW_FREQ_VIB")
    frequency_hz: float = 38500.0
    exposure_duration_sec: float = 5.0


class TreatmentRunResponse(BaseModel):
    run_id: str
    rover_id: str
    target_pest: str
    treatment_type: str
    status: str
    actuation_executed: bool
    safety_disclaimer: str


@router.get("/profiles")
async def list_treatment_profiles(
    current_user: User = Depends(require_roles([UserRole.OWNER, UserRole.OPERATOR, UserRole.REVIEWER])),
):
    """List standardized scientific non-chemical deterrence acoustic research profiles."""
    return [
        {
            "id": "TP-01-ARMYWORM",
            "pest": "Spodoptera frugiperda (Fall Armyworm)",
            "deterrence_type": "SWEPT_PULSE_US",
            "frequency_range_hz": [35000, 42000],
            "optimal_spl_db": 95.0,
            "status": "RESEARCH_LAB_PROTOTYPE",
        },
        {
            "id": "TP-02-CORN_BORER",
            "pest": "Ostrinia nubilalis (European Corn Borer)",
            "deterrence_type": "CONTINUOUS_US",
            "frequency_range_hz": [38000, 40000],
            "optimal_spl_db": 92.0,
            "status": "RESEARCH_LAB_PROTOTYPE",
        },
    ]


@router.post("/requests", response_model=TreatmentRunResponse, status_code=status.HTTP_201_CREATED)
async def request_treatment(
    body: TreatmentRequest,
    current_user: User = Depends(require_roles([UserRole.OWNER, UserRole.OPERATOR])),
    db: AsyncSession = Depends(get_db),
):
    """
    Submits a targeted deterrence treatment request.
    Strict Safety Policy: Operates in SIMULATION ONLY.
    Validates rover organization ownership and persists audit log (KN-12).
    """
    # 1. Verify rover exists in user's organization
    stmt = select(Rover).where(
        and_(
            Rover.id == body.rover_id,
            Rover.organization_id == current_user.organization_id,
        )
    )
    rover = (await db.execute(stmt)).scalar_one_or_none()
    if not rover:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rover {body.rover_id} not found in your organization.",
        )

    # 2. Generate unique run ID
    run_id = f"TR-SIM-{uuid.uuid4().hex[:12].upper()}"

    # 3. Persist treatment request in audit log
    audit = AuditLog(
        organization_id=current_user.organization_id,
        user_id=current_user.id,
        action="TREATMENT_SIMULATION_REQUESTED",
        entity_type="treatment_run",
        entity_id=run_id,
        payload_json=json.dumps({
            "rover_id": body.rover_id,
            "target_pest": body.target_pest,
            "treatment_type": body.treatment_type,
            "frequency_hz": body.frequency_hz,
            "duration_sec": body.exposure_duration_sec,
        }),
    )
    db.add(audit)
    await db.commit()

    return TreatmentRunResponse(
        run_id=run_id,
        rover_id=body.rover_id,
        target_pest=body.target_pest,
        treatment_type=body.treatment_type,
        status="SIMULATION_LOGGED",
        actuation_executed=False,  # Physical actuation strictly disabled
        safety_disclaimer="Simulation run recorded in research telemetry. Physical acoustic emitters locked.",
    )
