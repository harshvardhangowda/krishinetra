"""
KrishiNetra Backend - Treatment & Deterrence Workflow API
File: backend/app/api/routes/treatments.py
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.domains.auth_service import require_roles
from backend.app.infrastructure.db.models import User, UserRole

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
    """List standardized scientific non-chemical deterrence acoustic profiles."""
    return [
        {
            "id": "TP-01-ARMYWORM",
            "pest": "Spodoptera frugiperda (Fall Armyworm)",
            "deterrence_type": "SWEPT_PULSE_US",
            "frequency_range_hz": [35000, 42000],
            "optimal_spl_db": 95.0,
            "status": "VALIDATED_BIOASSAY",
        },
        {
            "id": "TP-02-CORN_BORER",
            "pest": "Ostrinia nubilalis (European Corn Borer)",
            "deterrence_type": "CONTINUOUS_US",
            "frequency_range_hz": [38000, 40000],
            "optimal_spl_db": 92.0,
            "status": "VALIDATED_BIOASSAY",
        },
    ]


@router.post("/requests", response_model=TreatmentRunResponse, status_code=status.HTTP_201_CREATED)
async def request_treatment(
    body: TreatmentRequest,
    current_user: User = Depends(require_roles([UserRole.OWNER, UserRole.OPERATOR])),
):
    """
    Submits a targeted deterrence treatment request.
    Strict Safety Policy: Operates in SIMULATION ONLY.
    Physical actuators remain disabled until approved agronomic bioassay field validation.
    """
    return TreatmentRunResponse(
        run_id="TR-SIM-20261123-01",
        rover_id=body.rover_id,
        target_pest=body.target_pest,
        treatment_type=body.treatment_type,
        status="SIMULATION_LOGGED",
        actuation_executed=False,  # Physical actuation strictly disabled
        safety_disclaimer="Simulation run recorded in research telemetry. Physical acoustic emitters locked.",
    )
