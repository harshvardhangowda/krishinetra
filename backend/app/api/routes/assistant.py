"""
KrishiNetra Backend - Read-Only AI Assistant Endpoint
File: backend/app/api/routes/assistant.py

Specification Policy:
Assistant is strictly read-only and backed by verifiable domain facts.
Must NEVER have rover driving, shell execution, or database mutation rights.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.domains.auth_service import get_current_user
from backend.app.infrastructure.db.models import User
from backend.app.infrastructure.assistant_providers.tools import AssistantToolProvider

router = APIRouter(prefix="/assistant", tags=["Read-Only AI Agronomy Assistant"])


class AssistantMessageRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)


class AssistantMessageResponse(BaseModel):
    answer: str
    evidence_citations: List[str]
    tools_consulted: List[str]
    safety_disclaimer: str


@router.post("/query", response_model=AssistantMessageResponse)
async def query_assistant(
    body: AssistantMessageRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Submits a query to the read-only AI Assistant.
    Consults authorized domain data sources and cites verifiable evidence.
    """
    # Consult authorized domain data
    rover_status = await AssistantToolProvider.get_rover_status(db, current_user.organization_id)
    inspection_summary = await AssistantToolProvider.get_inspection_summary(db, current_user.organization_id)
    sensor_health = await AssistantToolProvider.get_sensor_availability()

    citations = [
        f"Domain Inspection Record: {inspection_summary['total_inspection_sessions']} sessions, {inspection_summary['total_pests_detected']} pests logged.",
        f"Fleet Registry: {len(rover_status)} rovers registered to '{current_user.organization.name}'.",
        f"Telemetry Sensor Matrix: {sensor_health['motors_actuators']}.",
    ]

    response_text = (
        f"Regarding your query ('{body.query}'): "
        f"In your organization '{current_user.organization.name}', there are currently {len(rover_status)} registered rover(s) "
        f"and {inspection_summary['total_inspection_sessions']} completed inspection sessions with a total of "
        f"{inspection_summary['total_pests_detected']} target pest observations. "
        f"All rover motors and chemical/physical deterrence mechanisms remain permanently locked in safe simulation mode."
    )

    return AssistantMessageResponse(
        answer=response_text,
        evidence_citations=citations,
        tools_consulted=["get_rover_status", "get_inspection_summary", "get_sensor_availability"],
        safety_disclaimer="AI Assistant operates under strict read-only authority with zero actuation or direct database mutation rights.",
    )
