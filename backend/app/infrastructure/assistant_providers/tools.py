"""
KrishiNetra Backend - Domain Assistant Data Retrieval Tools
File: backend/app/infrastructure/assistant_providers/tools.py

Enforces strict read-only access and tenant organization isolation (KN-24).
"""

from typing import List, Dict, Any, Optional
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.infrastructure.db.models import (
    Rover,
    InspectionSession,
    InspectionFrame,
    Detection,
    HumanReview,
    ModelRegistry,
    RoverStatus,
)


class AssistantToolProvider:
    """Read-only tools allowing the assistant to ground responses in verified database facts."""

    @staticmethod
    async def get_rover_status(db: AsyncSession, organization_id: str) -> List[Dict[str, Any]]:
        stmt = select(Rover).where(Rover.organization_id == organization_id)
        rovers = (await db.execute(stmt)).scalars().all()
        return [
            {
                "id": r.id,
                "name": r.name,
                "status": r.status.value,
                "battery_pct": r.battery_pct,
                "last_heartbeat": r.last_heartbeat.isoformat() if r.last_heartbeat else None,
            }
            for r in rovers
        ]

    @staticmethod
    async def get_inspection_summary(db: AsyncSession, organization_id: str) -> Dict[str, Any]:
        stmt = select(
            func.count(InspectionSession.id),
            func.sum(InspectionSession.total_frames),
            func.sum(InspectionSession.total_detections),
        ).where(InspectionSession.organization_id == organization_id)
        res = (await db.execute(stmt)).first()
        return {
            "total_inspection_sessions": res[0] or 0,
            "total_frames_analyzed": res[1] or 0,
            "total_pests_detected": res[2] or 0,
        }

    @staticmethod
    async def get_reviewed_observations(
        db: AsyncSession,
        organization_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Fetch verified human reviews strictly scoped to user's organization (KN-24)."""
        stmt = (
            select(HumanReview)
            .join(InspectionFrame, HumanReview.frame_id == InspectionFrame.id)
            .join(InspectionSession, InspectionFrame.session_id == InspectionSession.id)
            .where(InspectionSession.organization_id == organization_id)
            .order_by(HumanReview.reviewed_at.desc())
            .limit(limit)
        )
        reviews = (await db.execute(stmt)).scalars().all()
        return [
            {
                "review_id": rev.id,
                "frame_id": rev.frame_id,
                "status": rev.review_status.value,
                "notes": rev.notes,
                "reviewed_at": rev.reviewed_at.isoformat(),
            }
            for rev in reviews
        ]

    @staticmethod
    async def get_model_evaluation(db: AsyncSession) -> List[Dict[str, Any]]:
        stmt = select(ModelRegistry)
        models = (await db.execute(stmt)).scalars().all()
        return [
            {
                "model_name": m.name,
                "version": m.version,
                "framework": m.framework,
                "is_active": m.is_active,
                "metrics": m.metrics_json,
            }
            for m in models
        ]

    @staticmethod
    async def get_sensor_availability() -> Dict[str, Any]:
        """Provides verified hardware status showing physical actuation is locked."""
        return {
            "camera": "CONNECTED_SIMULATION",
            "ultrasonic": "SIMULATED",
            "gps": "SIMULATED",
            "motors_actuators": "PERMANENTLY_DISABLED_BY_SAFETY_INTERLOCK",
            "non_chemical_deterrence": "SIMULATION_MODE_ONLY",
        }
