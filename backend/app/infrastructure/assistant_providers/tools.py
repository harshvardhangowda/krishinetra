"""
KrishiNetra Backend - Read-Only AI Assistant Providers & Tools
File: backend/app/infrastructure/assistant_providers/tools.py

Specification Requirement:
Strictly read-only assistant tools. Must NEVER have shell execution,
direct database mutation, rover driving, or actuator control access.
"""

from typing import Dict, Any, List
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.infrastructure.db.models import (
    Rover,
    InspectionSession,
    HumanReview,
    ModelRegistry,
    RoverStatus,
)


class AssistantToolProvider:
    """Provides evidence-backed, read-only analytical summaries to the AI assistant."""

    @staticmethod
    async def get_rover_status(db: AsyncSession, organization_id: str) -> List[Dict[str, Any]]:
        stmt = select(Rover).where(Rover.organization_id == organization_id)
        rovers = (await db.execute(stmt)).scalars().all()
        return [
            {
                "rover_id": r.id,
                "name": r.name,
                "status": r.status.value,
                "battery_pct": r.battery_pct,
                "hardware_version": r.hardware_version,
                "firmware_version": r.firmware_version,
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
    async def get_reviewed_observations(db: AsyncSession, limit: int = 10) -> List[Dict[str, Any]]:
        stmt = select(HumanReview).order_by(HumanReview.reviewed_at.desc()).limit(limit)
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
        return {
            "camera": "OPERATIONAL (Synthetic / Video Ingestion)",
            "ultrasonic_distance": "SIMULATED (Range: 0.15m - 2.50m)",
            "bme280_temp_humidity": "SIMULATED (26.5 C, 62% RH)",
            "motors_actuators": "DISABLED (Safety Interlock Active)",
        }
