"""
KrishiNetra Backend - System Capabilities & Diagnostics Routes
File: backend/app/api/routes/system.py
"""

from fastapi import APIRouter, Depends
from backend.app.domains.auth_service import get_current_user
from backend.app.infrastructure.db.models import User

router = APIRouter(prefix="/system", tags=["System & Platform"])


@router.get("/status")
async def get_system_status(
    current_user: User = Depends(get_current_user),
):
    """Detailed authenticated platform status report."""
    return {
        "platform": "KrishiNetra Enterprise AI Backend",
        "version": "1.0.0",
        "environment": "simulation-desktop",
        "database": "operational",
        "safety_interlock": "HARDWARE_ACTUATION_LOCKED",
        "user_organization": current_user.organization.name,
    }


@router.get("/capabilities")
async def get_system_capabilities():
    """Returns supported hardware capability flags (truthful platform reporting)."""
    return {
        "camera_input": {
            "synthetic_canopy": True,
            "recorded_clips_mp4": True,
            "physical_v4l2_camera": False,  # Truthful reporting: no fake physical camera
        },
        "pest_detection_models": {
            "yolov8n_pytorch": True,
            "yolov8n_onnx": True,
            "classes": ["army worm", "blister beetle", "corn borer"],
        },
        "actuation": {
            "mecanum_rover_motors": "SIMULATION_ONLY",
            "ultrasonic_deterrence": "SIMULATION_ONLY",
            "physical_hardware_actuators": "PERMANENTLY_DISABLED",
        },
    }
