"""
KrishiNetra Backend - API v1 Master Router
File: backend/app/api/v1/router.py
"""

from fastapi import APIRouter
from backend.app.api.v1.auth import router as auth_router
from backend.app.api.v1.rovers import router as rovers_router
from backend.app.api.v1.inspections import router as inspections_router
from backend.app.api.v1.ws import router as ws_router
from backend.app.api.routes.system import router as system_router
from backend.app.api.routes.telemetry import router as telemetry_router
from backend.app.api.routes.control import router as control_router
from backend.app.api.routes.treatments import router as treatments_router
from backend.app.api.routes.datasets import router as datasets_router
from backend.app.api.routes.assistant import router as assistant_router

api_v1_router = APIRouter()

# Core Authentication & Tenant Management
api_v1_router.include_router(auth_router)

# Platform System Status & Capability Discovery
api_v1_router.include_router(system_router)

# Rover Registry, Leases, and High-Level Commands
api_v1_router.include_router(rovers_router)

# Rover Telemetry, Sensors & Faults
api_v1_router.include_router(telemetry_router)

# Rover Motion, Arming, and Emergency STOP Control
api_v1_router.include_router(control_router)

# Inspections, Frames & AI Detections
api_v1_router.include_router(inspections_router)

# Treatment Requests & Bioassay Profiles
api_v1_router.include_router(treatments_router)

# Datasets & Trained Models Registry
api_v1_router.include_router(datasets_router)

# Read-Only AI Assistant with Evidence Citations
api_v1_router.include_router(assistant_router)

# Real-time WebSocket Channels
api_v1_router.include_router(ws_router)
