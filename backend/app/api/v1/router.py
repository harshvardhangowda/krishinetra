"""
KrishiNetra Backend - API v1 Master Router
File: backend/app/api/v1/router.py
"""

from fastapi import APIRouter
from backend.app.api.v1.auth import router as auth_router
from backend.app.api.v1.rovers import router as rovers_router
from backend.app.api.v1.inspections import router as inspections_router
from backend.app.api.v1.ws import router as ws_router

api_v1_router = APIRouter()
api_v1_router.include_router(auth_router)
api_v1_router.include_router(rovers_router)
api_v1_router.include_router(inspections_router)
api_v1_router.include_router(ws_router)
