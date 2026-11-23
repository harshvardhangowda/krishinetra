"""
KrishiNetra Backend - Production FastAPI Application
File: backend/app/main.py
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from backend.app.core.config import settings
from backend.app.core.database import engine, AsyncSessionLocal
from backend.app.infrastructure.db.base import Base
from backend.app.infrastructure.db.models import (
    Organization,
    User,
    Rover,
    UserRole,
    RoverStatus,
)
from backend.app.core.security import hash_password
from backend.app.api.v1.router import api_v1_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and graceful shutdown lifecycle."""
    # Ensure all relational database tables exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    # Teardown database pool
    await engine.dispose()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Production REST & WebSocket platform for KrishiNetra precision agriculture robotics and pest deterrence.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount versioned API routes
app.include_router(api_v1_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
async def root():
    """Service status and OpenAPI documentation link."""
    return {
        "service": settings.PROJECT_NAME,
        "version": "1.0.0",
        "status": "OPERATIONAL",
        "documentation": "/docs",
        "api_v1": settings.API_V1_STR,
    }


@app.get("/health/live", tags=["Health"])
async def health_liveness():
    """Kubernetes / Docker liveness probe: verifies process is alive."""
    return {"status": "ALIVE"}


@app.get("/health/ready", tags=["Health"])
async def health_readiness():
    """Readiness probe: validates active database connectivity."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "READY", "database": "CONNECTED"}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database unreachable: {str(exc)}",
        )
