"""
KrishiNetra Backend - Core Settings & Configuration
File: backend/app/core/config.py
"""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """Production configuration with environment variable override."""

    PROJECT_NAME: str = "KrishiNetra Precision Pest Management Platform"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "krishnetra-secure-production-secret-key-change-in-production"
    ALGORITHM: str = "HS256"

    # Database: defaults to async SQLite for local execution, or PostgreSQL via env
    DATABASE_URL: str = "sqlite+aiosqlite:///./krishnetra_backend.db"

    # Security & Sessions
    SESSION_EXPIRE_HOURS: int = 24
    CSRF_HEADER_NAME: str = "X-CSRF-Token"
    RATE_LIMIT_LOGIN_PER_MINUTE: int = 10

    # Rover Lease Configuration
    ROVER_LEASE_DEFAULT_TTL_SEC: int = 30
    ROVER_COMMAND_TIMEOUT_SEC: float = 5.0
    ROVER_HEARTBEAT_TIMEOUT_SEC: float = 10.0

    # Media Evidence Storage
    MEDIA_STORAGE_PATH: str = str(PROJECT_ROOT / "data" / "media_evidence")

    # CORS - Explicit allowlist for trusted origins (no wildcard permitted)
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # React / Next frontend
        "http://localhost:8501",  # Streamlit dashboard
        "http://localhost:8000",  # FastAPI swagger/docs
        "http://localhost:5500",  # VS Code Live Server
        "http://127.0.0.1:5500",  # VS Code Live Server (IP alias)
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",  # Vite dev server (IP alias)
        "http://localhost:8080",  # Local http-server
        "http://127.0.0.1:8080",  # Local http-server (IP alias)
    ]

    # Frontend Prediction API Configuration
    ALLOW_DEMO_PREDICTION: bool = True
    PREDICTION_MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB strict limit

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
