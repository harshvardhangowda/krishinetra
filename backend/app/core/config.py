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

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8501",  # Streamlit dashboard
        "http://localhost:8000",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
