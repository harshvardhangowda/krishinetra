"""
KrishiNetra Backend - Pytest Configuration & Test Fixtures
File: backend/tests/conftest.py
"""

import pytest_asyncio
from backend.app.core.database import engine, AsyncSessionLocal
from backend.app.infrastructure.db.base import Base
from backend.app.infrastructure.db.models import Organization, User, UserRole, Rover, RoverStatus
from backend.app.core.security import hash_password


@pytest_asyncio.fixture(scope="function", autouse=True)
async def initialize_test_database():
    """Ensure database schema is created and isolated test tenant populated per test function."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as db:
        org = Organization(
            name="KrishiNetra Test Labs",
            slug="test-org",
        )
        db.add(org)
        await db.flush()

        admin = User(
            organization_id=org.id,
            email="admin@krishinetra.ai",
            username="admin",
            full_name="Test Administrator",
            hashed_password=hash_password("KrishiNetra@2026"),
            role=UserRole.OWNER,
        )
        db.add(admin)

        rover = Rover(
            organization_id=org.id,
            name="Test Rover 01",
            serial_number="KNR-TEST-001",
            hardware_version="v1.0-sim",
            firmware_version="v1.0.0",
            status=RoverStatus.IDLE,
        )
        db.add(rover)
        await db.commit()

    yield

    # Cleanup after test
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
