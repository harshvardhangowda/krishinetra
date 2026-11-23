"""
KrishiNetra Backend - Pytest Configuration & Test Fixtures
File: backend/tests/conftest.py
"""

import pytest_asyncio
from backend.app.core.database import engine
from backend.app.infrastructure.db.base import Base
from backend.app.main import seed_initial_data


@pytest_asyncio.fixture(scope="session", autouse=True)
async def initialize_test_database():
    """Ensure database schema is created and initial seed data populated before running tests."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await seed_initial_data()
    yield
    # Cleanup after test session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()
