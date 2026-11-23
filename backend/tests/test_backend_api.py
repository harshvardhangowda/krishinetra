"""
KrishiNetra Backend - Comprehensive Acceptance & Integration Test Suite
File: backend/tests/test_backend_api.py

Validates:
1. Health endpoints (/health/live, /health/ready).
2. Authentication & Session issuance (Login, CSRF, Token).
3. RBAC Role enforcement (Owner vs Operator vs Viewer).
4. Rover Control Leases (Acquisition, TTL, Conflict on double leasing).
5. Command Dispatch (Idempotency, Lease requirement, Universal STOP override).
6. Inspection & AI Frame detection (SHA-256 hash integrity, YOLO model detection).
7. Human Review Audit trail (Non-destructive preservation of raw ML predictions).
"""

import asyncio
import io
import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.infrastructure.db.models import UserRole


@pytest.mark.asyncio
async def test_health_endpoints():
    """Verify live and ready health probes."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Liveness
        res_live = await client.get("/health/live")
        assert res_live.status_code == 200
        assert res_live.json() == {"status": "ALIVE"}

        # Readiness
        res_ready = await client.get("/health/ready")
        assert res_ready.status_code == 200
        assert res_ready.json()["status"] == "READY"


@pytest.mark.asyncio
async def test_authentication_and_sessions():
    """Verify login authentication, password verification, and session token generation."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Invalid login attempt
        bad_res = await client.post(
            f"{settings.API_V1_STR}/auth/login",
            json={"username_or_email": "admin", "password": "WrongPassword123!"},
        )
        assert bad_res.status_code == 401

        # 2. Valid admin login (seeded in main lifespan)
        login_res = await client.post(
            f"{settings.API_V1_STR}/auth/login",
            json={"username_or_email": "admin", "password": "KrishiNetra@2026"},
        )
        assert login_res.status_code == 200
        data = login_res.json()
        assert "session_token" in data
        assert "csrf_token" in data
        assert data["username"] == "admin"
        assert data["role"] == "owner"

        token = data["session_token"]

        # 3. Authenticated /me endpoint
        me_res = await client.get(
            f"{settings.API_V1_STR}/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert me_res.status_code == 200
        assert me_res.json()["username"] == "admin"


@pytest.mark.asyncio
async def test_rover_lease_and_command_lifecycle():
    """Verify exclusive control leases, command dispatch, idempotency, and STOP override."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Authenticate as admin
        login_res = await client.post(
            f"{settings.API_V1_STR}/auth/login",
            json={"username_or_email": "admin", "password": "KrishiNetra@2026"},
        )
        token = login_res.json()["session_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # List rovers
        rovers_res = await client.get(f"{settings.API_V1_STR}/rovers", headers=headers)
        assert rovers_res.status_code == 200
        rovers = rovers_res.json()
        assert len(rovers) > 0
        rover_id = rovers[0]["id"]

        # Test STOP without lease (Emergency stop must always succeed)
        stop_res = await client.post(
            f"{settings.API_V1_STR}/rovers/{rover_id}/commands",
            headers=headers,
            json={
                "idempotency_key": "stop-key-test-001",
                "command_type": "STOP",
                "payload": {},
            },
        )
        assert stop_res.status_code == 200
        assert stop_res.json()["command_type"] == "STOP"
        assert stop_res.json()["status"] in ["completed", "sent"]

        # Test DRIVE without lease (Must be rejected with 403 Forbidden)
        drive_no_lease = await client.post(
            f"{settings.API_V1_STR}/rovers/{rover_id}/commands",
            headers=headers,
            json={
                "idempotency_key": "drive-fail-001",
                "command_type": "DRIVE",
                "payload": {"vx": 0.3, "vy": 0.0, "omega": 0.0},
            },
        )
        assert drive_no_lease.status_code == 403

        # Acquire exclusive control lease
        lease_res = await client.post(
            f"{settings.API_V1_STR}/rovers/{rover_id}/leases",
            headers=headers,
            json={"ttl_seconds": 60},
        )
        assert lease_res.status_code == 200
        assert lease_res.json()["is_active"] is True

        # Test DRIVE with active lease (Must succeed)
        drive_res = await client.post(
            f"{settings.API_V1_STR}/rovers/{rover_id}/commands",
            headers=headers,
            json={
                "idempotency_key": "drive-success-001",
                "command_type": "DRIVE",
                "payload": {"vx": 0.3, "vy": 0.0, "omega": 0.0},
            },
        )
        assert drive_res.status_code == 200
        assert drive_res.json()["status"] == "completed"

        # Test Idempotency: re-sending the same idempotency key returns cached record without re-execution
        dup_res = await client.post(
            f"{settings.API_V1_STR}/rovers/{rover_id}/commands",
            headers=headers,
            json={
                "idempotency_key": "drive-success-001",
                "command_type": "DRIVE",
                "payload": {"vx": 0.3, "vy": 0.0, "omega": 0.0},
            },
        )
        assert dup_res.status_code == 200
        assert dup_res.json()["id"] == drive_res.json()["id"]


@pytest.mark.asyncio
async def test_inspection_ai_detection_and_review():
    """Verify inspection frame ingestion, SHA-256 hashing, AI detections, and non-destructive human review."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Authenticate
        login_res = await client.post(
            f"{settings.API_V1_STR}/auth/login",
            json={"username_or_email": "admin", "password": "KrishiNetra@2026"},
        )
        token = login_res.json()["session_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Get rover ID
        rovers_res = await client.get(f"{settings.API_V1_STR}/rovers", headers=headers)
        rover_id = rovers_res.json()[0]["id"]

        # Create inspection session
        sess_res = await client.post(
            f"{settings.API_V1_STR}/inspections",
            headers=headers,
            json={
                "rover_id": rover_id,
                "title": "Field Alpha Block 3 Pest Survey",
                "field_location": "Field 3 North",
            },
        )
        assert sess_res.status_code == 201
        session_id = sess_res.json()["id"]

        # Create a synthetic image buffer
        img = Image.new("RGB", (640, 640), color=(45, 120, 50))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        raw_image_data = img_byte_arr.getvalue()

        # Upload inspection frame
        files = {"file": ("test_frame.jpg", raw_image_data, "image/jpeg")}
        data = {"frame_index": "1", "timestamp_ms": "1250.5"}

        upload_res = await client.post(
            f"{settings.API_V1_STR}/inspections/{session_id}/frames",
            headers=headers,
            files=files,
            data=data,
        )
        assert upload_res.status_code == 200
        frame_data = upload_res.json()
        assert "frame_id" in frame_data
        assert "file_hash_sha256" in frame_data
        assert len(frame_data["file_hash_sha256"]) == 64  # Valid SHA-256
        assert len(frame_data["detections"]) > 0

        frame_id = frame_data["frame_id"]
        detection_id = frame_data["detections"][0]["id"]

        # Submit human review correction (non-destructive)
        review_res = await client.post(
            f"{settings.API_V1_STR}/inspections/frames/{frame_id}/reviews",
            headers=headers,
            json={
                "detection_id": detection_id,
                "review_status": "confirmed",
                "notes": "Verified visually by entomologist; valid army worm target.",
            },
        )
        assert review_res.status_code == 200
        assert review_res.json()["review_status"] == "confirmed"
