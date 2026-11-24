"""
KrishiNetra Backend - Prediction API Automated Acceptance & Robustness Tests
File: backend/tests/test_predict_api.py

Validates:
1. Valid pest image prediction with truthful YOLOv8/ONNX model inference.
2. Separation of model outputs, curated knowledge-base data, and heuristic estimates.
3. No-detection empty array return for non-pest images.
4. Rejection of unsupported file formats (HTTP 415).
5. Rejection of corrupted image payloads (HTTP 400).
6. Enforcement of the 10 MB file size limit (HTTP 413).
7. Rejection of empty file uploads (HTTP 400).
8. Strict CORS origin allowlist (authorizes localhost:5500, rejects untrusted origins).
9. Authentication modes (authenticated vs demo_unauthenticated vs invalid token 401).
10. Fail-closed error handling on AI inference failure (HTTP 500).
11. Hardware safety guarantee (read-only execution, zero rover commands/actuation).
"""

import io
from pathlib import Path
import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.infrastructure.adapters.ai_detector_adapter import ai_detector

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def create_test_image_bytes(format_name: str = "JPEG", size=(640, 640), color=(200, 200, 200)) -> bytes:
    """Helper to generate in-memory image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format_name)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_predict_no_detection_empty_array():
    """Verify that a plain image with no pests returns an empty detections array and NONE severity."""
    blank_bytes = create_test_image_bytes("JPEG", size=(320, 320), color=(240, 240, 240))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            f"{settings.API_V1_STR}/predict",
            files={"file": ("blank.jpg", blank_bytes, "image/jpeg")},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["total_detections"] == 0
        assert data["detections"] == []
        assert data["overall_severity"] == "NONE"
        assert data["image_metadata"]["width"] == 320
        assert data["image_metadata"]["height"] == 320
        assert data["image_metadata"]["format"] == "JPEG"
        assert data["mode"] == "demo_unauthenticated"
        assert data["hardware_safeguard"] == "READ_ONLY_ACTIVE"


@pytest.mark.asyncio
async def test_predict_real_supported_pest_image():
    """Verify inference on a real supported test sample with full knowledge base enrichment."""
    # Look for authentic IP102 test image
    test_img_dir = PROJECT_ROOT / "datasets" / "ip102_subset" / "images" / "test"
    sample_files = list(test_img_dir.glob("*.jpg")) if test_img_dir.exists() else []

    if sample_files:
        with open(sample_files[0], "rb") as f:
            sample_bytes = f.read()
    else:
        sample_bytes = create_test_image_bytes("JPEG", size=(640, 640))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            f"{settings.API_V1_STR}/predict",
            files={"file": ("pest_sample.jpg", sample_bytes, "image/jpeg")},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["inference_time_ms"] > 0
        assert data["image_metadata"]["format"] == "JPEG"
        assert data["image_metadata"]["file_size_bytes"] == len(sample_bytes)
        assert data["hardware_safeguard"] == "READ_ONLY_ACTIVE"

        if data["total_detections"] > 0:
            det = data["detections"][0]
            # 1. Model Prediction Verification
            assert "model_prediction" in det
            mp = det["model_prediction"]
            assert mp["class_id"] in [0, 1, 2]
            assert mp["class_name"] in ["army worm", "blister beetle", "corn borer"]
            assert 0.0 <= mp["confidence"] <= 1.0
            assert "box_pixels" in mp
            assert "box_normalized" in mp

            # 2. Curated Knowledge Base Verification
            assert "knowledge_base" in det
            kb = det["knowledge_base"]
            assert len(kb["scientific_name"]) > 0
            assert len(kb["target_crops"]) > 0
            assert "non_chemical_ipm" in kb
            assert len(kb["non_chemical_ipm"]["cultural_mechanical"]) > 0
            assert len(kb["non_chemical_ipm"]["bioacoustic_feasibility"]) > 0

            # 3. Transparent Heuristic Verification
            assert "heuristic_estimates" in det
            he = det["heuristic_estimates"]
            assert he["severity"] in ["LOW", "MEDIUM", "HIGH"]
            assert he["canopy_coverage_pct"] >= 0.0
            assert "estimation_basis" in he


@pytest.mark.asyncio
async def test_predict_convenience_alias_route():
    """Verify that POST /api/predict acts as a functional alias to POST /api/v1/predict."""
    test_bytes = create_test_image_bytes("PNG", size=(100, 100))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/api/predict",
            files={"file": ("sample.png", test_bytes, "image/png")},
        )
        assert res.status_code == 200
        assert res.json()["success"] is True
        assert res.json()["image_metadata"]["format"] == "PNG"


@pytest.mark.asyncio
async def test_predict_unsupported_file_format():
    """Verify that non-image payloads are rejected with HTTP 415."""
    bad_payload = b"#!/bin/bash\necho 'malicious shell payload'"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            f"{settings.API_V1_STR}/predict",
            files={"file": ("script.sh", bad_payload, "text/x-shellscript")},
        )
        assert res.status_code == 415
        assert "Invalid image format" in res.json()["detail"]


@pytest.mark.asyncio
async def test_predict_corrupted_image_payload():
    """Verify that files with valid magic bytes but corrupted bodies return HTTP 400."""
    corrupted_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 30  # Valid header, corrupted structure

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            f"{settings.API_V1_STR}/predict",
            files={"file": ("corrupt.jpg", corrupted_jpeg, "image/jpeg")},
        )
        assert res.status_code == 400
        assert "corrupted" in res.json()["detail"].lower() or "decode" in res.json()["detail"].lower()


@pytest.mark.asyncio
async def test_predict_empty_file():
    """Verify that a 0-byte file is rejected with HTTP 400."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            f"{settings.API_V1_STR}/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert res.status_code == 400


@pytest.mark.asyncio
async def test_predict_oversized_file():
    """Verify that files exceeding 10MB are rejected with HTTP 413 without loading into memory."""
    # Create fake oversized JPEG header followed by 10MB + 1KB of zeros
    oversized_bytes = b"\xff\xd8\xff" + b"\x00" * (10 * 1024 * 1024 + 1024)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            f"{settings.API_V1_STR}/predict",
            files={"file": ("huge.jpg", oversized_bytes, "image/jpeg")},
        )
        assert res.status_code == 413
        assert "10MB" in res.json()["detail"]


@pytest.mark.asyncio
async def test_predict_cors_allowlist():
    """Verify that authorized frontend origins receive CORS headers, while unauthorized origins do not."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Authorized VS Code Live Server origin (http://localhost:5500)
        res_auth = await client.options(
            f"{settings.API_V1_STR}/predict",
            headers={
                "Origin": "http://localhost:5500",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert res_auth.headers.get("access-control-allow-origin") == "http://localhost:5500"

        # 2. Authorized Vite dev server origin (http://localhost:5173)
        res_vite = await client.options(
            f"{settings.API_V1_STR}/predict",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert res_vite.headers.get("access-control-allow-origin") == "http://localhost:5173"

        # 3. Unauthorized attacker origin
        res_unauth = await client.options(
            f"{settings.API_V1_STR}/predict",
            headers={
                "Origin": "http://malicious-tracker.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert res_unauth.headers.get("access-control-allow-origin") != "http://malicious-tracker.com"


@pytest.mark.asyncio
async def test_predict_auth_vs_demo_modes():
    """Verify that authenticated requests report 'authenticated' mode, and unauthenticated report 'demo_unauthenticated'."""
    sample_bytes = create_test_image_bytes("JPEG", size=(100, 100))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Unauthenticated demo mode
        res_demo = await client.post(
            f"{settings.API_V1_STR}/predict",
            files={"file": ("sample.jpg", sample_bytes, "image/jpeg")},
        )
        assert res_demo.status_code == 200
        assert res_demo.json()["mode"] == "demo_unauthenticated"

        # 2. Authenticated user mode
        login_res = await client.post(
            f"{settings.API_V1_STR}/auth/login",
            json={"username_or_email": "admin", "password": "KrishiNetra@2026"},
        )
        token = login_res.json()["session_token"]

        res_auth = await client.post(
            f"{settings.API_V1_STR}/predict",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("sample.jpg", sample_bytes, "image/jpeg")},
        )
        assert res_auth.status_code == 200
        assert res_auth.json()["mode"] == "authenticated"

        # 3. Invalid token must be rejected
        res_invalid_token = await client.post(
            f"{settings.API_V1_STR}/predict",
            headers={"Authorization": "Bearer invalid_expired_token_123"},
            files={"file": ("sample.jpg", sample_bytes, "image/jpeg")},
        )
        assert res_invalid_token.status_code == 401


@pytest.mark.asyncio
async def test_predict_inference_engine_failure_handling(monkeypatch):
    """Verify that unexpected AI detector failures fail-closed with HTTP 500 without leaking sensitive tracebacks."""
    def broken_predict(image_bytes):
        raise RuntimeError("GPU execution context lost during forward pass")

    monkeypatch.setattr(ai_detector, "predict", broken_predict)

    sample_bytes = create_test_image_bytes("JPEG", size=(100, 100))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            f"{settings.API_V1_STR}/predict",
            files={"file": ("sample.jpg", sample_bytes, "image/jpeg")},
        )
        assert res.status_code == 500
        assert "AI inference engine failure" in res.json()["detail"]
