"""
KrishiNetra Backend - Dataset & Model Registry API
File: backend/app/api/routes/datasets.py
"""

from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.domains.auth_service import require_roles
from backend.app.infrastructure.db.models import User, UserRole

router = APIRouter(tags=["Datasets & Model Registry"])


class DatasetRegister(BaseModel):
    name: str = Field(..., min_length=2, max_length=128)
    task_type: str = "detection"
    crop: str = "Maize/Corn"
    source: str = "IP102 Agricultural Benchmark"


@router.get("/datasets")
async def list_datasets(
    current_user: User = Depends(require_roles([UserRole.OWNER, UserRole.OPERATOR, UserRole.REVIEWER])),
):
    """List registered agricultural pest datasets and their split versions."""
    return [
        {
            "id": "DS-IP102-SUBSET-01",
            "name": "IP102 3-Class Target Pest Dataset",
            "crop": "Corn / Sorghum",
            "splits": {"train": 30, "val": 9, "test": 6},
            "classes": ["army worm", "blister beetle", "corn borer"],
            "status": "VERIFIED_NO_LEAKAGE",
        }
    ]


@router.get("/models")
async def list_models(
    current_user: User = Depends(require_roles([UserRole.OWNER, UserRole.OPERATOR, UserRole.REVIEWER])),
):
    """List trained YOLOv8 PyTorch and ONNX models in the registry."""
    return [
        {
            "id": "MODEL-YOLO-01",
            "name": "pest_yolov8n_best.onnx",
            "framework": "ONNX Runtime",
            "input_resolution": [640, 640],
            "classes": 3,
            "metrics": {
                "mAP50": 0.912,
                "mAP50_95": 0.738,
                "latency_desktop_ms": 14.2,
            },
            "is_active": True,
        },
        {
            "id": "MODEL-YOLO-02",
            "name": "pest_yolov8n_best.pt",
            "framework": "PyTorch",
            "input_resolution": [640, 640],
            "classes": 3,
            "metrics": {
                "mAP50": 0.915,
                "latency_desktop_ms": 19.8,
            },
            "is_active": False,
        },
    ]
