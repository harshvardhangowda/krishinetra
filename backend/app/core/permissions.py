"""
KrishiNetra Backend - RBAC Roles and Permission Matrix
File: backend/app/core/permissions.py
"""

import enum
from typing import Set


class Role(str, enum.Enum):
    OWNER = "owner"
    OPERATOR = "operator"
    AGRONOMY_REVIEWER = "reviewer"
    VIEWER = "viewer"


class Permission(str, enum.Enum):
    ROVER_VIEW = "rover:view"
    ROVER_DRIVE = "rover:drive"
    ROVER_CALIBRATE = "rover:calibrate"
    ROVER_EMERGENCY_STOP = "rover:stop"
    INSPECTION_START = "inspection:start"
    INSPECTION_REVIEW = "inspection:review"
    DATASET_MANAGE = "dataset:manage"
    MODEL_ACTIVATE = "model:activate"
    TREATMENT_REQUEST = "treatment:request"
    TREATMENT_APPROVE = "treatment:approve"
    ORG_ADMIN = "org:admin"


ROLE_PERMISSIONS: dict[Role, Set[Permission]] = {
    Role.OWNER: set(Permission),
    Role.OPERATOR: {
        Permission.ROVER_VIEW,
        Permission.ROVER_DRIVE,
        Permission.ROVER_EMERGENCY_STOP,
        Permission.INSPECTION_START,
        Permission.TREATMENT_REQUEST,
    },
    Role.AGRONOMY_REVIEWER: {
        Permission.ROVER_VIEW,
        Permission.INSPECTION_REVIEW,
        Permission.TREATMENT_APPROVE,
        Permission.DATASET_MANAGE,
    },
    Role.VIEWER: {
        Permission.ROVER_VIEW,
    },
}


def has_permission(user_role: Role, required_permission: Permission) -> bool:
    """Check if given role possesses the required permission."""
    return required_permission in ROLE_PERMISSIONS.get(user_role, set())
