"""
KrishiNetra Backend - Structured Error Hierarchy & Handlers
File: backend/app/core/errors.py
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class KrishiNetraError(Exception):
    """Base exception for all domain and platform errors."""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class EntityNotFoundError(KrishiNetraError):
    def __init__(self, entity: str, entity_id: str):
        super().__init__(f"{entity} with ID '{entity_id}' not found.", code="NOT_FOUND")
        self.status_code = status.HTTP_404_NOT_FOUND


class ConflictError(KrishiNetraError):
    def __init__(self, message: str):
        super().__init__(message, code="CONFLICT")
        self.status_code = status.HTTP_409_CONFLICT


class LeaseViolationError(KrishiNetraError):
    def __init__(self, message: str):
        super().__init__(message, code="LEASE_VIOLATION")
        self.status_code = status.HTTP_403_FORBIDDEN


class SafetyInterlockViolationError(KrishiNetraError):
    def __init__(self, message: str):
        super().__init__(message, code="SAFETY_INTERLOCK_LOCKED")
        self.status_code = status.HTTP_423_LOCKED
