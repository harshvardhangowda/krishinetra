"""
KrishiNetra Backend - Authentication Pydantic Contracts
File: backend/app/contracts/auth.py
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from backend.app.infrastructure.db.models import UserRole


class LoginRequest(BaseModel):
    username_or_email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)


class LoginResponse(BaseModel):
    session_token: str
    csrf_token: str
    token_type: str = "Bearer"
    expires_at: datetime
    user_id: str
    username: str
    organization_id: str
    role: UserRole


class UserCreate(BaseModel):
    organization_id: str
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=64)
    full_name: str = Field(..., min_length=2, max_length=128)
    password: str = Field(..., min_length=8, max_length=128)
    role: UserRole = UserRole.VIEWER


class UserResponse(BaseModel):
    id: str
    organization_id: str
    email: EmailStr
    username: str
    full_name: str
    role: UserRole
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class OrganizationCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=128)
    slug: str = Field(..., min_length=2, max_length=64)


class OrganizationResponse(BaseModel):
    id: str
    name: str
    slug: str
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
