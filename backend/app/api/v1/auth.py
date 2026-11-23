"""
KrishiNetra Backend - Authentication & User Endpoints
File: backend/app/api/v1/auth.py
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.core.security import hash_password
from backend.app.contracts.auth import (
    LoginRequest,
    LoginResponse,
    UserCreate,
    UserResponse,
    OrganizationCreate,
    OrganizationResponse,
)
from backend.app.infrastructure.db.models import Organization, User, UserRole, Session
from backend.app.domains.auth_service import (
    authenticate_user,
    create_user_session,
    get_current_user_and_session,
    get_current_user,
    require_roles,
)

router = APIRouter(prefix="/auth", tags=["Authentication & Access"])


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """Authenticate with username/email and password. Returns session and CSRF tokens."""
    user = await authenticate_user(db, body.username_or_email, body.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials or inactive account.",
        )

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    session = await create_user_session(
        db=db,
        user=user,
        user_agent=user_agent,
        ip_address=client_ip,
    )

    return LoginResponse(
        session_token=session.session_token,
        csrf_token=session.csrf_token,
        expires_at=session.expires_at,
        user_id=user.id,
        username=user.username,
        organization_id=user.organization_id,
        role=user.role,
    )


@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout(
    auth_data: tuple[User, Session] = Depends(get_current_user_and_session),
    db: AsyncSession = Depends(get_db),
):
    """Terminate the current active session."""
    user, session = auth_data
    await db.delete(session)
    await db.commit()
    return {"status": "success", "message": "Logged out successfully."}


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: User = Depends(get_current_user),
):
    """Return currently authenticated user profile."""
    return current_user


@router.post("/organizations", response_model=OrganizationResponse, status_code=status.HTTP_201_CREATED)
async def create_organization(
    body: OrganizationCreate,
    db: AsyncSession = Depends(get_db),
):
    """Register a new organization tenant."""
    # Check slug collision
    stmt = select(Organization).where(Organization.slug == body.slug)
    existing = (await db.execute(stmt)).scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Organization with slug '{body.slug}' already exists.",
        )

    org = Organization(name=body.name, slug=body.slug)
    db.add(org)
    await db.commit()
    await db.refresh(org)
    return org


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    body: UserCreate,
    current_user: User = Depends(require_roles([UserRole.OWNER])),
    db: AsyncSession = Depends(get_db),
):
    """Owner registers a new organization member with specific RBAC role."""
    # Ensure owner only creates users in their own organization
    if body.organization_id != current_user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot create users outside your organization.",
        )

    # Check email / username uniqueness
    stmt = select(User).where((User.email == body.email) | (User.username == body.username))
    existing = (await db.execute(stmt)).scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username or email is already registered.",
        )

    new_user = User(
        organization_id=body.organization_id,
        email=body.email,
        username=body.username,
        full_name=body.full_name,
        hashed_password=hash_password(body.password),
        role=body.role,
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user
