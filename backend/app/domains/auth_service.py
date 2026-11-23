"""
KrishiNetra Backend - Authentication & RBAC Service
File: backend/app/domains/auth_service.py
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, List, Callable
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.core.security import (
    hash_password,
    verify_password,
    generate_secure_token,
)
from backend.app.infrastructure.db.models import (
    User,
    Session,
    Organization,
    UserRole,
    AuditLog,
    utcnow,
)

bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-Session-Token", auto_error=False)


async def authenticate_user(
    db: AsyncSession,
    username_or_email: str,
    password: str,
) -> Optional[User]:
    """Authenticate user by username or email and verify password hash."""
    stmt = (
        select(User)
        .where(
            (User.username == username_or_email) | (User.email == username_or_email)
        )
        .options(selectinload(User.organization))
    )
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        return None

    if not verify_password(password, user.hashed_password):
        return None

    return user


async def create_user_session(
    db: AsyncSession,
    user: User,
    user_agent: Optional[str] = None,
    ip_address: Optional[str] = None,
) -> Session:
    """Create a new authenticated session with CSRF protection token."""
    session_token = generate_secure_token(32)
    csrf_token = generate_secure_token(32)
    expires_at = utcnow() + timedelta(hours=settings.SESSION_EXPIRE_HOURS)

    session = Session(
        user_id=user.id,
        organization_id=user.organization_id,
        session_token=session_token,
        csrf_token=csrf_token,
        expires_at=expires_at,
        user_agent=user_agent,
        ip_address=ip_address,
    )
    db.add(session)

    # Log audit entry
    audit = AuditLog(
        organization_id=user.organization_id,
        user_id=user.id,
        action="AUTH_LOGIN_SUCCESS",
        entity_type="session",
        entity_id=session.id,
        ip_address=ip_address,
    )
    db.add(audit)

    await db.commit()
    await db.refresh(session)
    return session


async def get_current_user_and_session(
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    token_header: Optional[str] = Security(api_key_header),
    db: AsyncSession = Depends(get_db),
) -> tuple[User, Session]:
    """Extract and validate the active session from Bearer or X-Session-Token header."""
    token = None
    if bearer and bearer.credentials:
        token = bearer.credentials
    elif token_header:
        token = token_header

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide valid Bearer or X-Session-Token header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    stmt = (
        select(Session)
        .where(Session.session_token == token)
        .options(selectinload(Session.user).selectinload(User.organization))
    )
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token.",
        )

    # Verify expiration (aware datetime comparison)
    now_utc = utcnow()
    expires_at = session.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if now_utc > expires_at:
        await db.delete(session)
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has expired. Please re-authenticate.",
        )

    if not session.user or not session.user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated.",
        )

    return session.user, session


async def get_current_user(
    auth_data: tuple[User, Session] = Depends(get_current_user_and_session),
) -> User:
    """Dependency helper returning only the authenticated User."""
    return auth_data[0]


def require_roles(allowed_roles: List[UserRole]) -> Callable:
    """Enforce strict Role-Based Access Control (RBAC)."""
    async def role_checker(
        user: User = Depends(get_current_user),
    ) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Requires one of roles: {[r.value for r in allowed_roles]}.",
            )
        return user

    return role_checker
