"""
KrishiNetra Backend - Cryptographic Security & Password Management
File: backend/app/core/security.py
"""

import secrets
import hashlib
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Generate secure bcrypt password hash."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain text password against stored bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


def generate_secure_token(nbytes: int = 32) -> str:
    """Generate cryptographically secure URL-safe random string."""
    return secrets.token_urlsafe(nbytes)


def compute_sha256(data: bytes) -> str:
    """Compute standard SHA-256 hex digest for evidence integrity verification."""
    return hashlib.sha256(data).hexdigest()
