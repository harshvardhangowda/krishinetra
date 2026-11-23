"""
KrishiNetra Backend - Monotonic and UTC Clock Primitives
File: backend/app/core/clock.py
"""

from datetime import datetime, timezone
import time


def utcnow() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


def monotonic_time() -> float:
    """Return monotonic clock value in seconds for drift-free lease timing."""
    return time.monotonic()


def is_expired(expiry: datetime) -> bool:
    """Safely check if an expiry datetime is in the past."""
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    return utcnow() > expiry
