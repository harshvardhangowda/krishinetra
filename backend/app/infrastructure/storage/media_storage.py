"""
KrishiNetra Backend - Cryptographic Media Storage Service
File: backend/app/infrastructure/storage/media_storage.py
"""

from pathlib import Path
from typing import Tuple
from backend.app.core.config import settings
from backend.app.core.security import compute_sha256


class MediaStorageService:
    """Manages raw inspection frames with SHA-256 tamper-evident integrity."""

    def __init__(self, base_dir: str = settings.MEDIA_STORAGE_PATH) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def store_frame(
        self,
        session_id: str,
        frame_index: int,
        image_bytes: bytes,
        extension: str = "jpg",
    ) -> Tuple[str, str]:
        """
        Save frame to disk and return (relative_storage_path, sha256_hash).
        """
        sha256_hash = compute_sha256(image_bytes)
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        filename = f"frame_{frame_index:06d}_{sha256_hash[:12]}.{extension}"
        target_path = session_dir / filename
        target_path.write_bytes(image_bytes)

        rel_path = str(target_path.relative_to(self.base_dir.parent))
        return rel_path, sha256_hash


media_storage = MediaStorageService()
