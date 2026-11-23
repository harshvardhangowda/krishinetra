"""
KrishiNetra Backend - Storage Infrastructure
File: backend/app/infrastructure/storage/local_storage.py
"""

from pathlib import Path
from typing import Tuple
from backend.app.core.config import settings
from backend.app.core.security import compute_sha256


class LocalStorageProvider:
    """Manages disk-based private media storage with SHA-256 verification."""

    def __init__(self, root_dir: str = settings.MEDIA_STORAGE_PATH):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_file(self, folder: str, filename: str, content: bytes) -> Tuple[str, str]:
        """Save file bytes and return (relative_path, sha256_hash)."""
        sha256 = compute_sha256(content)
        target_dir = self.root / folder
        target_dir.mkdir(parents=True, exist_ok=True)

        target_file = target_dir / filename
        target_file.write_bytes(content)

        rel_path = str(target_file.relative_to(self.root.parent))
        return rel_path, sha256

    def read_file(self, rel_path: str) -> bytes:
        full_path = self.root.parent / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Media asset at {rel_path} does not exist.")
        return full_path.read_bytes()


storage_provider = LocalStorageProvider()
