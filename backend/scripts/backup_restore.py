"""
KrishiNetra Backend - Database Backup & Integrity Verification CLI
File: backend/scripts/backup_restore.py
"""

import os
import sys
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent


def backup_database(destination_dir: str = "./backups") -> Path:
    """Create timestamped copy of database and calculate SHA-256 integrity hash."""
    dest = Path(destination_dir)
    dest.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    db_file = BACKEND_ROOT / "krishnetra_backend.db"

    if not db_file.exists():
        print(f"[Backup] No active SQLite database found at {db_file}.")
        return dest

    target_file = dest / f"krishnetra_db_backup_{timestamp}.db"
    shutil.copy2(db_file, target_file)

    # Compute checksum
    hasher = hashlib.sha256()
    with open(target_file, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    checksum = hasher.hexdigest()

    manifest_file = dest / f"backup_manifest_{timestamp}.txt"
    manifest_file.write_text(f"File: {target_file.name}\nTimestamp: {timestamp}\nSHA256: {checksum}\n")

    print(f"[Backup] Successfully generated database backup: {target_file}")
    print(f"[Backup] Integrity Checksum: {checksum}")
    return target_file


if __name__ == "__main__":
    backup_database()
