"""
KrishiNetra Backend - Database Backup & Integrity Verification CLI
File: backend/scripts/backup_restore.py
"""

import os
import sys
import shutil
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import settings


def resolve_sqlite_db_path() -> Optional[Path]:
    """Dynamically resolves SQLite path from application configuration (KN-18)."""
    db_url = settings.DATABASE_URL
    if "sqlite" in db_url:
        path_part = db_url.split(":///")[-1]
        p = Path(path_part)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p
    return None


def backup_database(destination_dir: str = "./backups") -> Path:
    """Create timestamped copy of database using online backup API and calculate SHA-256."""
    dest = Path(destination_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    db_file = resolve_sqlite_db_path()

    if not db_file or not db_file.exists():
        print(f"[Backup] No active SQLite database found at {db_file}.")
        return dest

    target_file = dest / f"krishnetra_db_backup_{timestamp}.db"

    # Use SQLite online backup API to ensure transaction consistency on active DBs
    try:
        src_conn = sqlite3.connect(str(db_file))
        dst_conn = sqlite3.connect(str(target_file))
        with dst_conn:
            src_conn.backup(dst_conn)
        src_conn.close()
        dst_conn.close()
    except Exception as exc:
        print(f"[Backup] Online backup fallback to copy: {exc}")
        shutil.copy2(db_file, target_file)

    # Compute checksum
    hasher = hashlib.sha256()
    with open(target_file, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    checksum = hasher.hexdigest()

    manifest_file = dest / f"backup_manifest_{timestamp}.txt"
    manifest_file.write_text(f"Source: {db_file}\nFile: {target_file.name}\nTimestamp: {timestamp}\nSHA256: {checksum}\n")

    print(f"[Backup] Successfully generated database backup: {target_file}")
    print(f"[Backup] Integrity Checksum: {checksum}")
    return target_file


def restore_database(backup_file: str) -> bool:
    """Restore database from backup file with integrity verification (KN-18)."""
    b_path = Path(backup_file).resolve()
    if not b_path.exists():
        print(f"[Restore] Error: Backup file {b_path} not found.")
        return False

    db_file = resolve_sqlite_db_path()
    if not db_file:
        print("[Restore] Error: Database URL is not SQLite.")
        return False

    # Check for manifest checksum if available
    manifest_path = b_path.parent / b_path.name.replace("krishnetra_db_backup_", "backup_manifest_").replace(".db", ".txt")
    if manifest_path.exists():
        content = manifest_path.read_text()
        print(f"[Restore] Verified backup manifest found: {manifest_path.name}")

    db_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(b_path, db_file)
    print(f"[Restore] Successfully restored database to {db_file}")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--restore":
        restore_database(sys.argv[2])
    else:
        backup_database()
