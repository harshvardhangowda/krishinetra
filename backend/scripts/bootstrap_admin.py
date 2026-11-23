"""
KrishiNetra Backend - Secure Administrator Bootstrap CLI
File: backend/scripts/bootstrap_admin.py

Specification Policy:
Production administrator accounts must NEVER be silently auto-created with public
hardcoded credentials on server startup. This CLI enables operators to securely
bootstrap their organization and owner account with strong credentials.
"""

import argparse
import asyncio
import getpass
import re
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import select
from backend.app.core.database import AsyncSessionLocal, engine
from backend.app.infrastructure.db.base import Base
from backend.app.infrastructure.db.models import Organization, User, UserRole, Rover, RoverStatus
from backend.app.core.security import hash_password


def validate_password_strength(password: str) -> bool:
    """Enforce minimum 8 characters, at least one digit and one letter."""
    if len(password) < 8:
        return False
    if not re.search(r"[A-Za-z]", password) or not re.search(r"[0-9]", password):
        return False
    return True


async def bootstrap_admin(
    org_name: str,
    org_slug: str,
    username: str,
    email: str,
    password: str,
    rover_name: str = "KrishiNetra Rover Alpha",
    rover_serial: str = "KNR-ALPHA-001",
) -> None:
    if not validate_password_strength(password):
        print("[Error] Password must be at least 8 characters long and contain both letters and numbers.")
        sys.exit(1)

    # Ensure tables exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as db:
        # Check if org exists
        stmt = select(Organization).where(Organization.slug == org_slug)
        existing_org = (await db.execute(stmt)).scalar_one_or_none()

        if existing_org:
            org = existing_org
            print(f"[Bootstrap] Using existing organization: {org.name} ({org.slug})")
        else:
            org = Organization(name=org_name, slug=org_slug)
            db.add(org)
            await db.flush()
            print(f"[Bootstrap] Created organization: {org.name} ({org.slug})")

        # Check if user exists
        u_stmt = select(User).where((User.username == username) | (User.email == email))
        existing_user = (await db.execute(u_stmt)).scalar_one_or_none()
        if existing_user:
            print(f"[Error] User with username '{username}' or email '{email}' already exists.")
            sys.exit(1)

        owner = User(
            organization_id=org.id,
            email=email,
            username=username,
            full_name=f"{username.capitalize()} (Administrator)",
            hashed_password=hash_password(password),
            role=UserRole.OWNER,
        )
        db.add(owner)

        # Check rover
        r_stmt = select(Rover).where(Rover.serial_number == rover_serial)
        existing_rover = (await db.execute(r_stmt)).scalar_one_or_none()
        if not existing_rover:
            rover = Rover(
                organization_id=org.id,
                name=rover_name,
                serial_number=rover_serial,
                hardware_version="v1.0-sim",
                firmware_version="v1.0.0",
                status=RoverStatus.IDLE,
            )
            db.add(rover)

        await db.commit()
        print(f"[Bootstrap] Successfully created administrator account '{username}' ({email}) with role 'owner'.")
        print("[Bootstrap] Ready for production authentication.")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap KrishiNetra Organization & Administrator Account")
    parser.add_argument("--org-name", default="KrishiNetra Agro Labs", help="Organization name")
    parser.add_argument("--org-slug", default="krishinetra-labs", help="Unique organization slug")
    parser.add_argument("--username", help="Administrator username")
    parser.add_argument("--email", help="Administrator email address")
    parser.add_argument("--password", help="Administrator password (prompted if omitted)")
    args = parser.parse_args()

    username = args.username or input("Enter administrator username: ").strip()
    email = args.email or input("Enter administrator email: ").strip()
    password = args.password
    if not password:
        password = getpass.getpass("Enter administrator password (min 8 chars, alphanumeric): ")

    asyncio.run(bootstrap_admin(
        org_name=args.org_name,
        org_slug=args.org_slug,
        username=username,
        email=email,
        password=password,
    ))


if __name__ == "__main__":
    main()
