"""
KrishiNetra Backend - Real-Time Authenticated WebSocket Telemetry Feed
File: backend/app/api/v1/ws.py
"""

import asyncio
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from sqlalchemy import select, and_

from backend.app.core.database import AsyncSessionLocal
from backend.app.infrastructure.db.models import Session, Rover, utcnow

router = APIRouter(tags=["Real-time Telemetry"])


class ConnectionManager:
    """Manages active authenticated WebSocket connections by rover ID."""

    def __init__(self) -> None:
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, rover_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        if rover_id not in self.active_connections:
            self.active_connections[rover_id] = set()
        self.active_connections[rover_id].add(websocket)

    def disconnect(self, rover_id: str, websocket: WebSocket) -> None:
        if rover_id in self.active_connections:
            self.active_connections[rover_id].discard(websocket)

    async def broadcast(self, rover_id: str, message: dict) -> None:
        if rover_id in self.active_connections:
            for connection in list(self.active_connections[rover_id]):
                try:
                    await connection.send_json(message)
                except Exception:
                    self.active_connections[rover_id].discard(connection)


ws_manager = ConnectionManager()


@router.websocket("/ws/rovers/{rover_id}/telemetry")
async def rover_telemetry_ws(websocket: WebSocket, rover_id: str, token: str):
    """
    Authenticated WebSocket endpoint for real-time rover telemetry,
    heartbeats, and detection notifications.
    Enforces token validity and organization-scoped rover ownership.
    """
    # 1. Validate session token and rover organization authorization
    async with AsyncSessionLocal() as db:
        stmt = select(Session).where(Session.session_token == token)
        res = await db.execute(stmt)
        session = res.scalar_one_or_none()
        if not session or utcnow() > session.expires_at:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # 2. Verify that the requested rover belongs to user's organization
        rover_stmt = select(Rover).where(
            and_(
                Rover.id == rover_id,
                Rover.organization_id == session.organization_id,
            )
        )
        rover_res = await db.execute(rover_stmt)
        rover = rover_res.scalar_one_or_none()
        if not rover:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    await ws_manager.connect(rover_id, websocket)
    try:
        # Send initial connection acknowledgment
        await websocket.send_json({
            "type": "CONNECTION_ESTABLISHED",
            "rover_id": rover_id,
            "status": "SUBSCRIBED",
        })
        while True:
            data = await websocket.receive_text()
            # Echo heartbeat ping/pong
            await websocket.send_json({"type": "PONG", "received": data})
    except WebSocketDisconnect:
        ws_manager.disconnect(rover_id, websocket)
