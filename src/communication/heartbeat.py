"""
AI-Powered Precision Pest Management - Heartbeat & Link Liveness Manager
File: src/communication/heartbeat.py

Monitors bi-directional heartbeat pings between Edge AI (Raspberry Pi) and ESP32.
Detects communication timeouts and safely triggers COMMUNICATION_LOST -> NO_ACTION.
"""

import time
from enum import Enum
from typing import Optional, Tuple
from src.communication.command_protocol import HeartbeatMessage, HeartbeatAck


class CommunicationState(str, Enum):
    CONNECTED = "CONNECTED"
    COMMUNICATION_LOST = "COMMUNICATION_LOST"
    UNINITIALIZED = "UNINITIALIZED"


class HeartbeatManager:
    """Tracks link liveness and timeout conditions."""

    def __init__(
        self,
        heartbeat_interval_sec: float = 2.0,
        timeout_threshold_sec: float = 5.0,
    ):
        self.heartbeat_interval_sec = heartbeat_interval_sec
        self.timeout_threshold_sec = timeout_threshold_sec

        self.last_heartbeat_sent_time: float = 0.0
        self.last_heartbeat_ack_time: float = 0.0
        self.state: CommunicationState = CommunicationState.UNINITIALIZED
        self.missed_pings: int = 0

    def should_send_heartbeat(self, current_time: Optional[float] = None) -> bool:
        """Determine if it is time to emit a heartbeat ping."""
        now = current_time if current_time is not None else time.time()
        return (now - self.last_heartbeat_sent_time) >= self.heartbeat_interval_sec

    def generate_ping(self, current_time: Optional[float] = None) -> HeartbeatMessage:
        """Create a new heartbeat ping message."""
        now = current_time if current_time is not None else time.time()
        self.last_heartbeat_sent_time = now
        return HeartbeatMessage()

    def process_ack(self, ack: HeartbeatAck, current_time: Optional[float] = None) -> None:
        """Process received heartbeat acknowledgement from ESP32."""
        now = current_time if current_time is not None else time.time()
        self.last_heartbeat_ack_time = now
        self.missed_pings = 0
        self.state = CommunicationState.CONNECTED

    def check_liveness(self, current_time: Optional[float] = None) -> Tuple[CommunicationState, str]:
        """
        Check if the connection has timed out.

        Returns:
            (state, safe_decision_override)
        """
        now = current_time if current_time is not None else time.time()

        if self.state == CommunicationState.UNINITIALIZED:
            return CommunicationState.UNINITIALIZED, "NO_ACTION"

        elapsed_since_ack = now - self.last_heartbeat_ack_time
        if elapsed_since_ack > self.timeout_threshold_sec:
            self.state = CommunicationState.COMMUNICATION_LOST
            return CommunicationState.COMMUNICATION_LOST, "NO_ACTION"

        return CommunicationState.CONNECTED, "ACTIVE"
