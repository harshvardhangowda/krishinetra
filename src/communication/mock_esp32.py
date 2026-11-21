"""
AI-Powered Precision Pest Management - Mock ESP32 Device
File: src/communication/mock_esp32.py

Simulates ESP32 firmware behavior in software without requiring physical microcontroller
or actuator hardware. Enforces fail-safe default state.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from src.communication.command_protocol import (
    CommandPacket,
    AckResponse,
    AckStatus,
    HeartbeatMessage,
    HeartbeatAck,
    AllowedActions,
)


class MockESP32Device:
    """Software simulation of ESP32 edge microcontroller."""

    def __init__(
        self,
        device_id: str = "esp32-node-01",
        hardware_safety_state: str = "LOCKED",
        simulate_timeout: bool = False,
    ):
        self.device_id = device_id
        self.safety_state = hardware_safety_state
        self.simulate_timeout = simulate_timeout
        self.received_commands: List[Dict[str, Any]] = []
        self.command_history_log: List[AckResponse] = []

    def receive_command(self, raw_command: Any) -> AckResponse:
        """
        Process an incoming command and generate an appropriate AckResponse.

        Args:
            raw_command: CommandPacket instance, JSON string, or dict.

        Returns:
            AckResponse with status (ACK, REJECTED, INVALID_COMMAND, SAFETY_LOCKED, TIMEOUT)
        """
        if self.simulate_timeout:
            return AckResponse.create(
                command_id=getattr(raw_command, "command_id", "UNKNOWN"),
                status=AckStatus.TIMEOUT,
                device=self.device_id,
                safety_state=self.safety_state,
                reason="SIMULATED_DEVICE_TIMEOUT",
            )

        # 1. Parse into CommandPacket if needed
        cmd_packet: Optional[CommandPacket] = None
        if isinstance(raw_command, CommandPacket):
            cmd_packet = raw_command
        elif isinstance(raw_command, dict):
            try:
                cmd_packet = CommandPacket.from_dict(raw_command)
            except Exception as e:
                return AckResponse.create(
                    command_id=str(raw_command.get("command_id", "UNKNOWN")),
                    status=AckStatus.INVALID_COMMAND,
                    device=self.device_id,
                    safety_state=self.safety_state,
                    reason=f"SCHEMA_VALIDATION_ERROR: {e}",
                )
        elif isinstance(raw_command, str):
            try:
                cmd_packet = CommandPacket.from_json(raw_command)
            except Exception as e:
                return AckResponse.create(
                    command_id="UNKNOWN",
                    status=AckStatus.INVALID_COMMAND,
                    device=self.device_id,
                    safety_state=self.safety_state,
                    reason=f"JSON_PARSE_ERROR: {e}",
                )
        else:
            return AckResponse.create(
                command_id="UNKNOWN",
                status=AckStatus.INVALID_COMMAND,
                device=self.device_id,
                safety_state=self.safety_state,
                reason="UNSUPPORTED_DATA_TYPE",
            )

        self.received_commands.append(cmd_packet.to_dict())

        # 2. Validate Command Schema & Values
        is_valid, validation_err = cmd_packet.validate()
        if not is_valid:
            ack = AckResponse.create(
                command_id=cmd_packet.command_id,
                status=AckStatus.REJECTED,
                device=self.device_id,
                safety_state=self.safety_state,
                reason=validation_err,
            )
            self.command_history_log.append(ack)
            return ack

        # 3. Check Safety State
        if self.safety_state == "LOCKED" and cmd_packet.action != AllowedActions.NONE.value:
            ack = AckResponse.create(
                command_id=cmd_packet.command_id,
                status=AckStatus.SAFETY_LOCKED,
                device=self.device_id,
                safety_state=self.safety_state,
                reason="DEVICE_IN_SAFETY_LOCKED_STATE",
            )
            self.command_history_log.append(ack)
            return ack

        # 4. Successful acknowledgement (simulation only)
        ack = AckResponse.create(
            command_id=cmd_packet.command_id,
            status=AckStatus.ACK,
            device=self.device_id,
            safety_state=self.safety_state,
            reason="COMMAND_ACCEPTED_SIMULATION_ONLY",
        )
        self.command_history_log.append(ack)
        return ack

    def respond_heartbeat(self, ping: HeartbeatMessage) -> Optional[HeartbeatAck]:
        """Respond to a heartbeat ping message."""
        if self.simulate_timeout:
            return None
        return HeartbeatAck(
            message_type="heartbeat_ack",
            device=self.device_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="OK",
        )
