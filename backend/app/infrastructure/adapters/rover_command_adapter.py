"""
KrishiNetra Backend - Existing Firmware / Simulation Command Adapter
File: backend/app/infrastructure/adapters/rover_command_adapter.py
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe integration with existing protocol modules
try:
    from src.communication.command_protocol import (
        CommandPacket,
        AckResponse,
        AckStatus,
        AllowedActions,
    )
    from src.communication.mock_esp32 import MockESP32Device
    SIMULATION_AVAILABLE = True
except ImportError:
    SIMULATION_AVAILABLE = False


class RoverCommandAdapter:
    """Bridges FastAPI commands with the existing ESP32 mock simulation & protocol."""

    def __init__(self) -> None:
        if SIMULATION_AVAILABLE:
            self.mock_device = MockESP32Device(
                device_id="SIM-ROVER-01",
                hardware_safety_state="SIMULATION",
            )
        else:
            self.mock_device = None

    async def execute_command(
        self,
        command_type: str,
        payload: Dict[str, Any],
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Execute command against simulation/protocol and verify acknowledgment.
        Fail-closed: returns False if simulator is unavailable or command is unsupported.
        Returns: (success: bool, status_message: str, response_data: dict)
        """
        if not SIMULATION_AVAILABLE or self.mock_device is None:
            return False, "ESP32 simulator unavailable (Fail-Closed)", {"status": "UNAVAILABLE"}

        cmd_upper = command_type.upper()

        # Emergency STOP is non-blocking and always takes precedence
        if cmd_upper == "STOP":
            packet = CommandPacket.create(
                pest_class="emergency_stop",
                confidence=1.0,
                decision="EMERGENCY_STOP",
                action=AllowedActions.SIMULATED_STOP.value,
                safety_state="SIMULATION",
                source="fastapi_backend",
            )
            ack = self.mock_device.receive_command(packet)
            if ack.status == AckStatus.ACK.value:
                return True, "Emergency stop executed. All rover actuation locked.", ack.to_dict()
            return False, f"Emergency stop rejected: {ack.reason}", ack.to_dict()

        if cmd_upper in ["DETER", "SIMULATED_DETERRENCE"]:
            packet = CommandPacket.create(
                pest_class=payload.get("target_pest", "army worm"),
                confidence=float(payload.get("confidence", 0.9)),
                decision="TARGET",
                action=AllowedActions.SIMULATED_DETERRENCE.value,
                safety_state="SIMULATION",
                source="fastapi_backend",
            )
            ack = self.mock_device.receive_command(packet)
            if ack.status == AckStatus.ACK.value:
                return True, "Simulated deterrence executed safely.", ack.to_dict()
            return False, f"Command rejected by ESP32: {ack.reason}", ack.to_dict()

        if cmd_upper in ["DRIVE", "SERVO"]:
            packet = CommandPacket.create(
                pest_class="rover_navigation",
                confidence=1.0,
                decision=cmd_upper,
                action=AllowedActions.NONE.value,
                safety_state="SIMULATION",
                source="fastapi_backend",
            )
            ack = self.mock_device.receive_command(packet)
            if ack.status == AckStatus.ACK.value:
                return True, f"Simulated {cmd_upper} executed in safe software mode.", ack.to_dict()
            return False, f"Command rejected: {ack.reason}", ack.to_dict()

        # Fail-closed for any unsupported or unknown commands
        return False, f"Unsupported command type '{cmd_upper}'. Rejected under safety protocol.", {"status": "REJECTED"}


command_adapter = RoverCommandAdapter()
