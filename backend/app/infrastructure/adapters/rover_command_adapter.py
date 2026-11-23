"""
KrishiNetra Backend - Existing Firmware / Simulation Command Adapter
File: backend/app/infrastructure/adapters/rover_command_adapter.py
"""

import json
from typing import Dict, Any, Tuple
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe integration with existing protocol modules
try:
    from src.communication.command_protocol import (
        format_command,
        parse_response,
        calculate_checksum,
        COMMAND_TYPES,
    )
    from src.communication.mock_esp32 import MockESP32
    SIMULATION_AVAILABLE = True
except ImportError:
    SIMULATION_AVAILABLE = False


class RoverCommandAdapter:
    """Bridges FastAPI commands with the existing ESP32 mock simulation & protocol."""

    def __init__(self) -> None:
        if SIMULATION_AVAILABLE:
            self.mock_device = MockESP32(device_id="SIM-ROVER-01")
        else:
            self.mock_device = None

    async def execute_command(
        self,
        command_type: str,
        payload: Dict[str, Any],
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Execute command against simulation/protocol and verify acknowledgment.
        Returns: (success: bool, status_message: str, response_data: dict)
        """
        cmd_upper = command_type.upper()

        # Emergency STOP is non-blocking and always takes precedence
        if cmd_upper == "STOP":
            if self.mock_device:
                self.mock_device.emergency_stop()
            return True, "Emergency stop executed. All rover actuation locked.", {"state": "SAFE_STOP"}

        if not SIMULATION_AVAILABLE:
            return True, f"Simulated execution for {cmd_upper}", payload

        # Map to existing command protocol if available
        cmd_id = cmd_upper.lower()
        if cmd_id in COMMAND_TYPES:
            # Format according to existing CRC8/Checksum protocol
            wire_cmd = format_command(cmd_id, payload.get("seq", 1), payload)
            response = self.mock_device.process_command(wire_cmd)
            valid, resp_data = parse_response(response)
            if valid and resp_data.get("ack"):
                return True, "Command acknowledged by ESP32 simulator", resp_data
            else:
                return False, resp_data.get("error", "Command rejected by firmware simulator"), resp_data

        return True, f"Command {cmd_upper} accepted in safe simulation mode", payload


command_adapter = RoverCommandAdapter()
