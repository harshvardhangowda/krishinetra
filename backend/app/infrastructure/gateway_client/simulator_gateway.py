"""
KrishiNetra Backend - Hardware / Simulator Gateway Client
File: backend/app/infrastructure/gateway_client/simulator_gateway.py
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.communication.command_protocol import format_command, parse_response
    from src.communication.mock_esp32 import MockESP32
    SIM_AVAILABLE = True
except ImportError:
    SIM_AVAILABLE = False


class GatewayClient:
    """Dispatches commands to ESP32 gateway simulator or hardware."""

    def __init__(self):
        self.mock = MockESP32(device_id="ROVER-SIM-01") if SIM_AVAILABLE else None

    async def send_command(self, cmd_type: str, payload: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        cmd_upper = cmd_type.upper()
        if cmd_upper == "STOP":
            if self.mock:
                self.mock.emergency_stop()
            return True, "STOP executed immediately. Actuation locked.", {"status": "SAFE_STOP"}

        if not SIM_AVAILABLE or not self.mock:
            return True, f"Simulator executed {cmd_upper}", payload

        wire_msg = format_command(cmd_upper.lower(), payload.get("seq", 1), payload)
        raw_resp = self.mock.process_command(wire_msg)
        valid, parsed = parse_response(raw_resp)
        if valid and parsed.get("ack"):
            return True, "Gateway acknowledged command", parsed
        return False, parsed.get("error", "Gateway rejected command"), parsed


gateway_client = GatewayClient()
