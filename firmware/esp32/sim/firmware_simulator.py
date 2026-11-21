"""
AI-Powered Precision Pest Management - ESP32 Firmware Simulator
File: firmware/esp32/sim/firmware_simulator.py

Faithful host-side emulation of the ESP32 C++ firmware architecture.
Implements:
- Safety State Machine (BOOTING, DISCONNECTED, CONNECTED_LOCKED, COMMUNICATION_LOST, ERROR_LOCKED)
- JSON Command Parsing & Strict Validation
- Heartbeat Watchdog & Timeout Fail-Safe
- ACK Generation & System Status Telemetry
"""

import json
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class FirmwareSafetyState(str, Enum):
    BOOTING = "BOOTING"
    DISCONNECTED = "DISCONNECTED"
    CONNECTED_LOCKED = "CONNECTED_LOCKED"
    COMMUNICATION_LOST = "COMMUNICATION_LOST"
    ERROR_LOCKED = "ERROR_LOCKED"


class ESP32FirmwareSimulator:
    """Emulates ESP32 firmware lifecycle, MQTT message routing, and safety states."""

    def __init__(
        self,
        node_id: str = "esp32-node-01",
        firmware_version: str = "1.0.0-safe-sim",
        heartbeat_timeout_sec: float = 5.0,
    ):
        self.node_id = node_id
        self.firmware_version = firmware_version
        self.heartbeat_timeout_sec = heartbeat_timeout_sec

        self.state: FirmwareSafetyState = FirmwareSafetyState.BOOTING
        self.is_wifi_connected: bool = False
        self.is_mqtt_connected: bool = False
        self.last_heartbeat_time: float = 0.0
        self.has_received_heartbeat: bool = False

        self.published_acks: List[Dict[str, Any]] = []
        self.published_heartbeat_acks: List[Dict[str, Any]] = []
        self.published_status_logs: List[Dict[str, Any]] = []

        # Boot sequence
        self._boot()

    def _boot(self) -> None:
        """Simulate hardware bootup sequence."""
        self.state = FirmwareSafetyState.BOOTING
        # Post-boot initialization defaults to DISCONNECTED
        self.state = FirmwareSafetyState.DISCONNECTED

    def connect_network(self) -> None:
        """Simulate Wi-Fi and MQTT connection."""
        self.is_wifi_connected = True
        self.is_mqtt_connected = True
        # Connects into LOCKED mode only (Never activates actuation)
        self.state = FirmwareSafetyState.CONNECTED_LOCKED

    def disconnect_network(self) -> None:
        """Simulate network disconnection."""
        self.is_wifi_connected = False
        self.is_mqtt_connected = False
        self.state = FirmwareSafetyState.DISCONNECTED

    def handle_heartbeat_ping(self, payload: Any, current_time: Optional[float] = None) -> Dict[str, Any]:
        """Process incoming heartbeat ping on `krishnetra/esp32/heartbeat`."""
        now = current_time if current_time is not None else time.time()
        self.last_heartbeat_time = now
        self.has_received_heartbeat = True

        if self.state == FirmwareSafetyState.COMMUNICATION_LOST:
            self.state = FirmwareSafetyState.CONNECTED_LOCKED

        ack = {
            "message_type": "heartbeat_ack",
            "device": self.node_id,
            "timestamp": f"UPTIME_{int(now)}_S",
            "status": "OK",
        }
        self.published_heartbeat_acks.append(ack)
        return ack

    def check_watchdog(self, current_time: Optional[float] = None) -> FirmwareSafetyState:
        """Evaluate watchdog timeout condition."""
        if not self.has_received_heartbeat:
            return self.state

        now = current_time if current_time is not None else time.time()
        if self.state == FirmwareSafetyState.CONNECTED_LOCKED:
            if (now - self.last_heartbeat_time) > self.heartbeat_timeout_sec:
                self.state = FirmwareSafetyState.COMMUNICATION_LOST

        return self.state

    def handle_edge_command(self, raw_payload: Any) -> Dict[str, Any]:
        """
        Process command payload received on `krishnetra/edge/commands`.
        Returns structured ACK response.
        """
        # 1. Parse JSON safely without crashing
        data = None
        if hasattr(raw_payload, "to_dict"):
            data = raw_payload.to_dict()
        elif isinstance(raw_payload, dict):
            data = raw_payload
        elif isinstance(raw_payload, str):
            try:
                data = json.loads(raw_payload)
            except Exception as e:
                ack = self._build_ack("UNKNOWN", "REJECTED", f"JSON_PARSE_ERROR: {e}")
                self.published_acks.append(ack)
                return ack
        else:
            ack = self._build_ack("UNKNOWN", "REJECTED", "INVALID_PAYLOAD_TYPE")
            self.published_acks.append(ack)
            return ack

        command_id = str(data.get("command_id", "UNKNOWN")) if data.get("command_id") else "UNKNOWN"

        # 2. Schema field validation
        required_fields = ["command_id", "timestamp", "confidence", "action", "safety_state"]
        for f in required_fields:
            if f not in data or data[f] is None:
                ack = self._build_ack(command_id, "REJECTED", f"MISSING_FIELD: {f}")
                self.published_acks.append(ack)
                return ack

        # Check confidence bounds
        try:
            conf = float(data["confidence"])
            if conf < 0.0 or conf > 1.0:
                ack = self._build_ack(command_id, "REJECTED", f"CONFIDENCE_OUT_OF_BOUNDS: {conf}")
                self.published_acks.append(ack)
                return ack
        except (ValueError, TypeError):
            ack = self._build_ack(command_id, "REJECTED", "INVALID_CONFIDENCE_TYPE")
            self.published_acks.append(ack)
            return ack

        # Check action whitelist
        action_str = str(data["action"]).strip().upper()
        allowed_actions = {"NONE", "SIMULATED_DETERRENCE", "SIMULATED_STOP"}
        if action_str not in allowed_actions:
            ack = self._build_ack(command_id, "REJECTED", f"PROHIBITED_OR_UNKNOWN_ACTION: {data['action']}")
            self.published_acks.append(ack)
            return ack

        # Check physical actuator keywords
        prohibited = ["GPIO", "RELAY", "MOTOR", "SPEAKER", "PWM"]
        if any(kw in action_str for kw in prohibited):
            ack = self._build_ack(command_id, "REJECTED", "PHYSICAL_ACTUATOR_COMMAND_REJECTED")
            self.published_acks.append(ack)
            return ack

        # Successful simulation ACK
        ack = self._build_ack(command_id, "ACK", "COMMAND_ACCEPTED_SIMULATION_ONLY")
        self.published_acks.append(ack)
        return ack

    def generate_status_telemetry(self) -> Dict[str, Any]:
        """Generate periodic status payload for `krishnetra/system/status`."""
        telemetry = {
            "node_id": self.node_id,
            "firmware_version": self.firmware_version,
            "safety_state": self.state.value,
            "hardware_actuators": "DISABLED",
            "wifi_connected": self.is_wifi_connected,
            "mqtt_connected": self.is_mqtt_connected,
        }
        self.published_status_logs.append(telemetry)
        return telemetry

    def _build_ack(self, command_id: str, status: str, reason: str) -> Dict[str, Any]:
        return {
            "command_id": command_id,
            "device": self.node_id,
            "status": status,
            "safety_state": self.state.value,
            "reason": reason,
        }
