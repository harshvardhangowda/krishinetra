"""
AI-Powered Precision Pest Management - Safe Command Protocol
File: src/communication/command_protocol.py

Defines structured JSON schemas for Raspberry Pi -> ESP32 communication.
Enforces strict validation rules:
- Confidence range: 0.0 to 1.0
- Allowed simulated actions ONLY: NONE, SIMULATED_DETERRENCE, SIMULATED_STOP
- Physical hardware actuation commands are rejected.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class AllowedActions(str, Enum):
    NONE = "NONE"
    SIMULATED_DETERRENCE = "SIMULATED_DETERRENCE"
    SIMULATED_STOP = "SIMULATED_STOP"


class AckStatus(str, Enum):
    ACK = "ACK"
    REJECTED = "REJECTED"
    TIMEOUT = "TIMEOUT"
    INVALID_COMMAND = "INVALID_COMMAND"
    SAFETY_LOCKED = "SAFETY_LOCKED"


@dataclass
class CommandPacket:
    """Represents a structured command sent from Edge AI to ESP32."""
    command_id: str
    timestamp: str
    source: str
    model_version: str
    pest_class: str
    confidence: float
    decision: str
    action: str
    safety_state: str = "LOCKED"

    @classmethod
    def create(
        cls,
        pest_class: str,
        confidence: float,
        decision: str = "NO_ACTION",
        action: str = "NONE",
        model_version: str = "pest_yolov8n_best",
        source: str = "raspberry_pi",
        safety_state: str = "LOCKED",
    ) -> "CommandPacket":
        """Factory method to construct a valid CommandPacket with generated UUID and UTC ISO timestamp."""
        return cls(
            command_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            model_version=model_version,
            pest_class=pest_class,
            confidence=float(confidence),
            decision=decision,
            action=action,
            safety_state=safety_state,
        )

    def to_json(self) -> str:
        """Serialize command to JSON string."""
        return json.dumps(asdict(self))

    def to_dict(self) -> Dict[str, Any]:
        """Convert command to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommandPacket":
        """Construct CommandPacket from a dictionary with strict field checks."""
        required_fields = [
            "command_id",
            "timestamp",
            "source",
            "model_version",
            "pest_class",
            "confidence",
            "decision",
            "action",
            "safety_state",
        ]
        for f in required_fields:
            if f not in data or data[f] is None:
                raise ValueError(f"Missing required command field: '{f}'")

        # Validate confidence is float
        try:
            conf = float(data["confidence"])
        except (ValueError, TypeError):
            raise ValueError(f"Invalid confidence value: {data.get('confidence')}")

        return cls(
            command_id=str(data["command_id"]),
            timestamp=str(data["timestamp"]),
            source=str(data["source"]),
            model_version=str(data["model_version"]),
            pest_class=str(data["pest_class"]),
            confidence=conf,
            decision=str(data["decision"]),
            action=str(data["action"]),
            safety_state=str(data["safety_state"]),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "CommandPacket":
        """Deserialize from JSON string."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON command: {e}")
        return cls.from_dict(data)

    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        Validate all fields against the project safety protocol.

        Returns:
            (is_valid: bool, error_message: Optional[str])
        """
        if not self.command_id or not isinstance(self.command_id, str):
            return False, "Invalid or missing 'command_id'"

        if not self.timestamp or not isinstance(self.timestamp, str):
            return False, "Invalid or missing 'timestamp'"

        if not self.source:
            return False, "Missing 'source' field"

        if not (0.0 <= self.confidence <= 1.0):
            return False, f"Confidence out of bounds [0.0, 1.0]: {self.confidence}"

        # Allowed actions check
        valid_actions = {a.value for a in AllowedActions}
        if self.action not in valid_actions:
            return False, f"Prohibited or unknown action '{self.action}'. Allowed: {sorted(list(valid_actions))}"

        # Real actuator prohibition
        prohibited_keywords = ["ACTIVATE", "MOTOR", "RELAY", "GPIO", "SPEAKER", "LASER", "PULSE_HW"]
        action_upper = self.action.upper()
        for kw in prohibited_keywords:
            if kw in action_upper and not action_upper.startswith("SIMULATED_"):
                return False, f"Direct physical actuator action '{self.action}' is prohibited by safety protocol."

        return True, None


@dataclass
class AckResponse:
    """Represents an acknowledgement response returned by ESP32."""
    command_id: str
    device: str
    status: str  # ACK, REJECTED, TIMEOUT, INVALID_COMMAND, SAFETY_LOCKED
    timestamp: str
    safety_state: str = "LOCKED"
    reason: Optional[str] = None

    @classmethod
    def create(
        cls,
        command_id: str,
        status: AckStatus,
        device: str = "esp32",
        safety_state: str = "LOCKED",
        reason: Optional[str] = None,
    ) -> "AckResponse":
        return cls(
            command_id=command_id,
            device=device,
            status=status.value if isinstance(status, AckStatus) else str(status),
            timestamp=datetime.now(timezone.utc).isoformat(),
            safety_state=safety_state,
            reason=reason,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AckResponse":
        return cls(
            command_id=str(data.get("command_id", "")),
            device=str(data.get("device", "esp32")),
            status=str(data.get("status", "INVALID_COMMAND")),
            timestamp=str(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
            safety_state=str(data.get("safety_state", "LOCKED")),
            reason=data.get("reason"),
        )


@dataclass
class HeartbeatMessage:
    """Heartbeat message emitted by Raspberry Pi."""
    message_type: str = "heartbeat"
    source: str = "raspberry_pi"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class HeartbeatAck:
    """Heartbeat acknowledgement returned by ESP32."""
    message_type: str = "heartbeat_ack"
    device: str = "esp32"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "OK"

    def to_json(self) -> str:
        return json.dumps(asdict(self))
