"""
AI-Powered Precision Pest Management - Safety Interlock Layer
File: src/communication/safety_interlock.py

Strict hardware safety policy enforcement.
Rules:
1. Default state = LOCKED
2. Default action = NO_ACTION
3. Physical actuator commands are rejected
4. Unknown commands are rejected
5. Invalid JSON is rejected
6. Missing command_id is rejected
7. Missing timestamp is rejected
8. Missing safety_state is rejected
9. Communication timeout results in NO_ACTION
10. Model uncertainty must never directly activate hardware
11. Mock mode cannot accidentally control GPIO
12. No GPIO libraries used for actuator control in this phase
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from src.communication.command_protocol import CommandPacket, AllowedActions

logger = logging.getLogger("KrishnetraSafetyInterlock")


class SafetyState(str, Enum):
    LOCKED = "LOCKED"
    UNLOCKED = "UNLOCKED"


class SafetyInterlock:
    """Enforces safety rules and validates command compliance before transmission."""

    def __init__(
        self,
        default_state: SafetyState = SafetyState.LOCKED,
        min_confidence_cutoff: float = 0.60,
    ):
        self.state: SafetyState = default_state
        self.min_confidence_cutoff = min_confidence_cutoff
        self.rejection_log: list = []

    def inspect_and_filter(
        self,
        command: CommandPacket,
    ) -> Tuple[bool, Optional[str]]:
        """
        Inspect a CommandPacket against the 12 strict safety rules.

        Returns:
            (is_safe: bool, failure_reason: Optional[str])
        """
        # Rule 6, 7, 8: Field presence
        if not command.command_id:
            reason = "RULE_6_VIOLATION: Missing command_id"
            self._log_rejection(command, reason)
            return False, reason

        if not command.timestamp:
            reason = "RULE_7_VIOLATION: Missing timestamp"
            self._log_rejection(command, reason)
            return False, reason

        if not command.safety_state:
            reason = "RULE_8_VIOLATION: Missing safety_state"
            self._log_rejection(command, reason)
            return False, reason

        # Rule 10: Model uncertainty / confidence check
        if command.confidence < 0.0 or command.confidence > 1.0:
            reason = f"CONFIDENCE_OUT_OF_BOUNDS: {command.confidence}"
            self._log_rejection(command, reason)
            return False, reason

        if command.confidence < self.min_confidence_cutoff and command.action != AllowedActions.NONE.value:
            reason = (
                f"RULE_10_VIOLATION: Confidence {command.confidence:.2f} is below cutoff "
                f"{self.min_confidence_cutoff:.2f}. Action must be NONE."
            )
            self._log_rejection(command, reason)
            return False, reason

        # Rule 3: Physical actuator command prohibition
        action_str = command.action.strip().upper()
        if action_str not in {a.value for a in AllowedActions}:
            reason = f"RULE_4_VIOLATION: Unknown or prohibited action '{command.action}'"
            self._log_rejection(command, reason)
            return False, reason

        # Check for disallowed hardware trigger keywords
        prohibited = ["GPIO", "MOTOR", "RELAY", "SPEAKER", "PWM_WRITE", "ACTIVATE_HARDWARE"]
        if any(kw in action_str for kw in prohibited):
            reason = f"RULE_3_VIOLATION: Physical actuator control attempt detected: '{command.action}'"
            self._log_rejection(command, reason)
            return False, reason

        # Rule 1: Default locked state enforcement
        if self.state == SafetyState.LOCKED and command.safety_state != SafetyState.LOCKED.value:
            reason = "RULE_1_VIOLATION: Master safety state is LOCKED; command safety_state mismatch."
            self._log_rejection(command, reason)
            return False, reason

        return True, None

    def _log_rejection(self, command: Any, reason: str) -> None:
        """Record and log command rejection events."""
        entry = {
            "command_id": getattr(command, "command_id", "UNKNOWN"),
            "action": getattr(command, "action", "UNKNOWN"),
            "reason": reason,
        }
        self.rejection_log.append(entry)
        logger.warning(f"[SAFETY INTERLOCK REJECTED] Command ID: {entry['command_id']} | Reason: {reason}")
