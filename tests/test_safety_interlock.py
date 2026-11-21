"""
Unit Tests for Safety Interlock Layer (Phase 10)
File: tests/test_safety_interlock.py
"""

import unittest
from src.communication.command_protocol import CommandPacket, AllowedActions
from src.communication.safety_interlock import SafetyInterlock, SafetyState


class TestSafetyInterlock(unittest.TestCase):

    def setUp(self):
        self.interlock = SafetyInterlock(
            default_state=SafetyState.LOCKED,
            min_confidence_cutoff=0.60,
        )

    def test_valid_safe_command_passes(self):
        cmd = CommandPacket.create(
            pest_class="army worm",
            confidence=0.85,
            decision="NO_ACTION",
            action="NONE",
            safety_state="LOCKED",
        )
        is_safe, reason = self.interlock.inspect_and_filter(cmd)
        self.assertTrue(is_safe)
        self.assertIsNone(reason)

    def test_missing_command_id_rejected(self):
        cmd = CommandPacket.create(pest_class="corn borer", confidence=0.80)
        cmd.command_id = ""
        is_safe, reason = self.interlock.inspect_and_filter(cmd)
        self.assertFalse(is_safe)
        self.assertIn("RULE_6_VIOLATION", reason)

    def test_missing_timestamp_rejected(self):
        cmd = CommandPacket.create(pest_class="corn borer", confidence=0.80)
        cmd.timestamp = ""
        is_safe, reason = self.interlock.inspect_and_filter(cmd)
        self.assertFalse(is_safe)
        self.assertIn("RULE_7_VIOLATION", reason)

    def test_low_confidence_with_action_rejected(self):
        # Confidence < 0.60 with action != NONE must be rejected
        cmd = CommandPacket.create(
            pest_class="army worm",
            confidence=0.45,
            decision="TARGET_CANDIDATE",
            action="SIMULATED_DETERRENCE",
            safety_state="LOCKED",
        )
        is_safe, reason = self.interlock.inspect_and_filter(cmd)
        self.assertFalse(is_safe)
        self.assertIn("RULE_10_VIOLATION", reason)

    def test_physical_actuator_keyword_rejected(self):
        cmd = CommandPacket.create(
            pest_class="army worm",
            confidence=0.90,
            decision="TARGET",
            action="TRIGGER_SPEAKER_PWM",
            safety_state="LOCKED",
        )
        is_safe, reason = self.interlock.inspect_and_filter(cmd)
        self.assertFalse(is_safe)
        self.assertTrue("RULE_4_VIOLATION" in reason or "RULE_3_VIOLATION" in reason)


if __name__ == "__main__":
    unittest.main()
