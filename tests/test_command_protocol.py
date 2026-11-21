"""
Unit Tests for Safe Command Protocol (Phase 10)
File: tests/test_command_protocol.py
"""

import json
import unittest
from src.communication.command_protocol import (
    CommandPacket,
    AckResponse,
    AckStatus,
    HeartbeatMessage,
    HeartbeatAck,
    AllowedActions,
)


class TestCommandProtocol(unittest.TestCase):

    def test_valid_command_creation(self):
        cmd = CommandPacket.create(
            pest_class="army worm",
            confidence=0.87,
            decision="NO_ACTION",
            action="NONE",
            safety_state="LOCKED",
        )
        self.assertIsNotNone(cmd.command_id)
        self.assertEqual(cmd.source, "raspberry_pi")
        self.assertEqual(cmd.pest_class, "army worm")
        self.assertEqual(cmd.confidence, 0.87)
        self.assertEqual(cmd.safety_state, "LOCKED")

        is_valid, err = cmd.validate()
        self.assertTrue(is_valid)
        self.assertIsNone(err)

    def test_command_serialization_roundtrip(self):
        cmd = CommandPacket.create(
            pest_class="corn borer",
            confidence=0.82,
            decision="NO_ACTION",
            action="SIMULATED_DETERRENCE",
            safety_state="LOCKED",
        )
        json_str = cmd.to_json()
        parsed_cmd = CommandPacket.from_json(json_str)

        self.assertEqual(cmd.command_id, parsed_cmd.command_id)
        self.assertEqual(cmd.pest_class, parsed_cmd.pest_class)
        self.assertEqual(cmd.confidence, parsed_cmd.confidence)
        self.assertEqual(cmd.action, parsed_cmd.action)

    def test_invalid_confidence_bounds(self):
        cmd_high = CommandPacket.create(pest_class="army worm", confidence=1.25)
        is_valid, err = cmd_high.validate()
        self.assertFalse(is_valid)
        self.assertIn("Confidence out of bounds", err)

        cmd_neg = CommandPacket.create(pest_class="army worm", confidence=-0.10)
        is_valid, err = cmd_neg.validate()
        self.assertFalse(is_valid)
        self.assertIn("Confidence out of bounds", err)

    def test_prohibited_actuator_action(self):
        cmd = CommandPacket(
            command_id="test-123",
            timestamp="2026-10-07T14:00:00Z",
            source="raspberry_pi",
            model_version="pest_yolov8n_best",
            pest_class="army worm",
            confidence=0.90,
            decision="TRIGGER",
            action="ACTIVATE_RELAY",
            safety_state="LOCKED",
        )
        is_valid, err = cmd.validate()
        self.assertFalse(is_valid)
        self.assertIn("Prohibited or unknown action", err)

    def test_ack_response_creation(self):
        ack = AckResponse.create(
            command_id="cmd-999",
            status=AckStatus.ACK,
            device="esp32",
            safety_state="LOCKED",
        )
        self.assertEqual(ack.command_id, "cmd-999")
        self.assertEqual(ack.status, "ACK")
        self.assertEqual(ack.safety_state, "LOCKED")

    def test_heartbeat_creation(self):
        ping = HeartbeatMessage()
        self.assertEqual(ping.message_type, "heartbeat")
        self.assertEqual(ping.source, "raspberry_pi")

        pong = HeartbeatAck()
        self.assertEqual(pong.message_type, "heartbeat_ack")
        self.assertEqual(pong.device, "esp32")


if __name__ == "__main__":
    unittest.main()
