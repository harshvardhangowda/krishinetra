"""
Unit Tests for Mock ESP32 Device (Phase 10)
File: tests/test_mock_esp32.py
"""

import unittest
from src.communication.command_protocol import (
    CommandPacket,
    AckStatus,
    HeartbeatMessage,
)
from src.communication.mock_esp32 import MockESP32Device


class TestMockESP32(unittest.TestCase):

    def setUp(self):
        self.device = MockESP32Device(device_id="esp32-test-01", hardware_safety_state="LOCKED")

    def test_valid_no_action_command_ack(self):
        cmd = CommandPacket.create(
            pest_class="army worm",
            confidence=0.88,
            decision="NO_ACTION",
            action="NONE",
            safety_state="LOCKED",
        )
        ack = self.device.receive_command(cmd)
        self.assertEqual(ack.status, AckStatus.ACK.value)
        self.assertEqual(ack.command_id, cmd.command_id)
        self.assertEqual(ack.safety_state, "LOCKED")

    def test_invalid_command_type_rejected(self):
        ack = self.device.receive_command(12345)
        self.assertEqual(ack.status, AckStatus.INVALID_COMMAND.value)

    def test_malformed_dict_rejected(self):
        bad_dict = {"source": "rpi"}  # Missing required fields
        ack = self.device.receive_command(bad_dict)
        self.assertEqual(ack.status, AckStatus.INVALID_COMMAND.value)

    def test_unknown_action_rejected(self):
        cmd = CommandPacket(
            command_id="cmd-invalid-act",
            timestamp="2026-10-07T12:00:00Z",
            source="raspberry_pi",
            model_version="pest_yolov8n_best",
            pest_class="corn borer",
            confidence=0.75,
            decision="TARGET",
            action="UNKNOWN_ACTION_XYZ",
            safety_state="LOCKED",
        )
        ack = self.device.receive_command(cmd)
        self.assertEqual(ack.status, AckStatus.REJECTED.value)

    def test_heartbeat_response(self):
        ping = HeartbeatMessage()
        pong = self.device.respond_heartbeat(ping)
        self.assertIsNotNone(pong)
        self.assertEqual(pong.message_type, "heartbeat_ack")
        self.assertEqual(pong.status, "OK")

    def test_simulated_timeout_mode(self):
        timeout_device = MockESP32Device(simulate_timeout=True)
        cmd = CommandPacket.create(pest_class="army worm", confidence=0.85)
        ack = timeout_device.receive_command(cmd)
        self.assertEqual(ack.status, AckStatus.TIMEOUT.value)

        pong = timeout_device.respond_heartbeat(HeartbeatMessage())
        self.assertIsNone(pong)


if __name__ == "__main__":
    unittest.main()
