"""
Integration Tests for Communication Pipeline (Phase 10)
File: tests/test_communication.py
"""

import time
import unittest
from pathlib import Path

from src.decision.decision_engine import DecisionEngine
from src.communication.command_protocol import (
    CommandPacket,
    AckResponse,
    AckStatus,
    AllowedActions,
)
from src.communication.safety_interlock import SafetyInterlock, SafetyState
from src.communication.heartbeat import HeartbeatManager, CommunicationState
from src.communication.mqtt_client import MockMQTTClient
from src.communication.serial_client import MockSerialClient
from src.communication.mock_esp32 import MockESP32Device
from src.communication.telemetry_logger import CommunicationTelemetryLogger


class TestCommunicationPipeline(unittest.TestCase):

    def setUp(self):
        self.decision_engine = DecisionEngine(
            min_confidence_threshold=0.60,
            hardware_safety_lock=True,
        )
        self.safety_interlock = SafetyInterlock(
            default_state=SafetyState.LOCKED,
            min_confidence_cutoff=0.60,
        )
        self.mock_esp32 = MockESP32Device(device_id="esp32-node-01", hardware_safety_state="LOCKED")
        self.mqtt_client = MockMQTTClient()
        self.mqtt_client.connect()
        self.logger = CommunicationTelemetryLogger(local_log_dir="logs/test_comm_logs")

    def test_end_to_end_decision_to_esp32_ack(self):
        # 1. Simulate YOLO detection packet for Armyworm
        detection_packet = {
            "is_baseline_model": False,
            "detections": [
                {"class_name": "army worm", "confidence": 0.89, "bbox": [100, 100, 200, 200]}
            ],
        }

        # 2. Decision Engine evaluation
        decision_result = self.decision_engine.evaluate(detection_packet)
        self.assertEqual(decision_result["decision"], "TARGET_OBSERVED_SAFETY_LOCKED")
        self.assertEqual(decision_result["target_pest"], "army worm")

        # 3. Formulate safe command packet
        cmd = CommandPacket.create(
            pest_class=decision_result["target_pest"],
            confidence=decision_result["confidence"],
            decision="NO_ACTION",
            action=AllowedActions.NONE.value,
            safety_state="LOCKED",
        )

        # 4. Safety interlock inspection
        is_safe, err = self.safety_interlock.inspect_and_filter(cmd)
        self.assertTrue(is_safe)
        self.assertIsNone(err)

        # 5. Publish to MQTT & deliver to Mock ESP32
        received_acks = []

        def esp32_command_handler(topic, payload):
            ack = self.mock_esp32.receive_command(payload)
            received_acks.append(ack)
            # Log event
            self.logger.log_event(
                event_type="COMMAND_EXECUTED",
                status=ack.status,
                command_id=cmd.command_id,
                pest_class=cmd.pest_class,
                confidence=cmd.confidence,
                decision=cmd.decision,
                action=cmd.action,
                safety_state=cmd.safety_state,
            )

        self.mqtt_client.subscribe("krishnetra/edge/commands", esp32_command_handler)
        published = self.mqtt_client.publish("krishnetra/edge/commands", cmd.to_json())

        self.assertTrue(published)
        self.assertEqual(len(received_acks), 1)
        self.assertEqual(received_acks[0].status, AckStatus.ACK.value)
        self.assertEqual(received_acks[0].command_id, cmd.command_id)

    def test_heartbeat_manager_and_timeout(self):
        hb = HeartbeatManager(heartbeat_interval_sec=1.0, timeout_threshold_sec=3.0)

        t0 = 1000.0
        # Check initial uninitialized state
        state, decision = hb.check_liveness(current_time=t0)
        self.assertEqual(state, CommunicationState.UNINITIALIZED)
        self.assertEqual(decision, "NO_ACTION")

        # Generate ping and receive ACK from mock ESP32
        ping = hb.generate_ping(current_time=t0)
        pong = self.mock_esp32.respond_heartbeat(ping)
        self.assertIsNotNone(pong)

        hb.process_ack(pong, current_time=t0 + 0.1)
        state, decision = hb.check_liveness(current_time=t0 + 0.2)
        self.assertEqual(state, CommunicationState.CONNECTED)
        self.assertEqual(decision, "ACTIVE")

        # Advance time past timeout threshold (3.5s later without new ACK)
        state, decision = hb.check_liveness(current_time=t0 + 3.6)
        self.assertEqual(state, CommunicationState.COMMUNICATION_LOST)
        self.assertEqual(decision, "NO_ACTION")

    def test_mock_serial_loopback(self):
        serial_client = MockSerialClient(port="/dev/ttyUSB0", baudrate=115200)
        serial_client.open()

        cmd_json = '{"command_id": "ser-1", "action": "NONE"}'
        serial_client.feed_rx(cmd_json)

        received_line = serial_client.read_line()
        self.assertEqual(received_line, cmd_json)
        serial_client.close()


if __name__ == "__main__":
    unittest.main()
