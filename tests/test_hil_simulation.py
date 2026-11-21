"""
Integration Tests for Hardware-in-the-Loop (HIL) Simulation (Phase 12)
File: tests/test_hil_simulation.py
"""

import unittest
from src.decision.decision_engine import DecisionEngine
from src.communication.command_protocol import (
    CommandPacket,
    AllowedActions,
)
from src.communication.safety_interlock import SafetyInterlock, SafetyState
from src.communication.mqtt_client import MockMQTTClient
from src.communication.telemetry_logger import CommunicationTelemetryLogger
from firmware.esp32.sim.firmware_simulator import (
    ESP32FirmwareSimulator,
    FirmwareSafetyState,
)
from src.sensors.sensor_manager import SafeSensorManager


class TestHILSimulationPipeline(unittest.TestCase):

    def setUp(self):
        self.decision_engine = DecisionEngine(
            min_confidence_threshold=0.60,
            hardware_safety_lock=True,
        )
        self.safety_interlock = SafetyInterlock(
            default_state=SafetyState.LOCKED,
            min_confidence_cutoff=0.60,
        )
        self.firmware_sim = ESP32FirmwareSimulator(node_id="esp32-node-01")
        self.firmware_sim.connect_network()

        self.mqtt_bus = MockMQTTClient(client_id="hil-test-bus")
        self.mqtt_bus.connect()

        self.sensor_mgr = SafeSensorManager()
        self.logger = CommunicationTelemetryLogger(local_log_dir="logs/test_hil_logs")

    def test_full_hil_command_and_ack_pipeline(self):
        # 1. Detection Packet from Edge AI
        detection_packet = {
            "is_baseline_model": False,
            "detections": [
                {"class_name": "corn borer", "confidence": 0.84, "bbox": [50, 60, 150, 160]}
            ],
        }

        # 2. Decision Engine
        decision_result = self.decision_engine.evaluate(detection_packet)
        self.assertEqual(decision_result["decision"], "TARGET_OBSERVED_SAFETY_LOCKED")

        # 3. Formulate Safe Command
        cmd = CommandPacket.create(
            pest_class="corn borer",
            confidence=decision_result["confidence"],
            decision="NO_ACTION",
            action=AllowedActions.NONE.value,
            safety_state="LOCKED",
        )

        # 4. Safety Interlock
        is_safe, err = self.safety_interlock.inspect_and_filter(cmd)
        self.assertTrue(is_safe)
        self.assertIsNone(err)

        # 5. Route via MQTT to ESP32 Firmware Simulator
        received_acks = []

        def esp32_firmware_handler(topic, payload):
            ack = self.firmware_sim.handle_edge_command(payload)
            received_acks.append(ack)
            # Log to Telemetry
            self.logger.log_event(
                event_type="HIL_COMMAND_PROCESSED",
                status=ack["status"],
                command_id=cmd.command_id,
                pest_class=cmd.pest_class,
                confidence=cmd.confidence,
                decision=cmd.decision,
                action=cmd.action,
                safety_state=cmd.safety_state,
            )

        self.mqtt_bus.subscribe("krishnetra/edge/commands", esp32_firmware_handler)
        self.mqtt_bus.publish("krishnetra/edge/commands", cmd.to_json())

        # 6. Verify ACK
        self.assertEqual(len(received_acks), 1)
        self.assertEqual(received_acks[0]["status"], "ACK")
        self.assertEqual(received_acks[0]["command_id"], cmd.command_id)
        self.assertEqual(received_acks[0]["safety_state"], FirmwareSafetyState.CONNECTED_LOCKED.value)

    def test_hil_sensor_telemetry_flow(self):
        # Poll safe sensor HAL
        sensor_packet = self.sensor_mgr.get_telemetry_packet()
        self.assertEqual(sensor_packet["actuation_state"], "DISABLED_READ_ONLY")

        # Publish sensor telemetry on MQTT
        published_telemetry = []

        def telemetry_listener(topic, payload):
            published_telemetry.append(payload)

        self.mqtt_bus.subscribe("krishnetra/system/status", telemetry_listener)
        self.mqtt_bus.publish("krishnetra/system/status", str(sensor_packet))

        self.assertEqual(len(published_telemetry), 1)

    def test_hil_watchdog_timeout_recovery(self):
        # 1. Initial Heartbeat
        self.firmware_sim.handle_heartbeat_ping({}, current_time=50.0)
        self.assertEqual(self.firmware_sim.state, FirmwareSafetyState.CONNECTED_LOCKED)

        # 2. Simulate Link Loss (Watchdog Timeout at 56s)
        self.firmware_sim.check_watchdog(current_time=56.0)
        self.assertEqual(self.firmware_sim.state, FirmwareSafetyState.COMMUNICATION_LOST)

        # 3. Edge AI Command during timeout must still be processed safely
        cmd = CommandPacket.create(pest_class="army worm", confidence=0.88, action="NONE")
        ack = self.firmware_sim.handle_edge_command(cmd)
        self.assertEqual(ack["status"], "ACK")
        self.assertEqual(ack["safety_state"], FirmwareSafetyState.COMMUNICATION_LOST.value)

        # 4. Link Restored via Heartbeat
        self.firmware_sim.handle_heartbeat_ping({}, current_time=57.0)
        self.assertEqual(self.firmware_sim.state, FirmwareSafetyState.CONNECTED_LOCKED)


if __name__ == "__main__":
    unittest.main()
