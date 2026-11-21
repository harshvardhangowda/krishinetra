"""
Unit and Integration Tests for Phase 11 ESP32 Firmware Simulator
File: tests/test_firmware_simulation.py

Validates 12 specific firmware scenarios:
1. Valid NO_ACTION command
2. Invalid JSON
3. Missing command ID
4. Invalid confidence
5. Unknown action
6. Heartbeat success
7. Heartbeat timeout
8. MQTT disconnection and reconnection
9. Safe state after reboot
10. Rejection of attempted physical-actuator commands
11. Malformed messages do not crash the simulator
12. No automatic action after reconnect
"""

import unittest
from firmware.esp32.sim.firmware_simulator import (
    ESP32FirmwareSimulator,
    FirmwareSafetyState,
)


class TestESP32FirmwareSimulation(unittest.TestCase):

    def setUp(self):
        self.sim = ESP32FirmwareSimulator(node_id="esp32-node-01")
        self.sim.connect_network()

    # Scenario 1: Valid NO_ACTION command
    def test_scenario_01_valid_no_action_command(self):
        cmd = {
            "command_id": "cmd-001",
            "timestamp": "2026-10-07T12:00:00Z",
            "confidence": 0.85,
            "action": "NONE",
            "safety_state": "LOCKED",
        }
        ack = self.sim.handle_edge_command(cmd)
        self.assertEqual(ack["status"], "ACK")
        self.assertEqual(ack["command_id"], "cmd-001")
        self.assertEqual(ack["safety_state"], FirmwareSafetyState.CONNECTED_LOCKED.value)

    # Scenario 2: Invalid JSON string
    def test_scenario_02_invalid_json(self):
        bad_json = "{not a valid json object"
        ack = self.sim.handle_edge_command(bad_json)
        self.assertEqual(ack["status"], "REJECTED")
        self.assertIn("JSON_PARSE_ERROR", ack["reason"])

    # Scenario 3: Missing command ID
    def test_scenario_03_missing_command_id(self):
        cmd = {
            "timestamp": "2026-10-07T12:00:00Z",
            "confidence": 0.85,
            "action": "NONE",
            "safety_state": "LOCKED",
        }
        ack = self.sim.handle_edge_command(cmd)
        self.assertEqual(ack["status"], "REJECTED")
        self.assertIn("MISSING_FIELD: command_id", ack["reason"])

    # Scenario 4: Invalid confidence (>1.0 or <0.0)
    def test_scenario_04_invalid_confidence(self):
        cmd_high = {
            "command_id": "cmd-high",
            "timestamp": "2026-10-07T12:00:00Z",
            "confidence": 1.45,
            "action": "NONE",
            "safety_state": "LOCKED",
        }
        ack_high = self.sim.handle_edge_command(cmd_high)
        self.assertEqual(ack_high["status"], "REJECTED")
        self.assertIn("CONFIDENCE_OUT_OF_BOUNDS", ack_high["reason"])

        cmd_neg = {
            "command_id": "cmd-neg",
            "timestamp": "2026-10-07T12:00:00Z",
            "confidence": -0.20,
            "action": "NONE",
            "safety_state": "LOCKED",
        }
        ack_neg = self.sim.handle_edge_command(cmd_neg)
        self.assertEqual(ack_neg["status"], "REJECTED")
        self.assertIn("CONFIDENCE_OUT_OF_BOUNDS", ack_neg["reason"])

    # Scenario 5: Unknown action
    def test_scenario_05_unknown_action(self):
        cmd = {
            "command_id": "cmd-unknown",
            "timestamp": "2026-10-07T12:00:00Z",
            "confidence": 0.90,
            "action": "LASER_BLAST_XYZ",
            "safety_state": "LOCKED",
        }
        ack = self.sim.handle_edge_command(cmd)
        self.assertEqual(ack["status"], "REJECTED")
        self.assertIn("PROHIBITED_OR_UNKNOWN_ACTION", ack["reason"])

    # Scenario 6: Heartbeat success
    def test_scenario_06_heartbeat_success(self):
        ping = {"message_type": "heartbeat"}
        ack = self.sim.handle_heartbeat_ping(ping, current_time=100.0)
        self.assertEqual(ack["message_type"], "heartbeat_ack")
        self.assertEqual(ack["status"], "OK")
        self.assertEqual(self.sim.state, FirmwareSafetyState.CONNECTED_LOCKED)

    # Scenario 7: Heartbeat timeout
    def test_scenario_07_heartbeat_timeout(self):
        # Receive ping at t=100s
        self.sim.handle_heartbeat_ping({}, current_time=100.0)
        self.assertEqual(self.sim.state, FirmwareSafetyState.CONNECTED_LOCKED)

        # Check watchdog at t=106s (timeout threshold is 5.0s)
        new_state = self.sim.check_watchdog(current_time=106.0)
        self.assertEqual(new_state, FirmwareSafetyState.COMMUNICATION_LOST)
        self.assertEqual(self.sim.state, FirmwareSafetyState.COMMUNICATION_LOST)

    # Scenario 8: MQTT disconnection and reconnection
    def test_scenario_08_mqtt_disconnect_reconnect(self):
        self.sim.disconnect_network()
        self.assertEqual(self.sim.state, FirmwareSafetyState.DISCONNECTED)
        self.assertFalse(self.sim.is_mqtt_connected)

        # Reconnect
        self.sim.connect_network()
        self.assertEqual(self.sim.state, FirmwareSafetyState.CONNECTED_LOCKED)
        self.assertTrue(self.sim.is_mqtt_connected)

    # Scenario 9: Safe state after reboot
    def test_scenario_09_safe_state_after_reboot(self):
        rebooted_sim = ESP32FirmwareSimulator(node_id="esp32-node-02")
        # Post boot must be in DISCONNECTED state, before network connection
        self.assertEqual(rebooted_sim.state, FirmwareSafetyState.DISCONNECTED)
        rebooted_sim.connect_network()
        self.assertEqual(rebooted_sim.state, FirmwareSafetyState.CONNECTED_LOCKED)

    # Scenario 10: Rejection of attempted physical actuator commands
    def test_scenario_10_rejection_of_physical_actuators(self):
        cmd = {
            "command_id": "cmd-hw-01",
            "timestamp": "2026-10-07T12:00:00Z",
            "confidence": 0.95,
            "action": "TRIGGER_RELAY_GPIO4",
            "safety_state": "LOCKED",
        }
        ack = self.sim.handle_edge_command(cmd)
        self.assertEqual(ack["status"], "REJECTED")
        self.assertTrue(
            "PROHIBITED_OR_UNKNOWN_ACTION" in ack["reason"]
            or "PHYSICAL_ACTUATOR_COMMAND_REJECTED" in ack["reason"]
        )

    # Scenario 11: Malformed messages do not crash simulator
    def test_scenario_11_malformed_messages_do_not_crash(self):
        malformed_inputs = [
            None,
            12345,
            b"raw_bytes",
            [],
            "{broken json",
            {"missing_all_fields": True},
            {"confidence": "not_a_number"},
        ]
        for item in malformed_inputs:
            ack = self.sim.handle_edge_command(item)
            self.assertEqual(ack["status"], "REJECTED")
        # State must remain stable
        self.assertEqual(self.sim.state, FirmwareSafetyState.CONNECTED_LOCKED)

    # Scenario 12: No automatic action after reconnect
    def test_scenario_12_no_auto_action_after_reconnect(self):
        self.sim.disconnect_network()
        self.sim.connect_network()
        telemetry = self.sim.generate_status_telemetry()
        self.assertEqual(telemetry["safety_state"], FirmwareSafetyState.CONNECTED_LOCKED.value)
        self.assertEqual(telemetry["hardware_actuators"], "DISABLED")


if __name__ == "__main__":
    unittest.main()
