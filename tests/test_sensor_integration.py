"""
Unit Tests for Safe Sensor Integration (Phase 12)
File: tests/test_sensor_integration.py
"""

import unittest
from src.sensors.sensor_manager import (
    SafeSensorManager,
    SensorReading,
    SensorType,
    SensorStatus,
)


class TestSafeSensorIntegration(unittest.TestCase):

    def setUp(self):
        self.sensor_mgr = SafeSensorManager()

    def test_sensor_polling_returns_valid_readings(self):
        readings = self.sensor_mgr.poll_all_sensors()
        self.assertGreaterEqual(len(readings), 4)

        types_found = {r.sensor_type for r in readings}
        self.assertIn(SensorType.TEMPERATURE.value, types_found)
        self.assertIn(SensorType.HUMIDITY.value, types_found)
        self.assertIn(SensorType.BATTERY_VOLTAGE.value, types_found)
        self.assertIn(SensorType.AMBIENT_LIGHT.value, types_found)

    def test_sensor_reading_fields(self):
        readings = self.sensor_mgr.poll_all_sensors()
        for r in readings:
            self.assertIsInstance(r.sensor_id, str)
            self.assertIsInstance(r.sensor_type, str)
            self.assertIsInstance(r.timestamp, str)
            self.assertIsInstance(r.value, (int, float))
            self.assertIsInstance(r.unit, str)
            self.assertEqual(r.status, SensorStatus.OK.value)

    def test_telemetry_packet_structure(self):
        packet = self.sensor_mgr.get_telemetry_packet()
        self.assertEqual(packet["source"], "esp32_sensor_hal")
        self.assertEqual(packet["actuation_state"], "DISABLED_READ_ONLY")
        self.assertEqual(len(packet["readings"]), 4)

    def test_actuator_isolation(self):
        # Verify that sensor readings do not contain actuator control hooks
        packet = self.sensor_mgr.get_telemetry_packet()
        self.assertNotIn("actuator_command", packet)
        self.assertNotIn("trigger_relay", packet)


if __name__ == "__main__":
    unittest.main()
