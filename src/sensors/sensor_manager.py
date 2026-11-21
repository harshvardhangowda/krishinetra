"""
AI-Powered Precision Pest Management - Safe Sensor Hardware Abstraction Layer
File: src/sensors/sensor_manager.py

Provides a safe read-only hardware abstraction layer (HAL) for agricultural
and robot telemetry sensors:
- Ambient temperature / humidity (DHT22 / SHT31)
- Battery voltage / current telemetry (INA219)
- Ambient illumination level (BH1750)

STRICT SAFETY RULE:
Sensor readings are READ-ONLY and MUST NEVER directly trigger physical actuation.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any


class SensorType(str, Enum):
    TEMPERATURE = "TEMPERATURE"
    HUMIDITY = "HUMIDITY"
    BATTERY_VOLTAGE = "BATTERY_VOLTAGE"
    AMBIENT_LIGHT = "AMBIENT_LIGHT"


class SensorStatus(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"
    DISCONNECTED = "DISCONNECTED"


@dataclass
class SensorReading:
    """Represents a standardized, read-only sensor sample."""
    sensor_id: str
    sensor_type: str
    timestamp: str
    value: float
    unit: str
    status: str = SensorStatus.OK.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SafeSensorManager:
    """Safe read-only manager for field and rover telemetry."""

    def __init__(self):
        self.readings: Dict[str, SensorReading] = {}

    def poll_all_sensors(self) -> List[SensorReading]:
        """Collect current readings across all telemetry channels."""
        now_ts = datetime.now(timezone.utc).isoformat()

        temp_reading = SensorReading(
            sensor_id="DHT22_TEMP_01",
            sensor_type=SensorType.TEMPERATURE.value,
            timestamp=now_ts,
            value=26.5,
            unit="C",
            status=SensorStatus.OK.value,
        )

        hum_reading = SensorReading(
            sensor_id="DHT22_HUM_01",
            sensor_type=SensorType.HUMIDITY.value,
            timestamp=now_ts,
            value=62.0,
            unit="%",
            status=SensorStatus.OK.value,
        )

        vbat_reading = SensorReading(
            sensor_id="INA219_VBAT_01",
            sensor_type=SensorType.BATTERY_VOLTAGE.value,
            timestamp=now_ts,
            value=12.4,
            unit="V",
            status=SensorStatus.OK.value,
        )

        lux_reading = SensorReading(
            sensor_id="BH1750_LUX_01",
            sensor_type=SensorType.AMBIENT_LIGHT.value,
            timestamp=now_ts,
            value=450.0,
            unit="LUX",
            status=SensorStatus.OK.value,
        )

        self.readings = {
            temp_reading.sensor_id: temp_reading,
            hum_reading.sensor_id: hum_reading,
            vbat_reading.sensor_id: vbat_reading,
            lux_reading.sensor_id: lux_reading,
        }

        return list(self.readings.values())

    def get_latest_reading(self, sensor_id: str) -> Optional[SensorReading]:
        return self.readings.get(sensor_id)

    def get_telemetry_packet(self) -> Dict[str, Any]:
        """Package all sensor readings into a telemetry payload for MQTT/database."""
        readings_list = self.poll_all_sensors()
        return {
            "source": "esp32_sensor_hal",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sensor_count": len(readings_list),
            "readings": [r.to_dict() for r in readings_list],
            "actuation_state": "DISABLED_READ_ONLY",
        }
