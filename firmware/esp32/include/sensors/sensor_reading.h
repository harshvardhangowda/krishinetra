#pragma once

// =============================================================================
// Krishnetra Aavishkar: Safe Sensor Reading Data Structure
// File: firmware/esp32/include/sensors/sensor_reading.h
// Description: Read-only environmental & telemetry sensor data representation.
// =============================================================================

#include <Arduino.h>

struct SensorReading {
    String sensorId;
    String sensorType;   // "TEMPERATURE", "HUMIDITY", "BATTERY_VOLTAGE", "AMBIENT_LIGHT"
    unsigned long timestampMs;
    float value;
    String unit;         // "C", "%", "V", "LUX"
    String status;       // "OK", "DEGRADED", "ERROR", "DISCONNECTED"

    String toJson() const {
        String json = "{";
        json += "\"sensor_id\":\"" + sensorId + "\",";
        json += "\"sensor_type\":\"" + sensorType + "\",";
        json += "\"timestamp_ms\":" + String(timestampMs) + ",";
        json += "\"value\":" + String(value, 2) + ",";
        json += "\"unit\":\"" + unit + "\",";
        json += "\"status\":\"" + status + "\"";
        json += "}";
        return json;
    }
};
