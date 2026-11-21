// =============================================================================
// Krishnetra Aavishkar: Sensor Manager Implementation
// File: firmware/esp32/src/sensors/sensor_manager.cpp
// Description: Safe read-only telemetry sampling without actuator actuation.
// =============================================================================

#include "sensors/sensor_manager.h"

SensorManager::SensorManager() : lastReadTime(0) {}

void SensorManager::setup() {
    Serial.println(F("[SensorManager] Initializing read-only sensor telemetry layer..."));
    updateReadings();
}

void SensorManager::updateReadings() {
    readings.clear();
    readings.push_back(readTemperature());
    readings.push_back(readHumidity());
    readings.push_back(readBatteryVoltage());
    readings.push_back(readAmbientLight());
    lastReadTime = millis();
}

std::vector<SensorReading> SensorManager::getLatestReadings() const {
    return readings;
}

String SensorManager::getTelemetryJson() const {
    String json = "[";
    for (size_t i = 0; i < readings.size(); ++i) {
        json += readings[i].toJson();
        if (i + 1 < readings.size()) {
            json += ",";
        }
    }
    json += "]";
    return json;
}

SensorReading SensorManager::readTemperature() {
    SensorReading r;
    r.sensorId = "DHT22_TEMP_01";
    r.sensorType = "TEMPERATURE";
    r.timestampMs = millis();
    r.value = 26.5f; // Baseline nominal reading
    r.unit = "C";
    r.status = "OK";
    return r;
}

SensorReading SensorManager::readHumidity() {
    SensorReading r;
    r.sensorId = "DHT22_HUM_01";
    r.sensorType = "HUMIDITY";
    r.timestampMs = millis();
    r.value = 62.0f; // Baseline nominal reading
    r.unit = "%";
    r.status = "OK";
    return r;
}

SensorReading SensorManager::readBatteryVoltage() {
    SensorReading r;
    r.sensorId = "INA219_VBAT_01";
    r.sensorType = "BATTERY_VOLTAGE";
    r.timestampMs = millis();
    r.value = 12.4f; // 3S LiPo Nominal
    r.unit = "V";
    r.status = "OK";
    return r;
}

SensorReading SensorManager::readAmbientLight() {
    SensorReading r;
    r.sensorId = "BH1750_LUX_01";
    r.sensorType = "AMBIENT_LIGHT";
    r.timestampMs = millis();
    r.value = 450.0f; // Ambient daytime lux
    r.unit = "LUX";
    r.status = "OK";
    return r;
}
