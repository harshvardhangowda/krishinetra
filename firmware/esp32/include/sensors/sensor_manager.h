#pragma once

// =============================================================================
// Krishnetra Aavishkar: Sensor Manager Header
// File: firmware/esp32/include/sensors/sensor_manager.h
// Description: Read-only sensor abstraction layer. Never drives actuators.
// =============================================================================

#include <Arduino.h>
#include <vector>
#include "sensor_reading.h"

class SensorManager {
public:
    SensorManager();

    void setup();
    void updateReadings();
    std::vector<SensorReading> getLatestReadings() const;
    String getTelemetryJson() const;

private:
    std::vector<SensorReading> readings;
    unsigned long lastReadTime;

    // Simulated / Hardware Read Methods
    SensorReading readTemperature();
    SensorReading readHumidity();
    SensorReading readBatteryVoltage();
    SensorReading readAmbientLight();
};
