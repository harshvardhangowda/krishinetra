// =============================================================================
// Krishnetra Aavishkar: ESP32 Precision Pest Management Firmware
// File: firmware/esp32/src/main.cpp
// Description: Main firmware entry point for ESP32 microcontroller edge node.
//              Enforces safe simulation-only operation.
// =============================================================================

#include <Arduino.h>
#include "config.h"
#include "safety_state_machine.h"
#include "heartbeat_watchdog.h"
#include "wifi_manager.h"
#include "mqtt_manager.h"
#include "ack_reporter.h"

// Core Objects
SafetyStateMachine stateMachine;
HeartbeatWatchdog watchdog(HEARTBEAT_TIMEOUT_MS);
WifiManager wifiManager(stateMachine);
WiFiClient espWifiClient;
MqttManager mqttManager(espWifiClient, stateMachine, watchdog);

unsigned long lastStatusPublishMillis = 0;

void setup() {
    Serial.begin(SERIAL_BAUD_RATE);
    delay(1000);

    Serial.println();
    Serial.println(F("=================================================="));
    Serial.println(F("   KRISHNETRA ESP32 EDGE FIRMWARE (SAFE-SIM)      "));
    Serial.println(F("=================================================="));
    Serial.print(F("Node ID          : ")); Serial.println(DEVICE_ID);
    Serial.print(F("Firmware Version : ")); Serial.println(FIRMWARE_VERSION);
    Serial.print(F("Safety State     : ")); Serial.println(stateMachine.getStateString());
    Serial.println(F("Actuators        : PERMANENTLY DISABLED (Sim-Only)"));
    Serial.println(F("=================================================="));

    // Booting -> Disconnected
    stateMachine.transitionTo(SafetyState::DISCONNECTED, "Boot completed");

    // Network setup
    wifiManager.setup(WIFI_SSID_SECRET, WIFI_PASSWORD_SECRET);
    mqttManager.setup(MQTT_BROKER_SECRET, MQTT_PORT_SECRET);
}

void loop() {
    // 1. Maintain Wi-Fi link
    wifiManager.loop();

    // 2. Maintain MQTT link & process incoming messages
    mqttManager.loop();

    // 3. Heartbeat Watchdog check
    watchdog.checkTimeout(stateMachine);

    // 4. Periodic system status telemetry
    unsigned long now = millis();
    if (now - lastStatusPublishMillis > STATUS_PUBLISH_MS) {
        lastStatusPublishMillis = now;
        if (mqttManager.isConnected()) {
            String statusJson = AckReporter::generateStatusJson(
                DEVICE_ID,
                FIRMWARE_VERSION,
                stateMachine,
                wifiManager.getRssi()
            );
            mqttManager.publishStatus(statusJson.c_str());
        }
    }

    delay(10);
}
