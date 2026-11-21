#pragma once

// =============================================================================
// Krishnetra Aavishkar: ESP32 Firmware Configuration
// File: firmware/esp32/include/config.h
// =============================================================================

#include <Arduino.h>

#if __has_include("config_secrets.h")
#include "config_secrets.h"
#else
#define WIFI_SSID_SECRET     "DEFAULT_SSID"
#define WIFI_PASSWORD_SECRET "DEFAULT_PASSWORD"
#define MQTT_BROKER_SECRET   "127.0.0.1"
#define MQTT_PORT_SECRET     1883
#define MQTT_USER_SECRET     ""
#define MQTT_PASSWORD_SECRET ""
#endif

// Node Identification
#define DEVICE_ID            "esp32-node-01"
#define FIRMWARE_VERSION     "1.0.0-safe-sim"

// MQTT Topic Conventions (Consistent with Phase 10)
#define TOPIC_COMMANDS       "krishnetra/edge/commands"
#define TOPIC_ACK            "krishnetra/esp32/ack"
#define TOPIC_HEARTBEAT      "krishnetra/esp32/heartbeat"
#define TOPIC_STATUS         "krishnetra/system/status"

// Timing and Safety Timeouts
#define HEARTBEAT_TIMEOUT_MS 5000UL   // 5.0 seconds communication timeout
#define STATUS_PUBLISH_MS    10000UL  // Status telemetry published every 10s
#define MQTT_RECONNECT_MS    2000UL   // 2s reconnect backoff
#define SERIAL_BAUD_RATE     115200

// Strict Hardware Lock (Permanent Safety Constraint)
#define ACTUATOR_HARDWARE_ENABLED 0  // 0 = Disabled permanently
