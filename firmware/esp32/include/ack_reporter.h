#pragma once

// =============================================================================
// Krishnetra Aavishkar: ACK & Telemetry Reporter Header
// File: firmware/esp32/include/ack_reporter.h
// =============================================================================

#include <Arduino.h>
#include <ArduinoJson.h>
#include "safety_state_machine.h"

class AckReporter {
public:
    static String generateAckJson(
        const String& commandId,
        const char* status,
        const char* safetyState,
        const char* reason = nullptr,
        const char* deviceId = "esp32"
    );

    static String generateHeartbeatAckJson(
        const char* deviceId = "esp32"
    );

    static String generateStatusJson(
        const char* deviceId,
        const char* firmwareVersion,
        const SafetyStateMachine& stateMachine,
        int wifiRssi = 0
    );
};
