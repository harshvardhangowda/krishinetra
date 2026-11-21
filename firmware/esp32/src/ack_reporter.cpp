// =============================================================================
// Krishnetra Aavishkar: ACK & Telemetry Reporter Implementation
// File: firmware/esp32/src/ack_reporter.cpp
// =============================================================================

#include "ack_reporter.h"

String AckReporter::generateAckJson(
    const String& commandId,
    const char* status,
    const char* safetyState,
    const char* reason,
    const char* deviceId
) {
    StaticJsonDocument<512> doc;
    doc["command_id"] = commandId.length() > 0 ? commandId : "UNKNOWN";
    doc["device"] = deviceId;
    doc["status"] = status;
    doc["safety_state"] = safetyState;

    // ISO timestamp substitute (uptime in seconds)
    char tsBuf[32];
    snprintf(tsBuf, sizeof(tsBuf), "UPTIME_%lu_S", millis() / 1000UL);
    doc["timestamp"] = tsBuf;

    if (reason && strlen(reason) > 0) {
        doc["reason"] = reason;
    } else {
        doc["reason"] = "OK";
    }

    String output;
    serializeJson(doc, output);
    return output;
}

String AckReporter::generateHeartbeatAckJson(const char* deviceId) {
    StaticJsonDocument<256> doc;
    doc["message_type"] = "heartbeat_ack";
    doc["device"] = deviceId;
    doc["status"] = "OK";

    char tsBuf[32];
    snprintf(tsBuf, sizeof(tsBuf), "UPTIME_%lu_S", millis() / 1000UL);
    doc["timestamp"] = tsBuf;

    String output;
    serializeJson(doc, output);
    return output;
}

String AckReporter::generateStatusJson(
    const char* deviceId,
    const char* firmwareVersion,
    const SafetyStateMachine& stateMachine,
    int wifiRssi
) {
    StaticJsonDocument<512> doc;
    doc["node_id"] = deviceId;
    doc["firmware_version"] = firmwareVersion;
    doc["safety_state"] = stateMachine.getStateString();
    doc["uptime_ms"] = millis();
    doc["hardware_actuators"] = "DISABLED";
    doc["wifi_rssi"] = wifiRssi;

    String output;
    serializeJson(doc, output);
    return output;
}
