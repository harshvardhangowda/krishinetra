// =============================================================================
// Krishnetra Aavishkar: Command Parser & Validator Implementation
// File: firmware/esp32/src/command_parser.cpp
// =============================================================================

#include "command_parser.h"

ParsedCommand CommandParser::parseAndValidate(const char* jsonPayload, size_t length) {
    ParsedCommand cmd;
    cmd.isValid = false;
    cmd.confidence = 0.0f;

    StaticJsonDocument<1024> doc;
    DeserializationError error = deserializeJson(doc, jsonPayload, length);

    if (error) {
        cmd.commandId = "UNKNOWN";
        cmd.rejectionReason = String("JSON_PARSE_ERROR: ") + error.c_str();
        return cmd;
    }

    // 1. Check required fields presence
    if (!doc.containsKey("command_id") || doc["command_id"].isNull()) {
        cmd.commandId = "UNKNOWN";
        cmd.rejectionReason = "MISSING_FIELD: command_id";
        return cmd;
    }
    cmd.commandId = doc["command_id"].as<String>();
    if (cmd.commandId.length() == 0) {
        cmd.rejectionReason = "EMPTY_FIELD: command_id";
        return cmd;
    }

    if (!doc.containsKey("timestamp") || doc["timestamp"].isNull()) {
        cmd.rejectionReason = "MISSING_FIELD: timestamp";
        return cmd;
    }
    cmd.timestamp = doc["timestamp"].as<String>();

    if (!doc.containsKey("confidence") || doc["confidence"].isNull()) {
        cmd.rejectionReason = "MISSING_FIELD: confidence";
        return cmd;
    }
    cmd.confidence = doc["confidence"].as<float>();

    if (!doc.containsKey("action") || doc["action"].isNull()) {
        cmd.rejectionReason = "MISSING_FIELD: action";
        return cmd;
    }
    cmd.action = doc["action"].as<String>();

    if (!doc.containsKey("safety_state") || doc["safety_state"].isNull()) {
        cmd.rejectionReason = "MISSING_FIELD: safety_state";
        return cmd;
    }
    cmd.safetyState = doc["safety_state"].as<String>();

    // Optional metadata fields
    cmd.source = doc["source"] | "unknown";
    cmd.modelVersion = doc["model_version"] | "unknown";
    cmd.pestClass = doc["pest_class"] | "none";
    cmd.decision = doc["decision"] | "NO_ACTION";

    // 2. Value validation rules
    if (cmd.confidence < 0.0f || cmd.confidence > 1.0f || isnan(cmd.confidence)) {
        cmd.rejectionReason = String("CONFIDENCE_OUT_OF_BOUNDS: ") + String(cmd.confidence, 2);
        return cmd;
    }

    // 3. Action whitelist verification
    String act = cmd.action;
    act.toUpperCase();
    if (act != "NONE" && act != "SIMULATED_DETERRENCE" && act != "SIMULATED_STOP") {
        cmd.rejectionReason = String("PROHIBITED_OR_UNKNOWN_ACTION: ") + cmd.action;
        return cmd;
    }

    // 4. Physical actuator prohibition check
    if (act.indexOf("GPIO") >= 0 || act.indexOf("RELAY") >= 0 || 
        act.indexOf("MOTOR") >= 0 || act.indexOf("SPEAKER") >= 0 || 
        act.indexOf("PWM") >= 0) {
        cmd.rejectionReason = "PHYSICAL_ACTUATOR_COMMAND_REJECTED";
        return cmd;
    }

    cmd.isValid = true;
    cmd.rejectionReason = "";
    return cmd;
}
