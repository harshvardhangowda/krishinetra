#pragma once

// =============================================================================
// Krishnetra Aavishkar: Command Parser & Validator Header
// File: firmware/esp32/include/command_parser.h
// =============================================================================

#include <Arduino.h>
#include <ArduinoJson.h>

struct ParsedCommand {
    bool isValid;
    String commandId;
    String timestamp;
    String source;
    String modelVersion;
    String pestClass;
    float confidence;
    String decision;
    String action;
    String safetyState;
    String rejectionReason;
};

class CommandParser {
public:
    static ParsedCommand parseAndValidate(const char* jsonPayload, size_t length);
};
