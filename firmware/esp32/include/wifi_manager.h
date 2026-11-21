#pragma once

// =============================================================================
// Krishnetra Aavishkar: Wi-Fi Manager Header
// File: firmware/esp32/include/wifi_manager.h
// =============================================================================

#include <Arduino.h>
#include <WiFi.h>
#include "safety_state_machine.h"

class WifiManager {
public:
    WifiManager(SafetyStateMachine& stateMachine);

    void setup(const char* ssid, const char* password);
    void loop();
    bool isConnected();
    int getRssi();

private:
    SafetyStateMachine& stateMachine;
    const char* ssid;
    const char* password;
    unsigned long lastCheckMillis;
};
