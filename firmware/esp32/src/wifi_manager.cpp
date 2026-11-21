// =============================================================================
// Krishnetra Aavishkar: Wi-Fi Manager Implementation
// File: firmware/esp32/src/wifi_manager.cpp
// =============================================================================

#include "wifi_manager.h"

WifiManager::WifiManager(SafetyStateMachine& stateMachine)
    : stateMachine(stateMachine), ssid(nullptr), password(nullptr), lastCheckMillis(0) {}

void WifiManager::setup(const char* ssid, const char* password) {
    this->ssid = ssid;
    this->password = password;

    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    Serial.print(F("[WiFi] Connecting to: "));
    Serial.println(ssid);
}

void WifiManager::loop() {
    unsigned long now = millis();
    if (now - lastCheckMillis < 3000) return;
    lastCheckMillis = now;

    if (WiFi.status() != WL_CONNECTED) {
        if (stateMachine.getCurrentState() == SafetyState::CONNECTED_LOCKED) {
            stateMachine.transitionTo(SafetyState::DISCONNECTED, "Wi-Fi link dropped");
        }
    }
}

bool WifiManager::isConnected() {
    return WiFi.status() == WL_CONNECTED;
}

int WifiManager::getRssi() {
    return isConnected() ? WiFi.RSSI() : 0;
}
