#pragma once

// =============================================================================
// Krishnetra Aavishkar: MQTT Communication Manager Header
// File: firmware/esp32/include/mqtt_manager.h
// =============================================================================

#include <Arduino.h>
#include <PubSubClient.h>
#include <WiFiClient.h>
#include "safety_state_machine.h"
#include "heartbeat_watchdog.h"

class MqttManager {
public:
    MqttManager(
        WiFiClient& wifiClient,
        SafetyStateMachine& stateMachine,
        HeartbeatWatchdog& watchdog
    );

    void setup(const char* brokerHost, uint16_t brokerPort);
    void loop();
    bool isConnected();
    void publishStatus(const char* statusPayload);

private:
    PubSubClient mqttClient;
    SafetyStateMachine& stateMachine;
    HeartbeatWatchdog& watchdog;
    unsigned long lastReconnectAttempt;

    void reconnect();
    static void onMessageReceived(char* topic, byte* payload, unsigned int length);
    static MqttManager* instance;
    void handleIncomingMessage(const char* topic, const char* payload, unsigned int length);
};
