// =============================================================================
// Krishnetra Aavishkar: MQTT Communication Manager Implementation
// File: firmware/esp32/src/mqtt_manager.cpp
// =============================================================================

#include "mqtt_manager.h"
#include "config.h"
#include "command_parser.h"
#include "ack_reporter.h"

MqttManager* MqttManager::instance = nullptr;

MqttManager::MqttManager(
    WiFiClient& wifiClient,
    SafetyStateMachine& stateMachine,
    HeartbeatWatchdog& watchdog
) : mqttClient(wifiClient), stateMachine(stateMachine), watchdog(watchdog), lastReconnectAttempt(0) {
    instance = this;
}

void MqttManager::setup(const char* brokerHost, uint16_t brokerPort) {
    mqttClient.setServer(brokerHost, brokerPort);
    mqttClient.setCallback(MqttManager::onMessageReceived);
    mqttClient.setBufferSize(1024);
}

void MqttManager::loop() {
    if (!mqttClient.connected()) {
        if (stateMachine.getCurrentState() == SafetyState::CONNECTED_LOCKED) {
            stateMachine.transitionTo(SafetyState::DISCONNECTED, "MQTT broker connection lost");
        }
        reconnect();
    } else {
        mqttClient.loop();
    }
}

bool MqttManager::isConnected() {
    return mqttClient.connected();
}

void MqttManager::reconnect() {
    unsigned long now = millis();
    if (now - lastReconnectAttempt < MQTT_RECONNECT_MS) return;
    lastReconnectAttempt = now;

    Serial.print(F("[MQTT] Attempting connection to broker... "));
    if (mqttClient.connect(DEVICE_ID, MQTT_USER_SECRET, MQTT_PASSWORD_SECRET)) {
        Serial.println(F("CONNECTED!"));
        mqttClient.subscribe(TOPIC_COMMANDS);
        mqttClient.subscribe(TOPIC_HEARTBEAT);

        // Transition state (Remains strictly in LOCKED mode)
        stateMachine.transitionTo(SafetyState::CONNECTED_LOCKED, "MQTT connected & topics subscribed");
    } else {
        Serial.print(F("FAILED (rc="));
        Serial.print(mqttClient.state());
        Serial.println(F(")"));
    }
}

void MqttManager::publishStatus(const char* statusPayload) {
    if (mqttClient.connected()) {
        mqttClient.publish(TOPIC_STATUS, statusPayload);
    }
}

void MqttManager::onMessageReceived(char* topic, byte* payload, unsigned int length) {
    if (instance) {
        char buffer[1024];
        unsigned int len = length < sizeof(buffer) - 1 ? length : sizeof(buffer) - 1;
        memcpy(buffer, payload, len);
        buffer[len] = '\0';
        instance->handleIncomingMessage(topic, buffer, length);
    }
}

void MqttManager::handleIncomingMessage(const char* topic, const char* payload, unsigned int length) {
    // 1. Handle Heartbeat
    if (strcmp(topic, TOPIC_HEARTBEAT) == 0) {
        watchdog.recordHeartbeatReceived(stateMachine);
        String ackJson = AckReporter::generateHeartbeatAckJson(DEVICE_ID);
        mqttClient.publish(TOPIC_HEARTBEAT, ackJson.c_str());
        return;
    }

    // 2. Handle Edge AI Commands
    if (strcmp(topic, TOPIC_COMMANDS) == 0) {
        ParsedCommand cmd = CommandParser::parseAndValidate(payload, length);

        String ackJson;
        if (!cmd.isValid) {
            ackJson = AckReporter::generateAckJson(
                cmd.commandId,
                "REJECTED",
                stateMachine.getStateString(),
                cmd.rejectionReason.c_str(),
                DEVICE_ID
            );
        } else {
            // Valid simulated command received
            ackJson = AckReporter::generateAckJson(
                cmd.commandId,
                "ACK",
                stateMachine.getStateString(),
                "COMMAND_ACCEPTED_SIMULATION_ONLY",
                DEVICE_ID
            );
        }

        mqttClient.publish(TOPIC_ACK, ackJson.c_str());
    }
}
