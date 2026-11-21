// =============================================================================
// Krishnetra Aavishkar: Heartbeat Watchdog Implementation
// File: firmware/esp32/src/heartbeat_watchdog.cpp
// =============================================================================

#include "heartbeat_watchdog.h"

HeartbeatWatchdog::HeartbeatWatchdog(unsigned long timeoutMs)
    : timeoutMs(timeoutMs), lastHeartbeatMillis(0), hasReceivedInitialHeartbeat(false) {}

void HeartbeatWatchdog::recordHeartbeatReceived(SafetyStateMachine& stateMachine) {
    lastHeartbeatMillis = millis();
    hasReceivedInitialHeartbeat = true;

    if (stateMachine.getCurrentState() == SafetyState::COMMUNICATION_LOST) {
        stateMachine.transitionTo(SafetyState::CONNECTED_LOCKED, "Heartbeat restored");
    }
}

void HeartbeatWatchdog::checkTimeout(SafetyStateMachine& stateMachine) {
    if (!hasReceivedInitialHeartbeat) return;

    if (stateMachine.getCurrentState() == SafetyState::CONNECTED_LOCKED) {
        if ((millis() - lastHeartbeatMillis) > timeoutMs) {
            stateMachine.transitionTo(SafetyState::COMMUNICATION_LOST, "Heartbeat timeout exceeded");
        }
    }
}

bool HeartbeatWatchdog::isTimedOut() const {
    if (!hasReceivedInitialHeartbeat) return true;
    return (millis() - lastHeartbeatMillis) > timeoutMs;
}

unsigned long HeartbeatWatchdog::getLastHeartbeatTime() const {
    return lastHeartbeatMillis;
}
