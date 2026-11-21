#pragma once

// =============================================================================
// Krishnetra Aavishkar: Heartbeat Watchdog Header
// File: firmware/esp32/include/heartbeat_watchdog.h
// =============================================================================

#include <Arduino.h>
#include "safety_state_machine.h"

class HeartbeatWatchdog {
public:
    HeartbeatWatchdog(unsigned long timeoutMs = 5000);

    void recordHeartbeatReceived(SafetyStateMachine& stateMachine);
    void checkTimeout(SafetyStateMachine& stateMachine);
    bool isTimedOut() const;
    unsigned long getLastHeartbeatTime() const;

private:
    unsigned long timeoutMs;
    unsigned long lastHeartbeatMillis;
    bool hasReceivedInitialHeartbeat;
};
