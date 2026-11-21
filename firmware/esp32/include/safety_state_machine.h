#pragma once

// =============================================================================
// Krishnetra Aavishkar: Safety State Machine Header
// File: firmware/esp32/include/safety_state_machine.h
// =============================================================================

#include <Arduino.h>

enum class SafetyState {
    BOOTING,
    DISCONNECTED,
    CONNECTED_LOCKED,
    COMMUNICATION_LOST,
    ERROR_LOCKED
};

class SafetyStateMachine {
public:
    SafetyStateMachine();

    void transitionTo(SafetyState newState, const char* reason = nullptr);
    SafetyState getCurrentState() const;
    const char* getStateString() const;
    bool isActionAllowed() const;

private:
    SafetyState currentState;
};
