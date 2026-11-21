// =============================================================================
// Krishnetra Aavishkar: Safety State Machine Implementation
// File: firmware/esp32/src/safety_state_machine.cpp
// =============================================================================

#include "safety_state_machine.h"

SafetyStateMachine::SafetyStateMachine() : currentState(SafetyState::BOOTING) {}

void SafetyStateMachine::transitionTo(SafetyState newState, const char* reason) {
    if (currentState == newState) return;

    Serial.print(F("[SafetyStateMachine] State change: "));
    Serial.print(getStateString());
    Serial.print(F(" -> "));

    currentState = newState;
    Serial.print(getStateString());

    if (reason) {
        Serial.print(F(" (Reason: "));
        Serial.print(reason);
        Serial.print(F(")"));
    }
    Serial.println();
}

SafetyState SafetyStateMachine::getCurrentState() const {
    return currentState;
}

const char* SafetyStateMachine::getStateString() const {
    switch (currentState) {
        case SafetyState::BOOTING:            return "BOOTING";
        case SafetyState::DISCONNECTED:       return "DISCONNECTED";
        case SafetyState::CONNECTED_LOCKED:   return "CONNECTED_LOCKED";
        case SafetyState::COMMUNICATION_LOST: return "COMMUNICATION_LOST";
        case SafetyState::ERROR_LOCKED:       return "ERROR_LOCKED";
        default:                              return "ERROR_LOCKED";
    }
}

bool SafetyStateMachine::isActionAllowed() const {
    // Permanent fail-safe rule: Actuation is NEVER allowed in this phase.
    return false;
}
