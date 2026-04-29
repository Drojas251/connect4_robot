#ifndef END_EFFECTOR_H
#define END_EFFECTOR_H

#include <Arduino.h>
#include "Solenoid.h"

enum class EndEffectorState {
  READY,
  DISPENSING,
  RELOADING,
  ERROR
};

class EndEffector {
private:
  Solenoid& firstSolenoid;   // holds the last chip
  Solenoid& secondSolenoid;  // holds the second-to-last chip

  EndEffectorState state;

  unsigned long stateStartTime;
  unsigned long dispenseDurationMs;
  unsigned long reloadDurationMs;

public:
  EndEffector(
    Solenoid& first,
    Solenoid& second,
    unsigned long dispenseMs = 1000,
    unsigned long reloadMs = 1000
  )
    : firstSolenoid(first),
      secondSolenoid(second),
      state(EndEffectorState::READY),
      stateStartTime(0),
      dispenseDurationMs(dispenseMs),
      reloadDurationMs(reloadMs)
  {}

  void begin() {
    firstSolenoid.begin();
    secondSolenoid.begin();

    state = EndEffectorState::READY;
  }

  void dispense() {
    if (state != EndEffectorState::READY) {
      return;
    }

    // Dispense logic:
    // second solenoid stays extended to hold the stack
    // first solenoid retracts to release last chip
    secondSolenoid.extend();
    firstSolenoid.retract();

    state = EndEffectorState::DISPENSING;
    stateStartTime = millis();
  }

  void reload() {
    if (state != EndEffectorState::READY) {
      return;
    }

    // Reload logic:
    // first solenoid extends to catch next chip
    // second solenoid retracts to let one chip drop down
    firstSolenoid.extend();
    secondSolenoid.retract();

    state = EndEffectorState::RELOADING;
    stateStartTime = millis();
  }

  void update() {
    unsigned long now = millis();

    if (state == EndEffectorState::DISPENSING) {
      if (now - stateStartTime >= dispenseDurationMs) {
        // Return to loaded/ready state
        firstSolenoid.extend();
        secondSolenoid.extend();
        state = EndEffectorState::READY;
      }
    }

    if (state == EndEffectorState::RELOADING) {
      if (now - stateStartTime >= reloadDurationMs) {
        // Return to loaded/ready state
        firstSolenoid.extend();
        secondSolenoid.extend();

        state = EndEffectorState::READY;
      }
    }
  }

  bool isBusy() const {
    return state == EndEffectorState::DISPENSING ||
           state == EndEffectorState::RELOADING;
  }

  bool isReady() const {
    return state == EndEffectorState::READY;
  }

  EndEffectorState getState() const {
    return state;
  }

  const char* getStateString() const {
    switch (state) {
      case EndEffectorState::READY:
        return "READY";
      case EndEffectorState::DISPENSING:
        return "DISPENSING";
      case EndEffectorState::RELOADING:
        return "RELOADING";
      case EndEffectorState::ERROR:
        return "ERROR";
      default:
        return "UNKNOWN";
    }
  }
};

#endif