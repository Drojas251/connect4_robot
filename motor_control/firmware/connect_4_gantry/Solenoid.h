#ifndef SOLENOID_H
#define SOLENOID_H

#include <Arduino.h>

class Solenoid {
private:
  int pin;
  bool energizedExtends;

public:
  // doesEnergizeExtend = true:
  // applying power causes extension
  // false:
  // applying power causes retraction
  Solenoid(int controlPin, bool doesEnergizeExtend = false) {
    pin = controlPin;
    energizedExtends = doesEnergizeExtend;
  }

  void begin() {
    pinMode(pin, OUTPUT);
    extend();   // safe startup state
  }

  void extend() {
    digitalWrite(pin, energizedExtends ? HIGH : LOW);
  }

  void retract() {
    digitalWrite(pin, energizedExtends ? LOW : HIGH);
  }

  void holdExtended(unsigned long ms) {
    extend();
    delay(ms);
  }

  void holdRetracted(unsigned long ms) {
    retract();
    delay(ms);
  }
};
#endif