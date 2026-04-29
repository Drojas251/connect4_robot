#pragma once
#include <Arduino.h>

class LimitSwitch {
public:
  LimitSwitch(const char* name, uint8_t pin, bool active_low = true, bool use_pullup = false)
    : name_(name),
      pin_(pin),
      active_low_(active_low),
      use_pullup_(use_pullup),
      last_active_(false),
      initialized_(false) {}

  void begin() {
    pinMode(pin_, use_pullup_ ? INPUT_PULLUP : INPUT);
    last_active_ = isActive();
    initialized_ = true;
  }

  const char* name() const {
    return name_;
  }

  uint8_t pin() const {
    return pin_;
  }

  int raw() const {
    return digitalRead(pin_);
  }

  bool isActive() const {
    int v = raw();
    return active_low_ ? (v == LOW) : (v == HIGH);
  }

  bool updateAndCheckChanged() {
    bool active = isActive();

    if (!initialized_) {
      last_active_ = active;
      initialized_ = true;
      return false;
    }

    if (active != last_active_) {
      last_active_ = active;
      return true;
    }

    return false;
  }

  void publishState() const {
    Serial.print("LIMIT_STATE ");
    Serial.print(name_);
    Serial.print(" ");
    Serial.print(pin_);
    Serial.print(" ");
    Serial.print(raw());
    Serial.print(" ");
    Serial.println(isActive() ? 1 : 0);
  }

  void publishEvent() const {
    bool active = isActive();

    Serial.print("LIMIT_EVENT ");
    Serial.print(name_);
    Serial.print(" ");
    Serial.print(active ? "PRESSED" : "RELEASED");
    Serial.print(" ");
    Serial.print(pin_);
    Serial.print(" ");
    Serial.print(raw());
    Serial.print(" ");
    Serial.println(active ? 1 : 0);
  }

private:
  const char* name_;
  uint8_t pin_;
  bool active_low_;
  bool use_pullup_;
  bool last_active_;
  bool initialized_;
};