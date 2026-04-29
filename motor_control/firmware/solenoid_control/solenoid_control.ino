#include "Solenoid.h"
#include "EndEffector.h"

Solenoid firstSolenoid(9);    // holds last chip
Solenoid secondSolenoid(8);  // holds second-to-last chip

EndEffector endEffector(firstSolenoid, secondSolenoid);

unsigned long lastPrintTime = 0;

void setup() {
  Serial.begin(115200);
  endEffector.begin();

  Serial.println("End effector ready.");
}

void loop() {
  endEffector.update();

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "DISPENSE") {
      endEffector.dispense();
    }
    else if (cmd == "RELOAD") {
      endEffector.reload();
    }
    else if (cmd == "STATUS") {
      //Serial.print("STATE ");
      //Serial.println(endEffector.getStateString());
    }
  }

  if (millis() - lastPrintTime > 250) {
    lastPrintTime = millis();

    //Serial.print("STATE ");
    //Serial.println(endEffector.getStateString());
  }
}