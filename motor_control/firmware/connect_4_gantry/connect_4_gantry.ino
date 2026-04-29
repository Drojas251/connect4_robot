#include <AccelStepper.h>
#include <math.h>
#include "LimitSwitch.h"
#include "Solenoid.h"
#include "EndEffector.h"

// =======================
// END EFFECTOR / SOLENOIDS
// =======================
Solenoid firstSolenoid(9);    // holds last chip
Solenoid secondSolenoid(8);   // holds second-to-last chip

EndEffector endEffector(firstSolenoid, secondSolenoid);

// =======================
// PIN CONFIG
// =======================
constexpr uint8_t STEP_PIN = 3;
constexpr uint8_t DIR_PIN  = 4;
constexpr uint8_t EN_PIN   = 5;

// Limit switch: normally HIGH, active LOW
constexpr uint8_t HOME_LIMIT_PIN = 7;

// =======================
// MOTOR / GEARBOX RESOLUTION
// =======================
constexpr float MOTOR_STEPS_PER_REV  = 1600.0f;
constexpr float GEAR_RATIO           = 1.0f;
constexpr float OUTPUT_STEPS_PER_REV = MOTOR_STEPS_PER_REV * GEAR_RATIO;
constexpr float DEG_PER_REV          = 360.0f;

// =======================
// STATUS PUBLISHING RATE
// =======================
constexpr uint32_t STATE_PUBLISH_PERIOD_MS = 100;
constexpr uint32_t LIMIT_PUBLISH_PERIOD_MS = 100;
constexpr uint32_t EE_STATE_PUBLISH_PERIOD_MS = 250;

// If you have an external pull-up resistor, keep final argument false.
// If not, true enables INPUT_PULLUP.
LimitSwitch home_limit("home_min", HOME_LIMIT_PIN, true, true);

// =======================
// UNIT CONVERSIONS
// =======================
long degreesToSteps(float deg) {
  return lround((deg / DEG_PER_REV) * OUTPUT_STEPS_PER_REV);
}

float stepsToDegrees(long steps) {
  return (static_cast<float>(steps) / OUTPUT_STEPS_PER_REV) * DEG_PER_REV;
}

float degPerSecToStepsPerSec(float deg_per_sec) {
  return (deg_per_sec / DEG_PER_REV) * OUTPUT_STEPS_PER_REV;
}

float stepsPerSecToDegPerSec(float steps_per_sec) {
  return (steps_per_sec / OUTPUT_STEPS_PER_REV) * DEG_PER_REV;
}

float degPerSec2ToStepsPerSec2(float deg_per_sec2) {
  return (deg_per_sec2 / DEG_PER_REV) * OUTPUT_STEPS_PER_REV;
}

// =======================
// MOTION MODES
// =======================
enum class MotionMode : uint8_t {
  Idle = 0,
  Position,
  Jog
};

const char* modeToString(int mode) {
  switch (static_cast<MotionMode>(mode)) {
    case MotionMode::Idle:     return "IDLE";
    case MotionMode::Position: return "POSITION";
    case MotionMode::Jog:      return "JOG";
    default:                   return "UNKNOWN";
  }
}

// =======================
// ACCELSTEPPER
// =======================
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// =======================
// GLOBAL STATE
// =======================
String cmd_line;

MotionMode motion_mode = MotionMode::Idle;
bool prev_busy = false;
bool last_cmd_was_position = false;

float jog_target_vel_sps = 0.0f;
float jog_current_vel_sps = 0.0f;
float jog_accel_sps2 = 0.0f;

float last_reported_vel_sps = 0.0f;

uint32_t last_state_pub_ms = 0;
uint32_t last_limit_pub_ms = 0;
uint32_t last_ee_state_pub_ms = 0;
uint32_t last_motion_update_us = 0;

// =======================
// DRIVER HELPERS
// =======================
void enableDriver() {
  digitalWrite(EN_PIN, LOW);
}

bool isBusy() {
  if (motion_mode == MotionMode::Position) {
    return stepper.distanceToGo() != 0;
  }

  if (motion_mode == MotionMode::Jog) {
    return fabs(jog_current_vel_sps) > 1e-3f || fabs(jog_target_vel_sps) > 1e-3f;
  }

  return false;
}

// =======================
// PUBLISHERS
// =======================
void publishState() {
  float pos_deg = stepsToDegrees(stepper.currentPosition());
  float vel_deg_s = stepsPerSecToDegPerSec(last_reported_vel_sps);

  Serial.print("STATE ");
  Serial.print(pos_deg, 3);
  Serial.print(" ");
  Serial.print(vel_deg_s, 3);
  Serial.print(" ");
  Serial.print(isBusy() ? 1 : 0);
  Serial.print(" ");
  Serial.println(modeToString(static_cast<int>(motion_mode)));
}

void publishLimits() {
  home_limit.publishState();
}

void publishEndEffectorState() {
  Serial.print("EE_STATE ");
  Serial.println(endEffector.getStateString());
}

// =======================
// EVENTS
// =======================
void checkLimitEvents() {
  if (home_limit.updateAndCheckChanged()) {
    home_limit.publishEvent();
  }
}

void checkDoneEvent() {
  bool now_busy = isBusy();

  if (prev_busy && !now_busy && last_cmd_was_position) {
    float pos_deg = stepsToDegrees(stepper.currentPosition());
    Serial.print("DONE ");
    Serial.println(pos_deg, 3);

    last_cmd_was_position = false;
    motion_mode = MotionMode::Idle;
    last_reported_vel_sps = 0.0f;
  }

  prev_busy = now_busy;
}

// =======================
// COMMAND HANDLERS
// =======================
void handleMoveAbsDeg(float target_deg, float max_vel_deg_s, float max_accel_deg_s2) {
  float max_vel_sps = fabs(degPerSecToStepsPerSec(max_vel_deg_s));
  float max_accel_sps2 = fabs(degPerSec2ToStepsPerSec2(max_accel_deg_s2));
  long target_steps = degreesToSteps(target_deg);

  stepper.setMaxSpeed(max_vel_sps);
  stepper.setAcceleration(max_accel_sps2);
  stepper.moveTo(target_steps);

  motion_mode = MotionMode::Position;
  last_cmd_was_position = true;

  Serial.println("ACK MOVE_ABS_DEG");
}

void handleJogDegS(float vel_deg_s, float max_accel_deg_s2) {
  jog_target_vel_sps = degPerSecToStepsPerSec(vel_deg_s);
  jog_accel_sps2 = fabs(degPerSec2ToStepsPerSec2(max_accel_deg_s2));

  stepper.setMaxSpeed(max(10.0f, fabs(jog_target_vel_sps) * 1.2f));
  stepper.setSpeed(jog_current_vel_sps);

  motion_mode = MotionMode::Jog;
  last_cmd_was_position = false;

  Serial.println("ACK JOG_DEG_S");
}

void handleStop() {
  if (motion_mode == MotionMode::Position) {
    stepper.stop();
  } else if (motion_mode == MotionMode::Jog) {
    jog_target_vel_sps = 0.0f;
  }

  last_cmd_was_position = false;
  Serial.println("ACK STOP");
}

void handleSetZero() {
  stepper.setCurrentPosition(0);
  Serial.println("ACK SET_ZERO");
}

void handleDispense() {
  endEffector.dispense();
  Serial.println("ACK DISPENSE");
  publishEndEffectorState();
}

void handleReload() {
  endEffector.reload();
  Serial.println("ACK RELOAD");
  publishEndEffectorState();
}

// =======================
// COMMAND PARSER
// =======================
void parseCommand(const String& line) {
  String cmd = line;
  cmd.trim();

  if (cmd.length() == 0) return;

  if (cmd == "STOP") {
    handleStop();
    return;
  }

  if (cmd == "GET_STATE") {
    publishState();
    return;
  }

  if (cmd == "GET_LIMITS") {
    publishLimits();
    return;
  }

  if (cmd == "SET_ZERO") {
    handleSetZero();
    return;
  }

  if (cmd == "DISPENSE") {
    handleDispense();
    return;
  }

  if (cmd == "RELOAD") {
    handleReload();
    return;
  }

  if (cmd == "GET_EE_STATE" || cmd == "STATUS") {
    publishEndEffectorState();
    return;
  }

  if (cmd.startsWith("MOVE_ABS_DEG ")) {
    String rest = cmd.substring(String("MOVE_ABS_DEG ").length());
    rest.trim();

    int s1 = rest.indexOf(' ');
    int s2 = rest.indexOf(' ', s1 + 1);

    if (s1 < 0 || s2 < 0) {
      Serial.println("ERR BAD_ARGS MOVE_ABS_DEG");
      return;
    }

    float target_deg = rest.substring(0, s1).toFloat();
    float max_vel_deg_s = rest.substring(s1 + 1, s2).toFloat();
    float max_accel_deg_s2 = rest.substring(s2 + 1).toFloat();

    handleMoveAbsDeg(target_deg, max_vel_deg_s, max_accel_deg_s2);
    return;
  }

  if (cmd.startsWith("JOG_DEG_S ")) {
    String rest = cmd.substring(String("JOG_DEG_S ").length());
    rest.trim();

    int s1 = rest.indexOf(' ');

    if (s1 < 0) {
      Serial.println("ERR BAD_ARGS JOG_DEG_S");
      return;
    }

    float vel_deg_s = rest.substring(0, s1).toFloat();
    float max_accel_deg_s2 = rest.substring(s1 + 1).toFloat();

    handleJogDegS(vel_deg_s, max_accel_deg_s2);
    return;
  }

  Serial.print("ERR UNKNOWN_CMD ");
  Serial.println(cmd);
}

void pollSerial() {
  while (Serial.available() > 0) {
    char ch = static_cast<char>(Serial.read());

    if (ch == '\n') {
      parseCommand(cmd_line);
      cmd_line = "";
    } else if (ch != '\r') {
      cmd_line += ch;
    }
  }
}

// =======================
// MOTION UPDATE
// =======================
float rampToward(float current, float target, float max_delta) {
  float delta = target - current;

  if (delta > max_delta) return current + max_delta;
  if (delta < -max_delta) return current - max_delta;

  return target;
}

void updateJog(uint32_t now_us) {
  float dt_s = static_cast<float>(now_us - last_motion_update_us) * 1e-6f;

  if (dt_s < 0.0f) dt_s = 0.0f;
  if (dt_s > 0.05f) dt_s = 0.05f;

  float max_dv = jog_accel_sps2 * dt_s;
  jog_current_vel_sps = rampToward(jog_current_vel_sps, jog_target_vel_sps, max_dv);

  stepper.setSpeed(jog_current_vel_sps);
  stepper.runSpeed();

  last_reported_vel_sps = jog_current_vel_sps;

  if (fabs(jog_current_vel_sps) < 1e-3f && fabs(jog_target_vel_sps) < 1e-3f) {
    motion_mode = MotionMode::Idle;
    last_reported_vel_sps = 0.0f;
  }
}

void updateMotion(uint32_t now_us) {
  if (last_motion_update_us == 0) {
    last_motion_update_us = now_us;
  }

  if (motion_mode == MotionMode::Position) {
    long prev_pos = stepper.currentPosition();
    uint32_t prev_us = last_motion_update_us;

    stepper.run();

    long new_pos = stepper.currentPosition();
    float dt_s = static_cast<float>(now_us - prev_us) * 1e-6f;

    if (dt_s > 1e-6f) {
      last_reported_vel_sps = static_cast<float>(new_pos - prev_pos) / dt_s;
    } else {
      last_reported_vel_sps = 0.0f;
    }

    if (stepper.distanceToGo() == 0) {
      last_reported_vel_sps = 0.0f;
    }
  } else if (motion_mode == MotionMode::Jog) {
    updateJog(now_us);
  } else {
    last_reported_vel_sps = 0.0f;
  }

  last_motion_update_us = now_us;
}

// =======================
// ARDUINO SETUP / LOOP
// =======================
void setup() {
  pinMode(EN_PIN, OUTPUT);
  enableDriver();

  home_limit.begin();

  Serial.begin(115200);
  delay(500);

  stepper.setCurrentPosition(0);
  stepper.setEnablePin(EN_PIN);
  stepper.setPinsInverted(false, false, true);
  stepper.enableOutputs();

  endEffector.begin();

  Serial.println("READY");
  publishEndEffectorState();
}

void loop() {
  uint32_t now_us = micros();
  uint32_t now_ms = millis();

  pollSerial();

  endEffector.update();

  updateMotion(now_us);

  checkLimitEvents();

  if (now_ms - last_state_pub_ms >= STATE_PUBLISH_PERIOD_MS) {
    last_state_pub_ms = now_ms;
    publishState();
  }

  if (now_ms - last_limit_pub_ms >= LIMIT_PUBLISH_PERIOD_MS) {
    last_limit_pub_ms = now_ms;
    publishLimits();
  }

  if (now_ms - last_ee_state_pub_ms >= EE_STATE_PUBLISH_PERIOD_MS) {
    last_ee_state_pub_ms = now_ms;
    publishEndEffectorState();
  }

  checkDoneEvent();
}
// #include <AccelStepper.h>
// #include <math.h>
// #include "LimitSwitch.h"
// #include "Solenoid.h"
// #include "EndEffector.h"

// Solenoid firstSolenoid(9);    // holds last chip
// Solenoid secondSolenoid(8);  // holds second-to-last chip

// EndEffector endEffector(firstSolenoid, secondSolenoid);

// // =======================
// // PIN CONFIG
// // =======================
// constexpr uint8_t STEP_PIN = 3;
// constexpr uint8_t DIR_PIN  = 4;
// constexpr uint8_t EN_PIN   = 5;

// // Limit switch: normally HIGH, active LOW
// constexpr uint8_t HOME_LIMIT_PIN = 7;

// // =======================
// // MOTOR / GEARBOX RESOLUTION
// // =======================
// constexpr float MOTOR_STEPS_PER_REV  = 1600.0f;
// constexpr float GEAR_RATIO           = 1.0f;
// constexpr float OUTPUT_STEPS_PER_REV = MOTOR_STEPS_PER_REV * GEAR_RATIO;
// constexpr float DEG_PER_REV          = 360.0f;

// // =======================
// // STATUS PUBLISHING RATE
// // =======================
// constexpr uint32_t STATE_PUBLISH_PERIOD_MS = 100;
// constexpr uint32_t LIMIT_PUBLISH_PERIOD_MS = 100;

// // If you have an external pull-up resistor, keep use_pullup=false.
// // If not, change final argument to true.
// LimitSwitch home_limit("home_min", HOME_LIMIT_PIN, true, true);

// // =======================
// // UNIT CONVERSIONS
// // =======================
// long degreesToSteps(float deg) {
//   return lround((deg / DEG_PER_REV) * OUTPUT_STEPS_PER_REV);
// }

// float stepsToDegrees(long steps) {
//   return (static_cast<float>(steps) / OUTPUT_STEPS_PER_REV) * DEG_PER_REV;
// }

// float degPerSecToStepsPerSec(float deg_per_sec) {
//   return (deg_per_sec / DEG_PER_REV) * OUTPUT_STEPS_PER_REV;
// }

// float stepsPerSecToDegPerSec(float steps_per_sec) {
//   return (steps_per_sec / OUTPUT_STEPS_PER_REV) * DEG_PER_REV;
// }

// float degPerSec2ToStepsPerSec2(float deg_per_sec2) {
//   return (deg_per_sec2 / DEG_PER_REV) * OUTPUT_STEPS_PER_REV;
// }

// // =======================
// // MOTION MODES
// // =======================
// enum class MotionMode : uint8_t {
//   Idle = 0,
//   Position,
//   Jog
// };

// const char* modeToString(int mode) {
//   switch (static_cast<MotionMode>(mode)) {
//     case MotionMode::Idle:     return "IDLE";
//     case MotionMode::Position: return "POSITION";
//     case MotionMode::Jog:      return "JOG";
//     default:                   return "UNKNOWN";
//   }
// }

// // =======================
// // ACCELSTEPPER
// // =======================
// AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// // =======================
// // GLOBAL STATE
// // =======================
// String cmd_line;

// MotionMode motion_mode = MotionMode::Idle;
// bool prev_busy = false;
// bool last_cmd_was_position = false;

// float jog_target_vel_sps = 0.0f;
// float jog_current_vel_sps = 0.0f;
// float jog_accel_sps2 = 0.0f;

// float last_reported_vel_sps = 0.0f;

// uint32_t last_state_pub_ms = 0;
// uint32_t last_limit_pub_ms = 0;
// uint32_t last_motion_update_us = 0;

// // =======================
// // DRIVER HELPERS
// // =======================
// void enableDriver() {
//   digitalWrite(EN_PIN, LOW);
// }

// bool isBusy() {
//   if (motion_mode == MotionMode::Position) {
//     return stepper.distanceToGo() != 0;
//   }
//   if (motion_mode == MotionMode::Jog) {
//     return fabs(jog_current_vel_sps) > 1e-3f || fabs(jog_target_vel_sps) > 1e-3f;
//   }
//   return false;
// }

// // =======================
// // PUBLISHERS
// // =======================
// void publishState() {
//   float pos_deg = stepsToDegrees(stepper.currentPosition());
//   float vel_deg_s = stepsPerSecToDegPerSec(last_reported_vel_sps);

//   Serial.print("STATE ");
//   Serial.print(pos_deg, 3);
//   Serial.print(" ");
//   Serial.print(vel_deg_s, 3);
//   Serial.print(" ");
//   Serial.print(isBusy() ? 1 : 0);
//   Serial.print(" ");
//   Serial.println(modeToString(static_cast<int>(motion_mode)));
// }

// void publishLimits() {
//   home_limit.publishState();
// }

// void checkLimitEvents() {
//   if (home_limit.updateAndCheckChanged()) {
//     home_limit.publishEvent();
//   }
// }

// void checkDoneEvent() {
//   bool now_busy = isBusy();

//   if (prev_busy && !now_busy && last_cmd_was_position) {
//     float pos_deg = stepsToDegrees(stepper.currentPosition());
//     Serial.print("DONE ");
//     Serial.println(pos_deg, 3);
//     last_cmd_was_position = false;
//     motion_mode = MotionMode::Idle;
//     last_reported_vel_sps = 0.0f;
//   }

//   prev_busy = now_busy;
// }

// // =======================
// // COMMAND HANDLERS
// // =======================
// void handleMoveAbsDeg(float target_deg, float max_vel_deg_s, float max_accel_deg_s2) {
//   float max_vel_sps = fabs(degPerSecToStepsPerSec(max_vel_deg_s));
//   float max_accel_sps2 = fabs(degPerSec2ToStepsPerSec2(max_accel_deg_s2));
//   long target_steps = degreesToSteps(target_deg);

//   stepper.setMaxSpeed(max_vel_sps);
//   stepper.setAcceleration(max_accel_sps2);
//   stepper.moveTo(target_steps);

//   motion_mode = MotionMode::Position;
//   last_cmd_was_position = true;

//   Serial.println("ACK MOVE_ABS_DEG");
// }

// void handleJogDegS(float vel_deg_s, float max_accel_deg_s2) {
//   jog_target_vel_sps = degPerSecToStepsPerSec(vel_deg_s);
//   jog_accel_sps2 = fabs(degPerSec2ToStepsPerSec2(max_accel_deg_s2));

//   stepper.setMaxSpeed(fabs(jog_target_vel_sps)); 
//   stepper.setSpeed(jog_current_vel_sps);

//   motion_mode = MotionMode::Jog;
//   last_cmd_was_position = false;

//   Serial.println("ACK JOG_DEG_S");
// }

// void handleStop() {
//   if (motion_mode == MotionMode::Position) {
//     stepper.stop();
//   } else if (motion_mode == MotionMode::Jog) {
//     jog_target_vel_sps = 0.0f;
//   }

//   last_cmd_was_position = false;
//   Serial.println("ACK STOP");
// }

// void handleSetZero() {
//   stepper.setCurrentPosition(0);
//   Serial.println("ACK SET_ZERO");
// }

// void parseCommand(const String& line) {
//   String cmd = line;
//   cmd.trim();

//   if (cmd.length() == 0) return;

//   if (cmd == "STOP") {
//     handleStop();
//     return;
//   }

//   if (cmd == "GET_STATE") {
//     publishState();
//     return;
//   }

//   if (cmd == "GET_LIMITS") {
//     publishLimits();
//     return;
//   }

//   if (cmd == "SET_ZERO") {
//     handleSetZero();
//     return;
//   }

//   if (cmd.startsWith("MOVE_ABS_DEG ")) {
//     String rest = cmd.substring(String("MOVE_ABS_DEG ").length());
//     rest.trim();

//     int s1 = rest.indexOf(' ');
//     int s2 = rest.indexOf(' ', s1 + 1);

//     if (s1 < 0 || s2 < 0) {
//       Serial.println("ERR BAD_ARGS MOVE_ABS_DEG");
//       return;
//     }

//     float target_deg = rest.substring(0, s1).toFloat();
//     float max_vel_deg_s = rest.substring(s1 + 1, s2).toFloat();
//     float max_accel_deg_s2 = rest.substring(s2 + 1).toFloat();

//     handleMoveAbsDeg(target_deg, max_vel_deg_s, max_accel_deg_s2);
//     return;
//   }

//   if (cmd.startsWith("JOG_DEG_S ")) {
//     String rest = cmd.substring(String("JOG_DEG_S ").length());
//     rest.trim();

//     int s1 = rest.indexOf(' ');
//     if (s1 < 0) {
//       Serial.println("ERR BAD_ARGS JOG_DEG_S");
//       return;
//     }

//     float vel_deg_s = rest.substring(0, s1).toFloat();
//     float max_accel_deg_s2 = rest.substring(s1 + 1).toFloat();

//     handleJogDegS(vel_deg_s, max_accel_deg_s2);
//     return;
//   }

//   Serial.print("ERR UNKNOWN_CMD ");
//   Serial.println(cmd);
// }

// void pollSerial() {
//   while (Serial.available() > 0) {
//     char ch = static_cast<char>(Serial.read());

//     if (ch == '\n') {
//       parseCommand(cmd_line);
//       cmd_line = "";
//     } else if (ch != '\r') {
//       cmd_line += ch;
//     }
//   }
// }

// // =======================
// // MOTION UPDATE
// // =======================
// float rampToward(float current, float target, float max_delta) {
//   float delta = target - current;
//   if (delta > max_delta) return current + max_delta;
//   if (delta < -max_delta) return current - max_delta;
//   return target;
// }

// void updateJog(uint32_t now_us) {
//   float dt_s = static_cast<float>(now_us - last_motion_update_us) * 1e-6f;
//   if (dt_s < 0.0f) dt_s = 0.0f;
//   if (dt_s > 0.05f) dt_s = 0.05f;

//   float max_dv = jog_accel_sps2 * dt_s;
//   jog_current_vel_sps = rampToward(jog_current_vel_sps, jog_target_vel_sps, max_dv);

//   stepper.setSpeed(jog_current_vel_sps);
//   stepper.runSpeed();

//   last_reported_vel_sps = jog_current_vel_sps;

//   if (fabs(jog_current_vel_sps) < 1e-3f && fabs(jog_target_vel_sps) < 1e-3f) {
//     motion_mode = MotionMode::Idle;
//     last_reported_vel_sps = 0.0f;
//   }
// }

// void updateMotion(uint32_t now_us) {
//   if (last_motion_update_us == 0) {
//     last_motion_update_us = now_us;
//   }

//   if (motion_mode == MotionMode::Position) {
//     long prev_pos = stepper.currentPosition();
//     uint32_t prev_us = last_motion_update_us;

//     stepper.run();

//     long new_pos = stepper.currentPosition();
//     float dt_s = static_cast<float>(now_us - prev_us) * 1e-6f;
//     if (dt_s > 1e-6f) {
//       last_reported_vel_sps = static_cast<float>(new_pos - prev_pos) / dt_s;
//     } else {
//       last_reported_vel_sps = 0.0f;
//     }

//     if (stepper.distanceToGo() == 0) {
//       last_reported_vel_sps = 0.0f;
//     }
//   } else if (motion_mode == MotionMode::Jog) {
//     updateJog(now_us);
//   } else {
//     last_reported_vel_sps = 0.0f;
//   }

//   last_motion_update_us = now_us;
// }

// // =======================
// // ARDUINO SETUP / LOOP
// // =======================
// void setup() {
//   pinMode(EN_PIN, OUTPUT);
//   enableDriver();

//   home_limit.begin();

//   Serial.begin(115200);
//   delay(500);

//   stepper.setCurrentPosition(0);
//   stepper.setEnablePin(EN_PIN);
//   stepper.setPinsInverted(false, false, true);
//   stepper.enableOutputs();

//   endEffector.begin();

//   Serial.println("READY");
// }

// void loop() {
//   uint32_t now_us = micros();
//   uint32_t now_ms = millis();

//   pollSerial();
//   updateMotion(now_us);

//   checkLimitEvents();

//   if (now_ms - last_state_pub_ms >= STATE_PUBLISH_PERIOD_MS) {
//     last_state_pub_ms = now_ms;
//     publishState();
//   }

//   if (now_ms - last_limit_pub_ms >= LIMIT_PUBLISH_PERIOD_MS) {
//     last_limit_pub_ms = now_ms;
//     publishLimits();
//   }

//   checkDoneEvent();
// }