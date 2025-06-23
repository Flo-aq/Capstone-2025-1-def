
#include <VL53L0X.h>
////////////////////////////////////////////////////////////////
#define XSHUT_SENSOR_X A0
#define XSHUT_SENSOR_Y A1

VL53L0X tofSensorX;
VL53L0X tofSensorY;

////////////////////////////////////////////////////////////////
const int EN_PIN = 8;

const int STEP_X = 2;
const int DIR_X = 5;

const int STEP_Y = 3;
const int DIR_Y = 6;
////////////////////////////////////////////////////////////////
const int ENDSTOP_X = 9;
const int ENDSTOP_Y = 10;
////////////////////////////////////////////////////////////////
const int STEPS_PER_REV = 200 * 16;
const float LEAD_SCREW_PITCH = 8.0;
const float STEPS_PER_MM = STEPS_PER_REV / LEAD_SCREW_PITCH;
////////////////////////////////////////////////////////////////
const int MAX_POS_X = 195 - 51.86;
const int MAX_POS_Y = 211 - 27.39;
const int MIN_POS_X = 0;
const int MIN_POS_Y = 0;
////////////////////////////////////////////////////////////////
const int HOMING_SPEED_FAST = 30;     
const int HOMING_SPEED_SLOW = 150;
const int HOMING_BACK_DISTANCE = 10 * STEPS_PER_MM;
const unsigned long TIMEOUT_STEPS_X = 300 * STEPS_PER_MM;
const unsigned long TIMEOUT_STEPS_Y = 200 * STEPS_PER_MM;
////////////////////////////////////////////////////////////////
const int MIN_DISTANCE_TOF_X = 20;
const int MIN_DISTANCE_TOF_Y = 20;
const int MAX_DISTANCE_TOF_X = 100;
const int MAX_DISTANCE_TOF_Y = 100;
////////////////////////////////////////////////////////////////
const int MOVE_SPEED = 100;
////////////////////////////////////////////////////////////////
const unsigned long FAST_TOF_SPEED = 20000;
const unsigned long SLOW_TOF_SPEED = 200000;

const uint8_t TOF_SENSOR_X_ADDRESS = 0x30;
const uint8_t TOF_SENSOR_Y_ADDRESS = 0x31;

unsigned long lastResetTime = 0;
const unsigned long resetCooldown = 1000;

const int PRECISION_THRESHOLD_ON = 15;  // Umbral para activar alta precisión
const int PRECISION_THRESHOLD_OFF = 25; // Umbral para desactivar (más grande)
////////////////////////////////////////////////////////////////
const float ORIGIN_OFFSET_X = 51.86; // Ajusta según la posición real del origen
const float ORIGIN_OFFSET_Y = 27.39; // Ajusta según la posición real del origen
////////////////////////////////////////////////////////////////
float kp = 0.8;
float ki = 0.1;
float kd = 0.4;

const int MAX_ITERATIONS_PID = 500;

String inputBuffer = "";
bool commandComplete = false;

bool sensorTofXOk = false;
bool sensorTofYOk = false;

float current_pos_x = 0.0;
float current_pos_y = 0.0;

void processCommand(String command);
bool homing(int stepPin, int dirPin, int endstopPin, bool homingDirection);
void makeStep(int stepPin, int delayTime);
void moveMM(int stepPin, int dirPin, float distance, bool isXaxis);
void stopMotors();
float calculateMovedDistance(unsigned long steps, bool moveDirection);

void setup() {
  Serial.begin(115200);

  pinMode(EN_PIN, OUTPUT);
  pinMode(STEP_X, OUTPUT);
  pinMode(DIR_X, OUTPUT);
  pinMode(STEP_Y, OUTPUT);
  pinMode(DIR_Y, OUTPUT);
  pinMode(ENDSTOP_X, INPUT_PULLUP);
  pinMode(ENDSTOP_Y, INPUT_PULLUP);

  digitalWrite(EN_PIN, HIGH);
  digitalWrite(STEP_X, LOW);
  digitalWrite(STEP_Y, LOW);
}

void loop() {
  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();

    if (inChar == '\n') {
      commandComplete = true;
    } else {
      inputBuffer += inChar;
    }
  }

  if (commandComplete) {
    processCommand(inputBuffer);
    Serial.println("COMMAND EXECUTED");
    inputBuffer = "";
    commandComplete = false;
  }
}

void processCommand(String command) {
  command.trim();
  command.toUpperCase();

  if (command == "HX") {
    bool succesfull = homing(STEP_X, DIR_X, ENDSTOP_X, true);
    if (succesfull) {
      Serial.println("OK");
    } else {
      Serial.println("E: HOMING FAILED");
    }
  } else if (command == "HY") {
    bool succesfull = homing(STEP_Y, DIR_Y, ENDSTOP_Y, false);
    if (succesfull) {
      Serial.println("OK");
    } else {
      Serial.println("E: HOMING FAILED");
    }
  } else if (command.startsWith("MX ")) {
    float distance = command.substring(3).toFloat();
    moveMM(STEP_X, DIR_X, distance, true);
  } else if (command.startsWith("MY ")) {
    float distance = command.substring(3).toFloat();
    moveMM(STEP_Y, DIR_Y, distance, false);
  }  else if (command.startsWith("S ")) {
    stopMotors();
  } else if (command == "POSX") {
    Serial.print(" X:");
    Serial.println(current_pos_x);
  } else if (command == "POSY") {
    Serial.print(" Y:");
    Serial.println(current_pos_y);
  } else if (command == "FH") {
    if (!performFullHoming()) {
      Serial.println("E: HOMING FAILED");
      return;
    }
    Serial.println("OK");
  } else if (command == "SETORIGIN") {
    current_pos_x = 0.0;
    current_pos_y = 0.0;
    Serial.println("OK: ORIGIN SET");
  } else if (command == "HOME&SET&ORIGIN") {
    if (!performFullHoming()) {
      Serial.println("E: HOMING FAILED");
      return;
    }
    moveMM(STEP_X, DIR_X, ORIGIN_OFFSET_X, true);
    moveMM(STEP_Y, DIR_Y, ORIGIN_OFFSET_Y, false);

    current_pos_x = 0.0;
    current_pos_y = 0.0;

    Serial.println("OK");
  } else if (command == "XLIMIT") {
    Serial.println(MAX_POS_X);
  } else if (command == "YLIMIT") {
    Serial.println(MAX_POS_Y);
  } else {
    Serial.println("E: CNF");
  }
}


bool homing(int stepPin, int dirPin, int endstopPin, bool homingDirection) {
  digitalWrite(EN_PIN, LOW);
  delay(100);

  // F1
  digitalWrite(dirPin, homingDirection ? HIGH : LOW);

  int steps = 0;

  unsigned long TIMEOUT_STEPS;

  if (homingDirection) {
    TIMEOUT_STEPS = TIMEOUT_STEPS_X;
  } else {
    TIMEOUT_STEPS = TIMEOUT_STEPS_Y;
  }

  while (digitalRead(endstopPin) == HIGH) {
    makeStep(stepPin, HOMING_SPEED_FAST);
    steps++;

    if (steps > TIMEOUT_STEPS) {
      Serial.println("E: endstop");
      stopMotors();
      return false;
    }
  }

  delay(500);

  // F2
  digitalWrite(dirPin, homingDirection ? LOW : HIGH);

  for (int i = 0; i < HOMING_BACK_DISTANCE; i++) {
    makeStep(stepPin, HOMING_SPEED_FAST);
  }

  delay(500);

  //F3
  digitalWrite(dirPin, homingDirection ? HIGH : LOW);

  steps = 0;
  while (digitalRead(endstopPin) == HIGH) {
    makeStep(stepPin, HOMING_SPEED_SLOW);
    steps++;

    if (steps > TIMEOUT_STEPS) {
      Serial.println("E: endstop");
      stopMotors();
      return false;
    }
  
  }
  
  digitalWrite(EN_PIN, HIGH);
  return true;
}

bool performFullHoming() {
  bool succesfull_x = homing(STEP_X, DIR_X, ENDSTOP_X, true);
  delay(100);
  bool succesfull_y = homing(STEP_Y, DIR_Y, ENDSTOP_Y, false);
  delay(100);
  return succesfull_x && succesfull_y;
}


void makeStep(int stepPin, int delayTime) {
  digitalWrite(stepPin, HIGH);
  delayMicroseconds(delayTime);
  digitalWrite(stepPin, LOW);
  delayMicroseconds(delayTime);
}

void moveMM(int stepPin, int dirPin, float distance, bool isXaxis) {

  float finalPosition = isXaxis ? current_pos_x + distance : current_pos_y + distance;
  float adjustedDistance = distance;

  if (isXaxis) {
     if (finalPosition < MIN_POS_X) {
       adjustedDistance = -current_pos_x;
     } else if (finalPosition > MAX_POS_X) {
       adjustedDistance = MAX_POS_X - current_pos_x;
     }
   } else {
     if (finalPosition < MIN_POS_Y) {
       adjustedDistance = -current_pos_y;
     } else if (finalPosition > MAX_POS_Y) {
       adjustedDistance = MAX_POS_Y - current_pos_y;
     }
   }
   
  if (adjustedDistance == 0) {
    Serial.print("OK: 0");
    return;
  }

  digitalWrite(EN_PIN, LOW);
  delay(100);

  unsigned long steps = abs(adjustedDistance) * STEPS_PER_MM;

  bool moveDirection = (distance > 0);
  if (isXaxis) {
    digitalWrite(dirPin, moveDirection ? LOW : HIGH);
  } else {
    digitalWrite(dirPin, moveDirection ? HIGH : LOW);
  }

  int endstopPin = isXaxis ? ENDSTOP_X : ENDSTOP_Y;

  bool stopRequested = false;
  unsigned long completedSteps = 0;


  if (digitalRead(endstopPin) == LOW) {
    if (!moveDirection) {
      Serial.println("E: PHYSICAL LIMIT");
      stopMotors();
      return;
    }
  }

  for (unsigned long i = 0; i < steps && !stopRequested; i++) {
    if (digitalRead(endstopPin) == LOW) {
      if (!moveDirection) {
        stopRequested = true;
        break;
      }
    }

    makeStep(stepPin, MOVE_SPEED);
    completedSteps++;
  }

  float actualMovedDistance = (float)completedSteps / STEPS_PER_MM;
  if (!moveDirection) {
    actualMovedDistance = -actualMovedDistance;
  }

  if (isXaxis) {
    current_pos_x += actualMovedDistance;
  } else {
    current_pos_y += actualMovedDistance;
  }

  Serial.print("OK: ");
  Serial.println(actualMovedDistance);
  stopMotors();
}

void stopMotors() {
  digitalWrite(EN_PIN, HIGH);

  digitalWrite(STEP_X, LOW);
  digitalWrite(STEP_Y, LOW);

  delay(100);
}



float calculateMovedDistance(unsigned long steps, bool moveDirection) {
  float distance = (float)steps / STEPS_PER_MM;
  return moveDirection ? distance : -distance;
}
