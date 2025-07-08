
#include <VL53L0X.h>
#include <Wire.h>
#include <FastLED.h>

////////////////////////////////////////////////////////////////
#define XSHUT_SENSOR_X A9
#define XSHUT_SENSOR_Y A8

#define NUM_LEDS    16
#define LED_PIN     40
#define BRIGHTNESS  210


VL53L0X tofSensorX;
VL53L0X tofSensorY;

CRGB leds[NUM_LEDS];
bool ledsInitialized = false;

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
int MAX_POS_X = 300;
int MAX_POS_Y = 300;
int MIN_POS_X = 0;
int MIN_POS_Y = 0;
////////////////////////////////////////////////////////////////
const int HOMING_SPEED_FAST = 30; //30    
const int HOMING_SPEED_SLOW = 150; //150
const int HOMING_BACK_DISTANCE = 10 * STEPS_PER_MM;
const unsigned long TIMEOUT_STEPS_X = 300 * STEPS_PER_MM;
const unsigned long TIMEOUT_STEPS_Y = 300 * STEPS_PER_MM;
////////////////////////////////////////////////////////////////
int MIN_DISTANCE_TOF_X = 20;
int MIN_DISTANCE_TOF_Y = 20;
int MAX_DISTANCE_TOF_X = 100;
int MAX_DISTANCE_TOF_Y = 100;
////////////////////////////////////////////////////////////////
const int MOVE_SPEED = 60;//100
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
const float ORIGIN_OFFSET_X = 48; // Ajusta según la posición real del origen
const float ORIGIN_OFFSET_Y = 105; // Ajusta según la posición real del origen
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
bool performFullHoming();
bool initTofSensors();
bool initializeTofSensorX(unsigned long timingBudget);
bool initializeTofSensorY(unsigned long timingBudget);
bool isValidTofReading(VL53L0X& sensor, int distance);
int readAverageTofDistance(VL53L0X& sensor, int numSamples);
void moveWithPID(float targetDistance, int stepPin, int dirPin, VL53L0X& sensor, int endstopPin, bool isXaxis);

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
  Wire.begin(); 
  initTofSensors();
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

void initLEDs() {
  if (!ledsInitialized) {
    FastLED.addLeds<WS2811, LED_PIN, RGB>(leds, NUM_LEDS);
    FastLED.setBrightness(BRIGHTNESS);
    FastLED.setCorrection(0xFFFFFF);
    ledsInitialized = true;
  }
}

void turnOnLEDs() {
  initLEDs();
  fill_solid(leds, NUM_LEDS, CRGB::White);
  FastLED.show();
}

void turnOffLEDs() {
  if (ledsInitialized) {
    fill_solid(leds, NUM_LEDS, CRGB::Black);
    FastLED.show();
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
    Serial.println(ORIGIN_OFFSET_Y);
    current_pos_x = 0.0;
    current_pos_y = 0.0;
    
    moveMM(STEP_X, DIR_X, ORIGIN_OFFSET_X, true);
    moveMM(STEP_Y, DIR_Y, ORIGIN_OFFSET_Y, false);

    MAX_POS_X = 245 - ORIGIN_OFFSET_X;
    MAX_POS_Y = 174 - ORIGIN_OFFSET_Y;
    
    current_pos_x = 0.0;
    current_pos_y = 0.0;

    Serial.println("OK");
  } else if (command == "XLIMIT") {
    Serial.print("OK: ");
    Serial.println(MAX_POS_X);
  } else if (command == "YLIMIT") {
    Serial.print("OK: ");
    Serial.println(MAX_POS_Y);
  } else if (command.startsWith("PIDX ")) {
    int showPrint = command.substring(5).toInt();
    float distance = command.substring(7).toFloat();
    bool show = (showPrint != 0);  // Fix: declare show at this scope level and assign properly
    moveWithPID(distance, STEP_X, DIR_X, tofSensorX, ENDSTOP_X, true, show);
  } else if (command.startsWith("PIDY ")) {
    int showPrint = command.substring(5).toInt();
    float distance = command.substring(7).toFloat();
    bool show = (showPrint != 0);  // Fix: declare show at this scope level and assign properly
    moveWithPID(distance, STEP_Y, DIR_Y, tofSensorY, ENDSTOP_Y, false, show);
  } else if (command == "TOFX") {
    // Solo leer el sensor X
    if (!sensorTofXOk) {
      Serial.println("E: TOF X not initialized");
      return;
    }
    int distance = readAverageTofDistance(tofSensorX, 6);
    if (distance < 0) {
      Serial.println("E: Invalid TOF X reading");
    } else {
      Serial.print("OK: ");
      Serial.println(distance);
    }
  } else if (command == "TOFY") {
    // Solo leer el sensor Y
    if (!sensorTofYOk) {
      Serial.println("E: TOF Y not initialized");
      return;
    }
    int distance = readAverageTofDistance(tofSensorY, 6);
    if (distance < 0) {
      Serial.println("E: Invalid TOF Y reading");
    } else {
      Serial.print("OK: ");
      Serial.println(distance);
    }
  } 
  else if (command.startsWith("SET KP ")) {
    kp = command.substring(7).toFloat();
    Serial.print("OK: KP set to ");
    Serial.println(kp);
  }
  else if (command.startsWith("SET KI ")) {
    ki = command.substring(7).toFloat();
    Serial.print("OK: KI set to ");
    Serial.println(ki);
  }
  else if (command.startsWith("SET KD ")) {
    kd = command.substring(7).toFloat();
    Serial.print("OK: KD set to ");
    Serial.println(kd);
  }
  else if (command.startsWith("SET MAX_X ")) {
    MAX_POS_X = command.substring(9).toInt();
    Serial.print("OK: MAX_X set to ");
    Serial.println(MAX_POS_X);
  }
  else if (command.startsWith("SET MAX_Y ")) {
    MAX_POS_Y = command.substring(9).toInt();
    Serial.print("OK: MAX_Y set to ");
    Serial.println(MAX_POS_Y);
  }
  else if (command.startsWith("SET MIN_X ")) {
    MIN_POS_X = command.substring(9).toInt();
    Serial.print("OK: MIN_X set to ");
    Serial.println(MIN_POS_X);
  }
  else if (command.startsWith("SET MIN_Y ")) {
    MIN_POS_Y = command.substring(9).toInt();
    Serial.print("OK: MIN_Y set to ");
    Serial.println(MIN_POS_Y);
  }
  else if (command.startsWith("SET MIN_TOF_X ")) {
    MIN_DISTANCE_TOF_X = command.substring(13).toInt();
    Serial.print("OK: MIN_TOF_X set to ");
    Serial.println(MIN_DISTANCE_TOF_X);
  }
  else if (command.startsWith("SET MIN_TOF_Y ")) {
    MIN_DISTANCE_TOF_Y = command.substring(13).toInt();
    Serial.print("OK: MIN_TOF_Y set to ");
    Serial.println(MIN_DISTANCE_TOF_Y);
  }
  else if (command.startsWith("SET MAX_TOF_X ")) {
    MAX_DISTANCE_TOF_X = command.substring(13).toInt();
    Serial.print("OK: MAX_TOF_X set to ");
    Serial.println(MAX_DISTANCE_TOF_X);
  }
  else if (command.startsWith("SET MAX_TOF_Y ")) {
    MAX_DISTANCE_TOF_Y = command.substring(13).toInt();
    Serial.print("OK: MAX_TOF_Y set to ");
    Serial.println(MAX_DISTANCE_TOF_Y);
  }
  // Get current parameters
  else if (command == "GET PID") {
    Serial.print("KP:");
    Serial.print(kp);
    Serial.print(" KI:");
    Serial.print(ki);
    Serial.print(" KD:");
    Serial.println(kd);
  }
  else if (command == "GET LIMITS") {
    Serial.print("X:[");
    Serial.print(MIN_POS_X);
    Serial.print(",");
    Serial.print(MAX_POS_X);
    Serial.print("] Y:[");
    Serial.print(MIN_POS_Y);
    Serial.print(",");
    Serial.print(MAX_POS_Y);
    Serial.println("]");
  }
  else if (command == "GET TOF_LIMITS") {
    Serial.print("TOF_X:[");
    Serial.print(MIN_DISTANCE_TOF_X);
    Serial.print(",");
    Serial.print(MAX_DISTANCE_TOF_X);
    Serial.print("] TOF_Y:[");
    Serial.print(MIN_DISTANCE_TOF_Y);
    Serial.print(",");
    Serial.print(MAX_DISTANCE_TOF_Y);
    Serial.println("]");
  } else if (command == "LED ON" || command == "LED") {
    turnOnLEDs();
    Serial.println("OK: LED ON");
  }
  else if (command == "LED OFF") {
    turnOffLEDs();
    Serial.println("OK: LED OFF");
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
       adjustedDistance = MIN_POS_X - current_pos_x;
     } else if (finalPosition > MAX_POS_X) {
       adjustedDistance = MAX_POS_X - current_pos_x;
     }
   } else {
     if (finalPosition < MIN_POS_Y) {
       adjustedDistance = MIN_POS_Y - current_pos_y;
     } else if (finalPosition > MAX_POS_Y) {
       adjustedDistance = MAX_POS_Y - current_pos_y;
     }
   }
   
  if (adjustedDistance == 0) {
    Serial.println("OK: 0");
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

bool initializeTofSensorX(unsigned long timingBudget = FAST_TOF_SPEED) {
  pinMode(XSHUT_SENSOR_X, OUTPUT);
  digitalWrite(XSHUT_SENSOR_X, LOW);

  delay(100);

  bool sensorXOk = false;

  for (int attempt = 1; attempt <= 3; attempt++) {
    pinMode(XSHUT_SENSOR_X, INPUT);
    delay(100);

    if (tofSensorX.init()) {
      sensorXOk = true;
      tofSensorX.setTimeout(500);
      tofSensorX.setAddress(TOF_SENSOR_X_ADDRESS);
      tofSensorX.startContinuous();
      break;
    } else {
      digitalWrite(XSHUT_SENSOR_X, LOW);
      delay(200);
    }
  }

  if (!sensorXOk) {
    Serial.println("ERROR: TOF X");
    return false;
  }
  
  return true;
}

// Initialize TOF sensor Y with configurable timing budget
bool initializeTofSensorY(unsigned long timingBudget = FAST_TOF_SPEED) {
  pinMode(XSHUT_SENSOR_Y, OUTPUT);
  digitalWrite(XSHUT_SENSOR_Y, LOW);
  delay(100);

  bool sensorYOk = false;

  for (int attempt = 1; attempt <= 3; attempt++) {
    pinMode(XSHUT_SENSOR_Y, INPUT);
    delay(100);

    if (tofSensorY.init()) {
      sensorYOk = true;
      tofSensorY.setTimeout(500);
      tofSensorY.setAddress(TOF_SENSOR_Y_ADDRESS);
      tofSensorY.startContinuous();
      break;
    } else {
      digitalWrite(XSHUT_SENSOR_Y, LOW);
      delay(200);
    }
  }

  if (!sensorYOk) {
    Serial.println("ERROR: TOF Y");
    return false;
  }
  
  return true;
}

// Initialize both TOF sensors
bool initTofSensors() {
  sensorTofXOk = initializeTofSensorX();
  sensorTofYOk = initializeTofSensorY();
  return sensorTofXOk && sensorTofYOk;
}

// Check if TOF reading is valid
bool isValidTofReading(VL53L0X& sensor, int distance) {
  return !sensor.timeoutOccurred() && distance > 0 && distance < 8190;
}

// Read average TOF distance from multiple samples
int readAverageTofDistance(VL53L0X& sensor, int numSamples = 12) {
  long sum = 0;
  int validReadings = 0;
  
  for (int i = 0; i < numSamples; i++) {
    int reading = sensor.readRangeContinuousMillimeters();
    if (isValidTofReading(sensor, reading)) {
      sum += reading;
      validReadings++;
    }
    delay(10);
  }
  
  if (validReadings == 0) {
    return -1; // No valid readings
  }
  
  return sum / validReadings;
}

// Simple PID-controlled movement using TOF sensors
void moveWithPID(float targetDistance, int stepPin, int dirPin, VL53L0X& sensor, int endstopPin, bool isXaxis, bool showPrints) {
  if (abs(targetDistance) < 50) {
    if (showPrints) {
      Serial.println("Small distance detected, using direct movement");
    }
    moveMM(stepPin, dirPin, targetDistance, isXaxis);
    return;
  }
  if (showPrints) {
    Serial.print("PID movement: ");
    Serial.print(targetDistance);
    Serial.println(" mm");
  }
  
  bool& sensorOk = isXaxis ? sensorTofXOk : sensorTofYOk;
  if (!sensorOk) {
    Serial.println("E: TOF sensor not initialized");
    return;
  }
  
  // Get initial TOF distance (average of readings)
  int initialTofDistance = readAverageTofDistance(sensor);
  if (initialTofDistance < 0) {
    Serial.println("E: Invalid initial TOF reading");
    return;
  }
  
  const int MIN_TOF_DISTANCE = isXaxis ? MIN_DISTANCE_TOF_X : MIN_DISTANCE_TOF_Y;
  const int MAX_TOF_DISTANCE = isXaxis ? MAX_DISTANCE_TOF_X : MAX_DISTANCE_TOF_Y;

  if (initialTofDistance - targetDistance < MIN_TOF_DISTANCE) {
    targetDistance = -MIN_TOF_DISTANCE + initialTofDistance;
  } else if (initialTofDistance - targetDistance > MAX_TOF_DISTANCE) {
    targetDistance = -MAX_TOF_DISTANCE + initialTofDistance;
  } 

  float targetTofDistance = initialTofDistance - targetDistance;

  if (targetTofDistance < 25) {
    moveMM(stepPin, dirPin, targetDistance, isXaxis);
    return;
  }
  // Enable motors
  digitalWrite(EN_PIN, LOW);
  delay(100);
  
  int iterations = 0;
  unsigned long totalSteps = 0;
    
  if (showPrints) {
    Serial.print("Initial TOF: ");
    Serial.print(initialTofDistance);
    Serial.print(" mm, Target TOF: ");
    Serial.print(targetTofDistance);
    Serial.println(" mm");
  }
  
  bool moveTowardsEndstop = (targetDistance > 0);
  if (isXaxis) {
    digitalWrite(dirPin, moveTowardsEndstop ? LOW : HIGH);
  } else {
    digitalWrite(dirPin, moveTowardsEndstop ? HIGH : LOW);
  }
  
  // Initialize PID variables
  float error = 0;
  float prevError = 0;
  float integral = 0;
  int currentTofDistance = initialTofDistance;
  bool approachComplete = false;
  bool highPrecisionMode = false;

  int minStepDelay = 15;
  int maxStepDelay = 350;
  int stepDelay = 60; // Start with minimum delay
  
  bool currentDirection = moveTowardsEndstop; // True if moving towards endstop, false otherwise
  Serial.println(">> IT\tDIST\tERROR\tINTG\tDERIV\tOUTPUT\tPASOS");  // PRINT TEMPORAL

  while (iterations < MAX_ITERATIONS_PID) {
    if (digitalRead(endstopPin) == LOW) {
      Serial.println("E: PHYSICAL LIMIT");
      stopMotors();
      return;
    }

    currentTofDistance = readAverageTofDistance(sensor);
    if (currentTofDistance < 0) {
      Serial.println("E: Invalid TOF reading");
      stopMotors();
      approachComplete = true;
      break;
    }
    error = targetTofDistance - currentTofDistance;

    if (abs(error) < 25) {
      if (showPrints) {
        Serial.println("E: PID error below threshold, stopping movement");
      }
      approachComplete = true;

      break; // Stop if error is small enough
    }

    integral += error;
    integral = constrain(integral, -1000, 1000); // Limit integral to prevent windup

    float derivative = error - prevError;
    prevError = error;

    float output = kp * error + ki * integral + kd * derivative;

    currentDirection = (output > 0);
    if (isXaxis) {
      digitalWrite(dirPin, currentDirection ? LOW : HIGH);
    } else {
      digitalWrite(dirPin, currentDirection ? HIGH : LOW);
    }

    int stepsToMove = max(1, abs(constrain(output, -200, 200)) / STEPS_PER_MM);

    stepDelay = map(constrain(abs(error), 25, 200), 25, 200, maxStepDelay, minStepDelay);

    if (showPrints) {
      Serial.print(iterations);
      Serial.print("\t");
      Serial.print(currentTofDistance);
      Serial.print("\t");
      Serial.print(error);
      Serial.print("\t");
      Serial.print(integral);
      Serial.print("\t");
      Serial.print(derivative);
      Serial.print("\t");
      Serial.print(output);
      Serial.print("\t");
      Serial.println(stepsToMove);
    }

    for (int i = 0; i < stepsToMove && !approachComplete; i++) {
      float nextPos = isXaxis ? current_pos_x : current_pos_y;
      nextPos += currentDirection ? 1.0/STEPS_PER_MM : -1.0/STEPS_PER_MM;

      if ((isXaxis && (nextPos < MIN_POS_X || nextPos > MAX_POS_X)) || 
          (!isXaxis && (nextPos < MIN_POS_Y || nextPos > MAX_POS_Y))) {
        Serial.println("E: POSITION LIMIT");
        approachComplete = true;
        break;
      }

      makeStep(stepPin, stepDelay);
      totalSteps++;
      
      if (isXaxis) {
        current_pos_x += currentDirection ? 1.0/STEPS_PER_MM : -1.0/STEPS_PER_MM;
      } else {
        current_pos_y += currentDirection ? 1.0/STEPS_PER_MM : -1.0/STEPS_PER_MM;
      }
      
      // Check endstop
      if (digitalRead(endstopPin) == LOW) {
        Serial.println("E: ENDSTOP HIT");
        approachComplete = true;
        break;
      }
    }

    iterations++;
    delay(10);

  }
  float movedDistance = (float)totalSteps / STEPS_PER_MM;
  if (!currentDirection) {  // Si la dirección final era negativa
    movedDistance = -movedDistance;
  }
  
  // Move remaining distance directly
  float remainingDistance = targetDistance - movedDistance;
  
  if (abs(remainingDistance) > 2.0) { // Only move if remaining distance is significant
    if (showPrints) {
      Serial.print("PID complete. Moving remaining distance: ");
      Serial.println(remainingDistance);
    }
    moveMM(stepPin, dirPin, remainingDistance, isXaxis);
  }
  
  stopMotors();
}