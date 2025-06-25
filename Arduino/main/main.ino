
#include <VL53L0X.h>
////////////////////////////////////////////////////////////////
#define XSHUT_SENSOR_X A9
#define XSHUT_SENSOR_Y A8

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
    Serial.print("OK: ");
    Serial.println(MAX_POS_X);
  } else if (command == "YLIMIT") {
    Serial.print("OK: ");
    Serial.println(MAX_POS_Y);
  } else if (command.startsWith("TOFX ")) {
    float distance = command.substring(5).toFloat();
    moveWithPID(distance, STEP_X, DIR_X, tofSensorX, ENDSTOP_X, true);
  } else if (command.startsWith("TOFY ")) {
    float distance = command.substring(5).toFloat();
    moveWithPID(distance, STEP_Y, DIR_Y, tofSensorY, ENDSTOP_Y, false);
  } else if (command.startsWith("PIDX ")) {
    float distance = command.substring(5).toFloat();
    moveWithPID(distance, STEP_X, DIR_X, tofSensorX, ENDSTOP_X, true);
  } else if (command.startsWith("PIDY ")) {
    float distance = command.substring(5).toFloat();
    moveWithPID(distance, STEP_Y, DIR_Y, tofSensorY, ENDSTOP_Y, false);
  } else if (command == "TOFX") {
    // Solo leer el sensor X
    if (!sensorTofXOk) {
      Serial.println("E: TOF X not initialized");
      return;
    }
    int distance = readAverageTofDistance(tofSensorX);
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
    int distance = readAverageTofDistance(tofSensorY);
    if (distance < 0) {
      Serial.println("E: Invalid TOF Y reading");
    } else {
      Serial.print("OK: ");
      Serial.println(distance);
    }
  } else if (command == "INITTOF") {
    if (initTofSensors()) {
      Serial.println("OK: TOF sensors initialized");
    } else {
      Serial.println("E: TOF initialization failed");
    }
  }  else {
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
int readAverageTofDistance(VL53L0X& sensor, int numSamples = 6) {
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
void moveWithPID(float targetDistance, int stepPin, int dirPin, VL53L0X& sensor, int endstopPin, bool isXaxis) {
  Serial.print("PID movement: ");
  Serial.print(targetDistance);
  Serial.println(" mm");
  
  // Check if sensor is functioning
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
  
  // Enable motors
  digitalWrite(EN_PIN, LOW);
  delay(100);
  
  // Set distance limits based on axis
  const int MIN_TOF_DISTANCE = isXaxis ? MIN_DISTANCE_TOF_X : MIN_DISTANCE_TOF_Y;
  const int MAX_TOF_DISTANCE = isXaxis ? MAX_DISTANCE_TOF_X : MAX_DISTANCE_TOF_Y;
  
  // Initialize counters
  int iterations = 0;
  unsigned long totalSteps = 0;
  
  // Calculate target TOF distance
  // When moving toward sensor, TOF reading decreases
  int targetTofDistance = initialTofDistance - targetDistance;
  
  // Make sure target is within valid range
  targetTofDistance = constrain(targetTofDistance, MIN_TOF_DISTANCE, MAX_TOF_DISTANCE);
  
  Serial.print("Initial TOF: ");
  Serial.print(initialTofDistance);
  Serial.print(" mm, Target TOF: ");
  Serial.print(targetTofDistance);
  Serial.println(" mm");
  
  // Set direction based on target movement
  bool moveTowardsSensor = (targetDistance > 0);
  if (isXaxis) {
    digitalWrite(dirPin, moveTowardsSensor ? LOW : HIGH);
  } else {
    digitalWrite(dirPin, moveTowardsSensor ? HIGH : LOW);
  }
  
  // Initialize PID variables
  float error = 0;
  float prevError = 0;
  float integral = 0;
  int currentTofDistance = initialTofDistance;
  bool approachComplete = false;
  bool highPrecisionMode = false;
  
  // Main PID control loop
  while (iterations < MAX_ITERATIONS_PID && !approachComplete) {
    // Check if endstop triggered
    
    
    // Get current distance (average of readings)
    currentTofDistance = readAverageTofDistance(sensor);
    
    // Check for valid reading
    if (currentTofDistance < 0) {
      Serial.println("E: Invalid TOF reading during movement");
      break;
    }
    
    // Calculate error
    error = targetTofDistance - currentTofDistance;
    
    // Check if target reached
    if (moveTowardsSensor) {
      // When moving toward sensor, TOF reading decreases
      if (currentTofDistance <= targetTofDistance) {
        approachComplete = true;
      }
    } else {
      // When moving away from sensor, TOF reading increases
      if (currentTofDistance >= targetTofDistance) {
        approachComplete = true;
      }
    }
    
    // Exit if target reached
    if (approachComplete) {
      break;
    }
    
    // Check if out of range
    if (currentTofDistance < MIN_TOF_DISTANCE || currentTofDistance > MAX_TOF_DISTANCE) {
      Serial.println("E: TOF out of range");
      break;
    }
    
    // Calculate PID components
    integral += error;
    integral = constrain(integral, -1000, 1000); // Prevent integral windup
    float derivative = error - prevError;
    
    // Calculate output
    float output = kp * error + ki * integral + kd * derivative;
    
    // Determine direction
    bool pidDirection = (output > 0);
    if (isXaxis) {
      digitalWrite(dirPin, pidDirection ? LOW : HIGH);
    } else {
      digitalWrite(dirPin, pidDirection ? HIGH : LOW);
    }

    int stepsToMove = constrain(abs(output), 1, 50);
    
    // Check endstop before starting movement
    if (digitalRead(endstopPin) == LOW) {
        // If endstop is triggered and we're trying to move toward it, stop
        if ((isXaxis && !pidDirection) || (!isXaxis && pidDirection)) {
            Serial.println("E: PHYSICAL LIMIT");
            break;
        }
    }
    
    
    // Print debug info periodically
    if (iterations % 10 == 0) {
      Serial.print(iterations);
      Serial.print("\tDist:");
      Serial.print(currentTofDistance);
      Serial.print("\tErr:");
      Serial.print(error);
      Serial.print("\tI:");
      Serial.print(integral);
      Serial.print("\tD:");
      Serial.print(derivative);
      Serial.print("\tSteps:");
      Serial.println(stepsToMove);
    }
    
    // Execute movement
    for (int i = 0; i < stepsToMove; i++) {
        // If endstop is triggered during movement
        if (digitalRead(endstopPin) == LOW) {
            // Only stop if we're moving toward the endstop
            if ((isXaxis && !pidDirection) || (!isXaxis && pidDirection)) {
                Serial.println("E: ENDSTOP HIT");
                approachComplete = false; // End the PID loop
                break;
            }
        }
        
        makeStep(stepPin, 150);
        totalSteps++;
    }
    
    // Save current error for next iteration
    prevError = error;
    iterations++;
    
    // Small delay to allow readings to stabilize
    delay(20);
  }

  
  // Calculate actual movement
  float distanceMoved;
  if (iterations >= MAX_ITERATIONS_PID) {
    Serial.println("E: Max iterations reached");
    distanceMoved = initialTofDistance - currentTofDistance;
  } else if (approachComplete) {
    distanceMoved = targetDistance;
  } else {
    distanceMoved = initialTofDistance - currentTofDistance;
  }
  
  // Update position
  if (isXaxis) {
    current_pos_x += distanceMoved;
  } else {
    current_pos_y += distanceMoved;
  }
  
  // Stop motors
  stopMotors();
  
  // Print result
  Serial.print("PID complete - Iterations: ");
  Serial.print(iterations);
  Serial.print(", Final TOF: ");
  Serial.print(currentTofDistance);
  Serial.print(", Distance moved: ");
  Serial.println(distanceMoved);
  
  Serial.print("OK: ");
  Serial.println(distanceMoved);
}