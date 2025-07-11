
#include <VL53L0X.h>
#include <Wire.h>
#include <FastLED.h>

////////////////////////////////////////////////////////////////
#define XSHUT_SENSOR_X A9
#define XSHUT_SENSOR_Y A8

#define NUM_LEDS 16
#define LED_PIN 40
#define BRIGHTNESS 210

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
const int HOMING_SPEED_FAST = 30;  // 30
const int HOMING_SPEED_SLOW = 150; // 150
const int HOMING_BACK_DISTANCE = 10 * STEPS_PER_MM;
const unsigned long TIMEOUT_STEPS_X = 300 * STEPS_PER_MM;
const unsigned long TIMEOUT_STEPS_Y = 300 * STEPS_PER_MM;
////////////////////////////////////////////////////////////////
int MIN_DISTANCE_TOF_X = 130;
int MIN_DISTANCE_TOF_Y = 20;
int MAX_DISTANCE_TOF_X = 339;
int MAX_DISTANCE_TOF_Y = 166;
////////////////////////////////////////////////////////////////
const int MOVE_SPEED = 60; // 100
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
const float ORIGIN_OFFSET_X = 48;  // Ajusta según la posición real del origen
const float ORIGIN_OFFSET_Y = 105; // Ajusta según la posición real del origen
////////////////////////////////////////////////////////////////
float kp = 0.2;
float ki = 0.04;
float kd = 0.2;

const int MAX_ITERATIONS_PID = 10;

String inputBuffer = "";
bool commandComplete = false;

bool sensorTofXOk = false;
bool sensorTofYOk = false;

float current_pos_x = 0.0;
float current_pos_y = 0.0;

void processCommand(String command);
bool homing(int stepPin, int dirPin, int endstopPin, bool homingDirection);
void makeStep(int stepPin, int delayTime);
float moveMM(int stepPin, int dirPin, float distance, bool isXaxis, int print);
void stopMotors();
float calculateMovedDistance(unsigned long steps, bool moveDirection);
bool performFullHoming();
bool initTofSensors();
bool initializeTofSensorX(unsigned long timingBudget);
bool initializeTofSensorY(unsigned long timingBudget);
bool isValidTofReading(VL53L0X &sensor, int distance);
int readAverageTofDistance(VL53L0X &sensor, int numSamples);
void moveWithPID(float targetDistance, int stepPin, int dirPin, VL53L0X &sensor, int endstopPin, bool isXaxis, bool showPrints);

void setup()
{
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

void loop()
{
  while (Serial.available() > 0)
  {
    char inChar = (char)Serial.read();

    if (inChar == '\n')
    {
      commandComplete = true;
    }
    else
    {
      inputBuffer += inChar;
    }
  }

  if (commandComplete)
  {
    processCommand(inputBuffer);
    Serial.println("COMMAND EXECUTED");
    inputBuffer = "";
    commandComplete = false;
  }
}

void initLEDs()
{
  if (!ledsInitialized)
  {
    FastLED.addLeds<WS2811, LED_PIN, RGB>(leds, NUM_LEDS);
    FastLED.setBrightness(BRIGHTNESS);
    FastLED.setCorrection(0xFFFFFF);
    ledsInitialized = true;
  }
}

void turnOnLEDs()
{
  initLEDs();
  fill_solid(leds, NUM_LEDS, CRGB::White);
  FastLED.show();
}

void turnOffLEDs()
{
  if (ledsInitialized)
  {
    fill_solid(leds, NUM_LEDS, CRGB::Black);
    FastLED.show();
  }
}

void processCommand(String command)
{
  command.trim();
  command.toUpperCase();

  if (command == "HX")
  {
    bool succesfull = homing(STEP_X, DIR_X, ENDSTOP_X, true);
    if (succesfull)
    {
      Serial.println("OK");
    }
    else
    {
      Serial.println("E: HOMING FAILED");
    }
  }
  else if (command == "HY")
  {
    bool succesfull = homing(STEP_Y, DIR_Y, ENDSTOP_Y, false);
    if (succesfull)
    {
      Serial.println("OK");
    }
    else
    {
      Serial.println("E: HOMING FAILED");
    }
  }
  else if (command.startsWith("MX "))
  {
    float distance = command.substring(3).toFloat();
    float a = moveMM(STEP_X, DIR_X, distance, true, 1);
  }
  else if (command.startsWith("MY "))
  {
    float distance = command.substring(3).toFloat();
    float a = moveMM(STEP_Y, DIR_Y, distance, false, 1);
  }
  else if (command.startsWith("S "))
  {
    stopMotors();
  }
  else if (command == "POSX")
  {
    Serial.print(" X:");
    Serial.println(current_pos_x);
  }
  else if (command == "POSY")
  {
    Serial.print(" Y:");
    Serial.println(current_pos_y);
  }
  else if (command == "FH")
  {
    if (!performFullHoming())
    {
      Serial.println("E: HOMING FAILED");
      return;
    }
    Serial.println("OK");
  }
  else if (command == "SETORIGIN")
  {
    current_pos_x = 0.0;
    current_pos_y = 0.0;
    Serial.println("OK: ORIGIN SET");
  }
  else if (command == "HOME&SET&ORIGIN")
  {
    if (!performFullHoming())
    {
      Serial.println("E: HOMING FAILED");
      return;
    }
    current_pos_x = 0.0;
    current_pos_y = 0.0;

    float a = moveMM(STEP_X, DIR_X, ORIGIN_OFFSET_X, true, 0);
    float b = moveMM(STEP_Y, DIR_Y, ORIGIN_OFFSET_Y, false, 0);

    MAX_POS_X = 245 - ORIGIN_OFFSET_X;
    MAX_POS_Y = 174 - ORIGIN_OFFSET_Y;

    current_pos_x = 0.0;
    current_pos_y = 0.0;

    Serial.println("OK");
  }
  else if (command == "XLIMIT")
  {
    Serial.print("OK: ");
    Serial.println(MAX_POS_X);
  }
  else if (command == "YLIMIT")
  {
    Serial.print("OK: ");
    Serial.println(MAX_POS_Y);
  }
  else if (command.startsWith("PIDX "))
  {
    int showPrint = command.substring(5).toInt();
    float distance = command.substring(7).toFloat();
    bool show = (showPrint != 0); // Fix: declare show at this scope level and assign properly
    moveWithPID(distance, STEP_X, DIR_X, tofSensorX, ENDSTOP_X, true, show);
  }
  else if (command.startsWith("PIDY "))
  {
    int showPrint = command.substring(5).toInt();
    float distance = command.substring(7).toFloat();
    bool show = (showPrint != 0); // Fix: declare show at this scope level and assign properly
    moveWithPID(distance, STEP_Y, DIR_Y, tofSensorY, ENDSTOP_Y, false, show);
  }
  else if (command == "TOFX")
  {
    // Solo leer el sensor X
    if (!sensorTofXOk)
    {
      Serial.println("E: TOF X not initialized");
      return;
    }
    int distance = readAverageTofDistance(tofSensorX, 6);

    Serial.print("OK: ");
    Serial.println(distance);
  }
  else if (command == "TOFY")
  {
    // Solo leer el sensor Y
    if (!sensorTofYOk)
    {
      Serial.println("E: TOF Y not initialized");
      return;
    }
    int distance = readAverageTofDistance(tofSensorY, 6);

    Serial.print("OK: ");
    Serial.println(distance);
  }
  else if (command.startsWith("SET KP "))
  {
    kp = command.substring(7).toFloat();
    Serial.print("OK: KP set to ");
    Serial.println(kp);
  }
  else if (command.startsWith("SET KI "))
  {
    ki = command.substring(7).toFloat();
    Serial.print("OK: KI set to ");
    Serial.println(ki);
  }
  else if (command.startsWith("SET KD "))
  {
    kd = command.substring(7).toFloat();
    Serial.print("OK: KD set to ");
    Serial.println(kd);
  }
  else if (command.startsWith("SET MAX_X "))
  {
    MAX_POS_X = command.substring(9).toInt();
    Serial.print("OK: MAX_X set to ");
    Serial.println(MAX_POS_X);
  }
  else if (command.startsWith("SET MAX_Y "))
  {
    MAX_POS_Y = command.substring(9).toInt();
    Serial.print("OK: MAX_Y set to ");
    Serial.println(MAX_POS_Y);
  }
  else if (command.startsWith("SET MIN_X "))
  {
    MIN_POS_X = command.substring(9).toInt();
    Serial.print("OK: MIN_X set to ");
    Serial.println(MIN_POS_X);
  }
  else if (command.startsWith("SET MIN_Y "))
  {
    MIN_POS_Y = command.substring(9).toInt();
    Serial.print("OK: MIN_Y set to ");
    Serial.println(MIN_POS_Y);
  }
  else if (command.startsWith("SET MIN_TOF_X "))
  {
    MIN_DISTANCE_TOF_X = command.substring(13).toInt();
    Serial.print("OK: MIN_TOF_X set to ");
    Serial.println(MIN_DISTANCE_TOF_X);
  }
  else if (command.startsWith("SET MIN_TOF_Y "))
  {
    MIN_DISTANCE_TOF_Y = command.substring(13).toInt();
    Serial.print("OK: MIN_TOF_Y set to ");
    Serial.println(MIN_DISTANCE_TOF_Y);
  }
  else if (command.startsWith("SET MAX_TOF_X "))
  {
    MAX_DISTANCE_TOF_X = command.substring(13).toInt();
    Serial.print("OK: MAX_TOF_X set to ");
    Serial.println(MAX_DISTANCE_TOF_X);
  }
  else if (command.startsWith("SET MAX_TOF_Y "))
  {
    MAX_DISTANCE_TOF_Y = command.substring(13).toInt();
    Serial.print("OK: MAX_TOF_Y set to ");
    Serial.println(MAX_DISTANCE_TOF_Y);
  }
  // Get current parameters
  else if (command == "GET PID")
  {
    Serial.print("KP:");
    Serial.print(kp);
    Serial.print(" KI:");
    Serial.print(ki);
    Serial.print(" KD:");
    Serial.println(kd);
  }
  else if (command == "GET LIMITS")
  {
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
  else if (command == "GET TOF_LIMITS")
  {
    Serial.print("TOF_X:[");
    Serial.print(MIN_DISTANCE_TOF_X);
    Serial.print(",");
    Serial.print(MAX_DISTANCE_TOF_X);
    Serial.print("] TOF_Y:[");
    Serial.print(MIN_DISTANCE_TOF_Y);
    Serial.print(",");
    Serial.print(MAX_DISTANCE_TOF_Y);
    Serial.println("]");
  }
  else if (command == "LED ON" || command == "LED")
  {
    turnOnLEDs();
    Serial.println("OK: LED ON");
  }
  else if (command == "LED OFF")
  {
    turnOffLEDs();
    Serial.println("OK: LED OFF");
  }
  else
  {
    Serial.println("E: CNF");
  }
}

bool homing(int stepPin, int dirPin, int endstopPin, bool homingDirection)
{
  digitalWrite(EN_PIN, LOW);
  delay(100);

  // F1
  digitalWrite(dirPin, homingDirection ? HIGH : LOW);

  int steps = 0;

  unsigned long TIMEOUT_STEPS;

  if (homingDirection)
  {
    TIMEOUT_STEPS = TIMEOUT_STEPS_X;
  }
  else
  {
    TIMEOUT_STEPS = TIMEOUT_STEPS_Y;
  }

  while (digitalRead(endstopPin) == HIGH)
  {
    makeStep(stepPin, HOMING_SPEED_FAST);
    steps++;

    if (steps > TIMEOUT_STEPS)
    {
      Serial.println("E: endstop");
      stopMotors();
      return false;
    }
  }

  delay(500);

  // F2
  digitalWrite(dirPin, homingDirection ? LOW : HIGH);

  for (int i = 0; i < HOMING_BACK_DISTANCE; i++)
  {
    makeStep(stepPin, HOMING_SPEED_FAST);
  }

  delay(500);

  // F3
  digitalWrite(dirPin, homingDirection ? HIGH : LOW);

  steps = 0;
  while (digitalRead(endstopPin) == HIGH)
  {
    makeStep(stepPin, HOMING_SPEED_SLOW);
    steps++;

    if (steps > TIMEOUT_STEPS)
    {
      Serial.println("E: endstop");
      stopMotors();
      return false;
    }
  }

  digitalWrite(EN_PIN, HIGH);
  return true;
}

bool performFullHoming()
{
  bool succesfull_x = homing(STEP_X, DIR_X, ENDSTOP_X, true);
  delay(100);
  bool succesfull_y = homing(STEP_Y, DIR_Y, ENDSTOP_Y, false);
  delay(100);
  return succesfull_x && succesfull_y;
}

void makeStep(int stepPin, int delayTime)
{
  digitalWrite(stepPin, HIGH);
  delayMicroseconds(delayTime);
  digitalWrite(stepPin, LOW);
  delayMicroseconds(delayTime);
}

float moveMM(int stepPin, int dirPin, float distance, bool isXaxis, int showPrints)
{

  float finalPosition = isXaxis ? current_pos_x + distance : current_pos_y + distance;
  float adjustedDistance = distance;

  if (isXaxis)
  {
    if (finalPosition < MIN_POS_X)
    {
      adjustedDistance = MIN_POS_X - current_pos_x;
    }
    else if (finalPosition > MAX_POS_X)
    {
      adjustedDistance = MAX_POS_X - current_pos_x;
    }
  }
  else
  {
    if (finalPosition < MIN_POS_Y)
    {
      adjustedDistance = MIN_POS_Y - current_pos_y;
    }
    else if (finalPosition > MAX_POS_Y)
    {
      adjustedDistance = MAX_POS_Y - current_pos_y;
    }
  }

  if (adjustedDistance == 0)
  {
    if (showPrints != 0)
    {
      Serial.println("OK: 0");
    }
    return 0.0;
  }

  digitalWrite(EN_PIN, LOW);
  delay(100);

  unsigned long steps = abs(adjustedDistance) * STEPS_PER_MM;

  bool moveDirection = (distance > 0);
  if (isXaxis)
  {
    digitalWrite(dirPin, moveDirection ? LOW : HIGH);
  }
  else
  {
    digitalWrite(dirPin, moveDirection ? HIGH : LOW);
  }

  int endstopPin = isXaxis ? ENDSTOP_X : ENDSTOP_Y;

  bool stopRequested = false;
  unsigned long completedSteps = 0;

  if (digitalRead(endstopPin) == LOW)
  {
    if (!moveDirection)
    {
      if (showPrints != 0)
      {
        Serial.println("OK: 0");
      }
      stopMotors();
      return 0.0;
    }
  }

  for (unsigned long i = 0; i < steps && !stopRequested; i++)
  {
    if (digitalRead(endstopPin) == LOW)
    {
      if (!moveDirection)
      {
        stopRequested = true;
        break;
      }
    }

    makeStep(stepPin, MOVE_SPEED);
    completedSteps++;
  }

  float actualMovedDistance = (float)completedSteps / STEPS_PER_MM;
  if (!moveDirection)
  {
    actualMovedDistance = -actualMovedDistance;
  }

  if (isXaxis)
  {
    current_pos_x += actualMovedDistance;
  }
  else
  {
    current_pos_y += actualMovedDistance;
  }
  if (showPrints != 0)
  {
    Serial.print("OK: ");
    Serial.println(actualMovedDistance);
    return actualMovedDistance;
  }
  else {
    return actualMovedDistance;
  }
  stopMotors();
}

void stopMotors()
{
  digitalWrite(EN_PIN, HIGH);

  digitalWrite(STEP_X, LOW);
  digitalWrite(STEP_Y, LOW);

  delay(100);
}

float calculateMovedDistance(unsigned long steps, bool moveDirection)
{
  float distance = (float)steps / STEPS_PER_MM;
  return moveDirection ? distance : -distance;
}

bool initializeTofSensorX(unsigned long timingBudget = FAST_TOF_SPEED)
{
  pinMode(XSHUT_SENSOR_X, OUTPUT);
  digitalWrite(XSHUT_SENSOR_X, LOW);

  delay(100);

  bool sensorXOk = false;

  for (int attempt = 1; attempt <= 3; attempt++)
  {
    pinMode(XSHUT_SENSOR_X, INPUT);
    delay(100);

    if (tofSensorX.init())
    {
      sensorXOk = true;
      tofSensorX.setTimeout(500);
      tofSensorX.setAddress(TOF_SENSOR_X_ADDRESS);
      tofSensorX.startContinuous();
      break;
    }
    else
    {
      digitalWrite(XSHUT_SENSOR_X, LOW);
      delay(200);
    }
  }

  if (!sensorXOk)
  {
    Serial.println("E: TOF X");
    return false;
  }

  return true;
}

// Initialize TOF sensor Y with configurable timing budget
bool initializeTofSensorY(unsigned long timingBudget = FAST_TOF_SPEED)
{
  pinMode(XSHUT_SENSOR_Y, OUTPUT);
  digitalWrite(XSHUT_SENSOR_Y, LOW);
  delay(100);

  bool sensorYOk = false;

  for (int attempt = 1; attempt <= 3; attempt++)
  {
    pinMode(XSHUT_SENSOR_Y, INPUT);
    delay(100);

    if (tofSensorY.init())
    {
      sensorYOk = true;
      tofSensorY.setTimeout(500);
      tofSensorY.setAddress(TOF_SENSOR_Y_ADDRESS);
      tofSensorY.startContinuous();
      break;
    }
    else
    {
      digitalWrite(XSHUT_SENSOR_Y, LOW);
      delay(200);
    }
  }

  if (!sensorYOk)
  {
    Serial.println("E: TOF Y");
    return false;
  }

  return true;
}

// Initialize both TOF sensors
bool initTofSensors()
{
  sensorTofXOk = initializeTofSensorX();
  sensorTofYOk = initializeTofSensorY();
  return sensorTofXOk && sensorTofYOk;
}

// Check if TOF reading is valid
bool isValidTofReading(VL53L0X &sensor, int distance)
{
  return !sensor.timeoutOccurred() && distance > 0 && distance < 8190;
}

// Read average TOF distance from multiple samples
int readAverageTofDistance(VL53L0X &sensor, int numSamples = 20)
{
  long sum = 0;
  int validReadings = 0;

  for (int i = 0; i < numSamples; i++)
  {
    int reading = sensor.readRangeContinuousMillimeters();
    if (isValidTofReading(sensor, reading))
    {
      sum += reading;
      validReadings++;
    }
    delay(10);
  }

  if (validReadings == 0)
  {
    return -1; // No valid readings
  }

  return sum / validReadings;
}

// Simple PID-controlled movement using TOF sensors
void moveWithPID(float targetDistance, int stepPin, int dirPin, VL53L0X &sensor, int endstopPin, bool isXaxis, bool showPrints)
{
  // if targetDistance is less than 10 mm, complete movement with moveMM
  if (isXaxis && current_pos_x + targetDistance < MIN_POS_X)
  {
    targetDistance = MIN_POS_X - current_pos_x;
  }
  else if (isXaxis && current_pos_x + targetDistance > MAX_POS_X)
  {
    targetDistance = MAX_POS_X - current_pos_x;
  }
  else if (!isXaxis && current_pos_y + targetDistance < MIN_POS_Y)
  {
    targetDistance = MIN_POS_Y - current_pos_y;
  }
  else if (!isXaxis && current_pos_y + targetDistance > MAX_POS_Y)
  {
    targetDistance = MAX_POS_Y - current_pos_y;
  }

  if (targetDistance == 0)
  {
    Serial.println("OK: 0");
    return;
  }
  if (abs(targetDistance) < 10)
  {
    if (showPrints)
    {
      Serial.println("Small distance detected, using direct movement");
    }
    float movedDistance = moveMM(stepPin, dirPin, targetDistance, isXaxis, 0);
    Serial.print("OK: ");
    Serial.println(movedDistance);
    return;
  }
  // if i want to move towards the endstop, check if the endstop is triggered
  if (targetDistance > 0 && digitalRead(endstopPin) == LOW)
  {
    Serial.println("OK: 0");
    return;
  }

  if (showPrints)
  {
    Serial.print("PID movement: ");
    Serial.print(targetDistance);
    Serial.println(" mm");
  }
  // check if the sensor is initialized
  bool &sensorOk = isXaxis ? sensorTofXOk : sensorTofYOk;
  if (!sensorOk)
  {
    if (showPrints)
    {
      Serial.println("E: TOF sensor not initialized");
    }
    moveMM(stepPin, dirPin, targetDistance, isXaxis, 1);
    return;
  }

  // Get initial TOF distance (average of readings) and check if it's valid
  int initialTofDistance = readAverageTofDistance(sensor);
  if (initialTofDistance < 0 || initialTofDistance > 400)
  {
    if (showPrints)
    {
      Serial.println("E: Invalid TOF reading");
    }
    float movedDistance = moveMM(stepPin, dirPin, targetDistance, isXaxis, 0);
    Serial.print("OK: ");
    Serial.println(movedDistance);
    return;
  }

  int objectiveTofReading = initialTofDistance - int(targetDistance);
  bool moveDirection = (targetDistance > 0);
  bool approachCompleted = false;
  int iteration = 0;
  int currentTofReading = initialTofDistance;

  float error = 0.0;
  float previousError = 0.0;
  float integral = 0.0;
  float derivative = 0.0;
  float output = 0.0;

  if (isXaxis)
  {
    digitalWrite(dirPin, moveDirection ? LOW : HIGH);
  }
  else
  {
    digitalWrite(dirPin, moveDirection ? HIGH : LOW);
  }

  const int MIN_STEP_DELAY = 15; // Máxima velocidad (menor delay)
  const int MAX_STEP_DELAY = 100;
  int stepDelay = MIN_STEP_DELAY; // Start with maximum speed
  float totalMovedDistance = 0.0;

  if (showPrints)
  {
    Serial.print("Initial TOF: ");
    Serial.print(initialTofDistance);
    Serial.print(" mm, Target TOF: ");
    Serial.print(objectiveTofReading);
    Serial.println(" mm");
  }

  if (showPrints)
  {
    Serial.println("====== PID MOVEMENT START ======");
    Serial.print("Target distance: ");
    Serial.print(targetDistance);
    Serial.print(" mm, Axis: ");
    Serial.println(isXaxis ? "X" : "Y");
    Serial.println("IT\tTOF\tERROR\tINTG\tDERIV\tOUTPUT\tDIR\tSTEPS\tIT DIST\tTOTAL STEPS\tTOTAL DIST\tDELAY");
  }

  digitalWrite(EN_PIN, LOW);
  delay(50);

  while (iteration < MAX_ITERATIONS_PID && !approachCompleted)
  {
    currentTofReading = readAverageTofDistance(sensor);
    if (currentTofReading < 0 || currentTofReading > 400)
    {
      if (showPrints)
      {
        Serial.println("E: Invalid TOF reading during PID loop");
      }
      approachCompleted = true;
      break;
    }
    error = currentTofReading - objectiveTofReading;
    if (abs(error) <= 10)
    {
      approachCompleted = true;
    }
    if (!approachCompleted)
    {
      // Calculate PID control values
      integral += error;
      derivative = error - previousError;
      output = kp * error + ki * integral + kd * derivative;
      integral = constrain(integral, -100, 100); // Prevent integral windup
      if (error * previousError < 0)
      {
        integral = 0; // Reset integral if error changes sign
      }
      output = constrain(output, -100, 100);
      if (output > 0)
      {
        if (isXaxis && current_pos_x + output > MAX_POS_X)
        {
          output = MAX_POS_X - current_pos_x;
        }
        else if (!isXaxis && current_pos_y + output > MAX_POS_Y)
        {
          output = MAX_POS_Y - current_pos_y;
        }
        moveDirection = true;
      }
      else
      {
        if (isXaxis && current_pos_x + output < MIN_POS_X)
        {
          output = MIN_POS_X - current_pos_x;
        }
        else if (!isXaxis && current_pos_y + output < MIN_POS_Y)
        {
          output = MIN_POS_Y - current_pos_y;
        }
        moveDirection = false;
      }
      // map stepDelay according to output
      unsigned long stepsToMake = abs(output) * STEPS_PER_MM;
      if (isXaxis)
      {
        digitalWrite(dirPin, moveDirection ? LOW : HIGH);
      }
      else
      {
        digitalWrite(dirPin, moveDirection ? HIGH : LOW);
      }

      float absOutput = abs(output);
      if (absOutput <= 5)
      {
        stepDelay = MAX_STEP_DELAY; // Movimiento lento para pequeños ajustes
      }
      else if (absOutput >= 100)
      {
        stepDelay = MIN_STEP_DELAY; // Movimiento rápido para grandes correcciones
      }
      else
      {
        // Mapeo lineal entre 10-100 de output a 300-15 de delay (nota la inversión)
        stepDelay = map(constrain(absOutput, 5, 20), 5, 20, MAX_STEP_DELAY, MIN_STEP_DELAY);
      }
      unsigned long stepsMadeThisIteration = 0;
      float movedDistance = 0.0;
      for (int i = 0; i < stepsToMake; i++)
      {
        if (digitalRead(endstopPin) == LOW && !moveDirection)
        {
          approachCompleted = true;
          break;
        }
        makeStep(stepPin, stepDelay);
        stepsMadeThisIteration++;
      }
      if (moveDirection)
      {
        movedDistance += stepsMadeThisIteration / STEPS_PER_MM;
        totalMovedDistance += movedDistance;
      }
      else
      {
        movedDistance -= stepsMadeThisIteration / STEPS_PER_MM;
        totalMovedDistance -= movedDistance;
      }

      if (isXaxis)
      {
        current_pos_x += movedDistance;
      }
      else
      {
        current_pos_y += movedDistance;
      }
      if (showPrints)
      {
        Serial.print(iteration);
        Serial.print("\t");
        Serial.print(currentTofReading);
        Serial.print("\t");
        Serial.print(error);
        Serial.print("\t");
        Serial.print(integral);
        Serial.print("\t");
        Serial.print(derivative);
        Serial.print("\t");
        Serial.print(output);
        Serial.print("\t");
        Serial.print(moveDirection ? "+" : "-");
        Serial.print("\t");
        Serial.print(stepsMadeThisIteration);
        Serial.print("\t");
        Serial.print(movedDistance, 2);
        Serial.print("\t");
        Serial.print("\t");
        Serial.print(totalMovedDistance, 2);
        Serial.print("\t");
        Serial.print("\t");
        Serial.println(stepDelay);
      }
      previousError = error;
      iteration++;
    }
  }
  if (showPrints)
  {
    Serial.println("==== PID SUMMARY ====");
    Serial.print("Iterations: ");
    Serial.print(iteration);
    Serial.print(", Distance moved: ");
    Serial.print(totalMovedDistance);
    Serial.print(" mm, Final error: ");
    Serial.print(error);
    Serial.println(" mm");
  }
  if (abs(targetDistance) > totalMovedDistance)
  {
    delay(500);
    float toMove = targetDistance > 0 ? (targetDistance - totalMovedDistance) : -(abs(targetDistance) - totalMovedDistance);
    float movedDistance = moveMM(stepPin, dirPin, toMove, isXaxis, 0);
    totalMovedDistance += movedDistance;
    Serial.print("OK: ");
    Serial.println(totalMovedDistance);
    return;
  }
  else
  {
    Serial.print("OK: ");
    Serial.println(totalMovedDistance);
    stopMotors();
  }
}