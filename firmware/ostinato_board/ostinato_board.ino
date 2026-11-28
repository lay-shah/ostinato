// Motor board program for the Ostinato rover: an Arduino-compatible board with
// 5 V or 3.3 V logic, two DC motors with quadrature encoders, a motor driver that takes
// PWM and direction, two HC-SR04 distance sensors, and the emergency stop switch.
//
// Nothing has been built. Pin numbers and the constants marked CHECK are placeholders:
// set them for the parts you buy, then run `python -m ostinato.bench` with the wheels
// off the ground before driving. This is on top of a switch that cuts motor power,
// never instead of it.
//
// It compiles and runs on a computer against firmware/host/Arduino.h, with a simulated
// motor, in the tests.

#include <Arduino.h>

#include "motor_guard.h"
#include "wheel.h"

// CHECK: pins
const int LEFT_PWM = 5, LEFT_DIR = 4, RIGHT_PWM = 6, RIGHT_DIR = 7;
const int LEFT_A = 2, LEFT_B = 8, RIGHT_A = 3, RIGHT_B = 9;  // A pins must take interrupts
const int FRONT_TRIG = 10, FRONT_ECHO = 11, REAR_TRIG = 12, REAR_ECHO = 13;
const int ESTOP = A0;  // normally closed switch to ground: pressed or unplugged reads HIGH
const int BATTERY = A1;  // through a divider; see docs/WIRING.md

// CHECK: from the encoder's counts per turn and the wheel's size
const float MM_PER_TICK = 0.5f;
const uint32_t STEP_MS = 50;  // control and report period
// CHECK: battery mV for each count of analogRead. A 10k over 3.3k divider on a 5 V board:
// 5000 / 1023 * (10 + 3.3) / 3.3 = 19.7, for up to 20 V. Measure with a meter and adjust.
const float BATTERY_MV_PER_COUNT = 19.7f;
// CHECK: below this for CUTOFF_MS the board stops the motors until it is restarted, even
// if the computer is gone. 3.2 V a cell for a three-cell lithium battery. Readings under
// 1 V mean the divider is not wired, and are ignored.
const int BATTERY_CUTOFF_MV = 9600;
const uint32_t CUTOFF_MS = 1000;

ostinato::MotorGuard guard;
ostinato::Wheel left(MM_PER_TICK), right(MM_PER_TICK);
volatile long leftTicks = 0, rightTicks = 0;
uint32_t lastStep = 0;
int front = -1, rear = -1;  // mm; -1 when the sensor did not answer
bool readFront = true;
bool flat = false;      // latched: the battery stayed below the cutoff
uint32_t lowSince = 0;  // when it went below, or 0

void leftEdge() { leftTicks += digitalRead(LEFT_B) ? -1 : 1; }
void rightEdge() { rightTicks += digitalRead(RIGHT_B) ? 1 : -1; }  // mirrored on the other side

// One HC-SR04 reading in mm. -1 if it did not answer; 4000 if nothing is in range.
int distance(int trig, int echo) {
  digitalWrite(trig, LOW);
  delayMicroseconds(2);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);
  unsigned long us = pulseIn(echo, HIGH, 30000UL);
  if (us == 0) return -1;
  if (us >= 23000UL) return 4000;
  return (int)(us * 343UL / 2000UL);
}

void drive(int pwm, int pin, int dir) {
  digitalWrite(dir, pwm < 0 ? HIGH : LOW);
  analogWrite(pin, pwm < 0 ? -pwm : pwm);
}

void setup() {
  // Motors off before anything else.
  pinMode(LEFT_PWM, OUTPUT);
  pinMode(RIGHT_PWM, OUTPUT);
  analogWrite(LEFT_PWM, 0);
  analogWrite(RIGHT_PWM, 0);
  pinMode(LEFT_DIR, OUTPUT);
  pinMode(RIGHT_DIR, OUTPUT);
  pinMode(LEFT_A, INPUT_PULLUP);
  pinMode(LEFT_B, INPUT_PULLUP);
  pinMode(RIGHT_A, INPUT_PULLUP);
  pinMode(RIGHT_B, INPUT_PULLUP);
  pinMode(FRONT_TRIG, OUTPUT);
  pinMode(REAR_TRIG, OUTPUT);
  pinMode(FRONT_ECHO, INPUT);
  pinMode(REAR_ECHO, INPUT);
  pinMode(ESTOP, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(LEFT_A), leftEdge, RISING);
  attachInterrupt(digitalPinToInterrupt(RIGHT_A), rightEdge, RISING);
  Serial.begin(115200);
}

void loop() {
  uint32_t now = millis();
  while (Serial.available() > 0) guard.byte((char)Serial.read(), now);
  bool pressed = digitalRead(ESTOP) == HIGH;
  guard.tick(now, pressed || flat);
  if ((uint32_t)(now - lastStep) < STEP_MS) {
    if (!guard.armed()) {  // a stop takes effect at once, not at the next step
      left.stop();
      right.stop();
      drive(0, LEFT_PWM, LEFT_DIR);
      drive(0, RIGHT_PWM, RIGHT_DIR);
    }
    return;
  }
  uint32_t dt = now - lastStep;
  lastStep = now;
  noInterrupts();
  long l = leftTicks, r = rightTicks;
  interrupts();
  drive(left.update(guard.left(), l, dt), LEFT_PWM, LEFT_DIR);
  drive(right.update(guard.right(), r, dt), RIGHT_PWM, RIGHT_DIR);
  // One sensor per step, because each can take up to 30 ms.
  if (readFront) front = distance(FRONT_TRIG, FRONT_ECHO);
  else rear = distance(REAR_TRIG, REAR_ECHO);
  readFront = !readFront;
  char line[64];
  int state = pressed ? 2 : flat ? 3 : guard.armed() ? 1 : 0;
  int battery = (int)(analogRead(BATTERY) * BATTERY_MV_PER_COUNT + 0.5f);  // mV
  if (battery > 1000 && battery < BATTERY_CUTOFF_MV) {
    if (lowSince == 0) lowSince = now ? now : 1;
    if ((uint32_t)(now - lowSince) >= CUTOFF_MS) flat = true;
  } else {
    lowSince = 0;
  }
  int n = snprintf(line, sizeof line, "T,%lu,%ld,%ld,%d,%d,%d,%d", (unsigned long)now, l, r, front, rear, state,
                   battery);
  if (n > 0 && n < (int)sizeof line) {
    char end[8];
    snprintf(end, sizeof end, "*%04X\n", ostinato::crc16(line, (size_t)n));
    Serial.write((const uint8_t *)line, (size_t)n);
    Serial.write((const uint8_t *)end, 6);
  }
}
