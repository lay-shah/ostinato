// Runs the board program in simulated time. Reads lines from stdin:
//   <ms> B <hex bytes>     bytes arriving from the computer
//   <ms> E <0 or 1>        emergency stop released or pressed
//   <ms> F <mm or -1>      front distance; R for rear; -1 means no answer
//   <ms> L <0 or 1>        stall the left wheel
// Board output goes to stdout; every 50 ms, "<ms> <left mm/s> <right mm/s>" to stderr.
#include <math.h>
#include <stdlib.h>

#include <vector>

#include "Arduino.h"

namespace host {
uint32_t now_us = 0;
int level[32], pwm[32];
void (*isr[32])();
int echo_us[32];
const unsigned char *in;
size_t in_left = 0;
}  // namespace host
HostSerial Serial;

#include "../ostinato_board/ostinato_board.ino"

struct Motor {
  int pwm_pin, dir_pin, a_pin, b_pin, sign;
  double speed = 0, travel = 0;  // mm/s, mm
  bool stalled = false;
  long ticks = 0;
  void step(double dt) {
    int p = host::pwm[pwm_pin] * (host::level[dir_pin] ? -1 : 1);
    double target = stalled ? 0 : p / 0.4 * 0.85;  // 15% weaker than the controller assumes
    speed += (target - speed) * (dt / 0.08);       // 80 ms to respond
    if (stalled) speed = 0;
    travel += speed * dt;
    long now = (long)floor(travel / MM_PER_TICK);
    while (ticks != now) {
      bool forward = now > ticks;
      ticks += forward ? 1 : -1;
      host::level[b_pin] = (forward ? 0 : 1) ^ (sign < 0);
      host::isr[a_pin]();  // one rising edge of A, counted by B's level
    }
  }
};

int main() {
  host::level[ESTOP] = HIGH;
  setup();
  Motor l{LEFT_PWM, LEFT_DIR, LEFT_A, LEFT_B, 1}, r{RIGHT_PWM, RIGHT_DIR, RIGHT_A, RIGHT_B, -1};
  std::vector<unsigned char> buffer;
  unsigned long ms;
  char kind;
  char text[4096];
  uint32_t until = 0;
  auto run_to = [&](uint32_t target) {
    while (host::now_us / 1000 < target) {
      host::now_us += 1000;
      l.step(0.001);
      r.step(0.001);
      loop();
      if (host::now_us / 1000 % 50 == 0) fprintf(stderr, "%u %.1f %.1f\n", host::now_us / 1000, l.speed, r.speed);
    }
  };
  while (scanf("%lu %c %4095s", &ms, &kind, text) == 3) {
    run_to((uint32_t)ms);
    if (kind == 'B') {
      buffer.clear();
      for (size_t i = 0; text[i] && text[i + 1]; i += 2) {
        char pair[3] = {text[i], text[i + 1], 0};
        buffer.push_back((unsigned char)strtol(pair, NULL, 16));
      }
      host::in = buffer.data();
      host::in_left = buffer.size();
      loop();  // read it now, as a real board would soon
    } else if (kind == 'E') {
      host::level[ESTOP] = atoi(text) ? HIGH : LOW;
    } else if (kind == 'F' || kind == 'R') {
      int mm = atoi(text);
      host::echo_us[kind == 'F' ? FRONT_ECHO : REAR_ECHO] = mm < 0 ? 0 : mm * 2000 / 343;
    } else if (kind == 'L') {
      l.stalled = atoi(text) != 0;
    }
    until = (uint32_t)ms;
  }
  (void)until;
  fflush(stdout);
}
