// Just enough of Arduino to run the board program on a computer, with simulated motors,
// encoders, distance sensors, and emergency stop. For tests only.
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define HIGH 1
#define LOW 0
#define INPUT 0
#define OUTPUT 1
#define INPUT_PULLUP 2
#define RISING 3
#define A0 14

namespace host {
extern uint32_t now_us;
extern int level[32], pwm[32];
extern void (*isr[32])();
extern int echo_us[32];
extern const unsigned char *in;
extern size_t in_left;
}  // namespace host

inline uint32_t millis() { return host::now_us / 1000; }
inline void delayMicroseconds(unsigned) {}
inline void pinMode(int, int) {}
inline void digitalWrite(int pin, int v) { host::level[pin] = v; }
inline int digitalRead(int pin) { return host::level[pin]; }
inline void analogWrite(int pin, int v) { host::pwm[pin] = v; }
inline int digitalPinToInterrupt(int pin) { return pin; }
inline void attachInterrupt(int pin, void (*f)(), int) { host::isr[pin] = f; }
inline void noInterrupts() {}
inline void interrupts() {}
inline unsigned long pulseIn(int pin, int, unsigned long timeout) {
  int us = host::echo_us[pin];
  if (us <= 0) return 0;
  return (unsigned long)us > timeout ? 0 : (unsigned long)us;
}

struct HostSerial {
  void begin(long) {}
  int available() { return (int)host::in_left; }
  int read() {
    if (!host::in_left) return -1;
    host::in_left--;
    return *host::in++;
  }
  size_t write(const uint8_t *data, size_t n) { return fwrite(data, 1, n, stdout); }
};
extern HostSerial Serial;
