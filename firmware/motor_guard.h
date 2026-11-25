// The last check before the wheels, meant to run on the motor board.
// Portable C++11 with no hardware code, so it is tested on a computer.
// It does not replace a switch that cuts motor power.
//
// Commands:  D,<seq 0-255>,<left mm/s>,<right mm/s>*<CRC-16 in hex>\n
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

namespace ostinato {

const uint32_t QUIET_MS = 500;   // stop after this long without a good command
const long MAX_WHEEL_MM_S = 450; // 0.3 m/s plus turning; anything faster is refused
const size_t MAX_LINE = 48;

// CRC-16/CCITT-FALSE
inline uint16_t crc16(const char *data, size_t n) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < n; i++) {
    crc ^= (uint16_t)((uint8_t)data[i] << 8);
    for (int bit = 0; bit < 8; bit++)
      crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
  }
  return crc;
}

class MotorGuard {
public:
  // Call for every byte from the computer.
  void byte(char c, uint32_t now) {
    if (c == '\n') {
      if (!overflow_) line(now);
      length_ = 0;
      overflow_ = false;
    } else if (length_ < MAX_LINE) {
      buffer_[length_++] = c;
    } else {
      overflow_ = true;
    }
  }

  // Call for every control step, then send left() and right() to the motors.
  void tick(uint32_t now, bool estop) {
    estop_ = estop;
    if (estop || quiet(now)) halt();
  }

  long left() const { return left_; }
  long right() const { return right_; }
  // Armed means a stop command has arrived since the last halt.
  bool armed() const { return armed_; }
  uint8_t seq() const { return seq_; }

private:
  bool quiet(uint32_t now) const { return !heard_ || (uint32_t)(now - last_) > QUIET_MS; }

  // After any halt, the wheels stay still until the computer sends a stop.
  // This keeps an old command from moving the rover when a link returns.
  void halt() {
    left_ = right_ = 0;
    armed_ = false;
  }

  static bool number(const char *&p, char end, long low, long high, long &out) {
    const char *start = p;
    if (*p == '-') p++;
    const char *digits = p;
    while (*p >= '0' && *p <= '9') p++;
    if (p == digits || p - digits > 6 || *p != end) return false;
    out = strtol(start, NULL, 10);
    p++;
    return out >= low && out <= high;
  }

  void line(uint32_t now) {
    if (length_ < 6 || buffer_[length_ - 5] != '*') return;
    uint16_t sent = 0;
    for (size_t i = length_ - 4; i < length_; i++) {
      char c = buffer_[i];
      int v = c >= '0' && c <= '9' ? c - '0' : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
      if (v < 0) return;
      sent = (uint16_t)(sent << 4 | v);
    }
    size_t body = length_ - 5;
    if (memchr(buffer_, 0, body) || crc16(buffer_, body) != sent) return;
    buffer_[body] = 0;
    if (buffer_[0] != 'D' || buffer_[1] != ',') return;
    const char *p = buffer_ + 2;
    long seq, left, right;
    if (!number(p, ',', 0, 255, seq) || !number(p, ',', -MAX_WHEEL_MM_S, MAX_WHEEL_MM_S, left) ||
        !number(p, 0, -MAX_WHEEL_MM_S, MAX_WHEEL_MM_S, right))
      return;
    if (quiet(now)) halt(); // it may come before the next control step
    bool stop = left == 0 && right == 0;
    uint8_t ahead = (uint8_t)((uint8_t)seq - seq_);
    // A repeated or older command is ignored. A halted board takes any stop, so a
    // restarted computer can begin again.
    if (armed_ ? (ahead == 0 || ahead > 127) : !stop) return;
    if (estop_) return;
    seq_ = (uint8_t)seq;
    last_ = now;
    heard_ = armed_ = true;
    left_ = left;
    right_ = right;
  }

  char buffer_[MAX_LINE + 1];
  size_t length_ = 0;
  bool overflow_ = false, heard_ = false, armed_ = false, estop_ = true;
  uint32_t last_ = 0;
  uint8_t seq_ = 0;
  long left_ = 0, right_ = 0;
};

} // namespace ostinato
