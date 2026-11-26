// Wheel speed control: feed-forward plus proportional-integral on encoder counts.
// A wheel that is pushed but does not turn for STALL_MS is cut off until it is asked
// to stop: it may be against something, or someone.
#pragma once
#include <stdint.h>

namespace ostinato {

class Wheel {
public:
  // CHECK: FEED is PWM per mm/s, about 255 divided by the top speed with no load.
  static constexpr float FEED = 0.4f, GAIN = 0.15f, INTEGRAL = 0.6f;
  static const int MAX_PWM = 255;
  static const uint32_t STALL_MS = 1000;

  explicit Wheel(float mmPerTick) : mmPerTick_(mmPerTick) {}

  void stop() {
    sum_ = 0;
    pwm_ = 0;
    stuck_ = 0;
  }

  // Target in mm/s, the encoder count now, and ms since the last call. Returns PWM.
  int update(long target, long ticks, uint32_t dt) {
    float speed = dt ? (ticks - last_) * mmPerTick_ * 1000.0f / dt : 0;
    last_ = ticks;
    measured_ = speed;
    if (target == 0 || dt == 0 || dt > 500) {
      stop();
      stalled_ = false;
      return 0;
    }
    if (stalled_) return 0;
    float error = target - speed;
    // Too slow by more than half while pushing: count toward a stall.
    bool slow = (target > 0 ? speed : -speed) < 0.5f * (target > 0 ? target : -target);
    stuck_ = slow ? stuck_ + dt : 0;
    if (stuck_ >= STALL_MS) {
      stop();
      stalled_ = true;
      return 0;
    }
    float before = FEED * target + GAIN * error + INTEGRAL * sum_;
    // Integrate only while the output has room and the wheel is nearly keeping up, so
    // that a held wheel cannot store up a lurch for when it is let go.
    if (!slow && before > -MAX_PWM && before < MAX_PWM) sum_ += error * dt / 1000.0f;
    float limit = 0.3f * MAX_PWM / INTEGRAL;
    sum_ = sum_ > limit ? limit : sum_ < -limit ? -limit : sum_;
    float out = FEED * target + GAIN * error + INTEGRAL * sum_;
    // Never push against the wanted direction: that is braking, left to stopping.
    if ((target > 0 && out < 0) || (target < 0 && out > 0)) out = 0;
    pwm_ = (int)(out > MAX_PWM ? MAX_PWM : out < -MAX_PWM ? -MAX_PWM : out);
    return pwm_;
  }

  float measured() const { return measured_; }
  bool stalled() const { return stalled_; }

private:
  float mmPerTick_, sum_ = 0, measured_ = 0;
  long last_ = 0;
  int pwm_ = 0;
  uint32_t stuck_ = 0;
  bool stalled_ = false;
};

}  // namespace ostinato
