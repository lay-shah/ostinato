// Runs the board's code on a computer. Reads "<ms> B <byte>" or "<ms> T <estop>".
#include "motor_guard.h"
#include <cstdio>

int main() {
  ostinato::MotorGuard guard;
  unsigned long ms;
  char kind;
  int value;
  while (std::scanf("%lu %c %d", &ms, &kind, &value) == 3) {
    if (kind == 'B') {
      guard.byte((char)value, (uint32_t)ms);
    } else {
      guard.tick((uint32_t)ms, value != 0);
      std::printf("%ld %ld %d\n", guard.left(), guard.right(), guard.armed());
      std::fflush(stdout);
    }
  }
}
