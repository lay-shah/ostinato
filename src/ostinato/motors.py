"""Messages between the rover's computer and its motor board.

No board exists yet. `firmware/motor_guard.h` is the matching code for the board.
Commands: `D,<seq>,<left mm/s>,<right mm/s>*<CRC>`.
Reports: `T,<ms>,<left ticks>,<right ticks>,<front mm>,<rear mm>,<state>,<battery mV>*<CRC>`.
The state is 0 waiting for a stop, 1 ready, or 2 emergency stop pressed. Reports from
before the battery was added, without it, are still read.
"""

import binascii
import math
from dataclasses import dataclass

MAX_WHEEL = 450  # mm/s, as on the board
MAX_LINE = 80
SILENT = -1  # a distance sensor that did not answer
MAX_BATTERY = 60000  # mV; more is a damaged report


def seal(body):
    return f"{body}*{binascii.crc_hqx(body.encode(), 0xFFFF):04X}\n".encode()


def wheels(drive, track):
    """Left and right wheel speeds in mm/s. Slows both when one would be too fast."""
    if not (math.isfinite(drive.speed) and math.isfinite(drive.turn)):
        raise ValueError("invalid drive")
    left = 1000 * (drive.speed - drive.turn * track / 2)
    right = 1000 * (drive.speed + drive.turn * track / 2)
    scale = min(1.0, MAX_WHEEL / max(abs(left), abs(right), 1e-9))
    return int(left * scale), int(right * scale)


@dataclass(frozen=True)
class Report:
    ms: int
    left: int  # encoder ticks since the board started
    right: int
    front: float  # m, NaN when the sensor did not answer
    rear: float
    armed: bool
    estop: bool = False
    battery: float = math.nan  # V, NaN when not reported


def parse(line):
    """A Report, or None for anything damaged."""
    if len(line) > MAX_LINE or line[-5:-4] != b"*":
        return None
    try:
        body = line[:-5].decode("ascii")
        if seal(body)[:-1] != line:
            return None
        kind, *rest = body.split(",")
        numbers = [int(v) for v in rest]
        ms, left, right, front, rear, state, *battery = numbers
    except (UnicodeDecodeError, ValueError):
        return None
    if (
        kind != "T"
        or len(numbers) not in (6, 7)
        or ms < 0
        or min(front, rear) < SILENT
        or state not in (0, 1, 2)
    ):
        return None
    if battery and not 0 <= battery[0] <= MAX_BATTERY:
        return None
    front, rear = (math.nan if v == SILENT else v / 1000 for v in (front, rear))
    volts = battery[0] / 1000 if battery and battery[0] > 0 else math.nan
    return Report(ms, left, right, front, rear, state == 1, state == 2, volts)


class Board:
    """The motor board on a serial port. `port` has write(bytes) and read() -> bytes."""

    def __init__(self, port, track):
        if not 0 < track < 1:
            raise ValueError("track is the distance between the wheels in meters")
        self.port, self.track = port, track
        self._seq, self._buffer = 0, b""
        self._report = None

    def send(self, drive):
        """Call every control step, after `Guard`, even when stopped."""
        self.send_wheels(*wheels(drive, self.track))

    def send_wheels(self, left, right):
        """Each wheel's speed in mm/s. Only for bench tests: driving goes through `send`."""
        self._seq = (self._seq + 1) % 256
        # A halted board waits for a stop before it moves again.
        if self._report is None or not self._report.armed:
            left = right = 0
        left, right = (max(-MAX_WHEEL, min(MAX_WHEEL, int(v))) for v in (left, right))
        self.port.write(seal(f"D,{self._seq},{left},{right}"))

    def receive(self, t, guard):
        """Read reports, and tell the guard about the sensors. Return the new reports."""
        self._buffer += self.port.read()
        *lines, self._buffer = self._buffer.split(b"\n")
        self._buffer = self._buffer[-MAX_LINE:]
        found = []
        for line in lines:
            report = parse(line)
            # A board that restarts counts from zero again, and that is news too.
            # The same time twice means it is stuck repeating itself.
            if report and (self._report is None or report.ms != self._report.ms):
                self._report = report
                found.append(report)
        if any(r.estop for r in found):
            # Releasing the button must not set it moving: a person clears the guard.
            guard.emergency_stop()
        if found:
            guard.sensors(t, found[-1].front, found[-1].rear)
        return found
