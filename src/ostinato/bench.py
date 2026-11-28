"""First power-on checks for the motor board: `python -m ostinato.bench --port /dev/ttyACM0`.

Put the rover on a stand with its wheels off the ground, and keep the emergency stop
in hand. Each check says what it found. Do not drive the rover until all of them pass.
Needs `pip install pyserial`.
"""

import argparse
import time

from .motors import Board
from .rover import Guard

SPEED = 100  # mm/s, gentle
SPIN = 2.0  # seconds per wheel


class Bench:
    """Talks to the board through `port` and reports in plain words."""

    def __init__(self, port, track, ask=input, say=print, clock=time.monotonic, wait=time.sleep):
        self.board, self.guard = Board(port, track), Guard()
        self.track, self.ask, self.say, self.clock, self.wait = track, ask, say, clock, wait
        self.last = None
        self.failed = []

    def listen(self, seconds, speeds=(0, 0), send=True):
        """Send wheel speeds in mm/s every 50 ms; return the reports heard."""
        heard, end = [], self.clock() + seconds
        while self.clock() < end:
            if send:
                self.board.send_wheels(*speeds)
            heard += self.board.receive(self.clock(), self.guard)
            self.wait(0.05)
        if heard:
            self.last = heard[-1]
        return heard

    def check(self, ok, good, bad):
        self.say(("PASS  " if ok else "FAIL  ") + (good if ok else bad))
        if not ok:
            self.failed.append(bad)
        return ok

    def wheel(self, name, left, right):
        """Spin one wheel forward. Returns counts per second of (left, right)."""
        self.listen(0.3)  # a stop first, which the board needs before it moves
        before = self.last
        heard = self.listen(SPIN, (left, right))
        self.listen(0.3)
        if before is None or not heard:
            return None
        after = heard[-1]
        seconds = (after.ms - before.ms) / 1000 or 1
        return (after.left - before.left) / seconds, (after.right - before.right) / seconds

    def run(self):
        s = self.say
        s("Checking the board. Wheels must be off the ground.")
        heard = self.listen(1.0)
        if not self.check(
            len(heard) >= 10, f"{len(heard)} reports in 1 s", "no reports: check the cable, port, and power"
        ):
            return False
        r = heard[-1]
        self.check(
            not (r.front != r.front or r.rear != r.rear),
            f"distance sensors read {r.front:.2f} m and {r.rear:.2f} m",
            "a distance sensor does not answer: check its wiring",
        )
        self.check(
            r.battery == r.battery,
            f"battery reads {r.battery:.2f} V: check it against a meter, and set BATTERY_MV_PER_COUNT "
            "in the board program if they differ by more than 0.2 V",
            "the battery is not reported: check the divider into A1",
        )
        # The rover reverses as well as going forwards, so each sensor must face its own way.
        for side, other in (("front", "rear"), ("rear", "front")):
            self.ask(
                f"Hold a book about 30 cm from the {side} distance sensor, with nothing near the {other}, "
                "then press Enter here."
            )
            heard = self.listen(0.5)
            r = heard[-1] if heard else self.last
            near, far = getattr(r, side), getattr(r, other)
            self.check(
                near < 0.5 and not far < 0.5,
                f"the {side} sensor saw it, at {near:.2f} m",
                f"the {side} sensor read {near:.2f} m and the {other} {far:.2f} m: they may be swapped, "
                "or one may face the wrong way",
            )
        self.ask("Press the emergency stop, then press Enter here.")
        self.check(
            any(x.estop for x in self.listen(0.5)),
            "the board sees the emergency stop",
            "the board does not see the emergency stop: check the switch and its pin",
        )
        self.ask("Release the emergency stop, then press Enter here.")
        if not self.check(
            not self.listen(0.5)[-1:] or not self.last.estop,
            "emergency stop released",
            "the board still sees the emergency stop",
        ):
            return False
        for name, left, right, mine in (("left", SPEED, 0, 0), ("right", 0, SPEED, 1)):
            s(f"Spinning the {name} wheel forward at {SPEED} mm/s for {SPIN:g} s.")
            rates = self.wheel(name, left, right)
            if not self.check(rates is not None, "reports kept coming", "reports stopped while spinning"):
                return False
            own, other = rates[mine], rates[1 - mine]
            self.check(
                own > 0,
                f"the {name} wheel was counted going forward",
                f"the {name} wheel counted backward or not at all: swap its motor wires or encoder channels, or check the pins",
            )
            self.check(
                abs(other) < 0.1 * max(abs(own), 1),
                "the other wheel stayed still",
                f"the other wheel moved when only the {name} one should: the motors or encoders are swapped",
            )
            answer = self.ask(f"Did the {name} wheel turn forward, the way the rover drives? (yes or no) ")
            self.check(
                answer.strip().lower().startswith("y"),
                f"the {name} wheel turned forward",
                f"the {name} wheel turned the wrong way: swap its motor wires",
            )
            if own > 0:
                s(f"  {own / SPEED * 1000:.0f} counts per meter, if the wheel kept to the speed asked for.")
        s("Spinning both wheels, then going quiet, as if the computer froze.")
        self.listen(0.3)
        self.listen(1.0, (SPEED, SPEED))
        start = self.last
        self.listen(1.2, send=False)
        later = self.listen(0.6, send=False)
        still = len(later) >= 2 and (later[-1].left, later[-1].right) == (later[0].left, later[0].right)
        self.check(
            still and start is not None,
            "the wheels stopped within 1.2 s of silence",
            "the wheels kept turning without commands: do not drive it",
        )
        self.check(
            not self.last.armed if self.last else False,
            "the board waits for a stop before moving again",
            "the board did not halt",
        )
        self.listen(0.3)
        s(
            "All checks passed."
            if not self.failed
            else f"{len(self.failed)} checks failed. Fix them before driving."
        )
        return not self.failed


class SerialPort:
    def __init__(self, name):
        try:
            import serial
        except ImportError:
            raise ValueError("this needs pyserial: pip install pyserial") from None
        self._serial = serial.Serial(name, 115200, timeout=0, write_timeout=0.05)

    def write(self, data):
        self._serial.write(data)

    def read(self):
        return self._serial.read(4096)

    def close(self):
        self._serial.close()


def measure_distance(bench, meters=1.0):
    """Counts per meter, from pushing the rover by hand with the motors stopped."""
    bench.ask(
        f"Put the rover on the floor. Mark where it is, then press Enter, and push it straight {meters:g} m."
    )
    start = bench.last or (bench.listen(0.5) or [None])[-1]
    bench.ask("Press Enter when it has gone exactly that far.")
    end = (bench.listen(0.5) or [None])[-1]
    if start is None or end is None:
        raise ValueError("no reports")
    left, right = (end.left - start.left) / meters, (end.right - start.right) / meters
    bench.say(f"Left {left:.0f}, right {right:.0f} counts per meter. Use the average as ticks_per_meter.")
    if min(left, right) <= 0 or abs(left - right) > 0.05 * max(left, right):
        bench.say("They differ by more than 5%, or one went backward: check the wheels and encoders.")
    return left, right


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", required=True)
    parser.add_argument("--track", type=float, default=0.27, help="wheel to wheel, in m (default: 0.27)")
    parser.add_argument("--measure", action="store_true", help="also measure counts per meter by pushing it")
    args = parser.parse_args(argv)
    port = None
    try:
        port = SerialPort(args.port)
        bench = Bench(port, args.track)
        passed = bench.run()
        if passed and args.measure:
            measure_distance(bench)
    except KeyboardInterrupt:
        parser.exit(1, "\nStopped.\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    finally:
        if port is not None:
            port.close()
    if not passed:
        parser.exit(1)


if __name__ == "__main__":
    main()
