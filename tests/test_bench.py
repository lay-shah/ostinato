import math

import pytest

from ostinato.bench import Bench, measure_distance
from ostinato.motors import seal


class FakeBoard:
    """A board with wiring faults to choose from, on a clock the bench controls."""

    def __init__(
        self, swapped=False, backward=False, no_watchdog=False, estop_wired=True, silent_sensor=False
    ):
        self.t, self.out = 0.0, b""
        self.fault = {
            "swapped": swapped,
            "backward": backward,
            "no_watchdog": no_watchdog,
            "estop_wired": estop_wired,
        }
        self.silent = silent_sensor
        self.estop, self.armed, self.speeds, self.heard = False, False, (0, 0), -1.0
        self.ticks, self.next_report = [0.0, 0.0], 0.0

    def clock(self):
        return self.t

    def wait(self, seconds):
        end = self.t + seconds
        while self.t < end - 1e-9:
            self.t += 0.01
            if self.t - self.heard > 0.5 and not self.fault["no_watchdog"]:
                self.armed, self.speeds = False, (0, 0)
            if self.estop and self.fault["estop_wired"]:
                self.armed, self.speeds = False, (0, 0)
            left, right = self.speeds
            if self.fault["swapped"]:
                left, right = right, left
            if self.fault["backward"]:
                left = -left
            self.ticks[0] += left * 2 * 0.01  # 2 counts per mm
            self.ticks[1] += right * 2 * 0.01
            if self.t >= self.next_report:
                self.next_report += 0.05
                state = 2 if self.estop and self.fault["estop_wired"] else int(self.armed)
                front = -1 if self.silent else 800
                ms = round(self.t * 1000)
                self.out += seal(f"T,{ms},{int(self.ticks[0])},{int(self.ticks[1])},{front},1500,{state}")

    def write(self, data):
        body = data.rstrip(b"\n")[:-5].decode()
        _, _, left, right = body.split(",")
        left, right = int(left), int(right)
        if self.estop and self.fault["estop_wired"]:
            return
        if self.armed or left == right == 0:
            self.armed, self.speeds, self.heard = True, (left, right), self.t

    def read(self):
        data, self.out = self.out, b""
        return data


def bench(board, answers=None, said=None):
    answers = answers or {}

    def ask(question):
        if "Press the emergency stop" in question:
            board.estop = True
        elif "Release" in question:
            board.estop = False
        for key, answer in answers.items():
            if key in question:
                return answer
        return "yes"

    return Bench(
        board,
        0.27,
        ask=ask,
        say=(said.append if said is not None else lambda s: None),
        clock=board.clock,
        wait=board.wait,
    )


def test_a_good_board_passes():
    said = []
    assert bench(FakeBoard(), said=said).run()
    assert said[-1] == "All checks passed."
    rough = [float(s.split()[0]) for s in said if "counts per meter" in s]
    assert rough and all(r == pytest.approx(2000, rel=0.1) for r in rough)


@pytest.mark.parametrize(
    "fault,words",
    [
        ({"swapped": True}, "swapped"),
        ({"backward": True}, "counted backward"),
        ({"no_watchdog": True}, "do not drive it"),
        ({"estop_wired": False}, "does not see the emergency stop"),
        ({"silent_sensor": True}, "does not answer"),
    ],
)
def test_finds_wiring_faults(fault, words):
    said = []
    assert not bench(FakeBoard(**fault), said=said).run()
    assert any(words in s for s in said if s.startswith("FAIL"))
    assert "Fix them before driving" in said[-1] or "FAIL" in said[-1]


def test_asks_about_direction():
    said = []
    assert not bench(FakeBoard(), {"left wheel turn forward": "no"}, said).run()
    assert any("left wheel turned the wrong way" in s for s in said)


def test_no_board():
    class Nothing:
        t = 0.0

        def clock(self):
            return self.t

        def wait(self, s):
            self.t += s

        def write(self, data):
            pass

        def read(self):
            return b""

    n, said = Nothing(), []
    assert not Bench(n, 0.27, ask=lambda q: "", say=said.append, clock=n.clock, wait=n.wait).run()
    assert "no reports" in said[-1]


def test_measures_counts_per_meter():
    board, said = FakeBoard(), []
    b = bench(board, said=said)

    def push(question):
        if "exactly" in question:
            board.ticks = [board.ticks[0] + 2000, board.ticks[1] + 1990]
            board.wait(0.1)
        return ""

    b.ask = push
    b.listen(0.5)
    left, right = measure_distance(b)
    assert (left, right) == (pytest.approx(2000, abs=2), pytest.approx(1990, abs=2))
    assert not any("differ" in s for s in said)
    assert math.isfinite(left)
