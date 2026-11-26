"""The motor board program, run on this computer with simulated motors."""

import math
import shutil
import subprocess
from pathlib import Path

import pytest

from ostinato.motors import parse, seal

FIRMWARE = Path(__file__).parents[1] / "firmware"


@pytest.fixture(scope="module")
def board(tmp_path_factory):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++ compiler")
    program = tmp_path_factory.mktemp("sketch") / "board"
    include = [f"-I{FIRMWARE / d}" for d in ("host", ".", "ostinato_board")]
    subprocess.run(
        [
            compiler,
            "-std=c++14",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsanitize=undefined",
            *include,
            "-o",
            program,
            FIRMWARE / "host" / "run.cpp",
        ],
        check=True,
    )

    def run(events):
        text = "\n".join(
            f"{ms} B {what.hex()}" if isinstance(what, bytes) else f"{ms} {what}" for ms, what in events
        )
        done = subprocess.run([program], input=text + "\n", capture_output=True, text=True, check=True)
        reports = [parse(line.encode()) for line in done.stdout.split("\n") if line]
        assert None not in reports
        speeds = {
            int(a): (float(b), float(c)) for a, b, c in (line.split() for line in done.stderr.splitlines())
        }
        return reports, speeds

    return run


def commands(start, end, left, right, every=50, first=2):
    return [
        (ms, seal(f"D,{(first + i) % 256},{left},{right}")) for i, ms in enumerate(range(start, end, every))
    ]


READY = [(0, "F 900"), (0, "R 2000"), (5, "E 0"), (20, seal("D,1,0,0"))]


def test_holds_the_speed_asked_for(board):
    reports, speeds = board([*READY, *commands(70, 3000, 200, -120), (3000, "F 900")])
    for ms in range(1500, 3000, 50):
        assert speeds[ms][0] == pytest.approx(200, rel=0.05) and speeds[ms][1] == pytest.approx(
            -120, rel=0.05
        )
    last = reports[-1]
    assert last.armed and last.left > 0 > last.right and (last.front, last.rear) == (0.899, 1.999)
    # The counts match the distance: 0.5 mm each.
    assert last.left * 0.5 / 1000 == pytest.approx(0.2 * 2.6, rel=0.1)


def test_stops_soon_after_commands_stop(board):
    reports, speeds = board([*READY, *commands(70, 2000, 250, 250), (3500, "F 900")])
    assert speeds[2000][0] > 200
    assert all(abs(speeds[ms][0]) < 5 for ms in range(2850, 3500, 50))
    assert not reports[-1].armed


def test_emergency_stop(board):
    events = [*READY, *commands(70, 4000, 250, 250, first=2), (1500, "E 1"), (2500, "E 0"), (4000, "F 900")]
    reports, speeds = board(sorted(events, key=lambda e: e[0]))
    assert speeds[1450][0] > 200
    assert all(abs(speeds[ms][0]) < 5 for ms in range(1950, 4000, 50))
    states = {r.ms: (r.estop, r.armed) for r in reports}
    assert states[2000] == (True, False) and states[3000] == (False, False)


def test_unplugged_switch_counts_as_pressed(board):
    reports, speeds = board(
        [(0, "F 900"), (0, "R 900"), *commands(20, 2000, 200, 200, first=1), (2000, "F 900")]
    )
    assert max(abs(s[0]) for s in speeds.values()) == 0 and all(r.estop for r in reports)


def test_silent_sensor(board):
    reports, _ = board([*READY, (500, "F -1"), (1000, "R 900")])
    assert math.isnan(reports[-1].front) and reports[-1].rear == pytest.approx(2.0, abs=0.01)


def test_stalled_wheel_is_cut_off_and_never_lurches(board):
    events = [*READY, *commands(70, 6000, 200, 200), (500, "L 1"), (2500, "L 0"), (6000, "F 900")]
    _, speeds = board(sorted(events, key=lambda e: e[0]))
    # Freed after more than a second, it stays off while the same command repeats.
    assert all(speeds[ms][0] == 0 for ms in range(2550, 6000, 50))
    assert speeds[5950][1] == pytest.approx(200, rel=0.05)
    # Asked to stop and start again, it goes, without a lurch.
    events = [*READY, *commands(70, 1500, 200, 200), (400, "L 1"), (900, "L 0"), (1500, "F 900")]
    events += [(1500, seal("D,40,0,0")), *commands(1550, 4000, 200, 200, first=41), (4000, "F 900")]
    _, speeds = board(sorted(events, key=lambda e: e[0]))
    after = [speeds[ms][0] for ms in range(950, 4000, 50)]
    assert max(after) < 1.15 * 200 and after[-1] == pytest.approx(200, rel=0.05)


def test_never_faster_than_allowed(board):
    events = [*READY, *commands(70, 3000, 450, -450), (1000, "L 1"), (1300, "L 0"), (3000, "F 900")]
    _, speeds = board(sorted(events, key=lambda e: e[0]))
    assert max(abs(v) for pair in speeds.values() for v in pair) < 450 * 1.1
