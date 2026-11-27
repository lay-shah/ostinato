import math
import random
import subprocess

import pytest

from ostinato.motors import Board, Report, parse, seal, wheels
from ostinato.rover import STOP, Drive, Guard


@pytest.fixture(scope="module")
def run(board_program):
    program = board_program

    def run(events):
        """events: (ms, bytes) to receive or (ms, estop) for a control step."""
        text = []
        for ms, what in events:
            if isinstance(what, bytes):
                text += [f"{ms} B {b}" for b in what]
            else:
                text.append(f"{ms} T {int(what)}")
        done = subprocess.run([program], input="\n".join(text), capture_output=True, text=True, check=True)
        assert not done.stderr
        return [tuple(int(v) for v in line.split()) for line in done.stdout.splitlines()]

    return run


def d(seq, left, right):
    return seal(f"D,{seq},{left},{right}")


READY = [(0, False), (1, d(1, 0, 0))]


def test_moves_only_after_a_stop(run):
    assert run([(0, False), (1, d(1, 100, 100)), (2, False)]) == [(0, 0, 0)] * 2
    assert run([*READY, (2, d(2, 100, -50)), (3, False)])[-1] == (100, -50, 1)


def test_stops_when_quiet_and_stays_stopped(run):
    out = run([*READY, (2, d(2, 100, 100)), (502, False), (503, False), (504, d(3, 100, 100)), (505, False)])
    assert out[1:] == [(100, 100, 1), (0, 0, 0), (0, 0, 0)]
    again = run([*READY, (600, False), (601, d(9, 0, 0)), (602, d(10, 80, 80)), (603, False)])
    assert again[-1] == (80, 80, 1)


def test_clock_wraps(run):
    top = 2**32
    out = run([(top - 10, False), (top - 9, d(1, 0, 0)), (top - 8, d(2, 90, 90)), (100, False), (600, False)])
    assert out[1:] == [(90, 90, 1), (0, 0, 0)]


def test_estop(run):
    out = run(
        [
            *READY,
            (2, d(2, 99, 99)),
            (3, True),
            (4, d(3, 0, 0)),
            (5, d(4, 99, 99)),
            (6, True),
            (7, False),
            (8, d(5, 99, 99)),
            (9, False),
        ]
    )
    assert out[1:] == [(0, 0, 0)] * 4


def test_old_commands_do_not_keep_it_alive(run):
    move = d(2, 100, 100)
    out = run([*READY, (2, move), *((t, move) for t in range(50, 500, 50)), (503, False)])
    assert out[-1] == (0, 0, 0)
    assert run([*READY, (2, d(2, 100, 100)), (3, d(1, 50, 50)), (4, False)])[-1] == (100, 100, 1)
    assert run([(0, False), (1, d(255, 0, 0)), (2, d(0, 70, 70)), (3, False)])[-1] == (70, 70, 1)


@pytest.mark.parametrize(
    "body",
    [
        "D,2,451,0",
        "D,2,0,-451",
        "D,256,1,1",
        "D,-1,1,1",
        "D,2,1",
        "D,2,1,1,1",
        "D,2,,1",
        "D,2,1,1 ",
        "D,2,+1,1",
        "D,2,9999999999,1",
        "X,2,1,1",
        "D,2,-,1",
        "",
    ],
)
def test_refuses_bad_commands(run, body):
    assert run([*READY, (2, seal(body)), (3, False)])[-1] == (0, 0, 1)


def test_damage_never_moves_it(run):
    rng = random.Random(1)
    events = list(READY)
    for i in range(3000):
        line = bytearray(d(2, 100, 100))
        for _ in range(rng.randint(1, 3)):
            line[rng.randrange(len(line) - 1)] ^= 1 << rng.randrange(8)
        if bytes(line) == d(2, 100, 100):
            continue
        events.append((2, bytes(line) + b"\n"))
    assert run([*events, (3, False)])[-1] == (0, 0, 1)
    noise = [(2, bytes(rng.randrange(256) for _ in range(20000)) + b"\n")]
    assert run([*READY, *noise, (3, d(2, 60, 60)), (4, False)])[-1] == (60, 60, 1)


def test_wheels():
    assert wheels(Drive(0.2, 0), 0.3) == (200, 200)
    assert wheels(Drive(0, 1), 0.3) == (-150, 150)
    fast = wheels(Drive(0.3, 2), 0.4)
    assert max(map(abs, fast)) <= 450 and fast[0] < fast[1]
    with pytest.raises(ValueError):
        wheels(Drive(math.nan, 0), 0.3)


def test_parse():
    good = seal("T,1200,-5,7,800,-1,1")[:-1]
    r = parse(good)
    assert (r.ms, r.left, r.right, r.front, r.armed) == (1200, -5, 7, 0.8, True) and math.isnan(r.rear)
    for bad in (
        b"",
        good[:-1],
        good.replace(b"800", b"900"),
        seal("T,1,2,3,4,5,3")[:-1],
        seal("T,1,2,3,-2,5,1")[:-1],
        seal("T,1,2,3,4,5")[:-1],
        b"\xff" * 20,
        b"x" * 500,
        seal("T,1,2,3,4,5,1,-3")[:-1],
        seal("T,1,2,3,4,5,1,70000")[:-1],
        seal("T,1,2,3,4,5,1,12000,9")[:-1],
    ):
        assert parse(bad) is None
    assert math.isnan(r.battery)  # an older report, without it
    assert parse(seal("T,1200,-5,7,800,-1,1,11870")[:-1]).battery == pytest.approx(11.87)
    assert math.isnan(parse(seal("T,1200,-5,7,800,-1,1,0")[:-1]).battery)  # not wired


class Port:
    def __init__(self):
        self.sent, self.waiting = [], b""

    def write(self, data):
        self.sent.append(data)

    def read(self):
        data, self.waiting = self.waiting, b""
        return data


def test_board():
    port, guard = Port(), Guard()
    board = Board(port, 0.3)
    board.send(Drive(0.2, 0))
    assert port.sent[-1] == d(1, 0, 0)  # it does not know that the board is ready
    port.waiting = seal("T,10,0,0,900,900,1")[:8]
    assert board.receive(1.0, guard) == []
    port.waiting = seal("T,10,0,0,900,900,1")[8:] + b"junk\n"
    assert board.receive(1.1, guard) == [Report(10, 0, 0, 0.9, 0.9, True)]
    board.send(Drive(0.2, 0))
    assert port.sent[-1] == d(2, 200, 200)
    port.waiting = seal("T,10,0,0,900,900,1")  # stuck
    assert board.receive(5.0, guard) == []
    guard.link(5.0), guard.person(5.0, 3.0, False)
    assert guard.allow(5.0, Drive(0.1, 0)) == STOP and guard.reason == "no news from sensors"
    port.waiting = seal("T,20,0,0,900,900,0")
    board.receive(5.1, guard)
    board.send(Drive(0.2, 0))
    assert port.sent[-1] == d(3, 0, 0)


def test_board_and_firmware_agree(run):
    """Everything the computer sends is accepted, across the sequence wrapping."""
    port, guard = Port(), Guard()
    board = Board(port, 0.3)
    events = [(0, False)]
    for i in range(600):
        if i:
            port.waiting = seal(f"T,{i},0,0,900,900,1")
            board.receive(i / 20, guard)
        board.send(Drive(0.001 * (i % 300), 0))
        events += [(50 * i + 1, port.sent[-1]), (50 * i + 2, False)]
    out = run(events)
    assert out[0] == (0, 0, 0)
    assert all(left == right == pytest.approx(i % 300, abs=1) for i, (left, right, _) in enumerate(out[1:]))


def test_late_command_before_the_control_step(run):
    out = run([*READY, (2, d(2, 100, 100)), (900, d(3, 100, 100)), (901, False)])
    assert out[-1] == (0, 0, 0)


def test_releasing_the_emergency_stop_does_not_move_it():
    port, guard = Port(), Guard()
    board = Board(port, 0.3)
    for i, state in enumerate((1, 2, 0, 1, 1), 1):
        port.waiting = seal(f"T,{i},0,0,900,900,{state}")
        board.receive(i / 10, guard)
        guard.link(i / 10), guard.person(i / 10, 3.0, False)
        board.send(guard.allow(i / 10, Drive(0.2, 0)))
    assert guard.stopped and port.sent[-1] == d(5, 0, 0)
    guard.clear()
    port.waiting = seal("T,6,0,0,900,900,1")
    board.receive(0.6, guard)
    guard.link(0.6), guard.person(0.6, 3.0, False)
    assert guard.allow(0.6, Drive(0.2, 0)).speed > 0
