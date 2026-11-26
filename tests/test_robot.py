import json
import math
import random
import subprocess
import time
from typing import ClassVar

import numpy as np
import pytest

from ostinato.breathing import Reading
from ostinato.localize import Sighting
from ostinato.motors import seal
from ostinato.robot import Robot, Setup, main
from ostinato.rover import KEEP_OUT, STOP, Pose, State, wrap

ROOM = {
    "track": 0.27,
    "ticks_per_meter": 2000,
    "start": [2.4, 0.3, 143],
    "person": [0, 0],
    "views": [30, 150],
    "markers": {"1": [0, 3], "2": [3, 0], "3": [0, -3], "4": [-3, 0]},
    "camera": {"fov": 70, "pitch": 20},
}


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / "room.json"
    path.write_text(json.dumps(ROOM))
    return Setup.load(path)


class Hardware:
    """Wheels, encoders, and a camera that sees markers, around the real board code."""

    def __init__(self, program, setup, seed, loss=0.0):
        self.board = subprocess.Popen([program], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.setup, self.loss = setup, loss
        self.rng, self.chance = np.random.default_rng(seed), random.Random(seed)
        self.pose = Pose(setup.start.x + 0.04, setup.start.y - 0.03, setup.start.heading + 0.04)
        self.ticks, self.waiting, self.sent = [0.0, 0.0], b"", []
        self.restart_at = None
        self.ms0 = 0

    # The port that Robot uses
    def write(self, data):
        self.sent.append(data)

    def read(self):
        data, self.waiting = self.waiting, b""
        return data

    def run(self, t, dt, cut=False, estop=False):
        """Pass on what was sent, move the wheels for dt, and report."""
        ms = int(t * 1000) + 1
        for data in self.sent:
            if cut or self.chance.random() < self.loss:
                continue
            if self.chance.random() < self.loss:
                i = self.chance.randrange(len(data) - 1)
                data = data[:i] + bytes([self.chance.randrange(256)]) + data[i + 1 :]
            self.board.stdin.write("".join(f"{ms} B {b}\n" for b in data))
        self.sent.clear()
        self.board.stdin.write(f"{ms} T {int(estop)}\n")
        self.board.stdin.flush()
        left, right, armed = (int(v) for v in self.board.stdout.readline().split())
        v = [w / 1000 * (1 + self.rng.normal(0, 0.03)) for w in (left, right)]
        self.ticks = [a + b * dt * self.setup.ticks_per_meter for a, b in zip(self.ticks, v)]
        turn, middle = (v[1] - v[0]) / self.setup.track * dt, sum(v) / 2 * dt
        p = self.pose
        a = p.heading + turn / 2
        self.pose = Pose(p.x + middle * math.cos(a), p.y + middle * math.sin(a), wrap(p.heading + turn))
        state = 2 if estop else armed
        line = seal(f"T,{ms - self.ms0},{int(self.ticks[0])},{int(self.ticks[1])},3000,3000,{state}")
        if not cut and self.chance.random() >= self.loss:
            self.waiting += line
        return left or right

    def sightings(self):
        found = []
        for number, (x, y) in self.setup.markers.items():
            bearing = wrap(math.atan2(y - self.pose.y, x - self.pose.x) - self.pose.heading)
            if abs(wrap(bearing - math.pi / 2)) < 0.6:
                distance = math.dist((x, y), (self.pose.x, self.pose.y)) * (1 + self.rng.normal(0, 0.02))
                found.append(Sighting(number, distance, bearing + self.rng.normal(0, 0.01)))
        return found

    def chest(self, person, height):
        """Where the person's chest appears, or None if outside the picture."""
        from test_localize import project

        from ostinato.robot import CHEST_HEIGHT

        dx, dy = person[0] - self.pose.x, person[1] - self.pose.y
        look = self.pose.heading + self.setup.camera.pan * math.pi / 180
        ahead = dx * math.cos(look) + dy * math.sin(look)
        right = dx * math.sin(look) - dy * math.cos(look)
        if ahead <= 0.3:
            return None
        x, y = project(self.setup.camera, right, ahead, height - CHEST_HEIGHT)
        x, y = x + self.rng.normal(0, 4), y + self.rng.normal(0, 4)
        return (x, y) if 0 <= x < self.setup.camera.width and 0 <= y < self.setup.camera.height else None

    def close(self):
        self.board.kill()


def drive(program, setup, seed, until=State.MEASURING, seconds=200, **trouble):
    """Returns the robot, the nearest it came to the person, and how often the wheels
    turned when they should not have."""
    hw = Hardware(program, setup, seed, trouble.pop("loss", 0.0))
    robot, dt, nearest, wrong = Robot(setup, hw), 0.05, math.inf, 0
    cut, estop, restart = trouble.get("cut"), trouble.get("estop"), trouble.get("restart")
    really = trouble.get("really", setup.person)
    try:
        for i in range(int(seconds / dt)):
            t = i * dt
            if restart is not None and abs(t - restart) < dt / 2:
                hw.ms0 = int(t * 1000)
            measuring = robot.mission.state is State.MEASURING
            reading = (
                (
                    Reading(15.0, 0.9, "ok")
                    if trouble.get("breathing", True)
                    else Reading(None, 0.0, "chest not visible")
                )
                if measuring
                else None
            )
            turning = hw.run(t, dt, cut=cut is not None and t > cut, estop=estop is not None and t > estop)
            late = [x for x in (cut, estop) if x is not None and t > x + 0.6]
            wrong += bool(turning and late)
            chest = hw.chest(really, setup.lens_height) if measuring and setup.lens_height else None
            robot.step(t, sightings=hw.sightings() if i % 10 == 0 else (), reading=reading, chest=chest)
            nearest = min(nearest, math.dist((hw.pose.x, hw.pose.y), setup.person))
            if robot.mission.state is until or robot.mission.state is State.DONE:
                break
    finally:
        hw.close()
    return robot, nearest, wrong


@pytest.mark.parametrize("loss", [0.0, 0.1, 0.3])
def test_reaches_a_view_and_keeps_away(board_program, setup, loss):
    runs = [drive(board_program, setup, seed, loss=loss) for seed in range(6)]
    assert sum(r.mission.state is State.MEASURING for r, _, _ in runs) >= 5
    assert min(n for _, n, _ in runs) > KEEP_OUT


def test_measures_and_reports(board_program, setup):
    robot, _, _ = drive(board_program, setup, 1, until=State.DONE)
    assert "breaths/min" in robot.mission.outcome


@pytest.mark.parametrize("trouble", [{"cut": 4.0}, {"estop": 4.0}, {"restart": 4.0}])
def test_stops_for_trouble(board_program, setup, trouble):
    for seed in range(3):
        robot, nearest, wrong = drive(board_program, setup, seed, seconds=30, **trouble)
        assert wrong == 0 and robot.mission.state is not State.MEASURING and nearest > KEEP_OUT
    if "estop" in trouble:
        assert robot.guard.stopped
    if "restart" in trouble:
        assert robot.localizer.lost


def test_holds_still_until_the_board_answers(setup):
    class Silent:
        def __init__(self):
            self.sent = []

        def write(self, data):
            self.sent.append(data)

        def read(self):
            return b""

    port = Silent()
    robot = Robot(setup, port)
    assert all(robot.step(i / 20) == STOP for i in range(40))
    assert all(b",0,0*" in line for line in port.sent)


@pytest.mark.parametrize(
    "change",
    [
        {"track": 0},
        {"track": "wide"},
        {"start": [1, 2]},
        {"person": [0, "x"]},
        {"views": []},
        {"markers": {}},
        {"markers": {"a": [0, 0]}},
        {"markers": {"1": [0, 1e400]}},
        {"camera": {"zoom": 2}},
        {"camera": {"fov": 200}},
        {"ticks_per_meter": -1},
    ],
)
def test_bad_setup(tmp_path, change):
    path = tmp_path / "room.json"
    path.write_text(json.dumps({**ROOM, **change}).replace("Infinity", "1e400"))
    with pytest.raises(ValueError):
        Setup.load(path)


def test_setup_units(setup):
    assert setup.start.heading == pytest.approx(math.radians(143))
    assert setup.views == pytest.approx([math.radians(30), math.radians(150)])
    assert setup.camera.pitch == 20 and setup.markers[2] == (3.0, 0.0)


def test_command_needs_a_setup_and_port(tmp_path):
    with pytest.raises(SystemExit):
        main(["--setup", str(tmp_path / "missing.json"), "--port", "x"])


def test_without_markers_it_looks_once_then_stops(board_program, tmp_path):
    path = tmp_path / "room.json"
    path.write_text(json.dumps({**ROOM, "markers": {"9": [40, 40]}}))
    for seed in range(3):
        robot, nearest, _ = drive(board_program, Setup.load(path), seed, seconds=120)
        assert robot.mission.state is State.DONE and "a person must check" in robot.mission.outcome
        assert robot.searched >= 2 * math.pi and nearest > 1.5
        assert "no marker found" in robot.status()


@pytest.fixture
def mast(tmp_path):
    path = tmp_path / "room.json"
    path.write_text(json.dumps({**ROOM, "camera": {"fov": 70, "pitch": 30, "height": 0.8}}))
    return Setup.load(path)


def test_person_where_the_file_says(board_program, mast):
    for seed in range(4):
        robot, _, _ = drive(board_program, mast, seed, until=State.DONE)
        assert robot.person_problem is None and len(robot.seen_person) == 15
        assert "breaths/min" in robot.mission.outcome
    # With no rate, it goes on to the next view, which it may: the person was seen in place.
    robot, nearest, _ = drive(board_program, mast, 0, until=State.DONE, seconds=150, breathing=False)
    assert robot.person_problem is None and robot.mission.view == 1 and nearest > KEEP_OUT


@pytest.mark.parametrize("really", [(1.0, 0.0), (0.0, -0.9), (-0.7, 0.7), (0.9, 0.55)])
def test_stops_when_the_person_is_elsewhere(board_program, mast, really):
    """Seen in the wrong place, or not seen at all, which may mean beside the rover."""
    robot, _, _ = drive(board_program, mast, 1, until=State.DONE, seconds=120, really=really, breathing=False)
    assert robot.person_problem and robot.drive == STOP
    # It never reached the second view: the wheels stayed still once it doubted.
    assert [state for _, state, view in robot.mission.log if view == 1] in ([], ["driving", "done"])
    assert "setup file says" in robot.status()


def test_camera_height_is_checked(tmp_path):
    path = tmp_path / "room.json"
    for height in (0.1, 3, "high"):
        path.write_text(json.dumps({**ROOM, "camera": {"fov": 70, "height": height}}))
        with pytest.raises(ValueError):
            Setup.load(path)


class Camera:
    """Grey pictures; calls `each` before every one."""

    def __init__(self, frames, each=lambda n: None):
        self.n, self.frames, self.each = 0, frames, each
        self.released = False

    def set(self, *args):
        pass

    def read(self):
        self.n += 1
        self.each(self.n)
        return self.n <= self.frames, np.full((720, 1280, 3), 120, np.uint8)

    def release(self):
        self.released = True


class Port:
    """A board that is ready and whose wheels count up while asked to turn."""

    made: ClassVar[list] = []

    def __init__(self, name):
        self.sent, self.ms, self.ticks, self.closed = [], 0, 0, False
        Port.made.append(self)

    def write(self, data):
        self.sent.append(data)

    def read(self):
        self.ms += 50
        moving = self.sent and not self.sent[-1].split(b"*")[0].endswith(b",0,0")
        self.ticks += 20 if moving else 0
        return seal(f"T,{self.ms},{self.ticks},{self.ticks},3000,3000,1")

    def close(self):
        self.closed = True


class Finder:
    body = points = None

    def __call__(self, frame):
        return None

    def close(self):
        pass


def moving(data):
    return not data.split(b"*")[0].endswith(b",0,0")


@pytest.fixture
def program(monkeypatch, tmp_path):
    import cv2

    from ostinato import app, chest, robot

    (tmp_path / "room.json").write_text(json.dumps(ROOM))
    monkeypatch.setattr(robot, "SerialPort", Port)
    monkeypatch.setattr(chest, "PoseFinder", Finder)
    for name in ("imshow", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, lambda *a: None)
    Port.made.clear()

    def run(camera, keys=(), extra=()):
        presses = iter(keys)
        monkeypatch.setattr(app, "open_camera", lambda index: camera)
        monkeypatch.setattr(cv2, "waitKey", lambda ms: next(presses, 255))
        robot.main(["--setup", str(tmp_path / "room.json"), "--port", "x", *extra])
        return Port.made[-1]

    return run


def test_program_starts_stopped_and_cleans_up(program, capsys):
    camera = Camera(60)
    port = program(camera)
    assert not any(moving(d) for d in port.sent)
    assert port.closed and camera.released
    assert port.sent[-3:] == [d for d in port.sent[-3:] if not moving(d)]
    assert "only for testing" in capsys.readouterr().out


def test_program_goes_on_g(program):
    port = program(Camera(60, each=lambda n: time.sleep(0.01)), keys=[ord("g")])
    assert any(moving(d) for d in port.sent) and not moving(port.sent[-1])


def test_program_follows_the_remote(program, tmp_path):
    from ostinato.remote import Remote, load_key, make_key

    key = make_key(tmp_path / "key.txt")
    remote = Remote(load_key(key), "127.0.0.1")
    sent_at = {}

    def operator(n):
        # Go from picture 10 to 40, then silence, as if the laptop slept.
        remote.go = 10 <= n
        if n < 40:
            remote.step(time.monotonic())
            sent_at[n] = len(Port.made[-1].sent) if Port.made else 0
        time.sleep(0.02)

    try:
        port = program(Camera(80, each=operator), keys=[ord("g")] * 5, extra=["--key", str(key)])
    finally:
        remote.close()
    # Its own G is ignored with a remote; nothing moves before the remote says go.
    assert not any(moving(d) for d in port.sent[: sent_at[8]])
    assert any(moving(d) for d in port.sent[sent_at[12] : sent_at[39]])
    # About 0.3 s of silence later, it stops and stays stopped.
    assert not any(moving(d) for d in port.sent[sent_at[39] + 25 :])
