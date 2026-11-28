import json
import math
import random
import subprocess
import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from ostinato.breathing import Reading
from ostinato.localize import Sighting
from ostinato.motors import seal
from ostinato.robot import LOOKS, Robot, Setup, main
from ostinato.rover import KEEP_OUT, STOP, Drive, Pose, State, wrap

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
        self.battery = lambda t: 12.0  # V

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
        mv = round(self.battery(t) * 1000)
        line = seal(f"T,{ms - self.ms0},{int(self.ticks[0])},{int(self.ticks[1])},3000,3000,{state},{mv}")
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
    hw.battery = trouble.get("battery", hw.battery)
    robot, dt, nearest, wrong = Robot(setup, hw, ask=trouble.get("ask", False)), 0.05, math.inf, 0
    cut, estop, restart = trouble.get("cut"), trouble.get("estop"), trouble.get("restart")
    asks = robot.asks = []  # times the robot asked for a raised hand
    hand = trouble.get("hand")  # seconds after asking that a hand goes up, or None
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
            axis = None
            if chest and "lies" in trouble:  # the real direction to their head, in degrees
                a = math.radians(trouble["lies"])
                ends = [(really[0] + k * math.cos(a), really[1] + k * math.sin(a)) for k in (0.2, -0.3)]
                axis = tuple(hw.chest(p, setup.lens_height) for p in ends)
                axis = None if None in axis else axis
            # Asking lasts 10 s, as in the app.
            asking = bool(asks) and t - asks[-1] < 10
            raised = asking and hand is not None and t - asks[-1] >= hand
            robot.step(
                t,
                sightings=hw.sightings() if i % 10 == 0 else (),
                reading=reading,
                responded=raised,
                asking=asking,
                chest=chest,
                shaking=robot.mission.state is State.SETTLING and t < trouble.get("shaking", 0),
                axis=axis,
            )
            robot.moved_while_shaking = getattr(robot, "moved_while_shaking", False) or (
                robot.shaking and robot.drive != STOP
            )
            if robot.wants_ask:
                asks.append(t)
            robot.moved_while_asking = getattr(robot, "moved_while_asking", False) or (
                asking and robot.drive != STOP
            )
            nearest = min(nearest, setup.body.distance(hw.pose.x, hw.pose.y))
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


def test_waits_for_a_steady_picture(board_program, setup):
    plain, _, _ = drive(board_program, setup, 1)
    settled = next(t for t, state, _ in plain.mission.log if state == "settling")
    robot, _, _ = drive(board_program, setup, 1, shaking=settled + 4)
    times = {state: t for t, state, _ in robot.mission.log}
    assert robot.mission.state is State.MEASURING and not robot.moved_while_shaking
    assert times["measuring"] >= settled + 4 + 2 - 0.1
    robot.mission._go(0, State.SETTLING)
    robot.shaking = True
    assert robot.status().startswith("settling, waiting for the camera to stop shaking;")


def test_looks_for_markers_less_while_stopped(board_program, setup):
    robot, _, _ = drive(board_program, setup, 1)
    assert robot.mission.state is State.MEASURING and robot.drive == STOP and robot.localizer.good
    robot.moving, t = False, robot._t + 1
    looks = [robot.wants_markers(t + k * 0.1) for k in range(12)]
    assert looks == [True, False, False, False, False, True, False, False, False, False, True, False]
    robot.drive = Drive(0.1, 0.0)  # about to move: every picture
    assert all(robot.wants_markers(t + 2 + k * 0.01) for k in range(3))
    robot.drive = STOP
    robot.localizer.lost = True  # unsure where it is: every picture
    assert all(robot.wants_markers(t + 3 + k * 0.01) for k in range(3))


def test_stops_for_a_flat_battery_but_not_a_brief_sag(board_program, setup):
    low = Setup(**{**setup.__dict__, "battery_low": 10.5})
    # A sag while starting off: under 1 s below the limit.
    robot, _, _ = drive(board_program, low, 1, battery=lambda t: 9.8 if 3 < t < 3.8 else 12.0)
    assert robot.mission.state is State.MEASURING and robot.battery_problem is None
    # Flat from 4 s, while driving.
    robot, _, _ = drive(
        board_program, low, 1, until=State.DONE, seconds=40, battery=lambda t: 10.2 if t > 4 else 12
    )
    assert robot.battery_problem == "the battery is low (10.2 V)"
    assert robot.status() == "stopped: the battery is low (10.2 V); a person must check"
    assert "measuring" not in [state for _, state, _ in robot.mission.log]  # it never got there
    assert robot.drive == STOP and robot.record(40)["problem"] == "the battery is low (10.2 V)"
    # Without battery_low it carries on, but the check warns.
    robot, _, _ = drive(board_program, setup, 1, battery=lambda t: 9.0)
    assert robot.mission.state is State.MEASURING and robot.battery_problem is None


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
    states = [state for _, state, view in robot.mission.log if view == 1]
    assert states in ([], ["driving", "done"], ["driving", "going round the other way", "done"])
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
        self.sent, self.times, self.ms, self.ticks, self.closed = [], [], 0, 0, False
        Port.made.append(self)

    def write(self, data):
        self.sent.append(data)
        self.times.append(time.monotonic())

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
    sent_at, last = {}, []

    def operator(n):
        # Go from picture 10 to 40, then silence, as if the laptop slept.
        remote.go = 10 <= n
        if n < 40:
            remote.step(time.monotonic())
            last.append(time.monotonic())
            sent_at[n] = len(Port.made[-1].sent) if Port.made else 0
        time.sleep(0.02)

    try:
        port = program(Camera(80, each=operator), keys=[ord("g")] * 5, extra=["--key", str(key)])
    finally:
        remote.close()
    # Its own G is ignored with a remote; nothing moves before the remote says go.
    assert not any(moving(d) for d in port.sent[: sent_at[8]])
    assert any(moving(d) for d in port.sent[sent_at[12] : sent_at[39]])
    # Within 0.5 s of the last heartbeat, it stops and stays stopped.
    late = [d for d, at in zip(port.sent, port.times) if at > last[-1] + 0.55]
    assert late and not any(moving(d) for d in late)


def setup_with(tmp_path, **change):
    path = tmp_path / "room.json"
    path.write_text(json.dumps({**ROOM, **change}))
    return Setup.load(path)


def test_setup_check(tmp_path):
    from ostinato.robot import check_setup

    errors, warnings = check_setup(setup_with(tmp_path))
    assert errors == [] and all(
        "only marker" in w or "without a facing" in w or "no battery_low" in w or "no head direction" in w
        for w in warnings
    )
    assert any("markers 1, 2, 3, 4 without a facing" in w for w in warnings)
    errors, _ = check_setup(setup_with(tmp_path, start=[0.8, 0, 0]))
    assert any("start is 0.80 m" in e for e in errors)
    errors, _ = check_setup(setup_with(tmp_path, markers={"1": [5, 0]}))
    assert [e for e in errors if "sees no marker" in e] and len(errors) == 2
    errors, _ = check_setup(setup_with(tmp_path, markers={**ROOM["markers"], "5": [1.3, 0]}))
    assert any("marker 5 is on the rover's path" in e for e in errors)
    _, warnings = check_setup(setup_with(tmp_path, views=[30, 40]))
    assert any("within 20 degrees" in w for w in warnings)
    # A marker too small in the picture does not count.
    errors, _ = check_setup(setup_with(tmp_path, markers={"4": [-9, 0], "2": [9, 0]}))
    assert sum("sees no marker" in e for e in errors) == 2


def test_setup_check_knows_which_way_markers_face(tmp_path):
    from ostinato.robot import check_setup

    # View 1 sees only marker 4, 4.1 m away, nearly straight on when it faces +x.
    def marker_4(facing):
        markers = {**ROOM["markers"], "4": [-3, 0, facing]}
        return check_setup(setup_with(tmp_path, markers=markers))

    errors, warnings = marker_4(0)
    assert errors == [] and any("markers 1, 2, 3 without" in w for w in warnings)
    assert marker_4(50)[0] == []  # turned 41 degrees from view 1: still read
    for facing in (60, 180):  # turned 51 degrees, too far at 4.1 m; or backwards
        errors, _ = marker_4(facing)
        assert [e for e in errors if "view 1 sees no marker" in e]
    setup = setup_with(tmp_path, markers={"4": [-3, 0, 90], "2": [3, 0]})
    assert setup.markers == {4: (-3.0, 0.0), 2: (3.0, 0.0)} and setup.facings == {4: math.pi / 2}
    for bad in ([-3, 0, 90, 1], [-3, 0, 1e400]):
        with pytest.raises(ValueError, match="setup file"):
            setup_with(tmp_path, markers={"4": bad})


def test_the_example_room_is_complete_and_passes(capsys):
    from ostinato.robot import check_setup

    path = Path(__file__).parent.parent / "docs" / "room.example.json"
    setup = Setup.load(path)
    assert set(setup.facings) == set(setup.markers) and setup.lens_height == 0.8
    assert check_setup(setup) == ([], [])
    main(["--setup", str(path), "--check"])
    assert "No problems found." in capsys.readouterr().out


def test_setup_knows_how_the_person_lies(tmp_path):
    from ostinato.robot import check_setup

    setup = setup_with(tmp_path, person=[0, 0, 90])
    assert setup.person == (0.0, 0.0) and setup.head == pytest.approx(math.pi / 2)
    errors, warnings = check_setup(setup)
    assert errors == [] and not any("head direction" in w for w in warnings)
    # Started 1.5 m beyond the feet: 0.4 m from them.
    errors, _ = check_setup(setup_with(tmp_path, person=[0, 0, 90], start=[0, -1.5, 0]))
    assert any("start is 0.40 m from the person" in e for e in errors)
    # A view straight past the head or the feet sees the chest end on.
    _, warnings = check_setup(setup_with(tmp_path, person=[0, 0, 90], views=[80, 30, -100]))
    assert any("view 1 is beyond the head" in w for w in warnings)
    assert any("view 3 is beyond the feet" in w for w in warnings)
    for bad in ([0, 0, 90, 1], [0, 0, 1e400], [0]):
        with pytest.raises(ValueError, match="setup file"):
            setup_with(tmp_path, person=bad)


def test_guard_hears_the_distance_to_the_feet(tmp_path):
    from ostinato.localize import Localizer

    class Ear:
        def person(self, t, distance, approaching):
            self.heard = distance, approaching

    setup = setup_with(tmp_path, person=[0, 0, 90])
    # 2.3 m from the chest, straight past the feet, facing them.
    loc, ear = (
        Localizer(Pose(0, -2.3, math.pi / 2), setup.markers, setup.track, sigma=(0.01, 0.01, 0.01)),
        Ear(),
    )
    loc.tell(ear, 0.0, setup.body)
    assert ear.heard[0] == pytest.approx(1.2 - 2 * 0.01, abs=0.01) and ear.heard[1]
    loc.tell(ear, 0.0, setup.person)  # as before, the chest alone
    assert ear.heard[0] == pytest.approx(2.3 - 2 * 0.01, abs=0.01)


def test_status_shows_battery_and_pictures(board_program, setup):
    robot, _, _ = drive(board_program, setup, 1, battery=lambda t: 11.84)
    assert "; battery 11.8 V; " in robot.status()
    for k in range(31):
        robot.picture(100 + k / 6)  # 6 pictures a second
    assert "; 6 pictures/s (slow); " in robot.status()
    assert robot.status().endswith(robot.mission.outcome)


def test_remote_window_wraps_the_status_never_cuts_it():
    from ostinato import remote

    class Fake:
        go, status = False, "stopped: the person lies differently from the setup file; a person must check"

        def connected(self, now):
            return True

    rows = [text for text, _ in remote.lines(Fake(), 0.0)]
    assert rows[0] == "STOPPED" and rows[-1].startswith("G go")
    shown = " ".join(rows[2:-1])
    assert shown == Fake.status and all(len(r) <= 58 for r in rows[2:-1])


def test_stops_when_the_person_lies_another_way(board_program, tmp_path):
    lying = setup_with(tmp_path, person=[0, 0, 90], camera={"fov": 70, "pitch": 30, "height": 0.8})
    for seed in (1, 2, 3):
        robot, nearest, _ = drive(board_program, lying, seed, until=State.DONE, seconds=120, lies=95)
        assert robot.person_problem is None and "breaths/min" in robot.mission.outcome
        assert len(robot.seen_axis) == LOOKS and nearest > KEEP_OUT
    # Lying across the way the file says: the head may be nearer the rover's path.
    for seed in (1, 2, 3):
        robot, _, _ = drive(board_program, lying, seed, until=State.DONE, seconds=120, lies=0)
        assert robot.person_problem == "the person lies differently from the setup file"
        assert robot.drive == STOP


def test_says_when_the_board_cut_the_motors_for_the_battery(setup):
    class Board:
        def __init__(self):
            self.lines = [seal(f"T,{50 * i},0,0,3000,3000,{3 if i > 2 else 1},9400") for i in range(5)]

        def write(self, data):
            pass

        def read(self):
            return self.lines.pop(0) if self.lines else b""

    robot = Robot(setup, Board())
    for i in range(5):
        robot.step(i * 0.05)
    assert robot.battery_problem == "the board stopped the motors: the battery is below its cutoff"
    assert robot.status().startswith("stopped: the board stopped the motors") and robot.drive == STOP


def test_check_command(tmp_path, capsys):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(ROOM))
    main(["--setup", str(good), "--check"])
    assert "No problems found." in capsys.readouterr().out
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**ROOM, "markers": {"1": [5, 0]}}))
    with pytest.raises(SystemExit):
        main(["--setup", str(bad), "--check"])
    assert "Problem: view 1 sees no marker" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--setup", str(good)])


def test_run_log(program, tmp_path, capsys):
    log = tmp_path / "logs" / "run.jsonl"
    program(Camera(40, each=lambda n: time.sleep(0.03)), keys=[ord("g")], extra=["--log", str(log)])
    header, *rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert header == {"setup": ROOM}  # what it was set up with comes first
    assert (
        len(rows) >= 2
        and {"x", "y", "within", "guard", "state", "markers", "shaking", "outcome"} <= rows[0].keys()
    )
    assert all(isinstance(r["markers"], list) for r in rows)
    rates = [r["fps"] for r in rows if r["fps"]]
    assert rates and all(5 < rate < 40 for rate in rates)  # pictures 0.03 s apart, plus the work
    capsys.readouterr()
    main(["--setup", str(tmp_path / "room.json"), "--replay", str(log)])
    out = capsys.readouterr().out
    assert "driven" in out and "Nearest to the person" in out and "Ended driving" in out
    assert "different setup file" not in out
    other = tmp_path / "other.json"
    other.write_text(json.dumps({**ROOM, "person": [0.5, 0]}))
    main(["--setup", str(other), "--replay", str(log)])
    assert capsys.readouterr().out.startswith("Note: this log was made with a different setup file")
    # A log is never overwritten.
    with pytest.raises(SystemExit):
        program(Camera(5), extra=["--log", str(log)])


def test_replay_summary(tmp_path):
    from ostinato.robot import summarize

    rows = [
        {
            "t": 0,
            "x": 2.4,
            "y": 0.3,
            "within": 0.1,
            "lost": False,
            "problem": None,
            "guard": "emergency stop",
            "state": "driving",
            "outcome": "in progress",
        },
        {
            "t": 0.5,
            "x": 2.0,
            "y": 0.3,
            "within": 0.05,
            "lost": False,
            "problem": None,
            "guard": "ok",
            "state": "driving",
            "outcome": "in progress",
        },
        {
            "t": 1.0,
            "x": 1.4,
            "y": 0.3,
            "within": 0.05,
            "lost": True,
            "problem": "the person is not where the setup file says",
            "guard": "no news from person",
            "state": "done",
            "outcome": "could not reach any viewpoint: a person must check",
        },
    ]
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    text = summarize(path, setup_with(tmp_path))
    assert "about 1.0 m driven" in text and "1.33 m" in text
    assert "Lost its position at 1 s" in text and "Stopped because the person" in text
    assert "emergency stop: 0.5 s" in text and "shaking" not in text  # an older log
    settling = {**rows[1], "guard": "ok", "state": "settling", "outcome": "in progress"}
    shaky = [
        {**settling, "t": 1.5 + k / 2, "view": v, "shaking": k < n}
        for v, n in ((0, 3), (1, 1))
        for k in range(4)
    ]
    path.write_text("\n".join(json.dumps(r) for r in [*rows[:2], *shaky]))
    text = summarize(path, setup_with(tmp_path))
    assert "Waited for the camera to stop shaking: view 1 1.5 s, view 2 0.5 s" in text
    assert "Pictures a second" not in text  # no rates in this log
    timed = [{**r, "fps": f} for r, f in zip(shaky, (15, 15, 15, 8, 15, 15, 6, 15))]
    timed[6]["state"] = "measuring"
    path.write_text("\n".join(json.dumps(r) for r in [*rows[:2], *timed]))
    text = summarize(path, setup_with(tmp_path))
    assert "Pictures a second: typically 15, lowest 6." in text
    assert "Battery" not in text  # none in this log
    charged = [{**r, "battery": v} for r, v in zip(timed, (12.4, 12.3, 11.6, 12.1, 12.0, 11.9, 11.9, 11.8))]
    path.write_text("\n".join(json.dumps(r) for r in [*rows[:2], *charged]))
    assert "Battery: 12.4 V at the start, 11.8 V at the end, lowest 11.6 V." in summarize(
        path, setup_with(tmp_path)
    )
    assert "Below 10 a second for 1.0 s, 0.5 s of it while measuring." in text
    path.write_text("{broken")
    with pytest.raises(ValueError):
        summarize(path, setup_with(tmp_path))


def test_asks_for_a_hand_before_leaving_a_view(board_program, setup):
    robot, nearest, _ = drive(
        board_program, setup, 2, until=State.DONE, seconds=200, breathing=False, ask=True
    )
    # Once at each of the two views, and it never moved while asking.
    assert len(robot.asks) == 2 and not robot.moved_while_asking
    assert robot.mission.view == 1 and "a person must check" in robot.mission.outcome
    assert nearest > KEEP_OUT


def test_a_raised_hand_ends_it_where_it_is(board_program, setup):
    robot, _, _ = drive(
        board_program, setup, 2, until=State.DONE, seconds=200, breathing=False, ask=True, hand=3
    )
    assert len(robot.asks) == 1 and robot.mission.view == 0
    assert (
        robot.mission.state is State.DONE
        and robot.mission.outcome == "hand raised when asked; breathing was not measured"
    )
    assert robot.drive == STOP


def test_without_asking_it_never_asks(board_program, setup):
    robot, _, _ = drive(board_program, setup, 2, until=State.DONE, seconds=200, breathing=False)
    assert robot.asks == [] and robot.mission.view == 1
