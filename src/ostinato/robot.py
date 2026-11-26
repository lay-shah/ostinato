"""Run the rover: `python -m ostinato.robot --setup room.json --port /dev/ttyACM0`.

No rover has been built, so this has only run against simulated hardware. It needs
`pip install pyserial`. The hardware emergency stop that cuts motor power is the real
safety; everything here is on top of it.

The setup file, in meters and degrees, measured by a person:

    {"track": 0.27, "ticks_per_meter": 2000,
     "start": [2.4, 0.3, 140], "person": [0, 0], "views": [30, 150],
     "markers": {"1": [0, 3], "2": [3, 0]},
     "camera": {"calibration": "camera.json", "pitch": 20, "height": 0.8}}

"calibration" is from `python -m ostinato.calibrate`. Without it, give "fov", the
lens's angle across, and distances to markers are rougher.

"start" is where the rover is placed and which way it faces. "views" are directions
around the person, from the person, to measure from in order. Check a file with
`python -m ostinato.robot --setup room.json --check` before driving. "height" is the lens's
height above the floor. With it, the camera checks that the person is where the file
says, and the rover stops if not.
"""

import argparse
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calibrate import Calibration
from .localize import MARKER, Camera, Localizer, find_markers, ground, place
from .motors import Board
from .rover import KEEP_OUT, RING, STOP, Drive, Guard, Mission, Pose, State, wrap

CHEST_HEIGHT = 0.15  # m, the chest of a person lying down, above the floor
MOVED = 0.25  # m off before stopping: views are 1.25 m out, and 1 m must be kept
LOOKS = 15  # sightings of the person before judging
SEARCH_TURN = 0.4  # rad/s, turning in place to find a marker when unsure where it is


@dataclass(frozen=True)
class Setup:
    track: float
    ticks_per_meter: float
    start: Pose
    person: tuple[float, float]
    views: list[float]  # radians
    markers: dict[int, tuple[float, float]]
    camera: Camera = field(default_factory=Camera)
    lens_height: float | None = None  # m above the floor

    @classmethod
    def load(cls, path):
        try:
            raw = json.loads(Path(path).read_text())
            x, y, heading = (float(v) for v in raw["start"])
            camera = dict(raw.get("camera", {}))
            if not camera.keys() <= {"fov", "pitch", "pan", "height", "calibration"}:
                raise ValueError("camera takes calibration, fov, pitch, pan, and height")
            lens = camera.pop("calibration", None)
            camera = {k: float(v) for k, v in camera.items()}
            lens_height = camera.pop("height", None)
            if lens is not None:
                if "fov" in camera:
                    raise ValueError("give calibration or fov, not both")
                camera = Calibration.load(Path(path).parent / lens).camera(**camera)
            else:
                camera = Camera(**camera)
            setup = cls(
                float(raw["track"]),
                float(raw["ticks_per_meter"]),
                Pose(x, y, math.radians(heading)),
                tuple(float(v) for v in raw["person"]),
                [math.radians(float(v)) for v in raw["views"]],
                {int(k): tuple(float(v) for v in p) for k, p in raw["markers"].items()},
                camera,
                lens_height,
            )
        except (OSError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
            raise ValueError(f"setup file: {error}") from None
        numbers = [setup.track, setup.ticks_per_meter, *setup.person, *setup.views]
        numbers += [v for p in setup.markers.values() for v in p]
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError("setup file: every number must be finite")
        if not (0.1 <= setup.track <= 1 and setup.ticks_per_meter > 0 and len(setup.person) == 2):
            raise ValueError("setup file: check track, ticks_per_meter, and person")
        if not setup.views or not setup.markers or any(len(p) != 2 for p in setup.markers.values()):
            raise ValueError("setup file: needs views and at least one marker at [x, y]")
        if setup.lens_height is not None and not 0.2 <= setup.lens_height <= 2:
            raise ValueError("setup file: camera height must be 0.2 to 2 m")
        if not 0 < setup.camera.fov < 170 or abs(setup.camera.pitch) > 60:
            raise ValueError("setup file: check the camera")
        return setup


MIN_PIXELS = 30  # a marker smaller than this in the picture is often missed


def check_setup(setup):
    """Problems with a setup file, before driving. Returns (errors, warnings)."""
    errors, warnings = [], []
    px, py = setup.person
    gap = math.dist((setup.start.x, setup.start.y), setup.person)
    if gap < RING:
        errors.append(f"the start is {gap:.2f} m from the person; it must be at least {RING} m away")
    half = math.radians(setup.camera.fov) / 2 * 0.9  # markers at the very edge are often missed
    sees = setup.camera.matrix[0, 0] * MARKER / MIN_PIXELS
    for i, bearing in enumerate(setup.views, 1):
        x, y = px + RING * math.cos(bearing), py + RING * math.sin(bearing)
        facing = math.atan2(py - y, px - x)  # the camera's direction there
        seen = [
            n
            for n, (mx, my) in setup.markers.items()
            if 0.5 < math.dist((x, y), (mx, my)) < sees
            and abs(wrap(math.atan2(my - y, mx - x) - facing)) < half
        ]
        if seen:
            warnings += (
                [] if len(seen) > 1 else [f"view {i} sees only marker {seen[0]}; two make a better fix"]
            )
        else:
            errors.append(f"view {i} sees no marker: put one beyond the person, as seen from there")
    for a, b in zip(setup.views, setup.views[1:]):
        if abs(wrap(a - b)) < math.radians(20):
            warnings.append("two views are within 20 degrees of each other and see much the same")
    for n, (mx, my) in setup.markers.items():
        ring = abs(math.dist((mx, my), setup.person) - RING)
        if ring < 0.35:
            errors.append(f"marker {n} is on the rover's path around the person; move it")
        elif math.dist((mx, my), setup.person) < KEEP_OUT:
            warnings.append(f"marker {n} is close to the person; check it does not touch them")
    return errors, warnings


class Robot:
    """One control step at a time: sensors in, one command to the board out."""

    def __init__(self, setup, port, guard=None):
        self.setup = setup
        self.board = Board(port, setup.track)
        self.guard = guard or Guard()
        self.localizer = Localizer(setup.start, setup.markers, setup.track)
        self.mission = Mission(setup.person, list(setup.views))
        self.drive, self.moving = STOP, False
        self._last = None
        self.seen_person = deque(maxlen=LOOKS)
        self.confirmed = 0  # sightings of the person in place at this viewpoint
        self.person_problem = None
        self.searched = 0.0  # radians turned looking for a marker since it was last sure

    def step(self, t, link=True, sightings=(), reading=None, responded=False, asking=False, chest=None):
        """`link` is whether the operator is there. `chest` is the middle of the person's
        shoulders and hips in the picture, when seen. Returns the command sent."""
        if link:
            self.guard.link(t)
        for report in self.board.receive(t, self.guard):
            last, self._last = self._last, report
            if last is None:
                continue
            if report.ms < last.ms:
                # The board restarted. Its wheels may have rolled meanwhile, uncounted.
                self.localizer.lost = True
                continue
            per = self.setup.ticks_per_meter
            self.localizer.drive((report.left - last.left) / per, (report.right - last.right) / per)
            self.moving = (report.left, report.right) != (last.left, last.right)
        for sighting in sightings:
            self.localizer.see(sighting)
        self._check_person(chest)
        self.localizer.tell(self.guard, t, self.setup.person)
        moving = self.moving or self.drive != STOP
        was = self.mission.state
        want = self.mission.update(t, self.localizer.pose, moving, reading, responded, asking)
        if was is State.MEASURING and self.mission.state is not State.MEASURING:
            # Out of the picture, a person who moved may be right beside the rover.
            if (
                self.setup.lens_height is not None
                and self.confirmed < LOOKS
                and self.mission.state is not State.DONE
            ):
                self.person_problem = "the camera did not see the person where the setup file says"
            self.confirmed = 0
        if self.localizer.good:
            self.searched = 0.0
        elif self.mission.state in (State.DRIVING, State.FACING) and not self.localizer.lost:
            # Unsure where it is: turn on the spot, once around at most, to see a marker.
            # Never while settling or measuring.
            if self.searched < 2 * math.pi:
                self.searched += abs(self.drive.turn) * self._dt(t)
                want = Drive(0.0, SEARCH_TURN)
            else:
                want = STOP
        if self.person_problem:
            want = STOP  # the 1 m kept from them may be measured from the wrong place
        self._t = t
        self.drive = self.guard.allow(t, want)
        self.board.send(self.drive)
        return self.drive

    def _check_person(self, chest):
        if chest is None or self.setup.lens_height is None or not self.localizer.good or self.moving:
            return
        seen = ground(self.setup.camera, chest, self.setup.lens_height - CHEST_HEIGHT)
        if seen is None:
            return
        self.seen_person.append(place(self.localizer.pose, seen))
        if len(self.seen_person) == LOOKS:
            x, y = np.median(np.array(self.seen_person), axis=0)
            if math.dist((x, y), self.setup.person) > MOVED:
                self.person_problem = self.person_problem or "the person is not where the setup file says"
            else:
                self.confirmed += 1

    def _dt(self, t):
        last = getattr(self, "_t", None)
        return 0.0 if last is None else max(0.0, t - last)

    def status(self):
        place, _ = self.localizer.uncertainty
        if self.person_problem:
            return f"stopped: {self.person_problem}; a person must check"
        where = "lost: place it again" if self.localizer.lost else f"position within {100 * place:.0f} cm"
        if not (self.localizer.good or self.localizer.lost):
            where += ", looking for a marker" if self.searched < 2 * math.pi else ", no marker found"
        return f"{self.mission.state.value}; {self.guard.reason}; {where}; {self.mission.outcome}"


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


def main(argv=None):
    import cv2

    from .app import Monitor, open_camera
    from .chest import ChestTracker, PoseFinder
    from .response import ResponseCheck, Speaker

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--port", help="the motor board's serial port")
    parser.add_argument("--check", action="store_true", help="only check the setup file")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--key", type=Path, help="key file shared with `python -m ostinato.remote`")
    args = parser.parse_args(argv)
    from .remote import Listener, load_key

    if args.check:
        try:
            errors, warnings = check_setup(Setup.load(args.setup))
        except ValueError as error:
            parser.exit(1, f"{error}\n")
        for line in errors:
            print(f"Problem: {line}")
        for line in warnings:
            print(f"Note: {line}")
        print("Fix the problems before driving." if errors else "No problems found.")
        if errors:
            parser.exit(1)
        return
    if not args.port:
        parser.error("--port is needed, except with --check")
    listener = None
    try:
        setup = Setup.load(args.setup)
        if args.key:
            listener = Listener(load_key(args.key))
        port = SerialPort(args.port)
    except (OSError, ValueError) as error:
        if listener:
            listener.close()
        parser.exit(1, f"{error}\n")
    camera = open_camera(args.camera)
    if camera is None:
        port.close()
        parser.exit(1, "Could not open the camera.\n")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, setup.camera.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, setup.camera.height)
    finder = PoseFinder()
    monitor, check = Monitor(ChestTracker(finder)), ResponseCheck(Speaker())
    robot = Robot(setup, port)
    robot.guard.emergency_stop()
    title, start, was = "Ostinato rover", time.monotonic(), None
    if listener:
        print(
            f"Stopped. Waiting for the remote on port {listener.port}. Keep the hardware emergency stop in hand."
        )
    else:
        print("No --key: only for testing on a stand or tethered. G go, Space stop, Q quit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame.shape[:2] != (setup.camera.height, setup.camera.width):
                print("The camera stopped or gives the wrong size.")
                break
            t = time.monotonic() - start
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            measuring = robot.mission.state is State.MEASURING
            reading = responded = None
            if measuring:
                small = cv2.resize(
                    frame, (640, round(640 * frame.shape[0] / frame.shape[1])), interpolation=cv2.INTER_AREA
                )
                reading = monitor.update(small, t)
                responded = check.update(t, finder.body)
            chest = None
            if measuring and finder.points is not None:
                chest = tuple(finder.points.mean(axis=0) * (frame.shape[1] / 640))
            link = True
            if listener:
                listener.poll(t, robot.guard, robot.status())  # tells the guard itself
                link = False
            robot.step(
                t, link, find_markers(gray, setup.camera), reading, bool(responded), check.active, chest
            )
            if measuring and robot.mission.state is not State.MEASURING:
                monitor.reset()  # never carry readings from one view to the next
                check.cancel()
            line = robot.status()
            if line != was:
                print(f"{t:7.1f} s  {line}", flush=True)
                was = line
            cv2.imshow(title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                robot.guard.emergency_stop()
            elif key == ord("g") and not listener:
                robot.guard.clear()
            elif key == ord("q") or robot.mission.state is State.DONE:
                break
    except KeyboardInterrupt:
        pass
    finally:
        robot.guard.emergency_stop()
        for _ in range(3):
            robot.board.send(STOP)
        check.cancel()
        camera.release()
        finder.close()
        port.close()
        if listener:
            listener.close()
        cv2.destroyAllWindows()
    print(f"Outcome: {robot.mission.outcome}")


if __name__ == "__main__":
    main()
