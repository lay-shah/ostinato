"""Run the rover: `python -m ostinato.robot --setup room.json --port /dev/ttyACM0`.

No rover has been built, so this has only run against simulated hardware. It needs
`pip install pyserial`. The hardware emergency stop that cuts motor power is the real
safety; everything here is on top of it.

The setup file, in meters and degrees, measured by a person:

    {"track": 0.27, "ticks_per_meter": 2000,
     "start": [2.4, 0.3, 140], "person": [0, 0], "views": [30, 150],
     "markers": {"1": [0, 3], "2": [3, 0]},
     "camera": {"calibration": "camera.json", "pitch": 20}}

"calibration" is from `python -m ostinato.calibrate`. Without it, give "fov", the
lens's angle across, and distances to markers are rougher.

"start" is where the rover is placed and which way it faces. "views" are directions
around the person, from the person, to measure from in order.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from .calibrate import Calibration
from .localize import Camera, Localizer, find_markers
from .motors import Board
from .rover import STOP, Drive, Guard, Mission, Pose, State

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

    @classmethod
    def load(cls, path):
        try:
            raw = json.loads(Path(path).read_text())
            x, y, heading = (float(v) for v in raw["start"])
            camera = dict(raw.get("camera", {}))
            if not camera.keys() <= {"fov", "pitch", "pan", "calibration"}:
                raise ValueError("camera takes calibration, fov, pitch, and pan")
            lens = camera.pop("calibration", None)
            camera = {k: float(v) for k, v in camera.items()}
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
        if not 0 < setup.camera.fov < 170 or abs(setup.camera.pitch) > 60:
            raise ValueError("setup file: check the camera")
        return setup


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
        self.searched = 0.0  # radians turned looking for a marker since it was last sure

    def step(self, t, link=True, sightings=(), reading=None, responded=False, asking=False):
        """`link` is whether the operator is there. Returns the command sent."""
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
        self.localizer.tell(self.guard, t, self.setup.person)
        moving = self.moving or self.drive != STOP
        want = self.mission.update(t, self.localizer.pose, moving, reading, responded, asking)
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
        self._t = t
        self.drive = self.guard.allow(t, want)
        self.board.send(self.drive)
        return self.drive

    def _dt(self, t):
        last = getattr(self, "_t", None)
        return 0.0 if last is None else max(0.0, t - last)

    def status(self):
        place, _ = self.localizer.uncertainty
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
    parser.add_argument("--port", required=True, help="the motor board's serial port")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        setup = Setup.load(args.setup)
        port = SerialPort(args.port)
    except ValueError as error:
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
    print("Stopped. G to go, Space to stop, Q to quit. Keep the hardware emergency stop in hand.")
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
            robot.step(t, True, find_markers(gray, setup.camera), reading, bool(responded), check.active)
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
            elif key == ord("g"):
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
        cv2.destroyAllWindows()
    print(f"Outcome: {robot.mission.outcome}")


if __name__ == "__main__":
    main()
