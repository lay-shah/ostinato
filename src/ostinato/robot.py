"""Run the rover: `python -m ostinato.robot --setup room.json --port /dev/ttyACM0`.

No rover has been built, so this has only run against simulated hardware. It needs
`pip install pyserial`. The hardware emergency stop that cuts motor power is the real
safety; everything here is on top of it.

The setup file's format is explained at the top of `ostinato/setup_file.py`.
"""

import argparse
import json
import math
import time
from collections import deque
from itertools import pairwise
from pathlib import Path

import numpy as np

from .localize import GOOD, Localizer, find_markers, ground, place
from .motors import Board
from .rover import (
    RING,
    STOP,
    Drive,
    Guard,
    Mission,
    Pose,
    State,
    clearance,
    wrap,
)
from .setup_file import (  # noqa: F401 - also for those who import them from here
    BLIND,
    MIN_PIXELS,
    TURNED,
    TURNED_AWAY,
    TURNED_RANGE,
    Setup,
    check_setup,
    longest_blind,
    marker_reach,
    view_pose,
)
from .shake import Steadiness

CHEST_HEIGHT = 0.15  # m, the chest of a person lying down, above the floor
LIES_OFF = math.radians(45)  # the body turned this far from the setup file stops the rover
MOVED = 0.25  # m off before stopping: views are 1.25 m out, and 1 m must be kept
LOOKS = 15  # sightings of the person before judging
SEARCH_TURN = 0.4  # rad/s, turning in place to find a marker when unsure where it is
MIN_FPS = 10  # pictures a second; as `python -m ostinato.speed` aims for
BATTERY_WINDOW = 2.0  # s of battery reports judged together, so a sag when starting off passes
STILL_LOOKS = 0.5  # s between looks for markers while stopped: nothing changes but the noise


class Robot:
    """One control step at a time: sensors in, one command to the board out."""

    def __init__(self, setup, port, guard=None, ask=False):
        """With `ask`, it asks the person to raise a hand before leaving each view."""
        self.setup = setup
        self.board = Board(port, setup.track)
        self.guard = guard or Guard()
        self.localizer = Localizer(setup.start, setup.markers, setup.track)
        self.mission = Mission(setup.body, list(setup.views), ask_first=ask, obstacles=setup.obstacles)
        self.drive, self.moving, self.shaking = STOP, False, False
        self._frames = deque(maxlen=31)  # times of the last pictures
        self._last = None
        self.seen_person = deque(maxlen=LOOKS)
        self.seen_axis = deque(maxlen=LOOKS)  # directions from hips to shoulders, in the room
        self.confirmed = 0  # sightings of the person in place at this viewpoint
        self.person_problem = None
        self.battery_problem = None
        self.battery = deque()  # (t, volts) from the last BATTERY_WINDOW seconds
        self.searched = 0.0  # radians turned looking for a marker since it was last sure
        self.seen = []  # markers believed in the last step

    def step(
        self,
        t,
        link=True,
        sightings=(),
        reading=None,
        responded=False,
        asking=False,
        chest=None,
        shaking=False,
        axis=None,
    ):
        """`link` is whether the operator is there. `chest` is the middle of the person's
        shoulders and hips in the picture, when seen. `shaking` is whether the picture
        still moves. `axis` is (shoulders, hips) in the picture, each the middle of the
        two. Returns the command sent."""
        if link:
            self.guard.link(t)
        for report in self.board.receive(t, self.guard):
            last, self._last = self._last, report
            self._check_battery(t, report.battery)
            if report.cutoff and not self.battery_problem:
                self.battery_problem = "the board stopped the motors: the battery is below its cutoff"
            if last is None:
                continue
            if report.ms < last.ms:
                # The board restarted. Its wheels may have rolled meanwhile, uncounted.
                self.localizer.lost = True
                continue
            per = self.setup.ticks_per_meter
            self.localizer.drive((report.left - last.left) / per, (report.right - last.right) / per)
            self.moving = (report.left, report.right) != (last.left, last.right)
        self.seen = [s.marker for s in sightings if self.localizer.see(s)]
        self._check_person(chest)
        self._check_lying(axis)
        self.localizer.tell(self.guard, t, self.setup.body)
        moving = self.moving or self.drive != STOP
        was = self.mission.state
        self.shaking = shaking
        want = self.mission.update(t, self.localizer.pose, moving, reading, responded, asking, shaking)
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
        if self.person_problem or self.battery_problem:
            want = STOP  # the 1 m kept from them may be measured from the wrong place
        self._t = t
        self.drive = self.guard.allow(t, want)
        self.board.send(self.drive)
        return self.drive

    def picture(self, t):
        """Note that the camera gave a picture at time t."""
        self._frames.append(t)

    @property
    def fps(self):
        """Pictures a second, over the last 30. None until there are two."""
        if len(self._frames) < 2 or self._frames[-1] <= self._frames[0]:
            return None
        return (len(self._frames) - 1) / (self._frames[-1] - self._frames[0])

    def wants_markers(self, t):
        """Whether to look for markers in this picture. Every picture while it may move;
        twice a second while stopped, which saves time for measuring and keeps one view
        of the same marker from counting as many."""
        moving = self.moving or self.drive != STOP
        if moving or self.mission.state in (State.DRIVING, State.FACING) or not self.localizer.good:
            return True
        last = getattr(self, "_looked", None)
        if last is None or t - last >= STILL_LOOKS or t < last:
            self._looked = t
            return True
        return False

    @property
    def wants_ask(self):
        """Whether to start the spoken request for a raised hand now."""
        a = self.mission.assessor
        return a is not None and a.wants_ask and self.mission.state is State.MEASURING

    def _check_battery(self, t, volts):
        if not math.isfinite(volts):
            return
        self.battery.append((t, volts))
        while self.battery[0][0] < t - BATTERY_WINDOW:
            self.battery.popleft()
        if self.setup.battery_low is not None and t - self.battery[0][0] >= 0.75 * BATTERY_WINDOW:
            typical = float(np.median([v for _, v in self.battery]))
            if typical < self.setup.battery_low and not self.battery_problem:
                self.battery_problem = f"the battery is low ({typical:.1f} V)"

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

    def _check_lying(self, axis):
        """For someone lying down: do they lie the way the setup file says? The rover's
        distance from their head and feet depends on it."""
        s = self.setup
        if axis is None or s.head is None or s.lens_height is None or not self.localizer.good or self.moving:
            return
        ends = [ground(s.camera, p, s.lens_height - CHEST_HEIGHT) for p in axis]
        if None in ends:
            return
        (sx, sy), (hx, hy) = (place(self.localizer.pose, e) for e in ends)
        if not 0.2 < math.hypot(sx - hx, sy - hy) < 1.0:
            return  # not a torso lying flat: seen end on, or a bad pose
        self.seen_axis.append(math.atan2(sy - hy, sx - hx))
        if len(self.seen_axis) == LOOKS:
            angles = np.array(self.seen_axis)
            typical = math.atan2(np.sin(angles).mean(), np.cos(angles).mean())
            if abs(wrap(typical - s.head)) > LIES_OFF:
                self.person_problem = self.person_problem or "the person lies differently from the setup file"

    def _dt(self, t):
        last = getattr(self, "_t", None)
        return 0.0 if last is None else max(0.0, t - last)

    def record(self, t):
        """One line for the run log: where it thinks it is and why it did what it did.
        No pictures."""
        pose, place = self.localizer.pose, self.localizer.uncertainty[0]
        report = self.board._report
        return {
            "t": round(t, 3),
            "x": round(pose.x, 3),
            "y": round(pose.y, 3),
            "heading": round(pose.heading, 3),
            "within": round(place, 3),
            "good": self.localizer.good,
            "lost": self.localizer.lost,
            "markers": self.seen,
            "state": self.mission.state.value,
            "view": self.mission.view,
            "guard": self.guard.reason,
            "speed": round(self.drive.speed, 3),
            "turn": round(self.drive.turn, 3),
            "front": None if report is None or report.front != report.front else report.front,
            "rear": None if report is None or report.rear != report.rear else report.rear,
            "shaking": self.shaking,
            "fps": None if self.fps is None else round(self.fps, 1),
            "battery": None if not self.battery else round(self.battery[-1][1], 2),
            "problem": self.person_problem or self.battery_problem,
            "outcome": self.mission.outcome,
        }

    def status(self):
        place, _ = self.localizer.uncertainty
        if self.person_problem or self.battery_problem:
            return f"stopped: {self.person_problem or self.battery_problem}; a person must check"
        where = "lost: place it again" if self.localizer.lost else f"position within {100 * place:.0f} cm"
        if not (self.localizer.good or self.localizer.lost):
            where += ", looking for a marker" if self.searched < 2 * math.pi else ", no marker found"
        state = self.mission.state.value
        if self.shaking and self.mission.state is State.SETTLING:
            state += ", waiting for the camera to stop shaking"
        health = []
        if self.battery:
            health.append(f"battery {self.battery[-1][1]:.1f} V")
        if self.fps is not None:
            health.append(f"{self.fps:.0f} pictures/s" + (" (slow)" if self.fps < MIN_FPS else ""))
        parts = [state, self.guard.reason, where, *health, self.mission.outcome]
        return "; ".join(parts)


def fix_from_markers(setup, guess, sightings):
    """Where the camera must be, from two or more marker sightings, as a Pose: the best fit
    near `guess`. None with fewer than two, or if it does not settle."""
    known = [s for s in sightings if s.marker in setup.markers]
    if len(known) < 2:
        return None
    x = np.array([guess.x, guess.y, guess.heading], float)

    def residuals(v):
        out = []
        for s in known:
            mx, my = setup.markers[s.marker]
            out.append(s.distance - math.dist((v[0], v[1]), (mx, my)))
            out.append(wrap(s.bearing - (math.atan2(my - v[1], mx - v[0]) - v[2])))
        return np.array(out)

    for _ in range(20):
        r = residuals(x)
        jacobian = np.column_stack([(residuals(x + d) - r) / 1e-6 for d in np.eye(3) * 1e-6])
        step, *_ = np.linalg.lstsq(jacobian, -r, rcond=None)
        x += step
        if np.abs(step).max() < 1e-6:
            return Pose(float(x[0]), float(x[1]), wrap(float(x[2])))
    return None


def rehearsal(setup, pose, sightings, chest=None, axis=None):
    """Lines comparing what the camera sees, standing at `pose`, with the setup file."""
    cam = setup.camera
    half = math.radians(cam.fov) / 2
    lines = []
    seen = {s.marker: s for s in sightings}
    for n, (mx, my) in sorted(setup.markers.items()):
        distance = math.dist((pose.x, pose.y), (mx, my))
        bearing = wrap(math.atan2(my - pose.y, mx - pose.x) - pose.heading)
        across = math.degrees(wrap(bearing - math.radians(cam.pan)))
        expected = f"{distance:.2f} m, {abs(across):.0f} degrees {'left' if across > 0 else 'right'}"
        if n in seen:
            s = seen[n]
            off = math.degrees(wrap(s.bearing - bearing))
            lines.append(
                f"marker {n}: {s.distance:.2f} m, expected {expected}; "
                f"{round(100 * (s.distance - distance) / distance):+d}% and {round(off):+d} degrees off"
            )
        elif abs(across) < math.degrees(half):
            hidden = clearance([(pose.x, pose.y), (mx, my)], setup.obstacles) <= 0
            far = distance >= marker_reach(setup, n, pose.x, pose.y)
            why = (
                "hidden behind furniture in the setup file"
                if hidden
                else f"too far, or turned too far away, to be read from here ({expected})"
                if far
                else f"expected at {expected}"
            )
            lines.append(f"marker {n}: not seen, {why}")
    for n in sorted(set(seen) - set(setup.markers)):
        lines.append(f"marker {n}: seen, but not in the setup file")
    fix = fix_from_markers(setup, pose, sightings)
    if fix is None:
        lines.append("position: two markers are needed to place the camera")
    else:
        off = math.dist((fix.x, fix.y), (pose.x, pose.y))
        turn = math.degrees(wrap(fix.heading - pose.heading))
        verdict = (
            "" if off <= GOOD else "; check where the camera stands, and the markers' places in the file"
        )
        lines.append(
            f"position: the markers put the camera {off:.2f} m and {round(turn):+d} degrees from "
            f"where the file says it stands{verdict}"
        )
    if setup.lens_height is None:
        lines.append("person: give the camera height in the setup file to check where they are")
    elif chest is None:
        lines.append("person: not seen")
    else:
        found = ground(cam, chest, setup.lens_height - CHEST_HEIGHT)
        if found is None:
            lines.append("person: seen, but too far up the picture to place")
        else:
            x, y = place(pose, found)
            off = math.dist((x, y), setup.person)
            verdict = "" if off <= MOVED else f"; more than {MOVED} m, so the rover would stop"
            lines.append(
                f"person: chest {found.distance:.2f} m away, {off:.2f} m from where the file says{verdict}"
            )
            if setup.head is not None and axis is not None:
                ends = [ground(cam, p, setup.lens_height - CHEST_HEIGHT) for p in axis]
                if None not in ends:
                    (sx, sy), (hx, hy) = (place(pose, e) for e in ends)
                    lies = math.atan2(sy - hy, sx - hx)
                    turn = math.degrees(wrap(lies - setup.head))
                    lines.append(f"lying: {round(turn):+d} degrees from the file")
    return lines


def rehearse(setup, views, camera, finder, clock=time.monotonic, show=None):
    """Watch through the camera, standing where the rover would, at each of `views` in
    turn (0 is the start). Nothing moves. `show(frame)` may answer "next" or "quit".
    Ends with what was last seen at each."""
    views = [views] if isinstance(views, int) else list(views)
    last = {}
    try:
        for i, view in enumerate(views):
            more = "N goes to the next place, " if i + 1 < len(views) else ""
            answer = _rehearse_one(setup, view, camera, finder, clock, show, more, last)
            if answer != "next":
                break
    finally:
        if len(last) > 1:
            print("Summary, as last seen:")
            for view, lines in last.items():
                where = "start" if view == 0 else f"view {view}"
                print(f"  {where}: " + ("; ".join(lines) or "nothing to compare"))


def _rehearse_one(setup, view, camera, finder, clock, show, more, last):
    import cv2

    pose, steady, frames, printed = view_pose(setup, view), Steadiness(), deque(maxlen=31), None
    where = "the start" if view == 0 else f"view {view}"
    print(
        f"Stand the camera at {where}: ({pose.x:.2f}, {pose.y:.2f}) m, facing so the person is in the middle."
    )
    print(f"Nothing moves. {more}Q quits.\n", flush=True)
    while True:
        ok, frame = camera.read()
        if not ok or frame.shape[:2] != (setup.camera.height, setup.camera.width):
            print("The camera stopped or gives the wrong size.")
            return "quit"
        t = clock()
        frames.append(t)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        still = steady.update(t, gray)
        small = cv2.resize(
            frame, (640, round(640 * frame.shape[0] / frame.shape[1])), interpolation=cv2.INTER_AREA
        )
        chest = axis = None
        if finder(small) is not None:
            points = finder.points * (frame.shape[1] / 640)
            chest = tuple(points.mean(axis=0))
            axis = (tuple(points[:2].mean(axis=0)), tuple(points[2:].mean(axis=0)))
        if printed is None or t - printed >= 1.0:
            printed = t
            rate = (len(frames) - 1) / (frames[-1] - frames[0]) if frames[-1] > frames[0] else 0.0
            state = {True: "still", False: "moving", None: "too plain to tell if still"}[still]
            lines = rehearsal(setup, pose, find_markers(gray, setup.camera), chest, axis)
            last[view] = lines
            print("\n".join([f"{rate:.0f} pictures/s, picture {state}", *lines]) + "\n", flush=True)
        answer = show(frame) if show else None
        if answer in ("next", "quit"):
            return answer


def summarize(path, setup, raw=None):
    """A few lines about a saved run. `raw` is the setup file's contents, to compare with
    the one the log was made with."""
    rows = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            raise ValueError(f"line {n} of the log is damaged") from None
    made_with = rows.pop(0)["setup"] if rows and "setup" in rows[0] else None  # older logs have none
    if not rows:
        raise ValueError("the log is empty")
    path_m = sum(math.dist((a["x"], a["y"]), (b["x"], b["y"])) for a, b in pairwise(rows))
    nearest = min(setup.body.distance(r["x"], r["y"]) - 2 * r["within"] for r in rows)
    stops = {}
    for r in rows:
        if r["guard"] not in ("ok", "starting"):
            stops[r["guard"]] = stops.get(r["guard"], 0) + 1
    lines = (
        []
        if made_with is None or raw is None or made_with == raw
        else ["Note: this log was made with a different setup file, so distances may be wrong."]
    )
    lines += [
        f"{rows[-1]['t'] - rows[0]['t']:.0f} s, about {path_m:.1f} m driven, by its own estimate.",
        f"Nearest to the person, allowing for its uncertainty: {nearest:.2f} m.",
        f"Ended {rows[-1]['state']}: {rows[-1]['outcome']}.",
    ]
    if any(r["lost"] for r in rows):
        lines.append(f"Lost its position at {next(r['t'] for r in rows if r['lost']):.0f} s.")
    if any(r["problem"] for r in rows):
        lines.append(f"Stopped because {next(r['problem'] for r in rows if r['problem'])}.")
    for reason, count in sorted(stops.items(), key=lambda kv: -kv[1]):
        lines.append(f"Held by the guard, {reason}: {count / 2:.1f} s")
    shook = {}
    for r in rows:
        if r.get("shaking") and r["state"] == "settling":  # older logs have no "shaking"
            shook[r["view"]] = shook.get(r["view"], 0) + 1
    if shook:
        each = ", ".join(f"view {view + 1} {count / 2:.1f} s" for view, count in sorted(shook.items()))
        lines.append(f"Waited for the camera to stop shaking: {each}")
    volts = [r["battery"] for r in rows if r.get("battery")]  # older logs have none
    if volts:
        lines.append(
            f"Battery: {volts[0]:.1f} V at the start, {volts[-1]:.1f} V at the end, lowest {min(volts):.1f} V."
        )
    rates = [r["fps"] for r in rows if r.get("fps")]  # older logs have none
    if rates:
        slow = sum(rate < MIN_FPS for rate in rates)
        lines.append(f"Pictures a second: typically {np.median(rates):.0f}, lowest {min(rates):.0f}.")
        if slow:
            measuring = sum(
                bool(r.get("fps")) and r["fps"] < MIN_FPS and r["state"] == "measuring" for r in rows
            )
            lines.append(
                f"Below {MIN_FPS} a second for {slow / 2:.1f} s, {measuring / 2:.1f} s of it while measuring. "
                "Breathing may be missed then; see `python -m ostinato.speed`."
            )
    return "\n".join(lines)


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
    parser.add_argument("--map", type=Path, help="with --check: also draw the room from above, to a new PNG")
    parser.add_argument(
        "--log", type=Path, help="save positions and decisions, twice a second, without pictures"
    )
    parser.add_argument("--replay", type=Path, metavar="LOG", help="summarize a saved log and stop")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-prompt", action="store_true", help="never ask the person to raise a hand")
    parser.add_argument("--key", type=Path, help="key file shared with `python -m ostinato.remote`")
    parser.add_argument(
        "--rehearse",
        metavar="VIEW",
        help="no board: compare what the camera sees with the setup file, at view VIEW (0 is the "
        "start), or at each in turn with 'all'",
    )
    args = parser.parse_args(argv)
    from .remote import Listener, load_key

    if args.replay:
        try:
            print(summarize(args.replay, Setup.load(args.setup), json.loads(args.setup.read_text())))
        except (OSError, ValueError) as error:
            parser.exit(1, f"{error}\n")
        return
    if args.map and not args.check:
        parser.error("--map goes with --check")
    if args.check:
        try:
            setup = Setup.load(args.setup)
            errors, warnings = check_setup(setup)
            if args.map:
                from .room_map import draw

                with args.map.open("xb") as picture:  # never over an existing file
                    draw(setup, errors, warnings).save(picture, "PNG")
                print(f"Map saved to {args.map}")
        except (OSError, ValueError) as error:
            parser.exit(1, f"{error}\n")
        from .setup_file import visible_markers

        start = setup.start
        places = [("start", start.x, start.y, start.heading + math.radians(setup.camera.pan))]
        for i, bearing in enumerate(setup.views, 1):
            x, y = setup.body.around(bearing, RING)
            places.append((f"view {i}", x, y, math.atan2(setup.person[1] - y, setup.person[0] - x)))
        for name, x, y, facing in places:
            seen = sorted(visible_markers(setup, x, y, facing))
            print(
                f"{name} reads "
                + (
                    f"marker{'s' if len(seen) > 1 else ''} {', '.join(map(str, seen))}"
                    if seen
                    else "no marker"
                )
            )
        for line in errors:
            print(f"Problem: {line}")
        for line in warnings:
            print(f"Note: {line}")
        print("Fix the problems before driving." if errors else "No problems found.")
        if errors:
            parser.exit(1)
        return
    if args.rehearse is not None:
        try:
            setup = Setup.load(args.setup)
            if args.rehearse == "all":
                views = list(range(len(setup.views) + 1))
            elif args.rehearse.isdigit():
                views = [int(args.rehearse)]
            else:
                raise ValueError("--rehearse takes a view number, 0 for the start, or all")
            for view in views:
                view_pose(setup, view)
        except ValueError as error:
            parser.exit(1, f"{error}\n")
        camera = open_camera(args.camera)
        if camera is None:
            parser.exit(1, "Could not open the camera.\n")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, setup.camera.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, setup.camera.height)
        finder = PoseFinder()

        def show(frame):
            cv2.imshow("Ostinato rehearsal", frame)
            key = cv2.waitKey(1) & 0xFF
            return {ord("q"): "quit", ord("n"): "next"}.get(key)

        try:
            rehearse(setup, views, camera, finder, show=show)
        except KeyboardInterrupt:
            pass
        finally:
            finder.close()
            camera.release()
            cv2.destroyAllWindows()
        return
    if not args.port:
        parser.error("--port is needed, except with --check, --replay, or --rehearse")
    # Never drive on a setup file with problems; notes are only printed.
    try:
        errors, notes = check_setup(Setup.load(args.setup))
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    if errors:
        parser.exit(
            1, "".join(f"Problem: {e}\n" for e in errors) + "Not driving. Fix the setup file first.\n"
        )
    for note in notes:
        print(f"Note: {note}")
    log = None
    if args.log:
        try:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            log = args.log.open("x")
        except OSError as error:
            parser.exit(1, f"{error}\n")
    listener = None
    try:
        setup = Setup.load(args.setup)
        if args.key:
            listener = Listener(load_key(args.key))
        port = SerialPort(args.port)
        if log:
            # First, what the run was set up with, so a replay can tell if it differs.
            log.write(json.dumps({"setup": json.loads(args.setup.read_text())}) + "\n")
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
    robot = Robot(setup, port, ask=not args.no_prompt)
    robot.guard.emergency_stop()
    title, start, was, logged, steady = "Ostinato rover", time.monotonic(), None, -1.0, None
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
            robot.picture(t)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            measuring = robot.mission.state is State.MEASURING
            reading = responded = None
            if measuring:
                small = cv2.resize(
                    frame, (640, round(640 * frame.shape[0] / frame.shape[1])), interpolation=cv2.INTER_AREA
                )
                reading = monitor.update(small, t)
                responded = check.update(t, finder.body)
            shaking = False
            if robot.mission.state is State.SETTLING:
                steady = steady or Steadiness()  # fresh at each view
                shaking = steady.update(t, gray) is False
            else:
                steady = None
            chest = axis = None
            if measuring and finder.points is not None:
                points = finder.points * (frame.shape[1] / 640)
                chest = tuple(points.mean(axis=0))
                axis = (tuple(points[:2].mean(axis=0)), tuple(points[2:].mean(axis=0)))
            link = True
            if listener:
                listener.poll(t, robot.guard, robot.status())  # tells the guard itself
                link = False
            robot.step(
                t,
                link,
                find_markers(gray, setup.camera) if robot.wants_markers(t) else (),
                reading,
                bool(responded),
                check.active,
                chest,
                shaking,
                axis,
            )
            if measuring and robot.mission.state is not State.MEASURING:
                monitor.reset()  # never carry readings from one view to the next
                check.cancel()
            elif robot.wants_ask and not check.active:
                check.ask(t)
            if log and t - logged >= 0.5:
                log.write(json.dumps(robot.record(t)) + "\n")
                logged = t
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
        if log:
            log.close()
        cv2.destroyAllWindows()
    print(f"Outcome: {robot.mission.outcome}")
    if args.log:
        try:
            print("\n" + summarize(args.log, setup, json.loads(args.setup.read_text())))
            print(f"Saved {args.log}. Run --replay on it to see this again.")
        except (OSError, ValueError) as error:  # a run too short to log is not a failure
            print(f"No summary: {error}")


if __name__ == "__main__":
    main()
