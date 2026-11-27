"""Watch the simulated rover find a view and measure: `python -m ostinato.rover_demo`.

The map, the wheels, the motor board, the markers' sightings, and the person are
drawn. The robot program, position tracking, drive limits, chest tracking, breathing
estimate, and decisions are the real code. Nothing here is a robot.
"""

import argparse
import math
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .app import Monitor
from .chest import ChestTracker
from .decision import Action
from .demo import Demo
from .localize import Sighting
from .motors import seal
from .robot import Robot, Setup
from .rover import KEEP_OUT, RING, STOP, Drive, Pose, SimulatedBase, State, wrap
from .shake import Steadiness
from .ui import AMBER, BG, BLUE, CARD, DIM, GREEN, INK, LINE, font

W, H = 1080, 600
CAMERA = (24, 100, 544, 490)
MAP = (564, 100, 1056, 592)
SCALE = 82.0  # map pixels per meter
DT = 1 / 15
BEARINGS = [0.5, 2.6, -2.7]  # beside the person, not at the head or feet
MARKERS = {1: (0.0, 2.4), 2: (2.6, 0.0), 3: (0.0, -2.4), 4: (-2.6, 0.0)}
TRACK = 0.27
TICKS = 2000  # per meter
ASK_SECONDS = 10.0
WOBBLE = (8.0, 3.0, 0.5)  # px across the drawn picture, Hz, and seconds to die down, after stopping
LABEL = "SIMULATED: drawn person, map, and driving. No robot exists yet."


class Wheels:
    """A drawn motor board and wheels, on the port the robot program talks to.
    It obeys only checked commands, and stops when they stop coming."""

    def __init__(self, base, rng):
        self.base, self.rng = base, rng
        self.speeds, self.ticks, self.armed, self.heard = (0, 0), [0.0, 0.0], False, -1.0
        self.t, self._out = 0.0, b""

    def write(self, data):
        body = data.rstrip(b"\n")
        if seal(body[:-5].decode())[:-1] != body:
            return
        _, _, left, right = body[:-5].decode().split(",")
        left, right = int(left), int(right)
        if self.armed or (left == right == 0):
            self.armed, self.speeds, self.heard = True, (left, right), self.t

    def read(self):
        data, self._out = self._out, b""
        return data

    def run(self, t, dt):
        self.t = t
        if t - self.heard > 0.5:
            self.armed, self.speeds = False, (0, 0)
        left, right = (v / 1000 * (1 + self.rng.normal(0, 0.03)) for v in self.speeds)
        self.ticks = [a + b * dt * TICKS for a, b in zip(self.ticks, (left, right))]
        self.base.step(Drive((left + right) / 2, (right - left) / TRACK), dt)
        ms = round(t * 1000)
        self._out += seal(f"T,{ms},{int(self.ticks[0])},{int(self.ticks[1])},3000,3000,{int(self.armed)}")
        return self.speeds != (0, 0)


class RoverDemo:
    """One run. Call `step` once per frame."""

    def __init__(self, seed=1):
        rng = np.random.default_rng(seed)
        self.scene = Demo(lying=True)
        self.monitor = Monitor(ChestTracker(self.scene))
        self.base = SimulatedBase(Pose(2.6, -0.6, 2.4), (0.0, 0.0), rng, slip=0.0)
        self.wheels, self.rng = Wheels(self.base, rng), rng
        # Where a person placed it, a few centimeters out.
        setup = Setup(TRACK, TICKS, Pose(2.63, -0.57, 2.45), (0.0, 0.0), list(BEARINGS), dict(MARKERS))
        self.robot = Robot(setup, self.wheels, ask=True)
        self.asked_at = None  # the drawn person never raises a hand
        self.mission = self.robot.mission
        self.t, self.drive, self.trail, self.seen = 0.0, STOP, [], []
        self.frame = self.reading = None
        self.steady, self.stopped_at, self.shaking = None, None, False

    def sightings(self):
        """Markers in front of the camera, measured a little wrongly."""
        p, found = self.base.pose, []
        for number, (x, y) in MARKERS.items():
            bearing = wrap(math.atan2(y - p.y, x - p.x) - p.heading)
            if abs(wrap(bearing - math.pi / 2)) < 0.6:
                distance = math.dist((x, y), (p.x, p.y)) * (1 + self.rng.normal(0, 0.02))
                found.append(Sighting(number, distance, bearing + self.rng.normal(0, 0.01)))
        return found

    def step(self):
        self.t += DT
        self.wheels.run(self.t, DT)
        measuring = self.mission.state is State.MEASURING
        self.reading, self.shaking = None, False
        if self.mission.state is State.SETTLING:
            # The mast swings after the stop. The real steadiness check watches the picture.
            if self.steady is None:
                self.steady, self.stopped_at = Steadiness(), self.t
            self.scene.view = 1 if self.mission.view == 0 else 2
            size, hertz, decay = WOBBLE
            s = self.t - self.stopped_at
            dx = size * math.exp(-s / decay) * math.cos(2 * math.pi * hertz * s)
            frame = self.scene.draw(self.t)
            m = np.float32([[1, 0, dx], [0, 1, 0.3 * dx]])
            self.frame = cv2.warpAffine(frame, m, frame.shape[1::-1], borderMode=cv2.BORDER_REFLECT)
            self.shaking = self.steady.update(self.t, cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)) is False
        elif measuring:
            # Debris hides the chest from the first viewpoint only.
            self.scene.view = 1 if self.mission.view == 0 else 2
            self.frame = self.scene.draw(self.t)
            self.reading = self.monitor.update(self.frame, self.t)
        else:
            self.frame = None
        if self.mission.state is not State.SETTLING:
            self.steady = None
        self.seen = self.sightings() if round(self.t / DT) % 5 == 0 else []
        self.drive = self.robot.step(
            self.t, sightings=self.seen, reading=self.reading, asking=self.asking, shaking=self.shaking
        )
        if self.mission.state is not State.MEASURING and measuring:
            self.monitor.reset()  # readings from one view never carry over
            self.asked_at = None
        elif self.robot.wants_ask:
            self.asked_at = self.t
        self.trail.append((self.base.pose.x, self.base.pose.y))

    @property
    def done(self):
        return self.mission.state is State.DONE

    @property
    def asking(self):
        """Asking lasts 10 s, as in the app. Silent here."""
        return self.asked_at is not None and self.t - self.asked_at < ASK_SECONDS

    def status(self):
        """Return (headline, detail, color)."""
        view = f"View {self.mission.view + 1} of {len(BEARINGS)}"
        if self.done:
            return (
                f"{self.mission.outcome.capitalize()} (simulated)",
                "The rover stays where it stopped.",
                GREEN if self.measured else AMBER,
            )
        if self.mission.state is State.MEASURING and self.asking:
            return (
                f"{view}: asking the person to raise a hand",
                "Spoken on the rover; silent here. No hand proves nothing.",
                AMBER,
            )
        if self.mission.state is State.MEASURING:
            r = self.reading
            now = (
                f"{r.bpm:.0f} breaths/min"
                if r and r.bpm is not None
                else f"unable to assess: {r.reason}"
                if r
                else "starting"
            )
            return f"{view}: stopped and measuring", now.capitalize(), BLUE
        if self.mission.state is State.DRIVING:
            why = "Moving to a better view" if self.mission.view else "Driving to the first view"
            return (
                f"{view}: driving at {abs(self.drive.speed):.1f} m/s",
                f"{why}. Not measuring while moving.",
                INK,
            )
        if self.mission.state is State.SETTLING and self.shaking:
            return f"{view}: stopped, camera still shaking", "Measuring once the picture is still.", AMBER
        return f"{view}: {self.mission.state.value}", "Waiting until it is still before measuring.", INK

    def position(self):
        place = self.robot.localizer.uncertainty[0]
        if self.robot.localizer.lost:
            return "Position lost: a person must place it again"
        text = f"Position known within about {200 * place:.0f} cm"
        if not self.robot.localizer.good:
            text += ": turning to find a marker"
        return text

    @property
    def measured(self):
        a = self.mission.assessor
        return a is not None and a.decision.action is Action.REPORT

    def render(self):
        """The screen, in BGR."""
        image = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(image)
        headline, detail, color = self.status()
        d.rounded_rectangle((24, 14, W - 24, 86), 12, fill=CARD, outline=color, width=2)
        d.text((44, 20), headline, font=font(26), fill=color)
        d.text((44, 56), detail, font=font(16), fill=INK)
        self._map(d)
        x0, y0, x1, y1 = CAMERA
        d.rounded_rectangle(CAMERA, 12, fill=CARD, outline=LINE)
        if self.frame is None:
            text = "Camera off while driving" if not self.done else "Finished. Press R to run again."
            d.text(((x0 + x1) / 2, (y0 + y1) / 2), text, font=font(20), fill=DIM, anchor="mm")
        d.text((24, 506), f"{self.t:.0f} s     {self.position()}", font=font(18), fill=INK)
        d.text((24, 536), LABEL, font=font(15), fill=AMBER)
        d.text((24, 562), "R  Run again     Q  Quit", font=font(15), fill=DIM)
        screen = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if self.frame is not None:
            view = self.frame.copy()
            if self.monitor.tracker.box:
                a, b, c, e, _ = self.monitor.tracker.box
                cv2.rectangle(view, (a, b), (c, e), GREEN[::-1], 2)
            screen[y0:y1, x0:x1] = cv2.resize(view, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        return screen

    def _at(self, x, y):
        return (MAP[0] + MAP[2]) / 2 + x * SCALE, (MAP[1] + MAP[3]) / 2 - y * SCALE

    def _circle(self, d, radius, **style):
        x, y = self._at(0, 0)
        d.ellipse((x - radius * SCALE, y - radius * SCALE, x + radius * SCALE, y + radius * SCALE), **style)

    def _map(self, d):
        d.rounded_rectangle(MAP, 12, fill=CARD, outline=LINE)
        self._circle(d, KEEP_OUT, outline=AMBER, width=2)
        self._circle(d, RING, outline=LINE)
        x, y = self._at(0, 0)
        d.ellipse((x - 14, y - 44, x + 14, y - 18), fill=(186, 140, 104))
        d.rounded_rectangle((x - 18, y - 20, x + 18, y + 48), 8, fill=(84, 138, 190))
        d.text(
            ((MAP[0] + MAP[2]) / 2, MAP[3] - 10),
            "From above. Orange: 1 m keep-out, and where it thinks it is",
            font=font(14),
            fill=DIM,
            anchor="md",
        )
        for i, bearing in enumerate(BEARINGS):
            vx, vy = self._at(RING * math.cos(bearing), RING * math.sin(bearing))
            fill = BLUE if i == self.mission.view else DIM
            d.ellipse((vx - 11, vy - 11, vx + 11, vy + 11), outline=fill, width=2)
            d.text((vx, vy), str(i + 1), font=font(13), fill=fill, anchor="mm")
        seen = {s.marker for s in self.seen}
        for number, (mx, my) in MARKERS.items():
            x, y = self._at(mx, my)
            d.rectangle((x - 9, y - 9, x + 9, y + 9), fill=BLUE if number in seen else INK)
            d.text((x, y), str(number), font=font(12), fill=CARD, anchor="mm")
        if len(self.trail) > 1:
            d.line([self._at(*p) for p in self.trail], fill=DIM, width=2)
        e, spread = self.robot.localizer.pose, 2 * self.robot.localizer.uncertainty[0]
        ex, ey = self._at(e.x, e.y)
        r = max(4.0, spread * SCALE)
        d.ellipse((ex - r, ey - r, ex + r, ey + r), outline=AMBER, width=2)
        p = self.base.pose
        c, s = math.cos(p.heading), math.sin(p.heading)
        corners = [(0.18, 0.0), (-0.14, 0.12), (-0.14, -0.12)]
        d.polygon([self._at(p.x + a * c - b * s, p.y + a * s + b * c) for a, b in corners], fill=GREEN)
        if self.frame is not None:  # the camera looks to the left
            for spread in (-0.45, 0.45):
                a = p.heading + math.pi / 2 + spread
                d.line([self._at(p.x, p.y), self._at(p.x + math.cos(a), p.y + math.sin(a))], fill=BLUE)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    title = "Ostinato rover simulation"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(title, W, H)
    demo, due = RoverDemo(), time.monotonic()
    try:
        while True:
            if not demo.done:
                demo.step()
            cv2.imshow(title, demo.render())
            due = max(due + DT, time.monotonic())
            key = cv2.waitKey(max(1, int((due - time.monotonic()) * 1000))) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("r"):
                demo = RoverDemo()
            if time.monotonic() < due:
                time.sleep(max(0.0, due - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
