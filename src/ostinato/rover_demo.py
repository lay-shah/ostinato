"""Watch the simulated rover find a view and measure: `python -m ostinato.rover_demo`.

The map, the driving, and the person are drawn. The drive limits, chest tracking,
breathing estimate, and decisions are the real code. Nothing here is a robot.
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
from .rover import KEEP_OUT, RING, STOP, Guard, Mission, Pose, SimulatedBase, State
from .ui import AMBER, BG, BLUE, CARD, DIM, GREEN, INK, LINE, font

W, H = 1080, 600
CAMERA = (24, 100, 544, 490)
MAP = (564, 100, 1056, 592)
SCALE = 82.0  # map pixels per meter
DT = 1 / 15
BEARINGS = [0.5, 2.6, -2.7]  # beside the person, not at the head or feet
LABEL = "SIMULATED: drawn person, map, and driving. No robot exists yet."


class RoverDemo:
    """One run. Call `step` once per frame."""

    def __init__(self, seed=1):
        self.scene = Demo(lying=True)
        self.monitor = Monitor(ChestTracker(self.scene))
        self.base = SimulatedBase(Pose(2.6, -0.6, 2.4), (0.0, 0.0), np.random.default_rng(seed))
        self.guard, self.mission = Guard(), Mission((0.0, 0.0), list(BEARINGS))
        self.t, self.drive, self.trail = 0.0, STOP, []
        self.frame = self.reading = None

    def step(self):
        self.t += DT
        self.base.tell(self.guard, self.t)
        measuring = self.mission.state is State.MEASURING
        self.reading = None
        if measuring:
            # Debris hides the chest from the first viewpoint only.
            self.scene.view = 1 if self.mission.view == 0 else 2
            self.frame = self.scene.draw(self.t)
            self.reading = self.monitor.update(self.frame, self.t)
        else:
            self.frame = None
        want = self.mission.update(self.t, self.base.pose, self.drive != STOP, self.reading)
        if self.mission.state is not State.MEASURING and measuring:
            self.monitor.reset()  # readings from one view never carry over
        self.drive = self.guard.allow(self.t, want)
        self.base.step(self.drive, DT)
        self.trail.append((self.base.pose.x, self.base.pose.y))

    @property
    def done(self):
        return self.mission.state is State.DONE

    def status(self):
        """Return (headline, detail, color)."""
        view = f"View {self.mission.view + 1} of {len(BEARINGS)}"
        if self.done:
            return f"{self.mission.outcome.capitalize()} (simulated)", "The rover stays where it stopped.", GREEN if self.measured else AMBER
        if self.mission.state is State.MEASURING:
            r = self.reading
            now = f"{r.bpm:.0f} breaths/min" if r and r.bpm is not None else f"unable to assess: {r.reason}" if r else "starting"
            return f"{view}: stopped and measuring", now.capitalize(), BLUE
        if self.mission.state is State.DRIVING:
            why = "Moving to a better view" if self.mission.view else "Driving to the first view"
            return f"{view}: driving at {abs(self.drive.speed):.1f} m/s", f"{why}. Not measuring while moving.", INK
        return f"{view}: {self.mission.state.value}", "Waiting until it is still before measuring.", INK

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
        d.text((24, 506), f"{self.t:.0f} s", font=font(20), fill=INK)
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
        d.text((x, MAP[1] + 14), "Map from above. Orange: 1 m keep-out", font=font(14), fill=DIM, anchor="ma")
        for i, bearing in enumerate(BEARINGS):
            vx, vy = self._at(RING * math.cos(bearing), RING * math.sin(bearing))
            fill = BLUE if i == self.mission.view else DIM
            d.ellipse((vx - 11, vy - 11, vx + 11, vy + 11), outline=fill, width=2)
            d.text((vx, vy), str(i + 1), font=font(13), fill=fill, anchor="mm")
        if len(self.trail) > 1:
            d.line([self._at(*p) for p in self.trail], fill=DIM, width=2)
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
