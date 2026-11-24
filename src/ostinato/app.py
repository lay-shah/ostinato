"""Live breathing monitor: `ostinato`. Video is never saved."""

import argparse
import time
from collections import deque

import cv2
import numpy as np

from .breathing import Reading, estimate
from .chest import ChestTracker, PoseFinder
from .decision import Action, Assessor
from .response import ResponseCheck, Speaker
from .ui import H, W, compose

TITLE = "Ostinato"
WINDOW = 30.0  # seconds of signal used for each reading
EXTRA_VIEWS = 2  # times the app may ask for a different camera position


class Monitor:
    """Rolling window of chest motion."""

    def __init__(self, tracker):
        self.tracker = tracker
        self.samples = deque()
        self.reading = Reading(None, 0.0, "need more data")
        self.tracking = "starting"
        self._last_estimate = -np.inf

    def reset(self, reason="need more data"):
        self.samples.clear()
        self.tracker.reset()
        self.reading = Reading(None, 0.0, reason)
        self._last_estimate = -np.inf

    def update(self, frame, t):
        if self.samples and t - self.samples[-1][0] > 1.0:
            self.reset("stream interrupted")  # old and new do not join up
        value, self.tracking = self.tracker.update(frame)
        self.samples.append((t, value))
        while t - self.samples[0][0] > WINDOW:
            self.samples.popleft()
        # Never keep showing an old rate once tracking is lost.
        if not np.isfinite(value):
            self.reading = Reading(None, 0.0, self.tracking)
        elif t - self._last_estimate >= 1.0 or self.reading.reason == self.tracking:
            self._last_estimate = t
            self.reading = estimate(*zip(*self.samples))
        return self.reading


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-prompt", action="store_true", help="do not ask for a response on its own")
    parser.add_argument("--video", help="analyze a local video instead of the camera")
    args = parser.parse_args(argv)

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        parser.exit(1, "Could not open the camera or video. Check camera permission.\n")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.video and not fps > 0:
        parser.exit(1, "The video has no usable frame rate.\n")
    finder = PoseFinder()
    monitor = Monitor(ChestTracker(finder))
    assessor = Assessor(0.0, EXTRA_VIEWS)
    check = ResponseCheck(Speaker())
    asked = False  # at this viewpoint
    start, n = time.monotonic(), 0
    cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(TITLE, W, H)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] > 640:
                scale = 640 / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            t = n / fps if args.video else time.monotonic() - start
            n += 1
            reading = monitor.update(frame, t)
            responded = check.update(t, finder.body)
            decision = assessor.update(t, reading, responded)
            if decision.final:
                if check.active:
                    check.cancel()
            elif decision.action is Action.MOVE and not (asked or args.video or args.no_prompt):
                asked = True
                check.ask(t)
            view = EXTRA_VIEWS + 1 - assessor.views_left - (decision.action is Action.MOVE)
            screen, _ = compose(frame, monitor, decision, check, view, EXTRA_VIEWS + 1, WINDOW)
            cv2.imshow(TITLE, screen)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("v") and not (decision.final or args.video):
                asked = True
                check.ask(t)
            elif key == ord("m") and assessor.decision.action is Action.MOVE:
                monitor.reset()
                check.cancel()
                asked = False
                assessor.arrived(t)
            elif key == ord("r"):
                monitor.reset()
                check.cancel()
                asked = False
                assessor = Assessor(t, EXTRA_VIEWS)
    except KeyboardInterrupt:
        pass
    finally:
        check.cancel()
        cap.release()
        finder.close()
        cv2.destroyAllWindows()
    r = monitor.reading
    d = assessor.decision
    print(f"Decision: {d.action.value}, {d.status.value} ({d.reason})")
    print(f"Last reading: {r.status}" + (f", {r.bpm:.1f} breaths/min" if r.bpm is not None else f" ({r.reason})"))


if __name__ == "__main__":
    main()
