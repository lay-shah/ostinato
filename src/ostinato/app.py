"""Live breathing monitor: `ostinato`. Video is never saved."""

import argparse
import time
from collections import deque

import cv2
import numpy as np

from .breathing import Reading, estimate
from .chest import ChestTracker, PoseFinder
from .decision import Action, Assessor

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


def decision_text(decision):
    """Return (headline, detail). The headline says what a person should do."""
    if decision.action is Action.GET_HELP:
        when = f" at {decision.completed_at_s:.0f}s"
        return "GET HELP: a person must check", f"Past result{when}: {decision.status.value} ({decision.reason})"
    if decision.action is Action.REPORT:
        result = decision.status.value if decision.bpm is None else f"{decision.bpm:.1f} breaths/min"
        return f"PAST RESULT at {decision.completed_at_s:.0f}s: {result}", f"{decision.reason}; R to reassess"
    if decision.action is Action.MOVE:
        return "MOVE the camera, then press M", decision.reason
    return f"Decision: {decision.action.value}", decision.reason


def put(frame, text, y, scale, color, thickness=1):
    """Write one line, shrunk if needed to stay inside the frame."""
    room = frame.shape[1] - 24
    (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    if width > room:
        scale *= (room - 4) / width
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw(frame, monitor, decision):
    reading = monitor.reading
    if monitor.tracker.box:
        x0, y0, x1, y1, _ = monitor.tracker.box
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 200, 0), 2)
    if reading.bpm is not None:
        text = f"{reading.bpm:.0f} breaths/min (signal score {reading.confidence:.2f})"
        color = (80, 220, 80)
    else:
        text, color = f"Unable to assess: {reading.reason}", (0, 190, 255)
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 122), (25, 25, 25), -1)
    put(frame, f"Live: {text}", 30, 0.7, color, 2)
    note = "Research prototype. Not a medical device. M: moved  R: restart  Q: quit"
    headline, detail = decision_text(decision)
    urgent = decision.action in (Action.GET_HELP, Action.MOVE)
    put(frame, headline, 60, 0.65, (0, 190, 255) if urgent else (255, 255, 255), 2 if urgent else 1)
    put(frame, detail, 86, 0.5, (255, 255, 255))
    put(frame, note, 110, 0.5, (200, 200, 200))
    if len(monitor.samples) < 2:
        return
    t, y = map(np.array, zip(*monitor.samples))
    good = np.isfinite(y)
    if good.sum() < 2:
        return
    lo, hi = np.percentile(y[good], [2, 98])
    px = w * (t - t[0]) / WINDOW
    py = h - 15 - 70 * np.clip((y - lo) / max(hi - lo, 1e-6), 0, 1)
    for i in range(1, len(t)):
        if good[i - 1] and good[i]:
            a, b = (int(px[i - 1]), int(py[i - 1])), (int(px[i]), int(py[i]))
            cv2.line(frame, a, b, (80, 220, 80), 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
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
    start, n = time.monotonic(), 0
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
            decision = assessor.update(t, monitor.update(frame, t))
            draw(frame, monitor, decision)
            cv2.imshow("Ostinato", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m") and assessor.decision.action is Action.MOVE:
                monitor.reset()
                assessor.arrived(t)
            elif key == ord("r"):
                monitor.reset()
                assessor = Assessor(t, EXTRA_VIEWS)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        finder.close()
        cv2.destroyAllWindows()
    r = monitor.reading
    d = assessor.decision
    print(f"Decision: {d.action.value}, {d.status.value} ({d.reason})")
    print(f"Last reading: {r.status}" + (f", {r.bpm:.1f} breaths/min" if r.bpm is not None else f" ({r.reason})"))


if __name__ == "__main__":
    main()
