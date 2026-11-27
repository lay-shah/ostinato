"""How long the camera shakes after the robot stops: `python -m ostinato.shake`.

Point the mounted camera at a still, detailed scene with nobody in it. When told, push
the robot and stop it. No video is saved.

The rover also uses `Steadiness` at each view, so it measures only once the picture
is still, whatever the mast does.
"""

import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np

WIDTH = 1280  # results are in pixels of a picture this wide
STEADY = 0.3  # px; chest motion is about 1.3 px at the planned camera position
TAIL = 2.0  # seconds at the end that must be steady, to compare against
GOAL = 2.0  # seconds the rover waits before measuring


@dataclass(frozen=True)
class Result:
    settled: float | None  # seconds from the first large jolt until it stays steady
    jolt: float  # px, the largest movement between two pictures
    floor: float  # px, how steady it is at the end
    hertz: float | None  # how fast it wobbles, when that is clear
    reason: str


class ShiftTracker:
    """How far the whole picture has moved, in pixels of a picture WIDTH wide."""

    def __init__(self):
        self._last = self._points = None
        self.place = np.zeros(2)

    def update(self, gray):
        """Return the total (x, y), or None when too little can be followed."""
        scale = WIDTH / gray.shape[1]
        if self._last is not None and (self._points is None or len(self._points) < 60):
            self._points = cv2.goodFeaturesToTrack(self._last, 300, 0.01, 8)
        step = None
        if self._last is not None and self._points is not None and len(self._points) >= 20:
            new, ok, _ = cv2.calcOpticalFlowPyrLK(self._last, gray, self._points, None, winSize=(21, 21))
            back, ok2, _ = cv2.calcOpticalFlowPyrLK(gray, self._last, new, None, winSize=(21, 21))
            good = (ok.ravel() == 1) & (ok2.ravel() == 1)
            good &= np.linalg.norm((back - self._points).reshape(-1, 2), axis=1) < 0.3
            if good.sum() >= 20:
                step = np.median((new - self._points).reshape(-1, 2)[good], axis=0) * scale
            self._points = new[good]
        first, self._last = self._last is None, gray
        if first:
            return self.place.copy()
        if step is None:
            self._points = None
            return None
        self.place = self.place + step
        return self.place.copy()


class Steadiness:
    """Whether the picture is still enough to measure, while the rover waits at a view.

    Follows the whole picture, at half size, and calls it steady when it has stayed
    within LIMIT pixels for the last WINDOW seconds. That is twice STEADY, because the
    picture's own grain moves the estimate by a few tenths of a pixel. Returns None when the picture has
    too little detail to follow, so the rover falls back to waiting SETTLE seconds."""

    WINDOW = 0.5  # s
    LIMIT = 2 * STEADY  # px

    def __init__(self):
        self.tracker, self.recent = ShiftTracker(), []

    def update(self, t, gray):
        if gray.shape[1] > 640:
            gray = cv2.resize(
                gray, (640, round(640 * gray.shape[0] / gray.shape[1])), interpolation=cv2.INTER_AREA
            )
        place = self.tracker.update(gray)
        if place is None:
            self.recent = []
            return None
        self.recent = [(s, p) for s, p in self.recent if s >= t - self.WINDOW] + [(t, place)]
        if self.recent[-1][0] - self.recent[0][0] < self.WINDOW * 0.8:
            return False  # not watched long enough to say
        points = np.array([p for _, p in self.recent])
        return bool(np.linalg.norm(points - np.median(points, axis=0), axis=1).max() <= self.LIMIT)


def analyze(times, places):
    """Settling time from how far the picture moved. `places` are (x, y) in px."""
    t, p = np.asarray(times, float), np.asarray(places, float)
    if len(t) < 30 or t[-1] - t[0] < 2 * TAIL or not (np.isfinite(t).all() and np.isfinite(p).all()):
        return Result(None, 0.0, 0.0, None, "not enough pictures")
    if (np.diff(t) <= 0).any() or np.diff(t).max() > 0.5:
        return Result(None, 0.0, 0.0, None, "the camera stalled")
    steps = np.linalg.norm(np.diff(p, axis=0), axis=1)
    jolt = float(steps.max())
    # The first large step, not the largest: between pictures a fast wobble can look
    # largest late, which would make it seem to settle sooner than it did.
    at = int(np.flatnonzero(steps >= jolt / 2)[0]) + 1
    end = t >= t[-1] - TAIL
    rest = np.median(p[end], axis=0)
    away = np.linalg.norm(p - rest, axis=1)
    floor = float(np.percentile(away[end], 95))
    if jolt < 4 * STEADY:
        return Result(None, jolt, floor, None, "no jolt was seen: push it harder")
    if t[at] > t[-1] - TAIL:
        return Result(None, jolt, floor, None, "the jolt came too late")
    if floor > STEADY:
        return Result(None, jolt, floor, wobble(t[end], p[end]), "it was still moving at the end")
    last = int(np.flatnonzero(away > STEADY).max(initial=at))
    # A wobble passes through its rest position, so the last time it was away counts.
    settled = max(0.0, float(t[min(last + 1, len(t) - 1)] - t[at]))
    return Result(settled, jolt, floor, wobble(t[at:], p[at:]), "ok")


def wobble(t, p):
    """The main frequency, when one stands out. None otherwise."""
    if len(t) < 16:
        return None
    grid = np.linspace(t[0], t[-1], len(t))
    axis = int(np.argmax(np.ptp(p, axis=0)))
    y = np.interp(grid, t, p[:, axis])
    y = (y - np.median(y[len(y) // 2 :])) * np.hanning(2 * len(y))[len(y) :]
    power = np.abs(np.fft.rfft(y - y.mean(), 4096)) ** 2
    hertz = np.fft.rfftfreq(4096, grid[1] - grid[0])
    # Faster than half the picture rate cannot be told apart from slower.
    band = (hertz > 0.5) & (hertz < 0.45 / (grid[1] - grid[0]))
    if not band.any() or power[band].max() < 10 * np.median(power[band]):
        return None
    best = int(np.argmax(np.where(band, power, 0)))
    return float(hertz[best])


def describe(result):
    if result.settled is None:
        return f"No result: {result.reason} (jolt {result.jolt:.1f} px, end {result.floor:.2f} px)."
    text = f"Steady {result.settled:.1f} s after the jolt ({result.jolt:.1f} px; {result.floor:.2f} px at the end)."
    if result.hertz:
        text += f" It wobbles about {result.hertz:.1f} times a second."
    goal = "Within" if result.settled <= GOAL else "Longer than"
    return f"{text} {goal} the {GOAL:g} s the rover waits."


def capture(camera, seconds=8.0):
    video = cv2.VideoCapture(camera)
    try:
        if not video.isOpened():
            raise ValueError("the camera did not open")
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(20):  # let the exposure settle
            video.read()
        print("Push the robot and stop it now, then keep your hands off.")
        tracker, times, places, start = ShiftTracker(), [], [], time.monotonic()
        while time.monotonic() - start < seconds:
            ok, frame = video.read()
            if not ok:
                raise ValueError("the camera stopped")
            now = time.monotonic() - start
            place = tracker.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if place is None:
                raise ValueError(
                    "too little to follow: use a brighter, more detailed scene or a gentler push"
                )
            times.append(now)
            places.append(place)
        return times, places
    finally:
        video.release()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--times", type=int, default=3, help="how many pushes (default: 3)")
    args = parser.parse_args(argv)
    if args.times < 1:
        parser.error("--times must be at least 1")
    try:
        results = []
        for n in range(args.times):
            print(f"\nPush {n + 1} of {args.times}. Starting the camera.")
            results.append(analyze(*capture(args.camera)))
            print(describe(results[-1]))
        worst = [r.settled for r in results if r.settled is not None]
        if len(worst) == len(results):
            print(f"\nSlowest: {max(worst):.1f} s.")
        else:
            print("\nAt least one push gave no result, so this is not a pass.")
    except KeyboardInterrupt:
        parser.exit(1, "\nStopped.\n")
    except (ValueError, cv2.error) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
