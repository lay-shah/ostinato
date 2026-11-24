"""Ask a person to raise a hand, and watch for it.

A response is only ever positive evidence. Seeing none proves nothing: the person
may be deaf, trapped, hurt, out of view, or not understand the language.
"""

import math
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

PROMPT = "If you can hear me, please raise one hand."
MIN_VISIBILITY = 0.7
RAISED = 0.1  # wrist past the shoulder line toward the head, in shoulder widths
LOWERED = -0.2
DOWN_SECONDS = 0.5  # a hand must first be seen down for this long
HOLD_SECONDS = 0.6  # and then be held up for this long
LISTEN_SECONDS = 10.0
MAX_GAP = 1.0  # seconds without a view of the arm before its history is dropped
MAX_JUMP = 0.5  # body shift between frames, in shoulder widths: maybe someone else


@dataclass(frozen=True)
class Body:
    """Shoulder center and width in pixels, and each wrist's height toward the head."""

    center: tuple[float, float]
    width: float
    left: float | None  # shoulder widths; None if the wrist cannot be seen
    right: float | None


def body_from_landmarks(points, visibility, size):
    """`points` and `visibility` follow MediaPipe's 33 pose landmarks, in pixels.
    `size` is the frame (width, height).

    Height is measured from the shoulders toward the nose, so it does not depend on
    the person being upright. Returns None if the direction cannot be measured.
    """
    if min(visibility[i] for i in (0, 11, 12)) < MIN_VISIBILITY:
        return None
    nose, left, right = (np.asarray(points[i], float) for i in (0, 11, 12))
    center, across = (left + right) / 2, left - right
    width = float(np.linalg.norm(across))
    if width < 60:
        return None
    up = np.array([-across[1], across[0]]) / width
    toward_head = float((nose - center) @ up)
    if abs(toward_head) < 0.15 * width:
        return None
    up = up * math.copysign(1, toward_head)

    def height(i):
        x, y = points[i]
        value = float((np.array([x, y], float) - center) @ up) / width
        inside = 0 <= x < size[0] and 0 <= y < size[1]
        if inside:
            return value if visibility[i] >= MIN_VISIBILITY else None
        # A resting hand is often just out of frame. The model's guess is enough to
        # call it down, but a raised hand must really be seen.
        return value if value <= LOWERED else None

    return Body((float(center[0]), float(center[1])), width, height(15), height(16))


class _Arm:
    """Looks for one hand seen down, then raised and held."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._last = self._down_since = self._up_since = None
        self._was_down = False

    @property
    def raised_since(self):
        return self._up_since

    def update(self, t, height):
        """Return how long the hand has been held up after being down."""
        if self._last is not None and t - self._last > MAX_GAP:
            self.reset()
        if self._up_since is not None and (height is None or height < RAISED):
            # An interrupted raise must be followed by a fresh observed lower.
            self._was_down = False
        if height is None:
            self._up_since = self._down_since = None  # a hold must be seen throughout
            return 0.0
        self._last = t
        if height >= RAISED:
            self._down_since = None
            if not self._was_down:
                return 0.0
            if self._up_since is None:
                self._up_since = t
            return t - self._up_since
        self._up_since = None
        if height <= LOWERED:
            if self._down_since is None:
                self._down_since = t
            if t - self._down_since >= DOWN_SECONDS:
                self._was_down = True
        else:
            self._down_since = None
        return 0.0


class Speaker:
    """Speaks with a voice already on this computer. Nothing is sent anywhere."""

    TIMEOUT = 15.0
    READ_SECONDS = 4.0  # time to read the screen if there is no voice

    def __init__(self):
        self._command = next((c for c in ("say", "espeak-ng", "espeak") if shutil.which(c)), None)
        self._process = self._start = None
        self.failed = False

    def start(self, t):
        self.stop()
        self.failed, self._start = self._command is None, t
        if self._command:
            try:
                self._process = subprocess.Popen(
                    [self._command, PROMPT],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                self.failed = True

    def finished(self, t):
        if self._process is not None:
            code = self._process.poll()
            if code is None:
                if t - self._start < self.TIMEOUT:
                    return False
                self.stop()
                self.failed = True
            elif code:
                self.failed = True
            self._process = None
        return not self.failed or t - self._start >= self.READ_SECONDS

    def stop(self):
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process = None


class ResponseCheck:
    """One prompt at a time: idle, speaking, watching, then done."""

    def __init__(self, speaker):
        self.speaker = speaker
        self.state = "idle"
        self.outcome = None  # "response" or "none seen", for the latest prompt
        self._seen = False
        self._arms = (_Arm(), _Arm())
        self._body = self._watch_start = self._asked_at = None

    @property
    def active(self):
        return self.state in ("speaking", "watching")

    def ask(self, t):
        if not self.active:
            self.state, self._asked_at, self.outcome = "speaking", t, None
            self._seen = False
            self.speaker.start(t)

    def cancel(self):
        self.speaker.stop()
        self.state, self.outcome = "idle", None
        self._body = None
        for arm in self._arms:
            arm.reset()

    def update(self, t, body):
        """Call on every frame, even when idle. Return True once, if a response is seen."""
        if body is not None:
            old = self._body
            if old is not None and math.dist(body.center, old.center) > MAX_JUMP * old.width:
                self._seen = False
                for arm in self._arms:
                    arm.reset()
            self._body = body
        heights = (None, None) if body is None else (body.left, body.right)
        held = [arm.update(t, h) for arm, h in zip(self._arms, heights)]

        if self.active:
            for duration, arm in zip(held, self._arms):
                if duration >= HOLD_SECONDS and arm.raised_since >= self._asked_at:
                    self._seen = True
        if self.state == "speaking" and self.speaker.finished(t):
            self.state, self._watch_start = "watching", t
        if self.state != "watching":
            return False
        # The hand must be raised after the request, or while it was being made.
        if self._seen:
            self.state, self.outcome = "done", "response"
            return True
        if t - self._watch_start >= LISTEN_SECONDS:
            self.state, self.outcome = "done", "none seen"
        return False

    def message(self):
        """Return (headline, detail) while a prompt is running, else None."""
        if not self.active:
            return None
        how = "Voice unavailable: showing text only" if self.speaker.failed else "Asked out loud"
        if self.state == "speaking":
            return f"ASKING: {PROMPT}", how if self.speaker.failed else "Speaking now"
        return "WATCHING for a raised hand", f"{how}. Start with hands down, then raise one and hold."
