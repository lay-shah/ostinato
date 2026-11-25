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
MIN_SIDE = 0.4  # shoulder widths from the body midline; a hand at the face is closer
DOWN_SECONDS = 0.5  # a hand must first be seen down for this long
HOLD_SECONDS = 0.6  # and then be held up for this long
LISTEN_SECONDS = 10.0
MAX_GAP = 1.0  # seconds without a view of the arm before its history is dropped
LYING_TILT = 40.0  # degrees between the torso and image-down
MIN_TORSO = 0.6  # torso length needed to trust its direction, in shoulder widths
TORSO_RATIO = 1.5  # typical torso length, in shoulder widths
MIN_LIFT = 1.1  # wrist above its own shoulder, for a hand raised from lying down
RESTING_LIFT = 0.3
PAST_HEAD = 1.1  # toward the head, for a hand raised along the ground
MAX_JUMP = 0.5  # body shift between frames, in shoulder widths: maybe someone else


@dataclass(frozen=True)
class Body:
    """Shoulder center and width in pixels, and each wrist's height: toward the head, or off the
    ground if lying down."""

    center: tuple[float, float]
    width: float
    left: float | None  # shoulder widths; None if the wrist cannot be seen
    right: float | None


def lying_axes(points):
    """Return (shoulder center, unit vector toward the hips, scale), or None if upright.

    Seen from the side, a lying person's shoulders overlap, so the torso's length
    sets the scale when it implies wider shoulders than can be seen.
    """
    if not (np.abs(points) < 1e5).all():  # also rejects NaN
        return None
    across = points[0] - points[1]
    width = float(np.linalg.norm(across))
    center = (points[0] + points[1]) / 2
    torso = (points[2] + points[3]) / 2 - center
    length = float(np.linalg.norm(torso))
    if length < MIN_TORSO * width or length < 1:
        return None
    down = torso / length
    if np.degrees(np.arccos(np.clip(down[1], -1, 1))) < LYING_TILT:
        return None
    return center, down, max(width, length / TORSO_RATIO)


def _lying_body(points, visibility, size):
    """As `body_from_landmarks`, for a person lying across the image, or None.

    They may raise a hand past the head, or off the ground. Motion along the body is
    left out of the second, so an arm resting beside the body is never "raised".
    """
    if not min(visibility[i] for i in (23, 24)) >= MIN_VISIBILITY:
        return None
    axes = lying_axes(np.array([points[i] for i in (11, 12, 23, 24)], float))
    if axes is None:
        return None
    center, down, scale = axes
    nose = np.asarray(points[0], float) - center
    if scale < 60 or not -(nose @ down) >= 0.15 * scale:
        return None
    side = np.array([-down[1], down[0]])
    side = side if side[1] < 0 else -side  # image-up, across the body

    def height(wrist, shoulder):
        x, y = points[wrist]
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        point = np.array([x, y], float)
        inside = 0 <= x < size[0] and 0 <= y < size[1]
        value = -float((point - center) @ down) / scale
        lift = float((point - np.asarray(points[shoulder], float)) @ side) / scale
        if 0 < value < PAST_HEAD:
            value = 0.0  # perhaps touching the face, which rises off the ground too
        if lift >= MIN_LIFT:
            value = max(value, lift)
        elif lift > RESTING_LIFT:
            value = max(value, 0.0)  # neither raised nor down
        if inside:
            return value if visibility[wrist] >= MIN_VISIBILITY else None
        return value if value <= LOWERED else None

    return Body((float(center[0]), float(center[1])), scale, height(15, 11), height(16, 12))


def body_from_landmarks(points, visibility, size):
    """`points` and `visibility` follow MediaPipe's 33 pose landmarks, in pixels.
    `size` is the frame (width, height).

    Height is measured from the shoulders toward the nose, so it does not depend on
    the person being upright. Returns None if the direction cannot be measured.
    """
    if not min(visibility[i] for i in (0, 11, 12)) >= MIN_VISIBILITY:
        return None
    lying = _lying_body(points, visibility, size)
    if lying is not None:
        return lying
    nose, left, right = (np.asarray(points[i], float) for i in (0, 11, 12))
    center, across = (left + right) / 2, left - right
    width = float(np.linalg.norm(across))
    if not (width >= 60 and np.isfinite([*nose, *center]).all()):
        return None
    up = np.array([-across[1], across[0]]) / width
    toward_head = float((nose - center) @ up)
    if abs(toward_head) < 0.15 * width:
        return None
    up = up * math.copysign(1, toward_head)

    def height(i):
        x, y = points[i]
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        offset = np.array([x, y], float) - center
        value = float(offset @ up) / width
        if value > 0 and abs(float(offset @ across)) / width**2 < MIN_SIDE:
            value = 0.0  # near the face or chest: neither raised nor down
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
        self.high = False  # seen raised, and not seen lowered since

    @property
    def raised_since(self):
        return self._up_since

    def update(self, t, height):
        """Return how long the hand has been held up after being down."""
        if self._last is not None and t - self._last > MAX_GAP:
            self.reset()
        if height is None:
            self._up_since = self._down_since = None  # a hold must be seen throughout
            return 0.0
        self._last = t
        if height >= RAISED:
            self.high = True
            self._down_since = None
            if not self._was_down:
                return 0.0
            if self._up_since is None:
                self._up_since = t
            return t - self._up_since
        self._up_since = None
        if height <= LOWERED:
            self.high = False
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
            for arm in self._arms:
                if arm.high:  # already up, even if hidden right now: needs a fresh lower
                    arm.reset()
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
