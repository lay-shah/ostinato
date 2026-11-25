"""A scripted scene with a drawn person, for showing the app without a camera.

Frames and landmarks are synthetic. The tracker, estimator, decisions, and prompt
are the real ones.
"""

import time

import cv2
import numpy as np

from .decision import Action
from .response import Body

FPS = 15
SIZE = (640, 480)
BPM = 16
LABEL = "SYNTHETIC DEMO: not a real person"
MOVE_AFTER = 3.0  # seconds before the demo "moves the camera"
FADE_AFTER = 5.0  # seconds after a rate is reported


def _texture(seed, low, high):
    noise = np.random.default_rng(seed).integers(0, 255, (SIZE[1], SIZE[0]), np.uint8)
    noise = cv2.normalize(cv2.GaussianBlur(noise, (0, 0), 3), None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    low, high = np.array(low, np.float32), np.array(high, np.float32)
    return (low + noise[..., None] * (high - low)).astype(np.uint8)


class Demo:
    """Stands in for both the camera and the pose finder."""

    def __init__(self, clock=time.monotonic, sleep=time.sleep, lying=False):
        self.lying = lying
        self._clock, self._sleep = clock, sleep
        self._wall = _texture(1, (52, 56, 58), (112, 116, 118))
        self._shirt = _texture(2, (112, 66, 34), (190, 138, 84))
        self._rubble = _texture(3, (48, 54, 60), (120, 130, 138))
        self._torso = np.zeros((SIZE[1], SIZE[0]), np.uint8)
        outline = np.array([[245, 215], [395, 215], [440, 260], [430, 480], [210, 480], [200, 260]])
        cv2.fillPoly(self._torso, [outline], 1, cv2.LINE_AA)
        self._start = self._next = None
        self.restart()

    def restart(self):
        self.view = 1
        self.faded_at = self._waiting = None
        self.body = None

    def isOpened(self):
        return True

    def get(self, _):
        return FPS

    def release(self):
        pass

    def close(self):
        pass

    def read(self):
        now = self._clock()
        if self._start is None:
            self._start = self._next = now
        if now < self._next:
            self._sleep(self._next - now)
            now = self._next
        self._next = max(self._next + 1 / FPS, now)
        return True, self.draw(now - self._start)

    def draw(self, t):
        stop = t if self.faded_at is None else min(t, self.faded_at)
        dy = 2.2 * np.sin(2 * np.pi * BPM / 60 * stop)
        if self.lying:
            return self._draw_lying(dy)
        frame = self._wall.copy()
        cv2.ellipse(frame, (320, 130), (54, 70), 0, 0, 360, (104, 140, 186), -1, cv2.LINE_AA)
        cv2.ellipse(frame, (320, 96), (57, 42), 0, 180, 360, (34, 38, 46), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (300, 180), (340, 225), (94, 128, 172), -1)
        shift = np.float32([[1, 0, 0], [0, 1, dy]])
        shirt = cv2.warpAffine(self._shirt, shift, SIZE, borderMode=cv2.BORDER_REFLECT)
        frame[self._torso == 1] = shirt[self._torso == 1]
        if self.view == 1:  # debris hides the chest
            slab = np.array([[130, 250], [520, 205], [560, 480], [90, 480]])
            mask = np.zeros(self._torso.shape, np.uint8)
            cv2.fillPoly(mask, [slab], 1)
            frame[mask == 1] = self._rubble[mask == 1]
            cv2.polylines(frame, [slab], True, (30, 34, 38), 3, cv2.LINE_AA)
        return frame

    def _draw_lying(self, dy):
        """Someone on their back, seen from the side. Only the chest rises."""
        frame = self._wall.copy()
        frame[250:] = self._rubble[250:]
        cv2.line(frame, (0, 250), (SIZE[0], 250), (44, 48, 52), 3)
        legs = np.array([[400, 262], [420, 350], [640, 372], [640, 300]])
        cv2.fillPoly(frame, [legs], (62, 54, 48), cv2.LINE_AA)
        shirt = cv2.warpAffine(self._shirt, np.float32([[1, 0, 0], [0, 1, dy]]), SIZE, borderMode=cv2.BORDER_REFLECT)
        torso = np.zeros(self._torso.shape, np.uint8)
        outline = np.array([[180, 222], [200, 345], [330, 365], [425, 352], [410, 258], [300, 205]])
        cv2.fillPoly(torso, [outline], 1, cv2.LINE_AA)
        frame[torso == 1] = shirt[torso == 1]
        cv2.rectangle(frame, (150, 262), (186, 305), (94, 128, 172), -1)
        cv2.ellipse(frame, (108, 278), (58, 46), 0, 0, 360, (104, 140, 186), -1, cv2.LINE_AA)
        cv2.ellipse(frame, (88, 278), (42, 48), 0, 90, 270, (34, 38, 46), -1, cv2.LINE_AA)
        if self.view == 1:  # debris hides the chest
            slab = np.array([[140, 400], [210, 190], [470, 170], [500, 400]])
            cv2.fillPoly(frame, [slab], (70, 78, 84), cv2.LINE_AA)
            cv2.polylines(frame, [slab], True, (30, 34, 38), 3, cv2.LINE_AA)
        return frame

    def __call__(self, frame):
        """Landmarks, as the pose finder would return them."""
        if self.view == 1:
            self.body = None
            return None
        if self.lying:
            self.body = Body((195.0, 280.0), 140.0, -1.4, -1.4)
            return np.array([[200, 335], [190, 225], [400, 340], [395, 265]], float)
        self.body = Body((320.0, 235.0), 200.0, -1.4, -1.4)
        return np.array([[420, 235], [220, 235], [395, 480], [245, 480]], float)

    def step(self, t, decision):
        """Act out the scene. Return True when the camera should be 'moved'."""
        if decision.action is Action.MOVE:
            if self._waiting is None:
                self._waiting = t
            if t - self._waiting >= MOVE_AFTER:
                self._waiting = None
                self.view += 1
                return True
        if decision.bpm is not None and self.faded_at is None and t - decision.completed_at_s >= FADE_AFTER:
            self.faded_at = t
        return False
