import cv2
import numpy as np
import pytest

from ostinato.app import Monitor, draw
from ostinato.chest import ChestTracker
from ostinato.decision import Assessor

POINTS = np.array([[420, 160], [220, 160], [390, 420], [250, 420]], float)


def texture(seed, h, w):
    img = np.random.default_rng(seed).integers(0, 255, (h, w), np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.cvtColor(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)


BACK, BODY = texture(0, 480, 640), texture(1, 480, 640)
MASK = np.zeros((480, 640), np.uint8)
MASK[130:470, 190:450] = 1


def shift(img, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (640, 480), borderMode=cv2.BORDER_REFLECT)


def scene(dy, cam=(0, 0)):
    frame = np.where(MASK[..., None] == 1, shift(BODY, 0, dy), BACK)
    return shift(frame, *cam) if any(cam) else frame


def run(bpm, shake=0.0, amp=2.0, seconds=28, fps=15, visible=lambda t: True):
    now = [0.0]
    monitor = Monitor(ChestTracker(lambda f: POINTS if visible(now[0]) else None))
    for i in range(seconds * fps):
        t = now[0] = i / fps
        cam = (0, shake * np.sin(2 * np.pi * 0.4 * t))
        monitor.update(scene(amp * np.sin(2 * np.pi * bpm / 60 * t), cam), t)
    return monitor


def test_measures_moving_chest():
    assert run(15).reading.bpm == pytest.approx(15, abs=1)


def test_compensates_for_small_camera_motion():
    assert run(15, shake=1.5).reading.bpm == pytest.approx(15, abs=1)


def test_still_chest_is_not_given_a_rate():
    assert run(15, amp=0).reading.bpm is None


def test_camera_motion_alone_is_not_given_a_rate():
    assert run(15, amp=0, shake=1.5).reading.bpm is None


def test_old_rate_is_dropped_when_view_is_blocked():
    m = run(15, visible=lambda t: t < 27.5)
    assert m.reading.bpm is None
    assert m.reading.reason == "chest not visible"


def test_stalled_stream_restarts():
    m = run(15)
    m.update(scene(0), 45.0)
    assert m.reading.bpm is None and len(m.samples) == 1


def test_reset_discards_signal_from_old_view():
    m = run(15)
    m.reset()
    assert m.reading.bpm is None and not m.samples and m.tracker.box is None


def test_decision_line_draws():
    m = run(15, seconds=2)
    frame = scene(0)
    before = frame.copy()
    draw(frame, m, Assessor(0, 1).decision)
    assert (frame != before).any()
