import math

import cv2
import numpy as np
import pytest

from ostinato import rover_demo
from ostinato.demo import BPM
from ostinato.rover import KEEP_OUT, STOP, State
from ostinato.rover_demo import RoverDemo


def test_simulated_run_moves_to_a_second_view_and_measures():
    demo, states = RoverDemo(), []
    while not demo.done and demo.t < 120:
        demo.step()
        states.append((demo.mission.view, demo.mission.state))
        if demo.frame is not None:
            assert demo.drive == STOP  # never measures while moving
        assert math.hypot(demo.base.pose.x, demo.base.pose.y) >= KEEP_OUT
    assert (0, State.MEASURING) in states and demo.mission.view == 1
    assert demo.mission.assessor.decision.bpm == pytest.approx(BPM, abs=1)
    assert "simulated" in demo.status()[0]


def test_screen_shows_every_state(monkeypatch):
    drawn = []
    real = rover_demo.ImageDraw.ImageDraw.text
    monkeypatch.setattr(rover_demo.ImageDraw.ImageDraw, "text", lambda self, xy, text, **k: (drawn.append(text), real(self, xy, text, **k)))
    demo = RoverDemo()
    for i in range(330):
        demo.step()
        if i % 40 == 0:
            screen = demo.render()
            assert screen.shape == (rover_demo.H, rover_demo.W, 3) and screen.dtype == np.uint8
    assert rover_demo.LABEL in drawn
    assert any("driving" in t for t in drawn) and any("measuring" in t for t in drawn)


def test_window_runs_and_quits(monkeypatch):
    shown = []
    for name in ("namedWindow", "resizeWindow", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, lambda *a: None)
    monkeypatch.setattr(cv2, "imshow", lambda title, image: shown.append(image))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1)
    keys = iter([255, ord("r"), 255, ord("q")])
    monkeypatch.setattr(cv2, "waitKey", lambda ms: next(keys))
    rover_demo.main([])
    assert len(shown) == 4
