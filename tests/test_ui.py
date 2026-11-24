from collections import deque
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from test_decision import known_reasons

from ostinato import app, breathing, ui
from ostinato.breathing import Reading
from ostinato.decision import Action, Assessor
from ostinato.response import Body, ResponseCheck

OK = Reading(16.0, 0.9, "ok")


class Voice:
    failed = False

    def start(self, t):
        pass

    def finished(self, t):
        return True

    def stop(self):
        pass


def monitor(reading=OK, samples=()):
    return SimpleNamespace(reading=reading, tracker=SimpleNamespace(box=None), samples=deque(samples))


def finish(reason, views=0):
    a, t = Assessor(0, views), 0
    while a.decision.action is Action.WAIT:
        t += 1
        a.update(t, Reading(None, 0.0, reason))
    return a.decision


def texts(p):
    return [b[0] for b in p.boxes]


def check_fits(p):
    for text, box, limit in p.boxes:
        assert box[0] >= 0 and box[1] >= 0 and box[3] <= ui.H, text
        assert box[2] <= limit + 1, text


@pytest.mark.parametrize("reason", sorted(known_reasons()))
@pytest.mark.parametrize("shape", [(480, 640), (360, 640), (640, 360)])
def test_get_help_always_shows_and_text_fits(reason, shape):
    d = finish(reason)
    assert d.action is Action.GET_HELP
    img, p = ui.compose(np.zeros((*shape, 3), np.uint8), monitor(Reading(None, 0.0, reason)), d)
    assert img.shape == (ui.H, ui.W, 3)
    assert texts(p)[0].startswith("GET HELP")
    assert any("Past result" in t and reason in t for t in texts(p))
    check_fits(p)


def test_every_reason_has_a_hint():
    assert known_reasons() <= set(ui.HINTS)


def test_past_rate_is_not_shown_as_live():
    a = Assessor(0, 1)
    a.update(25, OK)
    d = a.update(28, OK)
    _, p = ui.compose(np.zeros((480, 640, 3), np.uint8), monitor(Reading(None, 0.0, "chest not visible")), d)
    assert "Unable to assess" in texts(p) and "16" not in texts(p)
    assert all("PAST" in t.upper() for t in texts(p) if "16.0" in t)
    check_fits(p)


def test_no_response_is_never_a_finding():
    check = ResponseCheck(Voice())
    check.ask(0)
    for i in range(200):
        check.update(i / 10, None)
    status, _ = ui.prompt_status(check)
    assert status == "None seen (proves nothing)"


def test_prompt_takes_over_banner_and_frame_is_untouched():
    check = ResponseCheck(Voice())
    check.ask(0)
    frame = np.full((480, 640, 3), 90, np.uint8)
    before = frame.copy()
    _, p = ui.compose(frame, monitor(), finish("chest not visible", 1), check, 1, 3)
    assert texts(p)[0].startswith("ASKING")
    assert (frame == before).all()
    check_fits(p)


def test_waveform_with_gaps():
    t = np.arange(0, 30, 0.1)
    y = np.sin(t)
    y[50:80] = np.nan
    ui.compose(np.zeros((480, 640, 3), np.uint8), monitor(samples=zip(t, y)), Assessor(0, 1).decision)


def run_app(monkeypatch, keys, body, args=(), seconds=30, fps=10):
    """Run the real main loop with a fake camera that never shows a chest."""
    clock, log, asks = [0.0], [], []

    class Camera:
        def isOpened(self):
            return True

        def get(self, prop):
            return fps

        def read(self):
            clock[0] += 1 / fps
            return clock[0] < seconds, np.zeros((480, 640, 3), np.uint8)

        def release(self):
            pass

    class Finder:
        body = None

        def __call__(self, frame):
            self.body = body(clock[0])

        def close(self):
            pass

    class Logged(Voice):
        def start(self, t):
            asks.append(round(clock[0], 1))

    def compose(frame, m, d, check, view, views, window):
        log.append((clock[0], d, view))
        return frame, None

    def key(_):
        return next((ord(k) for k, at in keys if abs(at - clock[0]) < 0.5 / fps), 255)

    monkeypatch.setattr(app, "PoseFinder", Finder)
    monkeypatch.setattr(app, "Speaker", Logged)
    monkeypatch.setattr(app, "compose", compose)
    monkeypatch.setattr(app.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(cv2, "VideoCapture", lambda x: Camera())
    for name in ("namedWindow", "resizeWindow", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, lambda *a: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1)
    monkeypatch.setattr(cv2, "waitKey", key)
    app.main(list(args))
    return log, asks


def test_app_asks_once_per_view_then_gets_help(monkeypatch):
    log, asks = run_app(monkeypatch, [("m", 18), ("m", 36)], lambda t: None, seconds=60)
    assert len(asks) == 3
    assert [v for _, _, v in log if v] and {v for _, _, v in log} == {1, 2, 3}
    assert log[-1][1].action is Action.GET_HELP and log[-1][1].bpm is None


def test_app_reports_a_raised_hand(monkeypatch):
    log, _ = run_app(monkeypatch, [], lambda t: Body((300, 200), 200, 0.8 if t > 8 else -1, None))
    assert log[-1][1].status.value == "responded" and log[-1][1].bpm is None


def test_app_prompt_options(monkeypatch):
    assert run_app(monkeypatch, [], lambda t: None, ["--no-prompt"])[1] == []
    assert len(run_app(monkeypatch, [("v", 2)], lambda t: None, ["--no-prompt"])[1]) == 1
    assert run_app(monkeypatch, [("v", 2)], lambda t: None, ["--video", "x"])[1] == []


def test_app_waits_for_camera_permission(monkeypatch, capsys):
    opened, naps = [], []

    class Cap:
        def isOpened(self):
            return len(opened) == 3

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda i: opened.append(i) or Cap())
    assert app.open_camera(0, naps.append) is not None and len(naps) == 2
    assert capsys.readouterr().out.count("allow camera access") == 1
    opened.clear()
    monkeypatch.setattr(Cap, "isOpened", lambda self: False)
    assert app.open_camera(0, naps.append) is None


def test_hand_raise_after_moving_needs_a_new_prompt(monkeypatch):
    log, _ = run_app(
        monkeypatch, [("m", 18)], lambda t: Body((300, 200), 200, 0.8 if 19 < t < 21 else -1, None), seconds=22
    )
    assert not log[-1][1].final


def test_needs_the_same_seconds_as_the_estimator():
    assert ui.MIN_SECONDS == breathing.MIN_SECONDS


def test_finished_rate_warns_when_no_longer_confirmed():
    a = Assessor(0, 1)
    a.update(25, OK)
    d = a.update(28, OK)
    frame = np.zeros((480, 640, 3), np.uint8)
    _, p = ui.compose(frame, monitor(Reading(None, 0.0, "motion faded")), d)
    assert any("Not confirmed now (motion faded)" in t for t in texts(p))
    check_fits(p)
    _, p = ui.compose(frame, monitor(OK), d)
    assert not any("Not confirmed" in t for t in texts(p))
