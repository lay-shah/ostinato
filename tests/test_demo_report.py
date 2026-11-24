import re
import stat
from datetime import datetime

import cv2
import numpy as np
import pytest

from ostinato import app
from ostinato.breathing import Reading
from ostinato.decision import Action, Assessor
from ostinato.demo import BPM, LABEL, Demo
from ostinato.report import Recorder, render, save
from ostinato.response import ResponseCheck


class Voice:
    failed = False

    def start(self, t):
        self.t = t

    def finished(self, t):
        return t - self.t > 2

    def stop(self):
        pass


def play_demo(monkeypatch, tmp_path, seconds, keys=()):
    """Run the real app on the demo, on a fake clock and without a window."""
    clock, log = [0.0], []

    def sleep(s):
        clock[0] += s

    real = app.compose

    def compose(*args):
        log.append((clock[0], args[2], args[1].reading, args[6:]))
        return (real(*args) if len(log) % 60 == 1 else (args[0], None))

    def key(_):
        clock[0] += 0.001
        if clock[0] > seconds:
            return ord("q")
        return next((ord(k) for k, at in keys if abs(at - clock[0]) < 0.034), 255)

    monkeypatch.setattr(app, "Demo", lambda: Demo(lambda: clock[0], sleep))
    monkeypatch.setattr(app, "Speaker", Voice)
    monkeypatch.setattr(app, "compose", compose)
    monkeypatch.setattr(app.time, "monotonic", lambda: clock[0])
    for name in ("namedWindow", "resizeWindow", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, lambda *a: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1)
    monkeypatch.setattr(cv2, "waitKey", key)
    app.main(["--demo", "--reports", str(tmp_path)])
    return log


def test_demo_tells_the_whole_story(monkeypatch, tmp_path):
    log = play_demo(monkeypatch, tmp_path, 85, keys=[("s", 80)])
    actions = [d.action for _, d, _, _ in log]
    assert actions.index(Action.MOVE) < actions.index(Action.REPORT)
    final = log[-1][1]
    assert final.bpm == pytest.approx(BPM, abs=1)
    assert log[-1][2].bpm is None and any(r.reason == "motion faded" for _, _, r, _ in log)
    assert all(LABEL in extra for *_, extra in log)
    files = list(tmp_path.iterdir())
    assert len(files) >= 1
    page = files[0].read_text()
    assert LABEL in page and "No response seen (proves nothing)" in page and "View 2" in page
    assert stat.S_IMODE(files[0].stat().st_mode) == 0o600


def test_demo_drawing_does_not_drift():
    d = Demo()
    assert d.draw(0.0).shape == (480, 640, 3)
    assert d(None) is None and d.body is None
    d.view = 2
    assert d(None).shape == (4, 2)


def recorder(reason="chest not visible"):
    r, a, check = Recorder(), Assessor(0, 0), ResponseCheck(Voice())
    for i in range(80):
        reading = Reading(None, 0.0, reason)
        r.update(i / 10, reading, a.update(i / 10, reading), check, 1)
    return r


def test_report_is_private_offline_and_safe():
    page = render(recorder("<script>x</script>"), "today")
    assert "<script>x" not in page and "&lt;script&gt;" in page
    page = render(recorder(), "today")
    assert not re.search(r"<script|https?:|src=|url\(|<img", page)
    assert not re.search(r"not breathing|\bdead\b|deceased|stopped breathing", page, re.IGNORECASE)
    assert "Unable to assess: a person must check" in page


def test_unfinished_and_empty_reports():
    with pytest.raises(ValueError):
        render(Recorder(), "today")
    r, a = Recorder(), Assessor(0, 1)
    reading = Reading(15.0, 0.9, "ok")
    r.update(0, reading, a.update(0, reading), ResponseCheck(Voice()), 1)
    page = render(r, "today")
    assert "Assessment not finished" in page and "15.0 breaths/min" in page
    r.restart(5)
    with pytest.raises(ValueError):
        render(r, "today")


def test_reports_never_overwrite(tmp_path):
    when = datetime(2026, 1, 2)
    save(recorder(), tmp_path, when)
    with pytest.raises(FileExistsError):
        save(recorder(), tmp_path, when)


def test_chart_handles_single_and_broken_runs():
    r = recorder()
    r.readings = [(0, 10.0, "ok"), (1, None, "x"), (2, 12.0, "ok"), (3, 12.0, "ok"), (4, None, "x")]
    page = render(r, "today")
    assert page.count("<polyline") == 1 and page.count("<circle") == 1
    assert np.isfinite([float(v) for v in re.findall(r'cy="([\d.]+)"', page)]).all()


def test_report_warns_when_rate_was_no_longer_confirmed():
    r, a, check = Recorder(), Assessor(10, 1), ResponseCheck(Voice())
    r.restart(10)
    ok, faded = Reading(16.0, 0.9, "ok"), Reading(None, 0.0, "motion faded")
    for t, reading in ((35, ok), (38, ok)):
        r.update(t, reading, a.update(t, reading), check, 1)
    page = render(r, "today")
    assert "measured at 28 s" in page and "not confirmed" not in page
    r.update(45, faded, a.update(45, faded), check, 1)
    assert "not confirmed when saved" in render(r, "today")
