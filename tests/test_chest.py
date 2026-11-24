import cv2
import numpy as np
import pytest

from ostinato.app import Monitor
from ostinato.breathing import Reading
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


def test_paired_trial_from_synthetic_frames_to_decision(monkeypatch, tmp_path, capsys):
    """Exercise the public CLI with simulated video, pose landmarks, clock, and counts."""
    import json
    from types import SimpleNamespace

    from ostinato import chest, viewpoints

    clock = [0.0]
    captures = []

    class Camera:
        def __init__(self, index):
            self.frame = 0
            self.closed = False
            captures.append(self)

        def isOpened(self):
            return True

        def read(self):
            self.frame += 1
            clock[0] += 1 / 12
            phase = 2 * np.pi * 16 / 60 * self.frame / 12
            return True, scene(2 * np.sin(phase))

        def release(self):
            self.closed = True

    class Pose:
        def __call__(self, frame):
            return None if len(captures) == 1 else POINTS

        def close(self):
            pass

    answers = iter(["", "", "8", "", "", "8"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    monkeypatch.setattr(cv2, "VideoCapture", Camera)
    monkeypatch.setattr(chest, "PoseFinder", Pose)
    monkeypatch.setattr(viewpoints, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    path = tmp_path / "synthetic.json"
    viewpoints.main(["trial", str(path), "--views", "occluded", "visible"])
    capsys.readouterr()
    assert all(camera.closed for camera in captures)
    viewpoints.main(["report", str(path)])
    report = json.loads(capsys.readouterr().out)
    assert report["coverage"] == 0.5
    assert report["accurate_second_view_recoveries"] == 1
    viewpoints.main(["replay", str(path)])
    runs = json.loads(capsys.readouterr().out)["runs"]
    assert [event["action"] for event in runs[0]["events"]] == ["move", "report"]
    assert runs[0]["time_to_usable_s"] > runs[1]["time_to_usable_s"] > 0


def test_trial_replay_matches_live_rules_through_brief_dropouts(monkeypatch):
    """Tracking drops for part of every second, between would-be one-second samples."""
    from types import SimpleNamespace

    from ostinato import chest, viewpoints
    from ostinato.decision import Action, Limits

    fps = 12
    clock = [0.0]
    frames = []

    def blocked(t):
        return 0.4 < t % 1 < 0.8

    class Camera:
        def __init__(self, index):
            pass

        def isOpened(self):
            return True

        def read(self):
            clock[0] += 1 / fps
            frames.append((clock[0], scene(2 * np.sin(2 * np.pi * 16 / 60 * clock[0]))))
            return True, frames[-1][1]

        def release(self):
            pass

    class Pose:
        def __call__(self, frame):
            return None if blocked(clock[0]) else POINTS

        def close(self):
            pass

    answers = iter(["", "", "8"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    monkeypatch.setattr(cv2, "VideoCapture", Camera)
    monkeypatch.setattr(chest, "PoseFinder", Pose)
    monkeypatch.setattr(viewpoints, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    # A clear estimate from 25 s isolates how dropouts affect confirmation.
    monkeypatch.setattr(
        "ostinato.app.estimate",
        lambda t, x, *_: Reading(16.0, 0.9, "ok") if t[-1] >= 25 else Reading(None, 0.0, "need more data"),
    )
    row = viewpoints.capture(0, "A")
    reasons = [r[3] for r in row.trace]
    assert len(row.trace) > 25 * fps and reasons.count("chest not visible") > 60
    assert any(r[1] is not None for r in row.trace)

    now = [0.0]
    monitor = Monitor(ChestTracker(lambda f: None if blocked(now[0]) else POINTS))
    live = Assessor(0, 1, Limits(view_seconds=viewpoints.WINDOW_SECONDS))
    for t, frame in frames:
        now[0] = t
        if t < viewpoints.WINDOW_SECONDS:
            assert live.update(t, monitor.update(frame, t)).bpm is None
    other = viewpoints.Observation("B", 30, None, 0, "chest not visible", 8, row.trace)
    run_a = viewpoints.replay([row, other])["runs"][0]
    assert run_a["events"][0]["action"] == Action.MOVE.value
    assert run_a["time_to_usable_s"] is None


def test_finished_rate_is_marked_as_past():
    from ostinato.ui import decision_text

    a = Assessor(0, 1)
    a.update(25, Reading(16.0, 0.9, "ok"))
    headline, _ = decision_text(a.update(28, Reading(16.0, 0.9, "ok")))
    assert headline == "PAST RESULT at 28s: 16.0 breaths/min"


def test_chest_cut_off_by_a_close_webcam_is_still_measured():
    from ostinato.chest import chest_box

    # Positions the pose model gave for a real seated frame, 640 x 360.
    pts = np.array([[412, 252], [240, 240], [380, 498], [255, 493]], float)
    _, y0, _, y1, _ = chest_box(pts, (360, 640))
    assert y1 == 360 and y1 - y0 > 90
    assert chest_box(pts + [0, 60], (360, 640)) is None  # too little chest left
    assert chest_box(pts + [260, 0], (360, 640)) is None


@pytest.mark.parametrize("breathing_px", [0, 2])
def test_moving_curtain_is_never_mistaken_for_breathing(breathing_px):
    """A swaying background with a still chest once gave a confident rate."""
    monitor = Monitor(ChestTracker(lambda f: POINTS))
    for i in range(28 * 15):
        t = i / 15
        sway = shift(BACK, 0, 3 * np.sin(2 * np.pi * 18 / 60 * t))
        body = shift(BODY, 0, breathing_px * np.sin(2 * np.pi * 13 / 60 * t))
        monitor.update(np.where(MASK[..., None] == 1, body, sway), t)
    assert monitor.reading.bpm is None
    if not breathing_px:
        assert monitor.reading.reason == "background moving"


def test_background_is_kept_in_step_with_samples():
    m = run(15, seconds=33)
    assert len(m.background) == len(m.samples)
    m.reset()
    assert not m.background


HEAD = np.zeros((480, 640), np.uint8)
HEAD[10:130, 265:375] = 1
FACE = texture(5, 480, 640)


@pytest.mark.parametrize(("head_px", "measured"), [(0.0, True), (1.2, True), (2.4, False)])
def test_rocking_moves_the_head_as_much_as_the_chest(head_px, measured):
    monitor = Monitor(ChestTracker(lambda f: POINTS))
    for i in range(28 * 15):
        wave = np.sin(2 * np.pi * 16 / 60 * i / 15)
        frame = np.where(HEAD[..., None] == 1, shift(FACE, 0, head_px * wave), scene(2 * wave))
        monitor.update(frame, i / 15)
    if measured:
        assert monitor.reading.bpm == pytest.approx(16, abs=1)
    else:
        assert monitor.reading.bpm is None and monitor.reading.reason == "whole body moving"


def test_resolution_change_and_bad_landmarks_do_not_crash():
    sizes = [(480, 640), (480, 640), (333, 501), (333, 501), (720, 1280), (480, 640)]
    tracker = ChestTracker(lambda f: POINTS * [f.shape[1] / 640, f.shape[0] / 480])
    for h, w in sizes:
        tracker.update(np.full((h, w, 3), 90, np.uint8))
    for value in (np.nan, np.inf, -np.inf):
        tracker = ChestTracker(lambda f, v=value: POINTS * v)
        assert tracker.update(scene(0))[1] == "chest not visible"
