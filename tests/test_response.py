import subprocess

import numpy as np
import pytest

from ostinato import response
from ostinato.response import Body, ResponseCheck, Speaker, body_from_landmarks


class Quiet:
    failed = False

    def __init__(self, seconds=2.0):
        self.seconds, self.t0, self.stopped = seconds, None, 0

    def start(self, t):
        self.t0 = t

    def finished(self, t):
        return t - self.t0 >= self.seconds

    def stop(self):
        self.stopped += 1


def play(height, seconds=20.0, ask=2.0, fps=15, center=lambda t: (300, 200), speaker=None):
    """Feed left-wrist heights (or None) and return (response time or None, check)."""
    check, hit = ResponseCheck(speaker or Quiet()), None
    for i in range(int(seconds * fps)):
        t = i / fps
        if ask is not None and abs(t - ask) < 0.5 / fps:
            check.ask(t)
        h = height(t)
        body = None if h == "lost" else Body(center(t), 200, h, None)
        if check.update(t, body):
            assert hit is None, "must fire once"
            hit = t
    return hit, check


def test_raise_after_prompt_is_a_response():
    hit, check = play(lambda t: 0.8 if t > 6 else -1)
    assert 6.5 < hit < 7 and check.outcome == "response"


def test_raise_during_prompt_counts_once_it_ends():
    hit, _ = play(lambda t: 0.8 if t > 3 else -1)
    assert hit == pytest.approx(4, abs=0.15)


@pytest.mark.parametrize(
    "height",
    [
        lambda t: -1,
        lambda t: None,
        lambda t: "lost",
        lambda t: 0.8,  # already up, never seen down
        lambda t: 0.8 if t > 1 else -1,  # raised before being asked
        lambda t: 0.8 if 6 < t < 6.4 else -1,  # too brief
        lambda t: 0.8 if t > 6 and t % 0.5 < 0.3 else -1,  # shaking
        lambda t: 0.05 if t > 6 else -1,  # not high enough
        lambda t: (0.8 if t % 0.4 < 0.2 else None) if t > 6 else -1,  # hold not seen throughout
        lambda t: 0.8 if t > 9 else (None if t > 4 else -1),  # down only before a long gap
        lambda t: 0.8 if t > 14.5 else -1,  # after the window
    ],
)
def test_not_a_response(height):
    hit, check = play(height)
    assert hit is None and check.outcome == "none seen" and not check.active


def test_no_prompt_no_response():
    assert play(lambda t: 0.8 if t > 6 else -1, ask=None)[0] is None


def test_body_jump_may_be_someone_else():
    hit, _ = play(lambda t: 0.8 if t > 6 else -1, center=lambda t: (300, 200) if t <= 6 else (500, 200))
    assert hit is None


def test_random_motion_rarely_counts():
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(100):
        noise = rng.normal(-0.8, 0.4, 300)
        hits += play(lambda t, n=noise: float(n[int(t * 15)]))[0] is not None
    assert hits == 0


def test_cancel_clears_history():
    _, check = play(lambda t: -1, seconds=5)
    check.cancel()
    assert check.state == "idle" and check.outcome is None and check.speaker.stopped
    check.ask(5.1)
    assert not any(check.update(5.1 + i / 15, Body((300, 200), 200, 0.9, None)) for i in range(120))


def landmarks(wrist, rotate=0.0):
    pts = np.zeros((33, 2))
    pts[0], pts[11], pts[12] = (0, -80), (60, 0), (-60, 0)
    pts[15], pts[16] = (90, wrist), (-90, 120)
    c, s = np.cos(rotate), np.sin(rotate)
    return pts @ np.array([[c, s], [-s, c]]) + 300, np.ones(33)


@pytest.mark.parametrize("rotate", np.radians([0, 45, 90, 180, 270]))
def test_height_is_the_same_lying_down(rotate):
    b = body_from_landmarks(*landmarks(-120, rotate), (640, 640))
    assert b.left == pytest.approx(1) and b.right == pytest.approx(-1)


def test_hidden_landmarks():
    for i in (0, 11, 12):
        pts, vis = landmarks(-120)
        vis[i] = 0.5
        assert body_from_landmarks(pts, vis, (640, 640)) is None
    pts, vis = landmarks(-120)
    vis[15] = 0.5
    assert body_from_landmarks(pts, vis, (640, 640)).left is None


def webcam(wrist, vis=0.97):
    """Positions the pose model gave for a real seated webcam frame, 640 x 360."""
    pts, v = np.zeros((33, 2)), np.ones(33)
    pts[0], pts[11], pts[12] = (324, 148), (412, 252), (240, 240)
    pts[15], pts[16] = (497, 500), wrist
    v[15], v[16] = 0.29, vis
    return body_from_landmarks(pts, v, (640, 360))


def test_natural_webcam_raise_counts_and_resting_hand_is_down():
    b = webcam((190, 194))
    assert b.right > response.RAISED and b.left < response.LOWERED


@pytest.mark.parametrize(
    ("wrist", "vis"),
    [((190, 194), 0.3), ((190, -20), 0.97), ((-20, 150), 0.97), ((190, 300), 0.3)],
)
def test_unseen_hand_is_neither_raised_nor_down(wrist, vis):
    assert webcam(wrist, vis).right is None


def test_hand_rising_from_below_the_frame_is_a_response():
    check, hits = ResponseCheck(Quiet()), []
    for i in range(150):
        t = i / 15
        if i == 30:
            check.ask(t)
        # Rising takes 0.6 s, during which motion blur hides the wrist.
        body = webcam((190, 480), 0.2) if t < 5 else webcam((190, 250), 0.3) if t < 5.6 else webcam((190, 194))
        if check.update(t, body):
            hits.append(t)
    assert len(hits) == 1 and 6.1 < hits[0] < 6.5


def fake_command(monkeypatch, result):
    monkeypatch.setattr(response.shutil, "which", lambda c: "x" if c == "say" else None)

    class P:
        def __init__(self, args, **kw):
            assert args == ["say", response.PROMPT]
            if result is OSError:
                raise OSError
            self.ended = False

        def poll(self):
            return -15 if self.ended else result

        def terminate(self):
            self.ended = True

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, "Popen", P)


def test_speaker(monkeypatch):
    fake_command(monkeypatch, 0)
    s = Speaker()
    s.start(0)
    assert s.finished(0.1) and not s.failed


@pytest.mark.parametrize("result", [1, OSError])
def test_speaker_failure_falls_back_to_reading_time(monkeypatch, result):
    fake_command(monkeypatch, result)
    s = Speaker()
    s.start(0)
    assert not s.finished(1) and s.failed
    assert s.finished(4)


def test_hung_speaker_is_stopped(monkeypatch):
    fake_command(monkeypatch, None)
    s = Speaker()
    s.start(0)
    assert not s.finished(14) and s.finished(15.1) and s.failed


def test_no_voice_installed(monkeypatch):
    monkeypatch.setattr(response.shutil, "which", lambda c: None)
    check = ResponseCheck(Speaker())
    check.ask(0)
    check.update(1, None)
    assert "Voice unavailable" in check.message()[1]


def test_pre_raised_hand_does_not_mask_other_hand():
    check = ResponseCheck(Quiet(0))
    hits = []
    for step in range(60):
        t = step / 10
        if step == 25:
            check.ask(t)
        left = .8 if step >= 10 else -1
        right = .8 if step >= 30 else -1
        if check.update(t, Body((300, 200), 200, left, right)):
            hits.append(t)
    assert len(hits) == 1 and hits[0] >= 3.6


@pytest.mark.parametrize("interruption", [None, 0.15, "lost"])
def test_pre_raised_hand_requires_fresh_lower_after_interruption(interruption):
    def height(t):
        if t < 1:
            return -1
        if 3 <= t < 3.2:
            return interruption
        return 0.8

    hit, check = play(height)
    assert hit is None and check.outcome == "none seen"


def test_pre_raised_hand_can_respond_after_being_lowered_again():
    hit, check = play(lambda t: -1 if t < 1 or 5 <= t < 6 else 0.8)
    assert 6.5 < hit < 7 and check.outcome == "response"


def test_completed_raise_survives_lowering_before_speech_finishes():
    hit, check = play(lambda t: .8 if 2.3 < t < 3.3 else -1)
    assert hit == pytest.approx(4.0, abs=.1)
    assert check.outcome == "response"
