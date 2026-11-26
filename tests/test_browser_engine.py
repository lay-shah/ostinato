import json
import math
import runpy
from pathlib import Path

import numpy as np
import pytest

ENGINE = Path(__file__).resolve().parents[1] / "docs" / "live-engine.py"
WIDTH, HEIGHT = 480, 360


@pytest.fixture
def engine():
    return runpy.run_path(str(ENGINE))["engine"]


def pose(raised=False):
    marks = [[0, 0, 0] for _ in range(33)]
    for index, x, y in (
        (0, 0.50, 0.19),
        (11, 0.35, 0.30),
        (12, 0.65, 0.30),
        (15, 0.35, 0.19 if raised else 0.74),
        (16, 0.65, 0.74),
        (23, 0.38, 0.67),
        (24, 0.62, 0.67),
    ):
        marks[index] = [x, y, 1]
    return marks


def frame(engine, t, rgba, marks):
    return json.loads(engine.frame(t, WIDTH, HEIGHT, rgba, marks))


def test_unseen_chest_changes_view_and_resets_samples(engine):
    blank = bytes(WIDTH * HEIGHT * 4)
    for t in range(6):
        state = frame(engine, t, blank, [])
    assert state["decision"]["action"] == "move"
    assert state["decision"]["status"] == "unable to assess"
    assert state["samples"][-1] == [5, None]

    state = json.loads(engine.command(6, "new_view"))
    assert state["view"] == 2
    assert state["decision"]["action"] == "wait"
    assert not state["samples"]

    engine.command(0, "start")
    state = frame(engine, 0, blank, [])
    assert state["view"] == 1
    assert state["elapsed"] == 0


def test_voice_completion_and_raised_wrist_report_a_response(engine):
    blank = bytes(WIDTH * HEIGHT * 4)
    for i in range(9):
        frame(engine, i / 10, blank, pose())
    engine.command(0.8, "prompt", True)
    assert frame(engine, 0.9, blank, pose())["prompt"]["state"] == "speaking"
    engine.command(0.9, "speech_done")
    assert frame(engine, 1, blank, pose())["prompt"]["state"] == "watching"

    for i in range(11, 20):
        state = frame(engine, i / 10, blank, pose(raised=True))
    assert state["decision"]["status"] == "responded"
    assert state["decision"]["bpm"] is None
    assert state["prompt"]["outcome"] == "response"


def test_weak_view_allows_a_held_raise_after_a_face_touch(engine):
    blank = bytes(WIDTH * HEIGHT * 4)
    marks = pose()
    marks[15] = [0.35, 1.2, 0.2]  # a resting wrist is below the camera frame
    for i in range(11):
        frame(engine, i / 10, blank, marks)
    engine.command(1, "prompt")

    marks[15] = [0.49, 0.19, 1]  # a visible wrist at the face
    for i in range(11, 56):
        state = frame(engine, i / 10, blank, marks)
    assert state["tracking"] == "not enough chest detail"
    assert state["decision"]["action"] == "wait"
    assert state["prompt"]["state"] == "watching"
    assert state["prompt"]["outcome"] is None

    marks[15] = [0.30, 0.24, 1]  # above the shoulder and out to the side
    for i in range(56, 65):
        state = frame(engine, i / 10, blank, marks)
    assert state["decision"]["status"] == "responded"
    assert state["prompt"]["outcome"] == "response"


def test_close_webcam_tracks_the_chest_up_to_the_frame_edge(engine):
    rgba = np.random.default_rng(124).integers(0, 256, (HEIGHT, WIDTH, 4), dtype=np.uint8)
    marks = pose()
    for index in (11, 12):
        marks[index][1] = 0.80
    for index in (23, 24):
        marks[index][1:] = [1.2, 0.2]

    frame(engine, 0, rgba, marks)
    state = frame(engine, 0.1, rgba, marks)
    assert state["tracking"] == "ok"
    assert state["box"][3] == HEIGHT

    for index in (11, 12):
        marks[index][1] = 0.84
    state = frame(engine, 0.2, rgba, marks)
    assert state["tracking"] == "chest not visible"
    assert state["box"] is None


@pytest.mark.parametrize("head_moves", [False, True], ids=["chest-only", "head-and-chest"])
def test_real_optical_flow_distinguishes_breathing_from_rocking(engine, head_moves):
    texture = np.random.default_rng(934855).choice([25, 225], size=(HEIGHT, WIDTH)).astype(np.uint8)
    rgba = np.empty((HEIGHT, WIDTH, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    marks = pose()
    # Move only the area around the chest. The tracker needs stationary
    # features on both sides to compensate for apparent camera movement.
    for i in range(291):
        t = i / 10
        shift = round(3 * math.sin(2 * math.pi * 0.25 * t))
        rgba[:, :, :3] = texture[:, :, None]
        rgba[100:224, 165:315, :3] = texture[100 + shift : 224 + shift, 165:315, None]
        if head_moves:
            rgba[5:75, 205:275, :3] = texture[5 + shift : 75 + shift, 205:275, None]
        state = frame(engine, t, rgba, marks)
    assert state["tracking"] == "ok"
    if head_moves:
        assert state["reading"]["reason"] == "whole body moving"
        assert state["decision"]["status"] == "unable to assess"
        assert state["decision"]["bpm"] is None
    else:
        assert state["decision"]["status"] == "breathing measured"
        assert state["decision"]["bpm"] == pytest.approx(15, abs=1)
