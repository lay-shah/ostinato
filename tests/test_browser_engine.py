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
        (0, .50, .19), (11, .35, .30), (12, .65, .30),
        (15, .35, .19 if raised else .74), (16, .65, .74),
        (23, .38, .67), (24, .62, .67),
    ):
        marks[index] = [x, y, 1]
    return marks


def frame(engine, t, rgba, marks, sound=False):
    return json.loads(engine.frame(t, WIDTH, HEIGHT, rgba, marks, sound))


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


def test_camera_waits_for_automatic_text_response_check_at_every_weak_view(engine):
    blank = bytes(WIDTH * HEIGHT * 4)
    engine.command(0, "start", camera=True)
    for view in range(1, 4):
        start = (view - 1) * 21
        for t in range(start, start + 5):
            state = frame(engine, t, blank, [])
            assert state["prompt"]["state"] == "idle"
        state = frame(engine, start + 5, blank, [])
        assert state["view"] == view
        assert state["decision"]["action"] == "wait"
        assert state["prompt"]["state"] == "speaking"
        assert "showing text only" in state["prompt"]["detail"]
        for t in range(start + 6, start + 19):
            state = frame(engine, t, blank, [])
            assert state["decision"]["action"] == "wait"
        assert state["prompt"]["state"] == "watching"
        state = frame(engine, start + 19, blank, [])
        assert state["prompt"]["outcome"] == "none seen"
        assert state["decision"]["status"] == "unable to assess"
        assert state["decision"]["action"] == ("move" if view < 3 else "get help")
        if view < 3:
            engine.command(start + 20, "new_view")

    assert json.loads(engine.command(63, "reset"))["prompt"]["state"] == "idle"
    for t in range(63, 69):
        state = frame(engine, t, blank, [])
    assert state["view"] == 1
    assert state["prompt"]["state"] == "speaking"

    engine.command(0, "start")
    for t in range(6):
        state = frame(engine, t, blank, [])
    assert state["decision"]["action"] == "move"
    assert state["prompt"]["state"] == "idle"


def test_automatic_voice_failure_falls_back_and_can_report_a_raise(engine):
    blank = bytes(WIDTH * HEIGHT * 4)
    engine.command(0, "start", camera=True)
    for i in range(56):
        state = frame(engine, i / 10, blank, pose(), sound=True)
        if state["prompt"]["state"] == "speaking":
            break
    assert state["prompt"]["detail"] == "Speaking now"

    failure = json.loads(engine.command(state["t"], "speech_failed"))
    assert "showing text only" in failure["prompt"]["detail"]
    for offset in range(1, 41):
        state = frame(engine, (i + offset) / 10, blank, pose(), sound=True)
        assert state["decision"]["action"] == "wait"
    assert state["prompt"]["state"] == "watching"

    for offset in range(41, 51):
        state = frame(engine, (i + offset) / 10, blank, pose(raised=True), sound=True)
    assert state["decision"]["status"] == "responded"
    assert state["prompt"]["outcome"] == "response"


def test_voice_completion_and_raised_wrist_report_a_response(engine):
    blank = bytes(WIDTH * HEIGHT * 4)
    for i in range(9):
        frame(engine, i / 10, blank, pose())
    engine.command(.8, "prompt", True)
    assert frame(engine, .9, blank, pose())["prompt"]["state"] == "speaking"
    engine.command(.9, "speech_done")
    assert frame(engine, 1, blank, pose())["prompt"]["state"] == "watching"

    for i in range(11, 20):
        state = frame(engine, i / 10, blank, pose(raised=True))
    assert state["decision"]["status"] == "responded"
    assert state["decision"]["bpm"] is None
    assert state["prompt"]["outcome"] == "response"


def test_weak_view_allows_a_held_raise_after_a_face_touch(engine):
    blank = bytes(WIDTH * HEIGHT * 4)
    marks = pose()
    marks[15] = [.35, 1.2, .2]  # a resting wrist is below the camera frame
    for i in range(11):
        frame(engine, i / 10, blank, marks)
    engine.command(1, "prompt")

    marks[15] = [.49, .19, 1]  # a visible wrist at the face
    for i in range(11, 56):
        state = frame(engine, i / 10, blank, marks)
    assert state["tracking"] == "not enough chest detail"
    assert state["decision"]["action"] == "wait"
    assert state["prompt"]["state"] == "watching"
    assert state["prompt"]["outcome"] is None

    marks[15] = [.30, .24, 1]  # above the shoulder and out to the side
    for i in range(56, 65):
        state = frame(engine, i / 10, blank, marks)
    assert state["decision"]["status"] == "responded"
    assert state["prompt"]["outcome"] == "response"


def test_close_webcam_tracks_the_chest_up_to_the_frame_edge(engine):
    rgba = np.random.default_rng(124).integers(0, 256, (HEIGHT, WIDTH, 4), dtype=np.uint8)
    marks = pose()
    for index in (11, 12):
        marks[index][1] = .80
    for index in (23, 24):
        marks[index][1:] = [1.2, .2]

    frame(engine, 0, rgba, marks)
    state = frame(engine, .1, rgba, marks)
    assert state["tracking"] == "ok"
    assert state["box"][3] == HEIGHT

    for index in (11, 12):
        marks[index][1] = .84
    state = frame(engine, .2, rgba, marks)
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
        shift = round(3 * math.sin(2 * math.pi * .25 * t))
        rgba[:, :, :3] = texture[:, :, None]
        rgba[100:224, 165:315, :3] = texture[100 + shift:224 + shift, 165:315, None]
        if head_moves:
            rgba[5:75, 205:275, :3] = texture[5 + shift:75 + shift, 205:275, None]
        state = frame(engine, t, rgba, marks)
    assert state["tracking"] == "ok"
    if head_moves:
        assert state["reading"]["reason"] == "whole body moving"
        assert state["decision"]["status"] == "unable to assess"
        assert state["decision"]["bpm"] is None
    else:
        assert state["decision"]["status"] == "breathing measured"
        assert state["decision"]["bpm"] == pytest.approx(15, abs=1)
