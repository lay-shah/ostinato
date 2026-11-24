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
    engine.command(.8, "prompt", True)
    assert frame(engine, .9, blank, pose())["prompt"]["state"] == "speaking"
    engine.command(.9, "speech_done")
    assert frame(engine, 1, blank, pose())["prompt"]["state"] == "watching"

    for i in range(11, 20):
        state = frame(engine, i / 10, blank, pose(raised=True))
    assert state["decision"]["status"] == "responded"
    assert state["decision"]["bpm"] is None
    assert state["prompt"]["outcome"] == "response"


def test_real_optical_flow_confirms_periodic_chest_motion(engine):
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
        state = frame(engine, t, rgba, marks)
    assert state["tracking"] == "ok"
    assert state["decision"]["status"] == "breathing measured"
    assert state["decision"]["bpm"] == pytest.approx(15, abs=1)
