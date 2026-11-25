import math

import cv2
import numpy as np
import pytest

from ostinato.shake import STEADY, ShiftTracker, analyze, describe


def shaking(size=20.0, hertz=4.0, decay=0.5, fps=30, seconds=8.0, start=1.0, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fps)
    s = np.clip(t - start, 0, None)
    x = np.where(t < start, 0.0, size * np.exp(-s / decay) * np.cos(2 * np.pi * hertz * s))
    return t, np.column_stack([x, 0.3 * x]) + rng.normal(0, noise, (len(t), 2))


@pytest.mark.parametrize("decay", [0.2, 0.5, 0.9])
@pytest.mark.parametrize("hertz", [2.0, 4.0, 8.0])
def test_settling_time(hertz, decay):
    r = analyze(*shaking(hertz=hertz, decay=decay, seconds=10))
    expected = decay * math.log(20 * math.hypot(1, 0.3) / STEADY)
    assert r.settled == pytest.approx(expected, abs=0.5 / hertz + 0.1)
    assert r.hertz == pytest.approx(hertz, rel=0.15)


def test_says_when_it_cannot_tell():
    assert "no jolt" in analyze(*shaking(size=0.5)).reason
    assert "still moving" in analyze(*shaking(decay=20)).reason
    assert "too late" in analyze(*shaking(start=7)).reason
    assert "not enough" in analyze([0, 1], [(0, 0), (1, 1)]).reason
    t, p = shaking()
    keep = (t < 3) | (t > 4)
    assert "stalled" in analyze(t[keep], p[keep]).reason
    p[5] = math.nan
    assert analyze(t, p).settled is None
    # A slow drift at the end is not steady, whatever caused it.
    t, p = shaking()
    assert analyze(t, p + np.outer(t, [0.6, 0])).settled is None


def test_describe():
    assert "Within the 2 s" in describe(analyze(*shaking(decay=0.2)))
    assert "Longer than" in describe(analyze(*shaking(decay=0.9, seconds=10)))
    assert describe(analyze(*shaking(size=0.1))).startswith("No result")


@pytest.mark.parametrize("width,height", [(1280, 720), (640, 360)])
def test_follows_a_picture(width, height):
    rng = np.random.default_rng(1)
    scene = cv2.resize(
        rng.integers(0, 255, (45, 80), np.uint8).astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    tracker, scale, seen = ShiftTracker(), width / 1280, []
    _, places = shaking(size=12, seconds=3, noise=0)
    for x, y in places:
        m = np.float32([[1, 0, x * scale], [0, 1, y * scale]])
        moved = cv2.warpAffine(
            scene, m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
        )
        noisy = np.clip(moved + rng.normal(0, 3, moved.shape), 0, 255).astype(np.uint8)
        seen.append(tracker.update(noisy))
    seen = np.array(seen) - seen[0]
    assert np.abs(seen - places).max() < 0.3


def test_blank_picture():
    tracker, blank = ShiftTracker(), np.full((360, 640), 90, np.uint8)
    tracker.update(blank)
    assert tracker.update(blank) is None
