import numpy as np
import pytest

from ostinato.breathing import estimate
from ostinato.synthetic import breathing

T = np.arange(0, 30, 1 / 30)


def breath(bpm, amp=0.005):
    return breathing(T, bpm, amp)


@pytest.mark.parametrize("bpm", [8, 12, 15, 20, 30, 36])
def test_rate(bpm):
    noise = np.random.default_rng(0).normal(0, 0.001, len(T))
    r = estimate(T, breath(bpm) + noise + 0.001 * T)
    assert r.bpm == pytest.approx(bpm, abs=1)


def test_irregular_timestamps_and_short_gap():
    rng = np.random.default_rng(1)
    t = np.sort(T + rng.uniform(-0.01, 0.01, len(T)))
    x = 0.005 * np.sin(2 * np.pi * 0.25 * t)
    x[300:315] = np.nan
    assert estimate(t, x).bpm == pytest.approx(15, abs=1)


def test_declines_still_signal():
    r = estimate(T, np.zeros_like(T))
    assert r.bpm is None and r.status == "unable to assess"


@pytest.mark.parametrize("walk", [False, True])
def test_false_readings_on_noise_are_rare(walk):
    rng = np.random.default_rng(2)
    count = 0
    for _ in range(200):
        x = rng.normal(0, 0.002, len(T))
        x = np.cumsum(x) if walk else x
        count += estimate(T, x).bpm is not None
    assert count <= 4


def test_declines_bad_data():
    assert estimate(T[:300], breath(15)[:300]).reason == "need more data"
    x = breath(15)
    x[200:260] = np.nan
    assert estimate(T, x).reason == "tracking interrupted"
    x = breath(15)
    x[-45:] = np.nan
    assert estimate(T, x).reason == "tracking interrupted"
    assert estimate(T, np.full_like(T, np.nan)).bpm is None
    x = breath(15)
    x[450:] += 0.2
    assert estimate(T, x).bpm is None


@pytest.mark.parametrize("harmonic", [0.0, 0.3])
@pytest.mark.parametrize("bpm", [6, 7, 10, 16, 24, 32, 40, 42])
def test_phase_1_accuracy_across_range(bpm, harmonic):
    rng = np.random.default_rng(10)
    errors = []
    for _ in range(20):
        x = breathing(T, bpm, harmonic=harmonic, phase=rng.uniform(0, 2 * np.pi))
        r = estimate(T, x + rng.normal(0, 0.0005, len(T)))
        if r.bpm is not None:
            errors.append(abs(r.bpm - bpm))
    assert len(errors) >= 18
    assert max(errors) <= 1


@pytest.mark.parametrize("bpm", [3, 4, 5, 50, 60])
def test_out_of_range_breathing_is_not_given_a_wrong_rate(bpm):
    rng = np.random.default_rng(11)
    for harmonic in (0.0, 0.3, 0.5):
        for _ in range(40):
            x = breathing(T, bpm, harmonic=harmonic, phase=rng.uniform(0, 2 * np.pi))
            assert estimate(T, x + rng.normal(0, 0.001, len(T))).bpm is None


def _stopped(bpm, seconds, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 30, 1 / 15)
    x = breathing(t, bpm, harmonic=0.2, phase=rng.uniform(0, 6))
    i = len(t) - int(seconds * 15)
    x[i:] = x[i]
    return estimate(t, x + np.cumsum(rng.normal(0, 0.0003, len(t))))


@pytest.mark.parametrize(("bpm", "seconds"), [(8, 12), (12, 8), (16, 6), (24, 5), (32, 5)])
def test_rate_is_dropped_soon_after_motion_stops(bpm, seconds):
    kept = sum(_stopped(bpm, seconds, seed).bpm is not None for seed in range(40))
    assert kept <= 4


@pytest.mark.parametrize("start", [0, 8, 14, 22])
def test_pause_anywhere_in_the_window_is_declined(start):
    t = np.arange(0, 30, 1 / 15)
    x = breathing(t, 16)
    pause = (t >= start) & (t < start + 8)
    x[pause] = x[pause][0]
    assert estimate(t, x).bpm is None


def test_varying_breath_depth_is_still_measured():
    ok = 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        t = np.arange(0, 30, 1 / 15)
        depth = np.interp(t, np.arange(0, 34, 4), np.clip(rng.normal(1, 0.2, 9), 0.5, 1.5))
        r = estimate(t, depth * breathing(t, rng.uniform(10, 24), phase=rng.uniform(0, 6)))
        ok += r.bpm is not None
    assert ok >= 54
