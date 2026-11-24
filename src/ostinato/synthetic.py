"""Synthetic chest motion with a known breathing rate, for testing."""

import numpy as np


def breathing(times, bpm, amplitude=0.005, harmonic=0.0, phase=0.0):
    """Return displacement in shoulder widths.

    `harmonic` is the relative size of the second harmonic, which makes
    inhalation and exhalation unequal, as in real breathing.
    """
    angle = 2 * np.pi * bpm / 60 * np.asarray(times, float) + phase
    return amplitude * (np.sin(angle) + harmonic * np.sin(2 * angle + 0.7))


def breaths(times, rng, bpm, timing=0.12, depth=0.15, pause=0.15, amplitude=0.01):
    """Return (displacement, true breaths/min), built breath by breath.

    Each breath is a quick inhale, a slower exhale, and a pause. `timing` and `depth`
    are relative spreads between breaths; `pause` is the typical share of a breath.
    """
    times = np.asarray(times, float)
    x, starts, t = np.zeros(len(times)), [], times[0] - rng.uniform(0, 60 / bpm)
    while t < times[-1]:
        length = max(0.8, 60 / bpm * rng.normal(1, timing))
        size = max(0.3, rng.normal(1, depth))
        active = length * (1 - min(0.8, max(0.0, rng.normal(pause, pause * 0.3))))
        inhale = 0.38 * active
        u = times - t
        rising, falling = (u >= 0) & (u < inhale), (u >= inhale) & (u < length)
        x[rising] = size * (1 - np.cos(np.pi * u[rising] / inhale)) / 2
        x[falling] = size * np.exp(-(u[falling] - inhale) / ((active - inhale) / 3.2))
        starts.append(t)
        t += length
    marks = np.array([*starts, t])
    first, last = np.interp([times[0], times[-1]], marks, np.arange(len(marks)))
    return amplitude * (x - x.mean()), float((last - first) * 60 / (times[-1] - times[0]))
