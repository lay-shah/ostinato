"""Synthetic chest motion with a known breathing rate, for testing."""

import numpy as np


def breathing(times, bpm, amplitude=0.005, harmonic=0.0, phase=0.0):
    """Return displacement in shoulder widths.

    `harmonic` is the relative size of the second harmonic, which makes
    inhalation and exhalation unequal, as in real breathing.
    """
    angle = 2 * np.pi * bpm / 60 * np.asarray(times, float) + phase
    return amplitude * (np.sin(angle) + harmonic * np.sin(2 * angle + 0.7))
