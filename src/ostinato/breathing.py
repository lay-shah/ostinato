"""Estimate breathing rate from a motion signal, or decline to."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

MIN_BPM = 6.0
MAX_BPM = 42.0
SEARCH_BPM = (4.5, 46.0)  # wider than reported, so limits are not edge peaks
LIMIT_TOLERANCE = 0.5  # breaths/min allowed past the limits for estimation error
MIN_PERIODICITY = 0.5  # correlation between the signal and itself one breath later
FS = 10.0  # uniform resampling rate, Hz
MIN_SECONDS = 25.0
MIN_COVERAGE = 0.8
MAX_GAP = 1.0  # longest tolerated hole in tracking, seconds
MIN_CONFIDENCE = 0.8
MIN_RMS = 2e-4  # smallest usable motion, in shoulder widths
MAX_RATE_SPREAD = 4.0  # disagreement between window halves, breaths/min


@dataclass(frozen=True)
class Reading:
    """`bpm` is None whenever the data cannot be trusted.

    An unmeasured rate says nothing about whether the person is breathing.
    """

    bpm: float | None
    confidence: float  # heuristic 0-1 spectral score, not a probability
    reason: str

    @property
    def status(self) -> str:
        return "measured" if self.bpm is not None else "unable to assess"


def _decline(reason: str, confidence: float = 0.0) -> Reading:
    return Reading(None, confidence, reason)


def _peak(x: np.ndarray) -> tuple[float, float]:
    """Return (breaths/min, fraction of band power near the peak)."""
    n_fft = 4096
    x = x - x.mean()
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)), n_fft)) ** 2
    bpm = np.fft.rfftfreq(n_fft, 1 / FS) * 60
    band = (bpm >= SEARCH_BPM[0]) & (bpm <= SEARCH_BPM[1])
    power, bpm = power[band], bpm[band]
    if power.sum() <= 0:
        return 0.0, 0.0
    k = int(power.argmax())
    if k in (0, len(power) - 1):  # edge maximum: drift or out-of-band motion
        return float(bpm[k]), 0.0
    a, b, c = power[k - 1 : k + 2]
    curve = a - 2 * b + c
    shift = 0.5 * (a - c) / curve if curve else 0.0
    rate = float(bpm[k] + shift * (bpm[1] - bpm[0]))
    half_width = 2 * 60 * FS / len(x)  # Hann main lobe
    # Breathing is not a pure sine, so its second harmonic counts too.
    near = (np.abs(bpm - rate) <= half_width) | (np.abs(bpm - 2 * rate) <= half_width)
    return rate, float(power[near].sum() / power.sum())


def _autocorrelation_rate(x: np.ndarray, rate: float) -> tuple[float, float]:
    """Return (breaths/min, correlation) at the best lag within 15% of `rate`.

    `x` must keep motion slower than the search band: when a harmonic is mistaken
    for the breathing rate, the slower true rhythm makes this correlation low.
    """
    period = 60 * FS / rate
    lags = np.arange(max(2, int(period * 0.85)), int(np.ceil(period * 1.15)) + 1)
    lags = lags[lags < len(x) - 2]
    if not len(lags):
        return 0.0, 0.0
    scores = np.array([np.corrcoef(x[:-k], x[k:])[0, 1] for k in lags])
    k = int(scores.argmax())
    lag = float(lags[k])
    if 0 < k < len(lags) - 1:
        a, b, c = scores[k - 1 : k + 2]
        curve = a - 2 * b + c
        lag += 0.5 * (a - c) / curve if curve else 0.0
    return 60 * FS / lag, float(scores[k])


def estimate(times, values) -> Reading:
    """Estimate from timestamps (s) and displacement (shoulder widths).

    Use NaN for frames in which the chest could not be tracked.
    """
    t = np.asarray(times, float)
    x = np.asarray(values, float)
    if t.shape != x.shape or t.ndim != 1:
        raise ValueError("times and values must be 1-D and the same length")
    if len(t) < 2 or t[-1] - t[0] < MIN_SECONDS - 0.2:
        return _decline("need more data")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing")

    good = np.isfinite(x)
    if good.mean() < MIN_COVERAGE:
        return _decline("chest not visible")
    tg, xg = t[good], x[good]
    edges = np.concatenate(([t[0]], tg, [t[-1]]))
    if np.diff(edges).max() > MAX_GAP:
        return _decline("tracking interrupted")

    grid = np.arange(tg[0], tg[-1], 1 / FS)
    y = np.interp(grid, tg, xg)
    y = y - np.polyval(np.polyfit(grid - grid[0], y, 1), grid - grid[0])

    sos = butter(2, [SEARCH_BPM[0] / 60 * 0.8, SEARCH_BPM[1] / 60 * 1.2], "bandpass", fs=FS, output="sos")
    filtered = sosfiltfilt(sos, y)
    rms = float(np.sqrt(np.mean(filtered**2)))
    if rms < MIN_RMS:
        return _decline("signal too weak")
    # A lurch dwarfs normal breaths.
    if np.abs(filtered).max() > 5 * np.median(np.abs(filtered)) + 1e-12:
        return _decline("too much movement")

    # Accumulated tracking error is a random walk, whose low-frequency power can
    # imitate slow breathing. Its velocity has a flat spectrum, so require a
    # clear rhythm in velocity as well as displacement.
    rate, score = _peak(filtered)
    velocity = np.gradient(filtered, 1 / FS)
    v_rate, v_score = _peak(velocity)
    score = min(1.0, max(0.0, min(score, v_score)))
    if score < MIN_CONFIDENCE:
        return _decline("signal not rhythmic enough", score)
    if abs(rate - v_rate) > MAX_RATE_SPREAD:
        return _decline("rate not stable", score)
    if not MIN_BPM - LIMIT_TOLERANCE <= rate <= MAX_BPM + LIMIT_TOLERANCE:
        return _decline("rate outside measurable range", score)
    low = butter(2, SEARCH_BPM[1] / 60 * 1.2, "lowpass", fs=FS, output="sos")
    ac_rate, periodicity = _autocorrelation_rate(sosfiltfilt(low, y), rate)
    if periodicity < MIN_PERIODICITY or abs(ac_rate - rate) > MAX_RATE_SPREAD:
        return _decline("methods disagree", score)
    mid = len(filtered) // 2
    first, _ = _peak(velocity[:mid])
    second, _ = _peak(velocity[mid:])
    if abs(first - second) > MAX_RATE_SPREAD:
        return _decline("rate not stable", score)
    return Reading(rate, score, "ok")
