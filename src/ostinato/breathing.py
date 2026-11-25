"""Estimate breathing rate from a motion signal, or decline to."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

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
FADE_PERIODS, MIN_FADE_SECONDS = 0.9, 2.5  # length of the stretch checked for faded motion
FADE_RATIO = 0.4  # least motion in any such stretch, relative to the typical one
MAX_MIRRORING = 0.4  # how far the signal may be explained by background motion
MIN_LEAK, MAX_LEAK = 0.3, 0.5  # camera motion left in the signal: relative size, and match
MAX_CAMERA_RATIO = 3.0  # camera motion, relative to the signal, that cannot be removed well enough
MAX_HEAD_RATIO, MIN_HEAD_MATCH = 0.85, 0.7  # head motion, relative to the chest, that means rocking
MIN_SWAY_MATCH = 0.7  # how closely sideways motion must follow the signal to mean swaying
MIN_SIDE_MATCH = 0.85  # least agreement between the halves of the chest
# Breath counting, used when the rhythm is too uneven for the spectrum alone.
COUNT_CONFIDENCE = 0.6  # least spectral score
COUNT_AGREEMENT = 2.0  # breaths/min between peaks, troughs, and the spectrum
COUNT_TIMING = 0.2  # largest relative spread of breath lengths
COUNT_LENGTHS = (0.6, 1.5)  # shortest and longest breath, relative to the typical one
COUNT_DEPTH = 0.35  # largest relative spread of breath depths
MIN_BREATHS = 4
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


def _faded(velocity: np.ndarray, rate: float) -> bool:
    """True if motion nearly vanished for about a breath, at any point.

    Without this, a window that is mostly old breathing keeps reporting a rate
    long after the motion has changed.
    """
    n = int(max(MIN_FADE_SECONDS, FADE_PERIODS * 60 / rate) * FS)
    total = np.concatenate(([0.0], np.cumsum(velocity**2)))
    rms = np.sqrt(np.maximum(total[n:] - total[:-n], 0) / n)
    return bool(rms.min() < FADE_RATIO * np.median(rms))


def _mirrors(signal: np.ndarray, background: np.ndarray) -> bool:
    """True if the signal looks like background motion turned upside down.

    Background motion is subtracted to cancel camera shake. Real shake also moves
    the chest, so nothing of it is left. A curtain or fan moves only the
    background, and subtracting it would invent a breathing signal.
    """
    if np.std(background) < 0.1 * np.std(signal):
        return False
    return bool(np.corrcoef(signal, background)[0, 1] < -MAX_MIRRORING)


def _leaks(signal: np.ndarray, background: np.ndarray) -> bool:
    """True if the signal follows the camera motion that was subtracted.

    On a faint background, camera motion is underestimated and only part of it is
    removed. What is left moves the same way, and can look like breathing.
    """
    if np.std(background) < MIN_LEAK * np.std(signal):
        return False
    if np.std(background) > MAX_CAMERA_RATIO * np.std(signal):
        return True
    return bool(np.corrcoef(signal, background)[0, 1] > MAX_LEAK)


def _rocks(signal: np.ndarray, head: np.ndarray) -> bool:
    """True if the head moves with the chest, and about as much.

    Breathing moves the chest more than the head. Rocking moves both.
    """
    if np.std(head) < MAX_HEAD_RATIO * np.std(signal):
        return False
    return bool(abs(np.corrcoef(signal, head)[0, 1]) > MIN_HEAD_MATCH)


def _splits(right: np.ndarray, left: np.ndarray) -> bool:
    """True unless both halves of the chest move together.

    Wind in clothing and moving shadows often pass across the chest. An injured chest
    can also move unevenly, and a person should look at that.
    """
    if min(np.std(right), np.std(left)) <= 0:
        return True
    return bool(np.corrcoef(right, left)[0, 1] < MIN_SIDE_MATCH)


def _sways(signal: np.ndarray, across: np.ndarray) -> bool:
    """True if the chest moves at least as much across the image, in step.

    Swaying is mostly sideways, but a little of it is always vertical too.
    """
    if np.std(across) < np.std(signal):
        return False
    return bool(abs(np.corrcoef(signal, across)[0, 1]) > MIN_SWAY_MATCH)


def _count(x: np.ndarray, rate: float) -> float | None:
    """Return breaths/min by counting, or None unless the breaths are alike."""
    swing = np.percentile(x, 90) - np.percentile(x, 10)
    peaks, found = find_peaks(x, prominence=0.35 * swing)
    troughs, _ = find_peaks(-x, prominence=0.35 * swing)
    rates = []
    for marks in (peaks, troughs):
        if len(marks) < MIN_BREATHS:
            return None
        lengths = np.diff(marks)
        if lengths.std() > COUNT_TIMING * lengths.mean():
            return None
        rates.append(60 * FS / lengths.mean())
    lengths, depths = np.diff(peaks), found["prominences"]
    low, high = COUNT_LENGTHS
    typical = np.median(lengths)
    if lengths.min() < low * typical or lengths.max() > high * typical:
        return None
    if depths.std() > COUNT_DEPTH * depths.mean() or lengths.sum() < 0.7 * len(x):
        return None
    counted = sum(rates) / 2
    if abs(rates[0] - rates[1]) > COUNT_AGREEMENT or abs(counted - rate) > COUNT_AGREEMENT:
        return None
    return float(counted)


def estimate(times, values, background=None, head=None, sideways=None, sides=None) -> Reading:
    """Estimate from timestamps (s) and displacement (shoulder widths).

    Use NaN for frames in which the chest could not be tracked. `background` is the
    background displacement already subtracted from `values`, and `head` the
    head displacement on the same scale, if known. `sideways` is the chest's
    displacement across the image.
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

    def companion(series):
        if series is None:
            return None
        values_there = np.interp(grid, tg, np.asarray(series, float)[good])
        return sosfiltfilt(sos, values_there - values_there.mean())

    sos = butter(2, [SEARCH_BPM[0] / 60 * 0.8, SEARCH_BPM[1] / 60 * 1.2], "bandpass", fs=FS, output="sos")
    filtered = sosfiltfilt(sos, y)
    rms = float(np.sqrt(np.mean(filtered**2)))
    if rms < MIN_RMS:
        return _decline("signal too weak")
    # A lurch dwarfs normal breaths.
    if np.abs(filtered).max() > 5 * np.median(np.abs(filtered)) + 1e-12:
        return _decline("too much movement")

    behind, above, across = companion(background), companion(head), companion(sideways)
    if behind is not None and _mirrors(filtered, behind):
        return _decline("background moving")
    if behind is not None and _leaks(filtered, behind):
        return _decline("camera moving")
    if above is not None and _rocks(filtered, above):
        return _decline("whole body moving")

    if sides is not None:
        right, left = (companion(side) for side in zip(*sides))
        if _splits(right, left):
            return _decline("chest moving unevenly")
    if across is not None and _sways(filtered, across):
        return _decline("whole body moving")

    # Accumulated tracking error is a random walk, whose low-frequency power can
    # imitate slow breathing. Its velocity has a flat spectrum, so require a
    # clear rhythm in velocity as well as displacement.
    rate, score = _peak(filtered)
    velocity = np.gradient(filtered, 1 / FS)
    v_rate, v_score = _peak(velocity)
    score = min(1.0, max(0.0, min(score, v_score)))
    if score < COUNT_CONFIDENCE:
        return _decline("signal not rhythmic enough", score)
    if abs(rate - v_rate) > MAX_RATE_SPREAD:
        return _decline("rate not stable", score)
    if not MIN_BPM - LIMIT_TOLERANCE <= rate <= MAX_BPM + LIMIT_TOLERANCE:
        return _decline("rate outside measurable range", score)
    low = butter(2, SEARCH_BPM[1] / 60 * 1.2, "lowpass", fs=FS, output="sos")
    smooth = sosfiltfilt(low, y)
    if _faded(np.gradient(smooth, 1 / FS), rate):
        return _decline("motion faded", score)
    ac_rate, periodicity = _autocorrelation_rate(smooth, rate)
    if periodicity < MIN_PERIODICITY or abs(ac_rate - rate) > MAX_RATE_SPREAD:
        return _decline("methods disagree", score)
    mid = len(filtered) // 2
    first, _ = _peak(velocity[:mid])
    second, _ = _peak(velocity[mid:])
    steady = abs(first - second) <= MAX_RATE_SPREAD
    if score >= MIN_CONFIDENCE and steady:
        return Reading(rate, score, "ok")
    counted = _count(filtered, rate)
    if counted is not None and MIN_BPM - LIMIT_TOLERANCE <= counted <= MAX_BPM + LIMIT_TOLERANCE:
        return Reading(counted, score, "ok")
    return _decline("rate not stable" if score >= MIN_CONFIDENCE else "signal not rhythmic enough", score)
