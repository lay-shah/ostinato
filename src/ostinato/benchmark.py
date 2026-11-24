"""Synthetic benchmark of the breathing estimator: `python -m ostinato.benchmark`.

Simulated signals only. It shows how the estimator behaves on motion we made up,
not how accurate it is on people.
"""

import argparse
import json

import numpy as np

from .breathing import estimate
from .synthetic import breaths

FPS = 15
TIMES = np.arange(0, 30, 1 / FPS)
DRIFT = 0.0003  # tracking error per frame, in shoulder widths
PATTERNS = {  # timing spread, depth spread, pause
    "regular": (0.05, 0.05, 0.05),
    "typical": (0.12, 0.15, 0.15),
    "long pauses": (0.12, 0.15, 0.35),
    "irregular": (0.25, 0.30, 0.20),
}


def _drift(rng, size=DRIFT):
    return np.cumsum(rng.normal(0, size, len(TIMES)))


def _band(rng, low, high):
    spectrum = np.fft.rfft(rng.normal(0, 1, len(TIMES)))
    freq = np.fft.rfftfreq(len(TIMES), 1 / FPS)
    spectrum[(freq < low) | (freq > high)] = 0
    x = np.fft.irfft(spectrum, len(TIMES))
    return 0.005 * x / x.std() + _drift(rng)


def _fidgets(rng):
    x = _drift(rng)
    for _ in range(rng.integers(6, 14)):
        x += rng.normal(0, 0.01) * np.exp(-(((TIMES - rng.uniform(0, 30)) / rng.uniform(0.3, 1.2)) ** 2))
    return x


def _posture(rng):
    return np.interp(TIMES, np.arange(0, 32, 1.5), rng.normal(0, 0.006, 22)) + _drift(rng)


NOT_BREATHING_MOTION = {
    "tracking drift": lambda rng: _drift(rng, 0.002),
    "white noise": lambda rng: rng.normal(0, 0.002, len(TIMES)),
    "fidgeting": _fidgets,
    "posture shifts": _posture,
    "random sway, 6-36/min": lambda rng: _band(rng, 0.1, 0.6),
    "random sway, 12-24/min": lambda rng: _band(rng, 0.2, 0.4),
    "random sway, 6-18/min": lambda rng: _band(rng, 0.1, 0.3),
}


def run(trials=200, seed=0):
    out = {"source": "synthetic benchmark", "trials": trials, "seed": seed, "breathing": {}, "other_motion": {}}
    for name, (timing, depth, pause) in PATTERNS.items():
        errors = []
        for i in range(trials):
            rng = np.random.default_rng([seed, 1, i])
            x, truth = breaths(TIMES, rng, rng.uniform(8, 30), timing, depth, pause)
            r = estimate(TIMES, x + _drift(rng))
            if r.bpm is not None:
                errors.append(abs(r.bpm - truth))
        out["breathing"][name] = {
            "reported": len(errors) / trials,
            "mean_error": float(np.mean(errors)) if errors else None,
            "p95_error": float(np.percentile(errors, 95)) if errors else None,
            "largest_error": float(max(errors)) if errors else None,
        }
    for name, make in NOT_BREATHING_MOTION.items():
        hits = sum(estimate(TIMES, make(np.random.default_rng([seed, 2, i]))).bpm is not None for i in range(trials))
        out["other_motion"][name] = {"false_rates": hits / trials}
    for bpm in (8, 12, 16, 24, 32):
        kept = {}
        for stop in (4, 6, 8, 10):
            n = 0
            for i in range(trials):
                rng = np.random.default_rng([seed, 3, i])
                x, _ = breaths(TIMES, rng, bpm)
                x[-stop * FPS :] = x[-stop * FPS]
                n += estimate(TIMES, x + _drift(rng)).bpm is not None
            kept[f"{stop} s"] = n / trials
        out.setdefault("rate_kept_after_motion_stops", {})[f"{bpm}/min"] = kept
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.trials < 1 or args.seed < 0:
        parser.error("trials must be positive and seed nonnegative")
    print(json.dumps(run(args.trials, args.seed), indent=2))


if __name__ == "__main__":
    main()
