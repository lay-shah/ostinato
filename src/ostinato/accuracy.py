"""Accuracy report from counted-breath trials: `ostinato-accuracy data/pair-*.json`."""

import argparse
import json
import math
import os
from html import escape
from pathlib import Path

import numpy as np

from .report import CSS
from .viewpoints import load_pair

FEW = 20  # reported windows below which numbers are called preliminary


def wilson(k, n, z=1.96):
    """95% interval for a share."""
    if n == 0:
        return None
    p = k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, mid - half), min(1.0, mid + half)]


def analyze(pairs, tolerance_bpm=2.0):
    if not math.isfinite(tolerance_bpm) or tolerance_bpm < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    windows = [w for pair in pairs for w in pair]
    if not windows:
        raise ValueError("no trials")
    done = [w for w in windows if w.bpm is not None]
    out = {
        "source": "paired webcam trials",
        "trials": len(pairs),
        "windows": len(windows),
        "reported": len(done),
        "coverage": len(done) / len(windows),
        "coverage_95ci": wilson(len(done), len(windows)),
        "tolerance_bpm": tolerance_bpm,
        "mean_absolute_error": None,
        "largest_error": None,
        "within_tolerance": None,
        "bias": None,
        "limits_of_agreement": None,
        "declines": {},
        "points": [[w.reference_bpm, w.bpm, w.view] for w in done],
    }
    for w in windows:
        if w.bpm is None:
            out["declines"][w.reason] = out["declines"].get(w.reason, 0) + 1
    if done:
        diff = np.array([w.bpm - w.reference_bpm for w in done])
        close = np.abs(diff) <= tolerance_bpm
        out.update(
            mean_absolute_error=float(np.abs(diff).mean()),
            largest_error=float(np.abs(diff).max()),
            within_tolerance=float(close.mean()),
            bias=float(diff.mean()),
        )
        if len(diff) > 2:
            spread = 1.96 * float(diff.std(ddof=1))
            out["limits_of_agreement"] = [out["bias"] - spread, out["bias"] + spread]
    return out


def _plot(result):
    points = [((ref + bpm) / 2, bpm - ref, view) for ref, bpm, view in result["points"]]
    if not points:
        return "<p>No reported windows to plot.</p>"
    limits = result["limits_of_agreement"] or []
    top = max(4.0, *(abs(d) * 1.2 for _, d, _ in points), *(abs(v) * 1.2 for v in limits))
    left = min(m for m, _, _ in points) - 2
    right = max(m for m, _, _ in points) + 2

    def x(v):
        return 50 + 690 * (v - left) / (right - left)

    def y(v):
        return 130 - 110 * v / top

    parts = ['<svg viewBox="0 0 760 290" role="img" aria-label="Difference between app and count, by rate">']
    lines = [(0.0, "no difference", ""), (result["bias"], f"mean {result['bias']:+.1f}", "6 4")]
    lines += [(v, f"{v:+.1f}", "2 4") for v in limits]
    for i, (value, text, dash) in enumerate(lines):
        tx, anchor = (50, "start") if i == 0 else (740, "end")  # so near-zero labels do not collide
        parts.append(
            f'<line x1="50" x2="740" y1="{y(value):.1f}" y2="{y(value):.1f}" stroke="var(--dim)" '
            f'stroke-dasharray="{dash}"/><text x="{tx}" y="{y(value) - 4:.1f}" text-anchor="{anchor}" font-size="12" '
            f'fill="var(--dim)">{text}</text>'
        )
    for m, d, view in points:
        parts.append(
            f'<circle cx="{x(m):.1f}" cy="{y(d):.1f}" r="5" fill="var(--ok)" fill-opacity=".75">'
            f"<title>{escape(view)}: {d:+.1f}</title></circle>"
        )
    parts.append(
        f'<text x="50" y="272" font-size="12" fill="var(--dim)">{left:.0f}</text>'
        f'<text x="740" y="272" text-anchor="end" font-size="12" fill="var(--dim)">{right:.0f}</text>'
        '<text x="395" y="284" text-anchor="middle" font-size="12" fill="var(--dim)">'
        "average of app and count (breaths/min); height is app minus count</text></svg>"
    )
    return "".join(parts)


def render(result):
    def number(v, unit=" breaths/min", sign=""):
        return "not available" if v is None else f"{v:{sign}.1f}{unit}"

    ci = result["coverage_95ci"]
    limits = result["limits_of_agreement"]
    rows = [
        ("Windows with a rate", (f"{result['reported']} of {result['windows']} ({result['coverage']:.0%}; "
                                f"95% interval {ci[0]:.0%} to {ci[1]:.0%})")),
        ("Mean absolute error", number(result["mean_absolute_error"])),
        ("Largest error", number(result["largest_error"])),
        ("Bias (app minus count)", number(result["bias"], sign="+")),
        ("95% limits of agreement", "not available" if not limits else f"{limits[0]:+.1f} to {limits[1]:+.1f}"),
        (f"Within {result['tolerance_bpm']:g} breaths/min",
         "not available" if result["within_tolerance"] is None else f"{result['within_tolerance']:.0%} of rates shown"),
    ]  # fmt: skip
    table = "".join(f"<tr><td>{escape(a)}</td><td>{escape(b)}</td></tr>" for a, b in rows)
    why = "".join(f"<tr><td>{n}</td><td>{escape(r)}</td></tr>" for r, n in sorted(result["declines"].items()))
    few = (
        f'<p class="warn"><b>Preliminary: fewer than {FEW} rates. Do not generalize from this.</b></p>'
        if result["reported"] < FEW
        else ""
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark">'
        f"<title>Ostinato accuracy</title><style>{CSS}</style></head><body><main>"
        f"<h1>Accuracy against counted breaths</h1>{few}"
        f"<p>{result['trials']} paired trials. Errors cover only windows where the app gave a rate, so read them "
        "beside the share that got one.</p>"
        f"<div class='box'><table>{table}</table></div><h2>App compared with count</h2><div class='box'>{_plot(result)}</div>"
        f"<h2>Windows without a rate</h2><div class='box'><table>{why or '<tr><td>None</td><td></td></tr>'}</table></div>"
        "<h2>Read this first</h2><p>Windows from the same person are related, so the intervals are too narrow when a few "
        "people give many windows. A hand count can be off by one breath, about 2 breaths/min over 30 seconds. "
        "This is a research prototype, not a medical device, and the limits were not set by a clinical standard.</p>"
        "</main></body></html>"
    )


def save(result, path):
    """Write a new page that only its owner can read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(handle, "w", encoding="utf-8") as f:
        f.write(render(result))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--tolerance-bpm", type=float, default=2.0)
    parser.add_argument("--html", type=Path, help="also save a page (a new file)")
    args = parser.parse_args(argv)
    try:
        if len({p.resolve() for p in args.files}) != len(args.files):
            raise ValueError("duplicate trial file")
        result = analyze([load_pair(p) for p in args.files], args.tolerance_bpm)
        if args.html:
            save(result, args.html)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
