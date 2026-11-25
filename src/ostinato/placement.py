"""Where a camera can see a lying person breathe: `python -m ostinato.placement`.

Geometry only, with an assumed chest rise. It says which camera positions are worth
trying, not which will work on a person.
"""

import argparse
import math
from dataclasses import dataclass

CHEST_HEIGHT = 0.2  # meters above the ground, lying down
SHOULDERS, TORSO = 0.38, 0.5  # meters between the shoulders, and shoulders to hips
RISE = 0.005  # meters, peak to peak; assumed for quiet breathing
MIN_PIXELS = 1.0  # peak-to-peak motion that the synthetic tests measure reliably
MIN_SCALE, MIN_SHOULDERS, TORSO_RATIO = 60, 0.3, 1.5  # as in the tracker
MAX_RAISE_ELEVATION = 40  # degrees; above this a raised hand may be missed


@dataclass(frozen=True)
class View:
    height: float
    distance: float
    elevation: float  # degrees above level
    pixels: float  # peak-to-peak chest motion in the image
    problem: str | None


def view(height, distance, fov=70.0, width=640, rise=RISE):
    """A level-roll camera at the person's side. `distance` is along the ground."""
    up = height - CHEST_HEIGHT
    reach = math.hypot(up, distance)
    angle = math.atan2(up, distance)
    focal = width / 2 / math.tan(math.radians(fov) / 2)
    per_meter = focal / reach
    pixels = per_meter * rise * math.cos(angle)
    scale = max(per_meter * SHOULDERS * abs(math.sin(angle)), per_meter * TORSO / TORSO_RATIO)
    problem = None
    if per_meter * SHOULDERS * math.sin(angle) < MIN_SHOULDERS * scale:
        problem = "too low: shoulders overlap"
    elif scale < MIN_SCALE:
        problem = "too far: person too small"
    elif pixels < MIN_PIXELS:
        problem = "chest motion too small"
    elif math.degrees(angle) > MAX_RAISE_ELEVATION:
        problem = "breathing only: too high for hand raises"
    return View(height, distance, math.degrees(angle), pixels, problem)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fov", type=float, default=70.0, help="horizontal field of view, degrees (default 70)"
    )
    parser.add_argument(
        "--width", type=int, default=640, help="pixels across the processed image (default 640)"
    )
    parser.add_argument(
        "--rise-mm", type=float, default=RISE * 1000, help="chest rise, peak to peak (default 5)"
    )
    args = parser.parse_args(argv)
    if not (0 < args.fov < 170 and args.width > 0 and args.rise_mm > 0):
        parser.error("field of view must be between 0 and 170, and the others above 0")
    print(f"Assumes a {args.rise_mm:g} mm chest rise, {args.fov:g} degrees across, {args.width} pixels wide.")
    print("Calculated, not measured. Nobody has been tested.\n")
    print(f"{'lens height':>11} {'distance':>9} {'angle':>6} {'motion':>8}  verdict")
    for height in (0.4, 0.6, 0.8, 1.0, 1.2):
        for distance in (0.8, 1.0, 1.25, 1.5, 2.0):
            v = view(height, distance, args.fov, args.width, args.rise_mm / 1000)
            verdict = v.problem or "worth trying"
            print(f"{height:>9.1f} m {distance:>7.2f} m {v.elevation:>5.0f}° {v.pixels:>5.1f} px  {verdict}")


if __name__ == "__main__":
    main()
