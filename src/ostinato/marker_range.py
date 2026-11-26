"""How far away, and how well, the camera reads a printed marker. In simulation.

`python -m ostinato.marker_range` draws a marker at many distances and angles, as a
1280 x 720, 70 degree camera would see it, blurred, dim, and grainy as asked, and reads
it back with the rover's own code. It prints how often the marker is found and how far
off the distance and direction are, next to what the rover assumes:

- the setup check counts a marker as seen out to MIN_PIXELS across (about 4.9 m here),
- the position estimate expects distances off by RANGE_NOISE and directions by
  BEARING_NOISE, one standard deviation.

Drawn markers are flat, evenly lit, and perfectly printed, and the lens is perfect.
Real ones read worse, so treat a range found here as the most to hope for. Photographs
from the real camera are the test that counts.
"""

import argparse
import math

import cv2
import numpy as np

from .localize import BEARING_NOISE, DICTIONARY, MARKER, RANGE_NOISE, Camera, find_markers
from .robot import MIN_PIXELS
from .rover import wrap

DISTANCES = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
TURNS = (0, 30, 50, 65)  # degrees the marker faces away from the camera
TRIES = 12
FOUND = 0.9  # share of tries a marker must be found to count as readable


def draw(camera, places, size=MARKER, blur=0.0, noise=0.0, light=1.0, seed=0):
    """Markers as the camera would see them, on a grey wall.

    places: (id, right, down, ahead, turn) in meters from the camera, before its tilt,
    with turn in radians about the upright. blur is in pixels (one standard deviation),
    noise in grey levels, and light scales the brightness (1 is a lit room)."""
    ss = 2
    w, h = camera.width * ss, camera.height * ss
    image = np.full((h, w), 150, np.uint8)
    tilt = math.radians(camera.pitch)
    for marker, right, down, ahead, turn in places:
        code = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(DICTIONARY), marker, 600)
        code = cv2.copyMakeBorder(code, 120, 120, 120, 120, cv2.BORDER_CONSTANT, value=255)
        half = size / 2 * 840 / 600
        quad = []
        for u, v in ((-half, -half), (half, -half), (half, half), (-half, half)):
            x, y, z = right + u * math.cos(turn), down + v, ahead + u * math.sin(turn)
            y, z = y * math.cos(tilt) - z * math.sin(tilt), y * math.sin(tilt) + z * math.cos(tilt)
            px = camera.matrix @ [x, y, z]
            quad.append(px[:2] / px[2] * ss)
        # Corners are the outer edges of the first and last pixels.
        source = np.float32([[-0.5, -0.5], [839.5, -0.5], [839.5, 839.5], [-0.5, 839.5]])
        m = cv2.getPerspectiveTransform(source, np.float32(quad) - 0.5)
        warped = cv2.warpPerspective(code, m, (w, h), flags=cv2.INTER_AREA)
        inside = cv2.warpPerspective(np.full_like(code, 255), m, (w, h))
        image[inside > 127] = warped[inside > 127]
    image = cv2.resize(image, (camera.width, camera.height), interpolation=cv2.INTER_AREA).astype(float)
    image *= light
    if blur:
        image = cv2.GaussianBlur(image, (0, 0), blur)
    image += np.random.default_rng(seed).normal(0, noise, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def attempt(camera, distance, turn, rng, blur=0.0, noise=0.0, light=1.0):
    """One marker at `distance` meters along the ground, somewhere in view, read back.
    Returns (distance error as a share, direction error in rad), or None if not found."""
    across = math.tan(math.radians(camera.fov) / 2) * 0.7
    side = rng.uniform(-across, across)
    ahead = distance / math.hypot(1, side)
    right = side * ahead
    down = rng.uniform(-0.1, 0.1) + ahead * math.tan(math.radians(camera.pitch))
    # Turned away from the camera, either way.
    facing = -math.atan2(right, ahead) + math.radians(turn) * rng.choice((-1, 1))
    seed = int(rng.integers(1 << 30))
    image = draw(camera, [(7, right, down, ahead, facing)], blur=blur, noise=noise, light=light, seed=seed)
    seen = [s for s in find_markers(image, camera) if s.marker == 7]
    if not seen:
        return None
    # Placed level, so this is already along the ground, as find_markers reports it.
    true = math.hypot(right, ahead)
    bearing = wrap(math.radians(camera.pan) - math.atan2(right, ahead))
    return (seen[0].distance - true) / true, wrap(seen[0].bearing - bearing)


def survey(camera, distances=DISTANCES, turns=TURNS, tries=TRIES, seed=0, **conditions):
    """{(distance, turn): (share found, distance errors, direction errors)}"""
    rng = np.random.default_rng(seed)
    table = {}
    for distance in distances:
        for turn in turns:
            results = [attempt(camera, distance, turn, rng, **conditions) for _ in range(tries)]
            hits = [r for r in results if r]
            table[distance, turn] = (
                len(hits) / tries,
                np.array([r[0] for r in hits]),
                np.array([r[1] for r in hits]),
            )
    return table


def readable(table, turn):
    """The farthest distance, with every nearer one, at which the marker is found
    nearly every time when turned `turn` degrees away. 0 if not even the nearest."""
    best = 0.0
    for distance in sorted({d for d, _ in table}):
        if table[distance, turn][0] < FOUND:
            break
        best = distance
    return best


def rms(values):
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else math.nan


def report(table, camera):
    distances = sorted({d for d, _ in table})
    turns = sorted({t for _, t in table})
    lines = ["How often it is found, by distance (rows) and how far it is turned away (columns):"]
    lines.append("         " + "".join(f"{t:>7}°" for t in turns))
    for d in distances:
        lines.append(f"  {d:4.1f} m " + "".join(f"{table[d, t][0] * 100:7.0f}%" for t in turns))
    lines.append("")
    lines.append("Errors when found (root mean square), against what the rover assumes:")
    lines.append(f"  distance: assumed {RANGE_NOISE:.0%} of it; direction: assumed {BEARING_NOISE:.3f} rad")
    for d in distances:
        ranges = np.concatenate([table[d, t][1] for t in turns])
        bearings = np.concatenate([table[d, t][2] for t in turns])
        if len(ranges):
            worse = rms(ranges) > RANGE_NOISE or rms(bearings) > BEARING_NOISE
            lines.append(
                f"  {d:4.1f} m: distance {rms(ranges):5.1%}, direction {rms(bearings):.3f} rad"
                + ("   <- worse than assumed" if worse else "")
            )
        else:
            lines.append(f"  {d:4.1f} m: never found")
    assumed = camera.matrix[0, 0] * MARKER / MIN_PIXELS
    lines.append("")
    lines.append(f"The setup check counts markers as seen out to {assumed:.1f} m.")
    for t in turns:
        far = readable(table, t)
        verdict = "never reliably" if not far else f"reliably to {far:.1f} m"
        short = far < assumed and far < max(distances)
        lines.append(
            f"  Turned {t:2d}°: read {verdict}" + ("  <- less than the check assumes" if short else "")
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--blur", type=float, default=1.0, help="pixels; 1 is a sharp, still camera")
    parser.add_argument("--noise", type=float, default=4.0, help="grey levels; dim rooms are grainier")
    parser.add_argument("--light", type=float, default=1.0, help="brightness; 0.3 is a dim room")
    parser.add_argument("--fov", type=float, default=70.0, help="degrees across")
    parser.add_argument("--pitch", type=float, default=0.0, help="degrees down")
    parser.add_argument("--tries", type=int, default=TRIES)
    args = parser.parse_args(argv)
    if not (args.blur >= 0 and args.noise >= 0 and 0 < args.light <= 1 and args.tries > 0):
        parser.error("blur and noise must be 0 or more, light above 0 and at most 1, tries above 0")
    if not (20 <= args.fov <= 120 and abs(args.pitch) <= 45):
        parser.error("fov must be 20 to 120 degrees and pitch at most 45")
    camera = Camera(fov=args.fov, pitch=args.pitch)
    table = survey(camera, DISTANCES, TURNS, args.tries, blur=args.blur, noise=args.noise, light=args.light)
    print(report(table, camera))
    print("\nDrawn markers on a perfect lens: real ones read worse. Check with photos.")


if __name__ == "__main__":
    main()
