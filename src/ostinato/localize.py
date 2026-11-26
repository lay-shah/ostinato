"""Where the rover is, from its wheels and printed markers.

Print markers: `python -m ostinato.localize --print 1 2 3 4`. Tape them upright on walls
or boxes and measure where each one is. Not tried on a robot.

Between markers the estimate drifts, so it carries its own uncertainty. `tell` says
nothing to the drive guard unless the estimate is good, and the guard then stops.
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .rover import Pose, wrap

DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER = 0.16  # m, side of the printed black square
DRIFT = 0.05  # how far off each wheel's distance may be, as a share of it
DRIFT_CHANGE = 0.02  # how much that share changes in a meter
SLIP = 0.02  # m in a meter, at random, for uneven ground
RANGE_NOISE = 0.04  # share of the distance to a marker
BEARING_NOISE = 0.02  # rad
GATE = 13.8  # a marker this far from where it should be is ignored (chi-squared, 99.9%)
MAX_REJECTS = 5  # this many in a row means the estimate, not the marker, is wrong
GOOD = 0.15  # m, the largest uncertainty (one standard deviation) to drive with
GOOD_HEADING = 0.25  # rad
MAX_STEP = 0.5  # m between updates; more means a bad encoder reading


@dataclass(frozen=True)
class Camera:
    """A camera on the mast that looks to the rover's left."""

    width: int = 1280
    height: int = 720
    fov: float = 70.0  # degrees across
    pitch: float = 0.0  # degrees downward
    pan: float = 90.0  # degrees left of forward
    # From `python -m ostinato.calibrate`. Without them the lens is assumed perfect.
    focal: tuple[float, float] | None = None  # px
    center: tuple[float, float] | None = None  # px
    distortion: tuple[float, ...] = ()

    @property
    def matrix(self):
        f = self.width / 2 / math.tan(math.radians(self.fov) / 2)
        fx, fy = self.focal or (f, f)
        cx, cy = self.center or (self.width / 2, self.height / 2)
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    @property
    def coefficients(self):
        return np.array(self.distortion, float) if self.distortion else None


@dataclass(frozen=True)
class Sighting:
    marker: int
    distance: float  # m along the ground, from the camera
    bearing: float  # rad, left of the rover's forward direction


def find_markers(gray, camera, size=MARKER):
    """Markers in a picture. The picture must be the size the camera says."""
    if gray.shape[:2] != (camera.height, camera.width):
        raise ValueError("picture size does not match the camera")
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY), parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return []
    ids = [int(i) for i in ids.ravel()]
    half = size / 2
    square = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], np.float32)
    tilt = math.radians(camera.pitch)
    found = []
    for marker, quad in zip(ids, corners):
        if ids.count(marker) > 1:
            continue  # two copies: no telling which is which
        ok, _, place = cv2.solvePnP(
            square, quad[0], camera.matrix, camera.coefficients, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok or not np.isfinite(place).all():
            continue
        right, down, ahead = place.ravel()
        ahead = ahead * math.cos(tilt) - down * math.sin(tilt)  # along the ground
        if ahead > 0:
            bearing = wrap(math.radians(camera.pan) - math.atan2(right, ahead))
            found.append(Sighting(marker, math.hypot(right, ahead), bearing))
    return found


class Localizer:
    """Extended Kalman filter over x, y, heading, and how far off each wheel is."""

    def __init__(self, pose, markers, track, sigma=(0.1, 0.1, 0.1)):
        """`markers` is {number: (x, y)} in meters; `pose` is where a person put the rover."""
        values = [*pose_array(pose), *sigma, track, *(v for p in markers.values() for v in p)]
        if not all(math.isfinite(v) for v in values) or track <= 0 or min(sigma) <= 0:
            raise ValueError("invalid starting pose, markers, or track")
        self.x = np.array([*pose_array(pose), 0.0, 0.0])
        self.P = np.diag(np.square([*sigma, DRIFT, DRIFT]))
        self.markers, self.track = dict(markers), track
        self.lost = False
        self._rejects = 0

    @property
    def pose(self):
        return Pose(float(self.x[0]), float(self.x[1]), float(self.x[2]))

    @property
    def uncertainty(self):
        """Largest standard deviation of position (m), and of heading (rad)."""
        largest = np.linalg.eigvalsh(self.P[:2, :2])[-1]
        return math.sqrt(max(0.0, largest)), math.sqrt(max(0.0, self.P[2, 2]))

    @property
    def good(self):
        place, heading = self.uncertainty
        return not self.lost and place <= GOOD and heading <= GOOD_HEADING

    def drive(self, left, right):
        """The distance each wheel turned since the last call, in meters."""
        if not (math.isfinite(left) and math.isfinite(right)) or max(abs(left), abs(right)) > MAX_STEP:
            self.lost = True
            return
        # A slipping or mismeasured wheel is off by the same share for a while. Treating
        # that as random would make the estimate far too sure of itself.
        rolled = left, right
        left, right = left * (1 + self.x[3]), right * (1 + self.x[4])
        middle, turn = (left + right) / 2, (right - left) / self.track
        a = self.x[2] + turn / 2
        c, s = math.cos(a), math.sin(a)
        self.x[:3] += middle * c, middle * s, turn
        self.x[2] = wrap(self.x[2])
        k = middle / 2 / self.track
        V = np.array(
            [
                [c / 2 + k * s, c / 2 - k * s],
                [s / 2 - k * c, s / 2 + k * c],
                [-1 / self.track, 1 / self.track],
            ]
        )
        F = np.eye(5)
        F[:2, 2] = -middle * s, middle * c
        F[:3, 3:] = V * rolled
        Q = np.zeros((5, 5))
        Q[:3, :3] = V @ np.diag([SLIP**2 * abs(d) for d in rolled]) @ V.T
        Q[[3, 4], [3, 4]] = [DRIFT_CHANGE**2 * abs(d) for d in rolled]
        self.P = F @ self.P @ F.T + Q

    def see(self, sighting):
        """Use one marker. Returns whether it was believed."""
        place = self.markers.get(sighting.marker)
        if place is None or not (math.isfinite(sighting.bearing) and 0.1 < sighting.distance < 10):
            return False
        dx, dy = place[0] - self.x[0], place[1] - self.x[1]
        q = dx * dx + dy * dy
        r = math.sqrt(q)
        if r < 0.1:
            return False
        error = np.array([sighting.distance - r, wrap(sighting.bearing - math.atan2(dy, dx) + self.x[2])])
        H = np.array([[-dx / r, -dy / r, 0, 0, 0], [dy / q, -dx / q, -1, 0, 0]])
        R = np.diag([(RANGE_NOISE * sighting.distance) ** 2, BEARING_NOISE**2])
        S = H @ self.P @ H.T + R
        if error @ np.linalg.solve(S, error) > GATE:
            self._rejects += 1
            self.lost |= self._rejects >= MAX_REJECTS
            return False
        self._rejects = 0
        K = np.linalg.solve(S, H @ self.P).T
        self.x += K @ error
        self.x[2] = wrap(self.x[2])
        A = np.eye(5) - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T
        return True

    def tell(self, guard, t, person):
        """Tell the guard how near the person is, erring toward nearer."""
        if not self.good:
            return
        dx, dy = person[0] - self.x[0], person[1] - self.x[1]
        ahead = dx * math.cos(self.x[2]) + dy * math.sin(self.x[2])
        guard.person(t, math.hypot(dx, dy) - 2 * self.uncertainty[0], ahead > 0)


def pose_array(pose):
    return [pose.x, pose.y, pose.heading]


def sheet(marker, size=MARKER, dpi=300):
    """A US Letter page with one marker, to print at 100%."""
    from PIL import Image, ImageDraw

    from .ui import font

    if not 0 <= marker < 50:
        raise ValueError("marker numbers are 0 to 49")
    side = round(size / 0.0254 * dpi)
    page = Image.new("L", (round(8.5 * dpi), 11 * dpi), 255)
    if side > page.width - dpi:
        raise ValueError("too large for the page")
    # A whole number of pixels per cell keeps the edges sharp.
    side -= side % 6
    image = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(DICTIONARY), marker, side)
    page.paste(Image.fromarray(image), ((page.width - side) // 2, 2 * dpi))
    d = ImageDraw.Draw(page)
    text = f"Ostinato marker {marker}\nPrint at 100%. The black square must be {size * 1000:.0f} mm wide.\nMeasure it. Mount it upright and flat, the right way up."
    d.multiline_text(
        (page.width / 2, 9 * dpi), text, font=font(60), fill=0, anchor="ma", align="center", spacing=20
    )
    return page


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--print", type=int, nargs="+", required=True, metavar="N", help="marker numbers")
    parser.add_argument("--output", type=Path, default=Path("markers.pdf"))
    args = parser.parse_args(argv)
    try:
        pages = [sheet(n) for n in dict.fromkeys(args.print)]
        pages[0].save(args.output, "PDF", resolution=300, save_all=True, append_images=pages[1:])
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
