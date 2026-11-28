"""The setup file: a room, measured by a person, for the rover. Check one with
`python -m ostinato.robot --setup room.json --check`, add `--map room.png` to draw it.

The setup file, in meters and degrees, measured by a person:

    {"track": 0.27, "ticks_per_meter": 2000,
     "start": [2.4, 0.3, 140], "person": [0, 0, 90], "views": [30, 150],
     "markers": {"1": [0, 3, 270], "2": [3, 0, 180]},
     "camera": {"calibration": "camera.json", "pitch": 20, "height": 0.8}}

"calibration" is from `python -m ostinato.calibrate`. Without it, give "fov", the
lens's angle across, and distances to markers are rougher.

"person" is [x, y, head]: the middle of the chest, and the direction from there to the
head. The rover then keeps its distance from the whole body, about 0.75 m toward the head
and 1.1 m toward the feet. Given as [x, y], it keeps away from the chest alone, which is
right only for someone sitting or standing.

Each marker is [x, y, facing]: where it is, and the direction its printed side faces.
The facing may be left out, as [x, y], but then the check cannot tell whether a marker is
turned too far from a view to be read.

"obstacles", optional, lists known furniture as [x, y, radius] each, a circle around it.
The check then counts markers hidden behind one as unseen, and the rover goes the long way
round when the short way passes one. Things not listed are found by its distance sensors.

"battery_low", in volts, is optional but wise: below it the rover holds its wheels and
asks for a person. About 3.5 V a cell for a lithium battery, so 10.5 for three cells.

"start" is where the rover is placed and which way it faces. "views" are directions
around the person, from the chest, to measure from in order. Beside the chest is best. Check a file with
`python -m ostinato.robot --setup room.json --check` before driving. "height" is the lens's
height above the floor. With it, the camera checks that the person is where the file
says, and the rover stops if not.
"""

import json
import math
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from .calibrate import Calibration
from .localize import MARKER, Camera
from .rover import CLEAR, KEEP_OUT, RING, TRAVEL, Body, Pose, clearance, clockwise, plan, wrap


@dataclass(frozen=True)
class Setup:
    track: float
    ticks_per_meter: float
    start: Pose
    person: tuple[float, float]
    views: list[float]  # radians
    markers: dict[int, tuple[float, float]]
    camera: Camera = field(default_factory=Camera)
    lens_height: float | None = None  # m above the floor
    facings: dict[int, float] = field(default_factory=dict)  # radians, where given
    battery_low: float | None = None  # V; below this the rover holds its wheels
    head: float | None = None  # radians from the chest to the head, for someone lying down
    obstacles: list = field(default_factory=list)  # known furniture, as (x, y, radius) in meters

    @property
    def body(self):
        return Body(self.person, self.head)

    @classmethod
    def load(cls, path):
        try:
            raw = json.loads(Path(path).read_text())
            x, y, heading = (float(v) for v in raw["start"])
            camera = dict(raw.get("camera", {}))
            if not camera.keys() <= {"fov", "pitch", "pan", "height", "calibration"}:
                raise ValueError("camera takes calibration, fov, pitch, pan, and height")
            lens = camera.pop("calibration", None)
            camera = {k: float(v) for k, v in camera.items()}
            lens_height = camera.pop("height", None)
            if lens is not None:
                if "fov" in camera:
                    raise ValueError("give calibration or fov, not both")
                camera = Calibration.load(Path(path).parent / lens).camera(**camera)
            else:
                camera = Camera(**camera)
            setup = cls(
                float(raw["track"]),
                float(raw["ticks_per_meter"]),
                Pose(x, y, math.radians(heading)),
                tuple(float(v) for v in raw["person"][:2]),
                [math.radians(float(v)) for v in raw["views"]],
                {int(k): tuple(float(v) for v in p[:2]) for k, p in raw["markers"].items()},
                camera,
                lens_height,
                {int(k): math.radians(float(p[2])) for k, p in raw["markers"].items() if len(p) == 3},
                None if raw.get("battery_low") is None else float(raw["battery_low"]),
                math.radians(float(raw["person"][2])) if len(raw["person"]) == 3 else None,
                [tuple(float(v) for v in o) for o in raw.get("obstacles", [])],
            )
            if len(raw["person"]) not in (2, 3):
                raise ValueError("setup file: person is [x, y] or [x, y, head]")
            sizes = [len(p) for p in raw["markers"].values()]
        except (OSError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
            raise ValueError(f"setup file: {error}") from None
        numbers = [setup.track, setup.ticks_per_meter, *setup.person, *setup.views]
        numbers += [] if setup.head is None else [setup.head]
        numbers += [v for o in setup.obstacles for v in o]
        numbers += [v for p in setup.markers.values() for v in p] + list(setup.facings.values())
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError("setup file: every number must be finite")
        if not (0.1 <= setup.track <= 1 and setup.ticks_per_meter > 0 and len(setup.person) == 2):
            raise ValueError("setup file: check track, ticks_per_meter, and person")
        if not setup.views or not setup.markers or any(n not in (2, 3) for n in sizes):
            raise ValueError("setup file: needs views and at least one marker at [x, y, facing]")
        if any(len(o) != 3 or not 0.05 <= o[2] <= 3 for o in setup.obstacles):
            raise ValueError("setup file: each obstacle is [x, y, radius], radius 0.05 to 3 m")
        if setup.battery_low is not None and not 3 <= setup.battery_low <= 50:
            raise ValueError("setup file: battery_low must be 3 to 50 V")
        if setup.lens_height is not None and not 0.2 <= setup.lens_height <= 2:
            raise ValueError("setup file: camera height must be 0.2 to 2 m")
        if not 0 < setup.camera.fov < 170 or abs(setup.camera.pitch) > 60:
            raise ValueError("setup file: check the camera")
        return setup


BLIND = 1.0  # m driven with no marker in view, after which the position may grow too unsure
MIN_PIXELS = 30  # a marker smaller than this in the picture is often missed
# A marker turned away from the camera is read less far. From `python -m ostinato.marker_range`
# with a 2 px blur: past TURNED, reliably only to TURNED_RANGE; past TURNED_AWAY, barely.
TURNED = math.radians(45)
TURNED_RANGE = 3.0  # m
TURNED_AWAY = math.radians(65)


def longest_blind(points, visible, pan, step=0.1, center=None):
    """The longest stretch of a route, in meters, along which the camera, `pan` left of
    the rover's front, sees no marker. `visible(x, y, facing)` lists markers. With
    `center`, the person's chest, it goes round clockwise backwards, as the rover does."""
    longest = run = 0.0
    for (x0, y0), (x1, y1) in pairwise(points):
        length = math.dist((x0, y0), (x1, y1))
        if length == 0:
            continue
        heading = math.atan2(y1 - y0, x1 - x0)
        if center is not None and clockwise(center, (x0, y0), (x1, y1)):
            heading += math.pi  # backwards
        for k in range(max(1, round(length / step))):
            f = (k + 0.5) / max(1, round(length / step))
            if visible(x0 + f * (x1 - x0), y0 + f * (y1 - y0), heading + pan):
                run = 0.0
            else:
                run += length / max(1, round(length / step))
                longest = max(longest, run)
    return longest


def marker_reach(setup, n, x, y):
    """How far marker n can be from a camera at (x, y) and still be read: less when it
    is turned away, and not at all when it is turned too far."""
    sees = setup.camera.matrix[0, 0] * MARKER / MIN_PIXELS
    if n not in setup.facings:
        return sees
    mx, my = setup.markers[n]
    turn = abs(wrap(math.atan2(y - my, x - mx) - setup.facings[n]))
    if turn > TURNED_AWAY:
        return 0.0
    return min(sees, TURNED_RANGE) if turn > TURNED else sees


def check_setup(setup):
    """Problems with a setup file, before driving. Returns (errors, warnings)."""
    errors, warnings = [], []
    px, py = setup.person
    body = setup.body
    gap = body.distance(setup.start.x, setup.start.y)
    if gap < RING:
        errors.append(f"the start is {gap:.2f} m from the person; it must be at least {RING} m away")
    half = math.radians(setup.camera.fov) / 2 * 0.9  # markers at the very edge are often missed

    def reach(n, x, y):
        return marker_reach(setup, n, x, y)

    def visible(x, y, facing):
        """Markers the camera reads from (x, y), looking toward `facing`."""
        return [
            n
            for n, (mx, my) in setup.markers.items()
            if 0.5 < math.dist((x, y), (mx, my)) < reach(n, x, y)
            and abs(wrap(math.atan2(my - y, mx - x) - facing)) < half
            and clearance([(x, y), (mx, my)], setup.obstacles) > 0  # not hidden behind one
        ]

    start = setup.start
    if not visible(start.x, start.y, start.heading + math.radians(setup.camera.pan)):
        warnings.append(
            "the camera sees no marker from the start, so the first leg runs on the wheels alone "
            "and it may stop to turn and look for one"
        )
    for i, bearing in enumerate(setup.views, 1):
        x, y = body.around(bearing, RING)
        seen = visible(x, y, math.atan2(py - y, px - x))  # the camera faces the chest there
        if seen:
            warnings += (
                [] if len(seen) > 1 else [f"view {i} sees only marker {seen[0]}; two make a better fix"]
            )
        else:
            errors.append(f"view {i} sees no marker: put one beyond the person, as seen from there")
    if setup.lens_height is None:
        warnings.append(
            "no camera height: the rover cannot check that the person is where the file says, "
            "or lies the way it says, and will not stop if not"
        )
    if setup.head is None:
        warnings.append(
            "person has no head direction: for someone lying down, give [x, y, head] so the rover "
            "keeps 1 m from the whole body, not only the chest"
        )
    else:
        for i, bearing in enumerate(setup.views, 1):
            along = abs(wrap(bearing - setup.head))
            if min(along, math.pi - along) < math.radians(30):
                end = "head" if along < math.pi / 2 else "feet"
                warnings.append(f"view {i} is beyond the {end}: the chest is seen end on, and from far")
    if setup.battery_low is None:
        warnings.append("no battery_low: it will not stop for a flat battery")
    unknown = sorted(set(setup.markers) - set(setup.facings))
    if unknown:
        warnings.append(
            f"marker{'s' if len(unknown) > 1 else ''} {', '.join(map(str, unknown))} "
            "without a facing; one turned more than 45 degrees from a view may be missed"
        )
    for i, bearing in enumerate(setup.views, 1):
        x, y = body.around(bearing, RING)
        if clearance([(x, y), (x, y)], setup.obstacles) < CLEAR:
            errors.append(f"view {i} is too near an obstacle: it needs {CLEAR} m from its edge")
    where = setup.start
    for i, bearing in enumerate(setup.views, 1):
        path, long_way = plan(where, body, bearing, setup.obstacles)
        if clearance([(where.x, where.y), *path], setup.obstacles) < CLEAR:
            errors.append(f"the way to view {i} passes an obstacle whichever way round it goes")
        elif long_way:
            warnings.append(f"the short way to view {i} passes an obstacle, so it will go the long way round")
        route = [(where.x, where.y), *path]
        blind = longest_blind(route[:-1], visible, math.radians(setup.camera.pan), center=setup.person)
        blind = max(blind, longest_blind(route[-2:], visible, math.radians(setup.camera.pan)))
        if blind > BLIND:
            warnings.append(
                f"on the way to view {i}, about {blind:.1f} m with no marker in view: it may stop to turn "
                "and look for one. A marker where the camera faces on that stretch would help"
            )
        where = Pose(*path[-1], 0.0)
    for a, b in zip(setup.views, setup.views[1:]):
        if abs(wrap(a - b)) < math.radians(20):
            warnings.append("two views are within 20 degrees of each other and see much the same")
    for n, (mx, my) in setup.markers.items():
        # Near the way round, or near where it comes in to a view.
        ring = min(abs(body.distance(mx, my) - RING), abs(body.distance(mx, my) - TRAVEL))
        if ring < 0.35:
            errors.append(f"marker {n} is on the rover's path around the person; move it")
        elif body.distance(mx, my) < KEEP_OUT:
            warnings.append(f"marker {n} is close to the person; check it does not touch them")
    return errors, warnings


def view_pose(setup, view):
    """Where the rover stands, and faces, at a viewpoint (1, 2, ...), or at the start (0)."""
    if view == 0:
        return setup.start
    if not 1 <= view <= len(setup.views):
        raise ValueError(f"the setup file has views 1 to {len(setup.views)}")
    x, y = setup.body.around(setup.views[view - 1], RING)
    toward = math.atan2(setup.person[1] - y, setup.person[0] - x)
    return Pose(x, y, wrap(toward - math.radians(setup.camera.pan)))
