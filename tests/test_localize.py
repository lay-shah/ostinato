import math

import cv2
import numpy as np
import pytest

from ostinato import localize
from ostinato.localize import Camera, Localizer, Sighting, find_markers, main, sheet
from ostinato.rover import KEEP_OUT, STOP, Drive, Guard, Mission, Pose, SimulatedBase, State, wrap

MARKERS = {1: (0.0, 3.0), 2: (3.0, 0.0), 3: (0.0, -3.0), 4: (-3.0, 0.0)}
TRACK = 0.3


def picture(camera, places, size=localize.MARKER, blur=0.0, noise=0.0):
    """Draw markers as the camera would see them. places: (id, right, down, ahead, yaw)
    in meters along the ground from the camera."""
    ss = 2
    w, h = camera.width * ss, camera.height * ss
    image = np.full((h, w), 150, np.uint8)
    tilt = math.radians(camera.pitch)
    for marker, right, down, ahead, yaw in places:
        code = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(localize.DICTIONARY), marker, 600
        )
        code = cv2.copyMakeBorder(code, 120, 120, 120, 120, cv2.BORDER_CONSTANT, value=255)
        half = size / 2 * 840 / 600
        quad = []
        for u, v in ((-half, -half), (half, -half), (half, half), (-half, half)):
            x, y, z = right + u * math.cos(yaw), down + v, ahead + u * math.sin(yaw)
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
    if blur:
        image = cv2.GaussianBlur(image, (0, 0), blur)
    image += np.random.default_rng(0).normal(0, noise, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("pitch", [0, 25])
@pytest.mark.parametrize("ahead", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("right,yaw", [(0.0, 0.0), (0.4, 0.5), (-0.4, -0.6)])
def test_finds_markers(pitch, ahead, right, yaw):
    camera = Camera(pitch=pitch)
    down = ahead * math.tan(math.radians(pitch))  # in the middle of the picture
    (seen,) = find_markers(picture(camera, [(7, right, down, ahead, yaw)], blur=1.0, noise=4), camera)
    assert seen.marker == 7
    assert seen.distance == pytest.approx(math.hypot(right, ahead), rel=0.04)
    assert seen.bearing == pytest.approx(math.pi / 2 - math.atan2(right, ahead), abs=0.02)


def test_ignores_copies_and_blank_pictures():
    camera = Camera()
    twice = picture(camera, [(7, -0.5, 0, 2, 0), (7, 0.5, 0, 2, 0), (8, 0, 0, 2, 0)])
    assert [s.marker for s in find_markers(twice, camera)] == [8]
    assert find_markers(np.full((720, 1280), 128, np.uint8), camera) == []
    with pytest.raises(ValueError):
        find_markers(np.zeros((480, 640), np.uint8), camera)


def look(pose, rng, marker, offset=0.0):
    x, y = MARKERS[marker]
    r = math.dist((pose.x, pose.y), (x, y))
    bearing = wrap(math.atan2(y - pose.y, x - pose.x) - pose.heading)
    return Sighting(marker, r * (1 + rng.normal(0, 0.02)) + offset, bearing + rng.normal(0, 0.01))


def wander(seed, seeing, slip=0.04, steps=1200):
    """Drive about with slipping wheels. Yields the truth and the estimate."""
    rng = np.random.default_rng(seed)
    true = Pose(2.0, 0.5, 1.0)
    start = Pose(2.0 + rng.normal(0, 0.1), 0.5 + rng.normal(0, 0.1), 1.0 + rng.normal(0, 0.1))
    loc = Localizer(start, MARKERS, TRACK)
    bias = 1 + rng.normal(0, slip, 2)
    for i in range(steps):
        left, right = 0.010, 0.0115 + 0.002 * math.sin(i / 60)
        gone = (left * bias[0], right * bias[1])
        middle, turn = sum(gone) / 2, (gone[1] - gone[0]) / TRACK
        a = true.heading + turn / 2
        true = Pose(true.x + middle * math.cos(a), true.y + middle * math.sin(a), wrap(true.heading + turn))
        loc.drive(left, right)
        if seeing and i % 10 == 0:
            loc.see(look(true, rng, 1 + (i // 100) % 4))
        yield true, loc


def off(true, loc):
    return math.dist((true.x, true.y), (loc.pose.x, loc.pose.y))


def test_markers_hold_the_estimate():
    """Near the markers it is right, and knows how right, even when a wheel keeps slipping."""
    for slip in (0.03, 0.06):
        errors = []
        for seed in range(10):
            for i, (true, loc) in enumerate(wander(seed, True, slip)):
                if i > 300 and math.hypot(true.x, true.y) < 5:
                    assert loc.good
                    errors.append((off(true, loc), off(true, loc) / loc.uncertainty[0]))
        error, spread = np.array(errors).T
        assert error.max() < 0.3 and np.mean(spread < 2) > 0.95


def test_without_markers_it_stops_trusting_itself_in_time():
    for seed in range(40):
        for true, loc in wander(seed, False):
            if not loc.good:
                break
            assert off(true, loc) < 0.6
        assert not loc.good


def test_moved_marker_is_rejected_then_lost():
    rng, pose = np.random.default_rng(1), Pose(0.0, 0.0, 0.0)
    loc = Localizer(pose, MARKERS, TRACK)
    for _ in range(10):
        assert loc.see(look(pose, rng, 1)) and loc.see(look(pose, rng, 2))
    assert not loc.see(look(pose, rng, 3, offset=1.0))
    assert loc.good and off(pose, loc) < 0.05
    for _ in range(5):
        loc.see(look(pose, rng, 3, offset=1.0))
    assert loc.lost and not loc.good


def test_bad_data():
    loc = Localizer(Pose(0, 0, 0), MARKERS, TRACK)
    for s in (
        Sighting(9, 1, 0),
        Sighting(1, math.nan, 0),
        Sighting(1, 1, math.nan),
        Sighting(1, -1, 0),
        Sighting(1, 99, 0),
    ):
        assert not loc.see(s)
    assert loc.good
    loc.drive(math.nan, 0)
    assert loc.lost
    for pose, markers, track in (
        (Pose(math.nan, 0, 0), {}, 0.3),
        (Pose(0, 0, 0), {1: (math.inf, 0)}, 0.3),
        (Pose(0, 0, 0), {}, 0),
    ):
        with pytest.raises(ValueError):
            Localizer(pose, markers, track)


def test_guard_stops_when_lost():
    guard, loc = Guard(), Localizer(Pose(3, 0, 0), MARKERS, TRACK)

    def ask(t):
        guard.link(t), guard.sensors(t, 9, 9), loc.tell(guard, t, (0, 0))
        return guard.allow(t, Drive(0.2, 0))

    ask(0.0)
    assert ask(0.1).speed > 0
    loc.drive(5, 5)
    assert [ask(t) for t in (0.5, 0.9)][-1] == STOP and guard.reason == "no news from person"


def mission(seed, blind=False):
    """The whole drive on estimates alone. Returns how it ended and the nearest it came."""
    rng = np.random.default_rng(seed)
    person = (0.0, 0.0)
    base = SimulatedBase(Pose(2.4, 0.3, 2.5), person, rng)
    loc = Localizer(Pose(2.45, 0.25, 2.55), MARKERS, TRACK)
    guard, run = Guard(), Mission(person, [0.5, 2.6])
    drive, nearest, t, dt = STOP, math.inf, 0.0, 0.05
    while t < 200 and run.state not in (State.MEASURING, State.DONE):
        guard.link(t), guard.sensors(t, 9, 9)
        loc.tell(guard, t, person)
        if not blind and int(t / dt) % 10 == 0:
            for m in MARKERS:
                s = look(base.pose, rng, m)
                if abs(wrap(s.bearing - math.pi / 2)) < 0.6:  # only what the camera faces
                    loc.see(s)
        drive = guard.allow(t, run.update(t, loc.pose, drive != STOP))
        before = base.pose
        base.step(drive, dt)
        turned = wrap(base.pose.heading - before.heading) * (1 + rng.normal(0, 0.05))
        gone = math.copysign(math.dist((base.pose.x, base.pose.y), (before.x, before.y)), drive.speed)
        loc.drive(gone - turned * TRACK / 2, gone + turned * TRACK / 2)
        nearest = min(nearest, math.dist((base.pose.x, base.pose.y), person))
        t += dt
    return run.state, nearest


def test_drives_a_mission_on_estimates():
    runs = [mission(seed) for seed in range(20)]
    assert sum(state is State.MEASURING for state, _ in runs) >= 19
    assert min(near for _, near in runs) > KEEP_OUT


def test_without_markers_a_mission_stops_away_from_the_person():
    runs = [mission(seed, blind=True) for seed in range(10)]
    assert all(state is not State.MEASURING for state, _ in runs)
    assert min(near for _, near in runs) > KEEP_OUT


def test_print(tmp_path, capsys):
    out = tmp_path / "m.pdf"
    main(["--print", "1", "2", "--output", str(out)])
    assert out.read_bytes().startswith(b"%PDF") and "Saved" in capsys.readouterr().out
    page = np.array(sheet(3))
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(localize.DICTIONARY))
    corners, ids, _ = detector.detectMarkers(
        cv2.resize(page, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    )
    assert ids.ravel().tolist() == [3]
    assert np.ptp(corners[0][0][:, 0]) * 4 / 300 * 25.4 == pytest.approx(160, abs=1.5)
    with pytest.raises(SystemExit):
        main(["--print", "50", "--output", str(out)])
