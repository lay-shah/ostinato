import json
import math

import cv2
import numpy as np
import pytest

from ostinato import calibrate
from ostinato.calibrate import Calibration, board, corners, sheet
from ostinato.localize import find_markers
from ostinato.robot import Setup

SIZE = (1280, 720)
K = np.array([[980.0, 0, 655], [0, 985.0, 352], [0, 0, 1]])
D = np.array([-0.12, 0.05, 0.0, 0.0, 0.0])


def distort(image):
    """Bend a perfect-lens picture as a lens with D would."""
    w, h = SIZE
    grid = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).reshape(-1, 1, 2).astype(np.float32)
    source = cv2.undistortPoints(grid, K, D, P=K).reshape(h, w, 2)
    return cv2.remap(image, source[..., 0], source[..., 1], cv2.INTER_LINEAR, borderValue=170)


def view(rotation, place, square=calibrate.SQUARE):
    """The board as the camera sees it, placed in front of it."""
    cols, rows = calibrate.SQUARES
    art = board(square).generateImage((cols * 120, rows * 120), marginSize=0)
    art = cv2.copyMakeBorder(art, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)
    m = 120 / square  # art pixels per meter
    corners3 = (
        np.float32(
            [
                [-60, -60, 0],
                [cols * 120 + 60, -60, 0],
                [cols * 120 + 60, rows * 120 + 60, 0],
                [-60, rows * 120 + 60, 0],
            ]
        )
        / m
    )
    projected, _ = cv2.projectPoints(corners3, np.float64(rotation), np.float64(place), K, None)
    source = np.float32([[0, 0], [art.shape[1], 0], [art.shape[1], art.shape[0]], [0, art.shape[0]]])
    h = cv2.getPerspectiveTransform(source, projected.reshape(4, 2).astype(np.float32))
    image = cv2.warpPerspective(art, h, SIZE, flags=cv2.INTER_AREA, borderValue=170)
    return distort(image)


def poses(n=16, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        z = rng.uniform(0.35, 0.6)
        x = rng.uniform(-0.35, 0.2) * z / 0.5
        y = rng.uniform(-0.2, 0.12) * z / 0.5
        out.append(((rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45), rng.uniform(-0.3, 0.3)), (x, y, z)))
    return out


@pytest.fixture(scope="module")
def views():
    found = [corners(view(r, p)) for r, p in poses()]
    return [f for f in found if f is not None]


def test_measures_the_lens(views):
    assert len(views) >= 14
    result = calibrate.calibrate(views, SIZE)
    assert result.focal == pytest.approx((980, 985), rel=0.01)
    assert result.center == pytest.approx((655, 352), abs=10)
    assert result.distortion[0] == pytest.approx(-0.12, abs=0.03)
    assert result.error < 0.5


def test_refuses_too_few_or_bunched_pictures(views):
    with pytest.raises(ValueError, match="at least"):
        calibrate.calibrate(views[:5], SIZE)
    middle = [corners(view(r, (-0.075, -0.05, 0.6))) for r, _ in poses()]
    with pytest.raises(ValueError, match="corners and edges"):
        calibrate.calibrate([m for m in middle if m], SIZE)
    assert corners(np.full((720, 1280), 128, np.uint8)) is None


def test_markers_are_closer_to_right_with_it(views, tmp_path):
    from test_localize import picture

    result = calibrate.calibrate(views, SIZE)
    result.save(tmp_path / "camera.json")
    measured = Calibration.load(tmp_path / "camera.json").camera()
    truth = measured.__class__(focal=(K[0, 0], K[1, 1]), center=(K[0, 2], K[1, 2]))
    guessed = measured.__class__(fov=60)
    image = distort(picture(truth, [(4, 0.5, 0.1, 2.0, 0.2)]))
    right = math.hypot(0.5, 2.0)
    (a,) = find_markers(image, measured)
    (b,) = find_markers(image, guessed)
    assert abs(a.distance - right) < 0.03 < abs(b.distance - right)


def test_setup_uses_it(tmp_path, views):
    calibrate.calibrate(views, SIZE).save(tmp_path / "camera.json")
    room = {
        "track": 0.27,
        "ticks_per_meter": 2000,
        "start": [1, 1, 0],
        "person": [0, 0],
        "views": [0],
        "markers": {"1": [3, 0]},
    }
    path = tmp_path / "room.json"
    path.write_text(json.dumps({**room, "camera": {"calibration": "camera.json", "pitch": 15}}))
    camera = Setup.load(path).camera
    assert camera.focal == pytest.approx((980, 985), rel=0.01) and camera.pitch == 15 and camera.distortion
    for bad in ({"calibration": "camera.json", "fov": 60}, {"calibration": "missing.json"}):
        path.write_text(json.dumps({**room, "camera": bad}))
        with pytest.raises(ValueError):
            Setup.load(path)
    (tmp_path / "broken.json").write_text(
        json.dumps({**json.loads((tmp_path / "camera.json").read_text()), "focal": [1, "x"]})
    )
    with pytest.raises(ValueError):
        Calibration.load(tmp_path / "broken.json")


def test_printed_board(tmp_path):
    page = np.array(sheet())
    small = cv2.resize(page, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    points, ids = corners(small)
    assert len(ids) == 24
    # One square is 25 mm on paper at 300 dots per inch.
    step = np.median(np.abs(np.diff(points.reshape(-1, 2)[:6, 0])))
    assert step / 0.4 / 300 * 25.4 == pytest.approx(25, abs=0.5)
    calibrate.main(["--print", "--output", str(tmp_path / "board.json")])
    assert (tmp_path / "board.pdf").read_bytes().startswith(b"%PDF")
    with pytest.raises(SystemExit):
        calibrate.main(["--output", str(tmp_path / "x.json")])
