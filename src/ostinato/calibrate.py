"""Measure the camera's lens: `python -m ostinato.calibrate`.

1. `python -m ostinato.calibrate --print` saves a board to print at 100%. Tape it flat
   to something stiff and measure one square.
2. `python -m ostinato.calibrate --square 25 --output camera.json` opens the camera. Hold
   the board at different distances and tilts, filling the corners of the picture too,
   and press Space for each picture. Press C to finish. Nothing is saved but the result.

Put the result in the rover's setup file as "calibration".
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .localize import Camera

DICTIONARY = cv2.aruco.DICT_5X5_100  # unlike the rover's markers, so neither is mistaken
SQUARES = (7, 5)
SQUARE = 0.025  # m, as printed
MARKER_SHARE = 0.75  # of a square
MIN_VIEWS = 12
MIN_CORNERS = 8  # of the 24 inside corners, for a picture to count
MAX_ERROR = 1.0  # px, typical distance between where corners were and where they should be
MIN_COVER = 0.6  # share of the picture that corners must have reached


def board(square=SQUARE):
    return cv2.aruco.CharucoBoard(
        SQUARES, square, square * MARKER_SHARE, cv2.aruco.getPredefinedDictionary(DICTIONARY)
    )


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    focal: tuple[float, float]
    center: tuple[float, float]
    distortion: tuple[float, ...]
    error: float  # px
    views: int

    def camera(self, **placement):
        """A Camera with this lens, mounted as `placement` says (pitch, pan)."""
        fov = math.degrees(2 * math.atan(self.width / 2 / self.focal[0]))
        return Camera(
            self.width,
            self.height,
            fov,
            focal=self.focal,
            center=self.center,
            distortion=self.distortion,
            **placement,
        )

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        try:
            raw = json.loads(Path(path).read_text())
            c = cls(
                int(raw["width"]),
                int(raw["height"]),
                tuple(float(v) for v in raw["focal"]),
                tuple(float(v) for v in raw["center"]),
                tuple(float(v) for v in raw["distortion"]),
                float(raw["error"]),
                int(raw["views"]),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"calibration file: {error}") from None
        numbers = [*c.focal, *c.center, *c.distortion, c.error]
        if not (all(math.isfinite(v) for v in numbers) and c.width > 0 and c.height > 0 and min(c.focal) > 0):
            raise ValueError("calibration file: invalid numbers")
        if len(c.focal) != 2 or len(c.center) != 2 or len(c.distortion) not in (4, 5, 8):
            raise ValueError("calibration file: wrong shape")
        return c


def corners(gray, square=SQUARE):
    """The board's inside corners in a picture: (points, ids), or None."""
    detector = cv2.aruco.CharucoDetector(board(square))
    points, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < MIN_CORNERS:
        return None
    return points, ids


def calibrate(views, size, square=SQUARE):
    """`views` are (points, ids) from `corners`; `size` is (width, height).
    Returns a Calibration, or raises ValueError saying what to do."""
    if len(views) < MIN_VIEWS:
        raise ValueError(f"take at least {MIN_VIEWS} pictures; there are {len(views)}")
    everywhere = np.concatenate([p.reshape(-1, 2) for p, _ in views])
    width, height = size
    cells = {(int(4 * x / width), int(4 * y / height)) for x, y in everywhere}
    if len(cells) < MIN_COVER * 16:
        raise ValueError("move the board into the corners and edges of the picture too")
    b = board(square)
    objects, images = [], []
    for points, ids in views:
        o, i = b.matchImagePoints(points, ids)
        objects.append(o)
        images.append(i)
    error, matrix, distortion, _, _ = cv2.calibrateCamera(objects, images, size, None, None)
    if not (np.isfinite(matrix).all() and np.isfinite(distortion).all()):
        raise ValueError("the calibration failed: take new pictures")
    if error > MAX_ERROR:
        raise ValueError(f"corners are {error:.1f} px off: keep the board flat and still, in good light")
    return Calibration(
        width,
        height,
        (float(matrix[0, 0]), float(matrix[1, 1])),
        (float(matrix[0, 2]), float(matrix[1, 2])),
        tuple(float(v) for v in distortion.ravel()),
        float(error),
        len(views),
    )


def sheet(square=SQUARE, dpi=300):
    from PIL import Image, ImageDraw

    from .ui import font

    cols, rows = SQUARES
    px = round(square / 0.0254 * dpi)
    page = Image.new("L", (round(8.5 * dpi), 11 * dpi), 255)
    if cols * px > page.width - dpi / 2:
        raise ValueError("too large for the page")
    image = board(square).generateImage((cols * px, rows * px), marginSize=0)
    page.paste(Image.fromarray(image), ((page.width - cols * px) // 2, dpi))
    text = (
        f"Ostinato calibration board. Print at 100%.\n"
        f"Each square should be {square * 1000:.0f} mm. Measure one and pass --square."
    )
    d = ImageDraw.Draw(page)
    d.multiline_text(
        (page.width / 2, 2 * dpi + rows * px), text, font=font(50), fill=0, anchor="ma", align="center"
    )
    return page


def capture(camera, square):
    from .app import open_camera

    video = open_camera(camera)
    if video is None:
        raise ValueError("the camera did not open")
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    title, views, size = "Ostinato calibration", [], None
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                raise ValueError("the camera stopped")
            size = frame.shape[1], frame.shape[0]
            found = corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), square)
            view = frame.copy()
            if found:
                cv2.aruco.drawDetectedCornersCharuco(view, *found)
            for points, _ in views:
                for x, y in points.reshape(-1, 2):
                    cv2.circle(view, (int(x), int(y)), 2, (0, 160, 255), -1)
            text = f"{len(views)} pictures. Space: take one. C: finish. Q: quit."
            cv2.putText(view, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
            cv2.imshow(title, view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and found:
                views.append(found)
            elif key == ord("c"):
                return views, size
            elif key == ord("q"):
                raise KeyboardInterrupt
    finally:
        video.release()
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--print", action="store_true", help="save the board to print")
    parser.add_argument("--square", type=float, help="measured square size in mm")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("camera.json"))
    args = parser.parse_args(argv)
    try:
        if args.print:
            path = args.output.with_suffix(".pdf") if args.output.suffix == ".json" else args.output
            sheet().save(path, "PDF", resolution=300)
            print(f"Saved {path}")
            return
        if args.square is None or not 5 <= args.square <= 100:
            parser.error("measure a printed square and pass --square in mm")
        views, size = capture(args.camera, args.square / 1000)
        result = calibrate(views, size, args.square / 1000)
        result.save(args.output)
        print(
            f"Saved {args.output}: corners {result.error:.2f} px off on average, {result.camera().fov:.0f} degrees across."
        )
    except KeyboardInterrupt:
        parser.exit(1, "Stopped. Nothing saved.\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
