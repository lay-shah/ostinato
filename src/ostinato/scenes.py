"""Video stress test on drawn scenes: `python -m ostinato.scenes`.

Runs the real tracker and estimator on textures with known motion and given
landmarks. It checks for false rates and lost ones. It is not evidence of accuracy
on people.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from .app import Monitor
from .chest import ChestTracker

SIZE = (640, 480)
FPS = 10
SECONDS = 32
# Motion in these is breathing, so a rate near the truth is right.
BREATHING = ("breathing", "breathing and camera shake", "breathing and sideways sway")
# Motion in these is not, so any rate is false.
FALSE = (
    "still",
    "camera shake, faint wall",
    "moving background",
    "rocking",
    "swaying side to side",
    "wind across clothing",
    "passing shadow",
    "changing light",
    "zoom",
)
POSTURES = ("upright", "lying")


def texture(rng):
    noise = cv2.GaussianBlur(rng.integers(0, 255, (SIZE[1], SIZE[0]), np.uint8), (0, 0), 3)
    return cv2.cvtColor(cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)


def person(rng, posture):
    """Return landmarks, a body mask, a head mask, and the unit vector to the hips."""
    angle = np.radians(rng.uniform(-15, 15) if posture == "upright" else rng.choice([-1, 1]) * rng.uniform(60, 120))
    down, across = np.array([-np.sin(angle), np.cos(angle)]), np.array([np.cos(angle), np.sin(angle)])
    if posture == "lying":
        across = across * rng.uniform(0.55, 1.0)  # shoulders overlap when seen from the side
    top = np.array([320, 240]) - 85 * down
    points = [top + 65 * across, top - 65 * across, top + 190 * down + 45 * across, top + 190 * down - 45 * across]
    outline = [top - 85 * across, top + 85 * across, top + 200 * down + 75 * across, top + 200 * down - 75 * across]
    body, head = (np.zeros((SIZE[1], SIZE[0]), np.uint8) for _ in range(2))
    cv2.fillPoly(body, [np.int32(outline)], 1)
    cv2.circle(head, np.int32(top - 70 * down).tolist(), 42, 1, -1)
    return np.array(points), body, head, down


def run(task):
    """Return (scene, posture, true rate or None, rates shown after 26 s)."""
    scene, posture, seed = task
    cv2.setNumThreads(1)
    rng = np.random.default_rng(seed)
    points, body, head, down = person(rng, posture)
    bpm, size, phase = rng.uniform(9, 28), rng.uniform(1.0, 2.5), rng.uniform(0, 2 * np.pi)
    back = texture(rng)
    if scene == "camera shake, faint wall":
        back = np.clip(150 + (back.astype(float) - 128) * 0.06, 0, 255).astype(np.uint8)
    picture = np.where(body[..., None] == 1, texture(rng), back)
    face = np.full_like(back, 120) if "faint" in scene else texture(rng)
    picture = np.where(head[..., None] == 1, face, picture)
    y, x = np.mgrid[0 : SIZE[1], 0 : SIZE[0]].astype(np.float32)
    # The chest rises most, and the belly less.
    along = (x - points[:2].mean(0)[0]) * down[0] + (y - points[:2].mean(0)[1]) * down[1]
    chest = cv2.GaussianBlur((np.clip(1 - ((along - 60) / 130) ** 2, 0, 1) * body).astype(np.float32), (0, 0), 5)
    whole = cv2.GaussianBlur(((body | head) > 0).astype(np.float32), (0, 0), 3)
    person_only = whole > 0.5
    pivot, ripple = 320 + rng.uniform(-60, 60), rng.uniform(25, 120)

    def move(image, dx, dy):
        return cv2.remap(image, (x - dx).astype(np.float32), (y - dy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    monitor, shown = Monitor(ChestTracker(lambda frame: points + rng.normal(0, 0.4, points.shape))), []
    for i in range(SECONDS * FPS):
        wave = np.sin(2 * np.pi * bpm / 60 * i / FPS + phase)
        other = np.sin(2 * np.pi * bpm * 0.37 / 60 * i / FPS)
        frame = picture
        if scene.startswith("breathing"):
            frame = move(picture, whole * (size * other if "sway" in scene else 0.0), chest * size * wave)
            if "shake" in scene:
                frame = move(frame, 0 * x, 0 * y + 0.8 * np.sin(3.3 * i / FPS))
        elif scene == "camera shake, faint wall":
            frame = move(picture, 0 * x, 0 * y + 1.5 * wave)
        elif scene == "moving background":
            frame = np.where(person_only[..., None], picture, move(picture, 0 * x, 0 * y + 3 * wave))
        elif scene == "rocking":
            frame = move(picture, 0 * x, whole * size * wave)
        elif scene == "swaying side to side":
            turn = 1.5 * size / 250 * wave
            frame = move(picture, whole * (460 - y) * turn, whole * (x - pivot) * turn)
        elif scene == "wind across clothing":
            frame = move(picture, 0 * x, body * 0.8 * np.sin(x / ripple + 2 * np.pi * bpm / 60 * i / FPS))
        elif scene == "passing shadow":
            frame = picture * (1 - 0.45 / (1 + np.exp(-(y - (300 + 60 * wave)) / 25)))[..., None]
        elif scene == "changing light":
            frame = picture * (1 + 0.25 * wave)
        elif scene == "zoom":
            turn = cv2.getRotationMatrix2D((320, 240), 0, 1 + 0.004 * wave)
            frame = cv2.warpAffine(picture, turn, SIZE, borderMode=cv2.BORDER_REFLECT)
        noisy = np.clip(frame + rng.normal(0, 2, frame.shape), 0, 255).astype(np.uint8)
        reading = monitor.update(noisy, i / FPS)
        if i >= 26 * FPS:
            shown.append(reading.bpm)
    return scene, posture, bpm if scene in BREATHING else None, shown


def summarize(results):
    out = {}
    for scene in (*BREATHING, *FALSE):
        for posture in POSTURES:
            rows = [r for r in results if r[:2] == (scene, posture)]
            if not rows:
                continue
            rated = [(truth, [b for b in shown if b is not None]) for _, _, truth, shown in rows]
            given = [(truth, rates) for truth, rates in rated if rates]
            entry = {"clips": len(rows), "given_a_rate": len(given)}
            if scene in BREATHING:
                errors = [abs(b - truth) for truth, rates in given for b in rates]
                entry["largest_error"] = max(errors, default=None)
            out[f"{scene}, {posture}"] = entry
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=12, help="clips of each kind (default 12)")
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    if args.clips < 1:
        parser.error("--clips must be at least 1")
    tasks = [(s, p, args.seed + i) for s in (*BREATHING, *FALSE) for p in POSTURES for i in range(args.clips)]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(run, tasks))
    print(json.dumps({"source": "synthetic scenes", "scenes": summarize(results)}, indent=2))


if __name__ == "__main__":
    main()
