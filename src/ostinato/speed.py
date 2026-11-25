"""How fast this computer runs Ostinato: `python -m ostinato.speed`. No camera needed.

Use it before choosing the robot's computer. Without `--image`, the picture is drawn
and the pose model finds nobody, which is faster than measuring a person.
"""

import argparse
import platform
import time

import cv2
import numpy as np

from .breathing import estimate
from .chest import ChestTracker, PoseFinder
from .demo import Demo

TARGET_FPS = 10.0  # slower worked on synthetic video; this leaves room


def milliseconds(call, count):
    call()  # the first call sets things up
    start = time.perf_counter()
    for _ in range(count):
        call()
    return (time.perf_counter() - start) / count * 1000


def measure(finder, frames=60, image=None):
    """Return milliseconds for each stage of one frame. `finder` is a pose finder."""
    demo = Demo()
    demo.view = 2
    scenes = [demo.draw(i / 15) for i in range(30)]
    turn = [0]

    def scene():
        turn[0] += 1
        return scenes[turn[0] % len(scenes)]

    tracker = ChestTracker(demo)
    t = np.arange(0, 30, 1 / 15)
    x = np.sin(2 * np.pi * 0.25 * t)
    return {
        "pose model": milliseconds(lambda: finder(scene() if image is None else image), frames),
        "chest tracking": milliseconds(lambda: tracker.update(scene()), frames),
        # Runs once a second, so it is not part of every frame.
        "breathing estimate (once a second)": milliseconds(lambda: estimate(t, x), 20),
    }


def frame_rate(stages):
    frame = stages["pose model"] + stages["chest tracking"]
    return (1000 - stages["breathing estimate (once a second)"]) / frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--image", help="photo of one person, for a realistic pose time")
    args = parser.parse_args(argv)
    if args.frames < 1:
        parser.error("--frames must be at least 1")
    image = None
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            parser.error("could not read the image")
        image = cv2.resize(image, (640, max(1, round(image.shape[0] * 640 / image.shape[1]))))
    finder = PoseFinder()
    try:
        stages = measure(finder, args.frames, image)
        found = image is not None and finder(image) is not None
    finally:
        finder.close()
    fps = frame_rate(stages)
    print(f"\n{platform.machine()}, {platform.system()}, Python {platform.python_version()}")
    for name, ms in stages.items():
        print(f"  {name:<36}{ms:7.1f} ms")
    print(f"About {fps:.0f} frames/s without camera or drawing time; aim for {TARGET_FPS:.0f} or more.")
    if not found:
        print("Nobody was found, so a real person will be slower. Use --image with a photo of one person.")
    if fps < TARGET_FPS:
        print("Probably too slow. Try a faster computer.")


if __name__ == "__main__":
    main()
