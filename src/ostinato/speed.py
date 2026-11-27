"""How fast this computer runs Ostinato: `python -m ostinato.speed`. No camera needed.

Use it before choosing the robot's computer. Without `--image`, the picture is drawn
and the pose model finds nobody, which is faster than measuring a person.

The rover does more with every picture than the desktop app: it also looks for markers
in the full 1280 x 720 picture, and, while waiting to measure, checks the picture is
still. So it gives two frame rates, and the rover's is the one for choosing a computer.
"""

import argparse
import platform
import time

import cv2
import numpy as np

from .breathing import estimate
from .chest import ChestTracker, PoseFinder
from .demo import Demo
from .localize import Camera, find_markers
from .marker_range import draw
from .shake import Steadiness

TARGET_FPS = 10.0  # slower worked on synthetic video; this leaves room
PER_FRAME = ("pose model", "chest tracking")
ROVER = ("markers, 1280 x 720 (rover)", "picture to grey and half size (rover)")


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
    camera = Camera()
    grey = draw(camera, [(1, -0.5, 0.0, 3.0, 0.0), (2, 0.6, 0.1, 4.0, 0.3)], blur=1.0, noise=4.0)
    colour = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    steady, clock = Steadiness(), [0]

    def settle():
        clock[0] += 1
        steady.update(clock[0] / 15, grey)

    def convert():
        cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
        cv2.resize(colour, (640, 360), interpolation=cv2.INTER_AREA)

    t = np.arange(0, 30, 1 / 15)
    x = np.sin(2 * np.pi * 0.25 * t)
    return {
        "pose model": milliseconds(lambda: finder(scene() if image is None else image), frames),
        "chest tracking": milliseconds(lambda: tracker.update(scene()), frames),
        # Runs once a second, so it is not part of every frame.
        "breathing estimate (once a second)": milliseconds(lambda: estimate(t, x), 20),
        ROVER[0]: milliseconds(lambda: find_markers(grey, camera), frames),
        ROVER[1]: milliseconds(convert, frames),
        # Only while settling, when nothing is measured, so never with the pose model.
        "steadiness (rover, while settling)": milliseconds(settle, frames),
    }


def frame_rate(stages, rover=False):
    """Pictures a second while measuring. The rover also finds markers in each one."""
    frame = sum(stages[name] for name in PER_FRAME + (ROVER if rover else ()))
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
    fps, rover = frame_rate(stages), frame_rate(stages, rover=True)
    print(f"\n{platform.machine()}, {platform.system()}, Python {platform.python_version()}")
    for name, ms in stages.items():
        print(f"  {name:<40}{ms:7.1f} ms")
    print(f"Desktop app: about {fps:.0f} frames/s while measuring.")
    print(f"Rover: about {rover:.0f} frames/s while measuring. Aim for {TARGET_FPS:.0f} or more.")
    print("Both leave out the camera's own time and drawing the window.")
    if not found:
        print("Nobody was found, so a real person will be slower. Use --image with a photo of one person.")
    if rover < TARGET_FPS:
        print("Probably too slow for the rover. Try a faster computer.")


if __name__ == "__main__":
    main()
