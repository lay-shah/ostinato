"""Guided breath-count trials: `python -m ostinato.trial`. Video is never saved."""

import argparse
import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from . import accuracy, viewpoints

VIEWS = {
    "seated": ("front-1m", "angled-1m"),
    "lying": ("side-low", "side-high"),
}
SETUP = {
    "front-1m": "Camera straight ahead, about 1 m away.",
    "angled-1m": "Camera about 45 degrees to one side, about 1 m away.",
    "side-low": "Camera at their side, level, about 0.6 m high and 1 m away. Secure it.",
    "side-high": "Camera at their side, level, about 0.9 m high and 1 m away. Secure it.",
}


def say(text):
    """Speak, so that the observer can watch the chest. Returns once it has been said."""
    command = next((c for c in ("say", "espeak-ng", "espeak") if shutil.which(c)), None)
    if command is None:
        return
    try:
        subprocess.run(
            [command, text], timeout=10, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def next_path(folder, posture):
    used = [int(m[1]) for p in folder.glob("trial-*.json") if (m := re.match(r"trial-(\d+)", p.name))]
    return folder / f"trial-{max(used, default=0) + 1:03d}-{posture}.json"


def choose(question, options):
    while True:
        answer = input(f"{question} ({' or '.join(options)}): ").strip().lower()
        found = [o for o in options if answer and o.startswith(answer)]
        if len(found) == 1:
            return found[0]


def record(camera, view):
    """One view, with the option to repeat it if the camera stops."""
    while True:
        print(f"\n{SETUP[view]}")
        try:
            return viewpoints.capture(camera, view, announce=say)
        except ValueError as error:
            print(error)
            if choose("Repeat this view?", ["yes", "no"]) == "no":
                return None


def run(folder, camera):
    """Record trials until the user stops. Return the files written."""
    if choose("Has everyone taking part agreed to it?", ["yes", "no"]) == "no":
        print("Ask them first.")
        return []
    posture = choose("Is the person seated or lying down?", list(VIEWS))
    print("The person should breathe normally. The observer counts breaths and looks only at the chest.")
    folder.mkdir(parents=True, exist_ok=True)
    saved = []
    while True:
        path = next_path(folder, posture)
        number = int(path.name.split("-")[1])
        # Alternate which view is first, so that order does not favor one.
        views = VIEWS[posture] if number % 2 else VIEWS[posture][::-1]
        pair = []
        for view in views:
            row = record(camera, view)
            if row is None:
                break
            pair.append(row)
        if len(pair) == 2:
            data = {"source": "paired-webcam-v3", "observations": [asdict(row) for row in pair]}
            with path.open("x") as output:
                json.dump(data, output, indent=2, allow_nan=False)
            saved.append(path)
            print(f"Saved {path}. Results are shown at the end, so they cannot sway the next count.")
        else:
            print("Not saved: a trial needs both views.")
        if choose("Another trial?", ["yes", "no"]) == "no":
            return saved


def summary(result):
    lines = [f"\nA rate was given in {result['reported']} of {result['windows']} windows."]
    if result["mean_absolute_error"] is not None:
        lines.append(
            f"Average error {result['mean_absolute_error']:.1f} breaths/min, largest {result['largest_error']:.1f}, "
            f"{result['within_tolerance']:.0%} within {result['tolerance_bpm']:g}."
        )
    if len(result["by_view"]) > 1:
        lines += ["", "By camera position:", *(f"  {line}" for line in accuracy.compare_lines(result))]
    for reason, n in sorted(result["declines"].items()):
        lines.append(f"No rate in {n}: {reason}")
    if result["reported"] < accuracy.FEW:
        lines.append(
            f"Preliminary: fewer than {accuracy.FEW} rates. A count itself can be off by about 2 breaths/min."
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder", type=Path, default=Path("data"), help="where trials are saved (default: data)"
    )
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        saved = run(args.folder, args.camera)
        if saved:
            result = accuracy.analyze([viewpoints.load_pair(p) for p in saved])
            print(summary(result))
            page = args.folder / f"accuracy-{time.strftime('%Y%m%d-%H%M%S')}.html"
            accuracy.save(result, page)
            print(f"Full report for this session: {page}")
    except (KeyboardInterrupt, EOFError):
        parser.exit(1, "\nStopped. Trials already saved are kept.\n")
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
