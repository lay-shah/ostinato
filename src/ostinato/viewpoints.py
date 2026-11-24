"""Paired camera trials and an offline, explicitly simulated viewpoint comparison."""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

WINDOW_SECONDS = 30


@dataclass(frozen=True)
class Observation:
    view: str
    elapsed_s: float
    bpm: float | None
    signal_quality: float
    reason: str
    counted_breaths: int

    @property
    def reference_bpm(self):
        return self.counted_breaths * 60 / self.elapsed_s


def summarize(pairs):
    readings = [reading for pair in pairs for reading in pair]
    accepted = [reading for reading in readings if reading.bpm is not None]
    errors = [abs(reading.bpm - reading.reference_bpm) for reading in accepted]
    declined_first = [pair for pair in pairs if pair[0].bpm is None]
    return {
        "paired_trials": len(pairs),
        "reported_windows": len(accepted),
        "total_windows": len(readings),
        "coverage": len(accepted) / len(readings) if readings else None,
        "mean_absolute_error_bpm_on_reported": mean(errors) if errors else None,
        "first_view_declines": len(declined_first),
        "second_view_reports_after_first_decline": sum(
            second.bpm is not None for _, second in declined_first
        ),
    }


def capture(camera, view):
    import cv2

    from .breathing import estimate
    from .chest import ChestTracker, PoseFinder

    input(f"Set up view {view}. Observer ready? Press Enter to begin. ")
    cap = cv2.VideoCapture(camera)
    finder = None
    try:
        if not cap.isOpened():
            raise ValueError("Camera unavailable")
        finder = PoseFinder()
        tracker = ChestTracker(finder)
        times, motion = [], []
        print("START counting now", flush=True)
        start = time.monotonic()
        while time.monotonic() - start < WINDOW_SECONDS:
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Camera stopped; repeat the trial")
            elapsed = time.monotonic() - start
            if elapsed >= WINDOW_SECONDS:
                break
            if frame.shape[1] > 640:
                scale = 640 / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            value, _ = tracker.update(frame)
            times.append(elapsed)
            motion.append(value)
        elapsed = time.monotonic() - start
        print("STOP counting", flush=True)
    finally:
        cap.release()
        if finder is not None:
            finder.close()

    # A blocked camera read can leave the manual and camera windows out of sync.
    edges = [0.0, *times, elapsed]
    if not times or max(b - a for a, b in zip(edges, edges[1:])) > 1:
        raise ValueError("Camera timing interrupted; repeat the trial")
    reading = estimate(times, motion)
    while True:
        raw = input("Observer's breath count for this window: ").strip()
        if raw.isascii() and raw.isdigit():
            break
        print("Enter a nonnegative whole number.")
    return Observation(view, elapsed, reading.bpm, reading.confidence, reading.reason, int(raw))


def simulate(name, sites, policy):
    """One target, two candidate viewpoints on a line; travel speed is one unit/s."""
    remaining = list(sites)
    position = elapsed = 0
    events = []
    while remaining:
        # Visibility is a pre-observation geometric cue. The estimator outcome is hidden.
        def rank(site):
            distance = abs(site[1] - position)
            return (site[2] if policy == "visibility-first" else 0, distance, site[0])

        chosen = min(remaining, key=rank)
        remaining.remove(chosen)
        view, location, _, outcome = chosen
        elapsed += abs(location - position) + WINDOW_SECONDS
        position = location
        events.append({"view": view, "elapsed_s": elapsed, "result": outcome})
        if outcome == "usable":
            break
    success = events[-1]["result"] == "usable"
    return {"scenario": name, "policy": policy, "events": events,
            "time_to_reading_s": elapsed if success else None, "spent_s": elapsed}


def demo():
    scenarios = {
        "near view obstructed": [("A", 1, True, "obstructed"), ("B", 4, False, "usable")],
        "both clear": [("A", 1, False, "usable"), ("B", 4, False, "usable")],
        "clear view has weak signal": [("A", 1, True, "usable"), ("B", 4, False, "weak signal")],
        "neither usable": [("A", 1, True, "obstructed"), ("B", 4, False, "weak signal")],
    }
    return {"source": "simulation", "runs": [
        simulate(name, sites, policy) for name, sites in scenarios.items()
        for policy in ("nearest-first", "visibility-first")
    ]}


def load_pair(path):
    data = json.loads(path.read_text())
    if data["source"] != "paired-webcam-v1":
        raise ValueError(f"{path}: expected a paired webcam trial")
    pair = [Observation(**row) for row in data["observations"]]
    if len(pair) != 2 or len({row.view for row in pair}) != 2:
        raise ValueError(f"{path}: expected two distinct views")
    for row in pair:
        if (not math.isfinite(row.elapsed_s) or row.elapsed_s <= 0
                or type(row.counted_breaths) is not int or row.counted_breaths < 0
                or (row.bpm is not None and not math.isfinite(row.bpm))):
            raise ValueError(f"{path}: invalid observation")
    return pair


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("simulate")
    trial = commands.add_parser("trial")
    trial.add_argument("output", type=Path)
    trial.add_argument("--views", nargs=2, required=True, metavar=("FIRST", "SECOND"))
    trial.add_argument("--camera", type=int, default=0)
    report = commands.add_parser("report")
    report.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "simulate":
            result = demo()
        elif args.command == "report":
            if len({p.resolve() for p in args.files}) != len(args.files):
                parser.error("duplicate trial file")
            result = summarize([load_pair(path) for path in args.files])
        else:
            if len(set(args.views)) != 2 or args.output.exists():
                parser.error("provide two distinct views and a new output file")
            pair = [capture(args.camera, view) for view in args.views]
            data = {"source": "paired-webcam-v1", "observations": [asdict(row) for row in pair]}
            with args.output.open("x") as output:
                json.dump(data, output, indent=2, allow_nan=False)
            result = summarize([pair])
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
