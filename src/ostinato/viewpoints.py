"""Paired camera trials and an offline, explicitly simulated viewpoint comparison."""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from itertools import pairwise
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

    trace: tuple = ()

    @property
    def reference_bpm(self):
        return self.counted_breaths * 60 / self.elapsed_s


def summarize(pairs, tolerance_bpm=2.0):
    if not math.isfinite(tolerance_bpm) or tolerance_bpm < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    readings = [reading for pair in pairs for reading in pair]
    accepted = [reading for reading in readings if reading.bpm is not None]
    errors = [abs(reading.bpm - reading.reference_bpm) for reading in accepted]
    declined_first = [pair for pair in pairs if pair[0].bpm is None]
    recovery_errors = [abs(second.bpm - second.reference_bpm) for _, second in declined_first
                       if second.bpm is not None]
    return {
        "recovery_tolerance_bpm": tolerance_bpm,
        "accurate_second_view_recoveries": sum(error <= tolerance_bpm for error in recovery_errors),
        "recovered_absolute_errors_bpm": recovery_errors,
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
        times, motion, trace = [], [], []
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
            value, reason = tracker.update(frame)
            times.append(elapsed)
            motion.append(value)
            if not trace or elapsed - trace[-1][0] >= 1.0:
                if math.isfinite(value):
                    r = estimate(times, motion)
                    trace.append((elapsed, r.bpm, r.confidence, r.reason))
                else:
                    trace.append((elapsed, None, 0.0, reason))
        elapsed = time.monotonic() - start
        print("STOP counting", flush=True)
    finally:
        cap.release()
        if finder is not None:
            finder.close()

    # A blocked camera read can leave the manual and camera windows out of sync.
    edges = [0.0, *times, elapsed]
    if not times or max(b - a for a, b in pairwise(edges)) > 1:
        raise ValueError("Camera timing interrupted; repeat the trial")
    reading = estimate(times, motion)
    trace.append((elapsed, reading.bpm, reading.confidence, reading.reason))
    while True:
        raw = input("Observer's breath count for this window: ").strip()
        if raw.isascii() and raw.isdigit():
            break
        print("Enter a nonnegative whole number.")
    return Observation(view, elapsed, reading.bpm, reading.confidence, reading.reason, int(raw), tuple(trace))


def simulate(name, sites, policy):
    """One target, two candidate viewpoints on a line; travel speed is one unit/s."""
    remaining = list(sites)
    position = elapsed = 0
    events = []
    while remaining:
        # Visibility is a pre-observation geometric cue. The estimator outcome is hidden.
        def rank(site, position=position):
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
    if data["source"] not in ("paired-webcam-v1", "paired-webcam-v2"):
        raise ValueError(f"{path}: expected a paired webcam trial")
    pair = [Observation(**row) for row in data["observations"]]
    if len(pair) != 2 or len({row.view for row in pair}) != 2:
        raise ValueError(f"{path}: expected two distinct views")
    for row in pair:
        if (not math.isfinite(row.elapsed_s) or row.elapsed_s <= 0
                or type(row.counted_breaths) is not int or row.counted_breaths < 0
                or (row.bpm is not None and not math.isfinite(row.bpm))):
            raise ValueError(f"{path}: invalid observation")
        if not math.isfinite(row.signal_quality) or not 0 <= row.signal_quality <= 1:
            raise ValueError(f"{path}: invalid signal quality")
        previous = -1.0
        for t, bpm, quality, reason in row.trace:
            if not (0 <= t <= row.elapsed_s and previous < t and 0 <= quality <= 1
                    and (bpm is None or math.isfinite(bpm)) and isinstance(reason, str)):
                raise ValueError(f"{path}: invalid sample in replay trace")
            previous = t
    return pair


def replay(pair):
    """Compare the recorded and reversed view order with zero modeled travel time."""
    from .breathing import Reading
    from .decision import Action, Assessor, Limits

    if any(not view.trace for view in pair):
        raise ValueError("Replay needs timed readings from a v2 trial; v1 supports report only")
    runs = []
    for order in (pair, pair[::-1]):
        assessor = Assessor(0, 1, Limits(view_seconds=WINDOW_SECONDS))
        start = 0.0
        events = []
        for view in order:
            for t, bpm, quality, reason in view.trace:
                now = start + t
                decision = assessor.update(now, Reading(bpm, quality, reason))
                if decision.final or decision.action is Action.MOVE:
                    break
            events.append({"view": view.view, "elapsed_s": now, "action": decision.action.value,
                           "reason": decision.reason, "bpm": decision.bpm})
            if decision.action is not Action.MOVE:
                break
            assessor.arrived(now)
            start = now
        runs.append({"first_view": order[0].view, "events": events,
                     "time_to_usable_s": now if decision.bpm is not None else None})
    return {"source": "paired-trial replay", "modeled_travel_s": 0, "runs": runs}


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
    report.add_argument("--tolerance-bpm", type=float, default=2.0)
    replay_cmd = commands.add_parser("replay")
    replay_cmd.add_argument("file", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "simulate":
            result = demo()
        elif args.command == "report":
            if len({p.resolve() for p in args.files}) != len(args.files):
                parser.error("duplicate trial file")
            result = summarize([load_pair(path) for path in args.files], args.tolerance_bpm)
        elif args.command == "replay":
            result = replay(load_pair(args.file))
        else:
            if len(set(args.views)) != 2 or args.output.exists():
                parser.error("provide two distinct views and a new output file")
            pair = [capture(args.camera, view) for view in args.views]
            data = {"source": "paired-webcam-v2", "observations": [asdict(row) for row in pair]}
            with args.output.open("x") as output:
                json.dump(data, output, indent=2, allow_nan=False)
            result = summarize([pair])
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
