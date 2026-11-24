"""2D simulated search with several casualties. Every result here is simulated.

Planners only see `Known` data. Whether a view really works, and who is really
urgent, stays in `Person` and is used only to produce readings and scores.
Person locations are assumed known. Nothing here models real sensors or injuries.
"""

import argparse
import heapq
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .breathing import MIN_SECONDS, Reading
from .decision import Action, Assessor, Status

N = 40  # grid cells per side
CELL = 0.5  # meters
SPEED = 1.0  # meters/s, as in viewpoints.py
VIEW_DISTANCE = 2.0  # meters
EXTRA_VIEWS = 3
ORDERS = ("nearest", "ranked")
VIEWS = ("nearest", "predicted")
BASELINE = "nearest-first"
MOVING = Reading(None, 0.0, "stream interrupted")
CUES = {"useful": 0.25, "weak": 0.08, "useless": 0.0}  # how far urgency shifts the cue


@dataclass(frozen=True)
class Known:
    id: int
    cell: tuple[int, int]
    facing: float  # estimated chest direction, radians
    cue: float  # noisy urgency cue, 0-1


@dataclass(frozen=True)
class Person:
    known: Known
    urgent: bool
    facing: float
    weak_signal: bool
    blocked: frozenset  # view indices hidden by unmapped debris
    bpm: float


@dataclass(frozen=True)
class Scene:
    walls: np.ndarray
    start: tuple[int, int]
    people: tuple[Person, ...]

    def no_go(self):
        """Walls plus a one-cell margin around every casualty: never drive there."""
        grid = self.walls.copy()
        for p in self.people:
            r, c = p.known.cell
            grid[max(0, r - 1) : r + 2, max(0, c - 1) : c + 2] = True
        return grid


def angle_gap(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def make_scene(seed, cue="useful"):
    rng = np.random.default_rng(seed)
    walls = np.zeros((N, N), bool)
    for _ in range(9):
        h, w = rng.integers(2, 10, 2)
        r, c = rng.integers(2, N - 2, 2)
        walls[r : r + h, c : c + w] = True
    start = (0, 0)
    free = [tuple(map(int, x)) for x in np.argwhere(~walls) if max(x) >= 6]
    people, cells = [], []
    for i in range(int(rng.integers(4, 8))):
        while True:
            cell = free[int(rng.integers(len(free)))]
            if all(math.dist(cell, c) >= 4 for c in cells):
                break
        cells.append(cell)
        urgent = bool(rng.random() < 0.35)
        facing = float(rng.uniform(-math.pi, math.pi))
        # Draw the same random numbers for every cue level, so scenes stay paired.
        cue_value = 0.5 + CUES[cue] * (1 if urgent else -1) + 0.25 * float(rng.normal())
        known = Known(i, cell, facing + float(rng.normal(0, 0.35)), min(0.95, max(0.05, cue_value)))
        blocked = frozenset(int(v) for v in range(8) if rng.random() < 0.2)
        bpm = float(rng.uniform(26, 38) if urgent else rng.uniform(10, 24))
        people.append(Person(known, urgent, facing, bool(rng.random() < 0.1), blocked, bpm))
    return Scene(walls, start, tuple(people))


def is_free(walls, cell):
    r, c = cell
    return 0 <= r < N and 0 <= c < N and not walls[r, c]


def paths_from(walls, start):
    """Dijkstra. Return (meters to each reachable cell, previous cell)."""
    cost, prev, heap = {start: 0.0}, {}, [(0.0, start)]
    while heap:
        d, cell = heapq.heappop(heap)
        if d > cost[cell]:
            continue
        r, c = cell
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nxt = (r + dr, c + dc)
                if nxt == cell or not is_free(walls, nxt):
                    continue
                # Do not cut wall corners.
                if dr and dc and (walls[r + dr, c] or walls[r, c + dc]):
                    continue
                total = d + CELL * math.hypot(dr, dc)
                if total < cost.get(nxt, math.inf) - 1e-9:
                    cost[nxt], prev[nxt] = total, cell
                    heapq.heappush(heap, (total, nxt))
    return cost, prev


def trace(prev, start, goal):
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1]


def line_clear(walls, a, b):
    n = max(2, int(math.dist(a, b) * 8))
    return not any(
        walls[round(a[0] + (b[0] - a[0]) * i / n), round(a[1] + (b[1] - a[1]) * i / n)] for i in range(n + 1)
    )


def view_angle(index):
    return index * math.pi / 4


def mapped_views(walls, no_go, known):
    """Views that are safe to stand at and unobstructed on the map: {index: cell}."""
    out = {}
    for i in range(8):
        cell = (
            known.cell[0] + round(VIEW_DISTANCE / CELL * math.sin(view_angle(i))),
            known.cell[1] + round(VIEW_DISTANCE / CELL * math.cos(view_angle(i))),
        )
        if is_free(no_go, cell) and line_clear(walls, known.cell, cell):
            out[i] = cell
    return out


def predicted_usable(known, index):
    gap = math.degrees(angle_gap(known.facing, view_angle(index)))
    return 0.8 if gap < 60 else 0.4 if gap < 90 else 0.1


def choose_view(known, views, cost, policy):
    """Return the index of the best reachable view, or None."""
    reachable = [i for i in views if views[i] in cost]
    if not reachable:
        return None
    if policy == "nearest":
        return min(reachable, key=lambda i: (cost[views[i]], i))
    return max(
        reachable, key=lambda i: (predicted_usable(known, i) / (cost[views[i]] / SPEED + MIN_SECONDS), -i)
    )


def choose_person(candidates, cost, policy):
    """`candidates` is [(Known, views)]. Travel is to the closest view, in both policies."""

    def meters(item):
        return min(cost[c] for c in item[1].values() if c in cost)

    if policy == "nearest":
        return min(candidates, key=lambda x: (meters(x), x[0].id))
    return max(candidates, key=lambda x: (x[0].cue / (meters(x) / SPEED + MIN_SECONDS), -x[0].id))


def observe(person, index, seconds):
    """Simulated reading after `seconds` at a view."""
    if index in person.blocked or angle_gap(person.facing, view_angle(index)) > math.radians(75):
        return Reading(None, 0.0, "chest not visible")
    if seconds < MIN_SECONDS:
        return Reading(None, 0.0, "need more data")
    if person.weak_signal:
        return Reading(None, 0.0, "signal too weak")
    return Reading(person.bpm, 0.9, "ok")


class _Robot:
    def __init__(self, scene):
        self.no_go, self.cell, self.t, self.track = scene.no_go(), scene.start, 0.0, [scene.start]
        self._cache = {}

    def paths(self):
        if self.cell not in self._cache:
            self._cache[self.cell] = paths_from(self.no_go, self.cell)
        return self._cache[self.cell]

    def drive(self, goal, each_step=lambda: False):
        """Drive until the goal, or until `each_step` returns True. Return True on arrival."""
        _, prev = self.paths()
        for nxt in trace(prev, self.cell, goal)[1:]:
            self.t += CELL * math.dist(self.cell, nxt) / SPEED
            self.cell = nxt
            self.track.append(nxt)
            if each_step():
                return False
        return True


def assess(robot, person, views, policy):
    known, views = person.known, dict(views)
    cost, _ = robot.paths()
    index = choose_view(known, views, cost, policy)
    robot.drive(views[index])
    reached = robot.t
    a = Assessor(robot.t, min(EXTRA_VIEWS, len(views) - 1))
    used, arrived = [index], robot.t
    del views[index]
    while not a.decision.final:
        robot.t += 1.0
        reading = observe(person, index, robot.t - arrived)
        d = a.update(robot.t, reading)
        while d.action is Action.MOVE:
            cost, _ = robot.paths()
            index = choose_view(known, views, cost, policy)
            if index is None:
                d = a.move_failed(robot.t)
                continue
            target = views.pop(index)
            if robot.drive(target, lambda: a.update(robot.t, MOVING).final):
                d, arrived = a.arrived(robot.t), robot.t
                used.append(index)
            else:
                break
    d = a.decision
    score = reading.confidence if d.bpm is not None else None
    return {
        "person": known.id, "reached_s": reached, "finished_s": robot.t, "views": used,
        "status": d.status.value, "bpm": d.bpm, "score": score, "reason": d.reason,
    }  # fmt: skip


def run(scene, order, view):
    robot, results = _Robot(scene), []
    waiting = {p.known.id: p for p in scene.people}
    cost, _ = robot.paths()
    views = {}
    for i, p in list(waiting.items()):
        views[i] = {k: c for k, c in mapped_views(scene.walls, robot.no_go, p.known).items() if c in cost}
        if not views[i]:  # never driven to, under any policy
            del waiting[i]
            results.append({
                "person": i, "reached_s": None, "finished_s": None, "views": [],
                "status": Status.UNABLE.value, "bpm": None, "score": None, "reason": "no reachable view",
            })  # fmt: skip
    while waiting:
        cost, _ = robot.paths()
        known, _ = choose_person([(p.known, views[i]) for i, p in waiting.items()], cost, order)
        results.append(assess(robot, waiting.pop(known.id), views[known.id], view))
    return {"policy": name(order, view), "results": results, "total_s": robot.t, "track": robot.track}


def name(order, view):
    if (order, view) == ("nearest", "nearest"):
        return BASELINE
    return f"{order} person, {view} view"


def rank_for_medics(results):
    """Attention order with reasons. Illustrative, not a clinical triage method."""

    def key(r):
        if r["bpm"] is None:
            return (0, 0)
        return (1, -r["bpm"]) if not 10 <= r["bpm"] <= 30 else (2, -r["bpm"])

    out = []
    for r in sorted(results, key=lambda r: (*key(r), r["person"])):
        if r["bpm"] is None:
            why = f"unable to assess ({r['reason']}): needs a human check"
        else:
            note = "outside" if key(r)[0] == 1 else "within"
            why = f"{r['bpm']:.0f} breaths/min, {note} 10-30"
        out.append({"person": r["person"], "why": why})
    return out


def _interval(x):
    rng = np.random.default_rng(0)
    means = rng.choice(x, (4000, len(x))).mean(axis=1)
    return [round(float(v), 1) for v in np.percentile(means, [2.5, 97.5])]


def compare(seeds, cue="useful", progress=False):
    """Paired comparison. Primary measure: seconds to reach urgent people."""
    policies = [(o, v) for o in ORDERS for v in VIEWS]
    reach = {name(*p): [] for p in policies}  # one mean per scene
    finish = {k: [] for k in reach}
    total = {k: [] for k in reach}
    measured = {k: [0, 0] for k in reach}
    reachable = urgent_count = 0
    for seed in seeds:
        if progress:
            print(f"{cue} cue, scene {seed}", file=sys.stderr, flush=True)
        scene = make_scene(seed, cue)
        urgent = {p.known.id for p in scene.people if p.urgent}
        urgent_count += len(urgent)
        for p in policies:
            k, out = name(*p), run(scene, *p)
            hits = [r for r in out["results"] if r["person"] in urgent and r["reached_s"] is not None]
            if hits:
                reach[k].append(np.mean([r["reached_s"] for r in hits]))
                finish[k].append(np.mean([r["finished_s"] for r in hits]))
            if p == policies[0]:
                reachable += len(hits)
            total[k].append(out["total_s"])
            measured[k][0] += sum(r["bpm"] is not None for r in out["results"])
            measured[k][1] += len(out["results"])
    base = np.array(reach[BASELINE])
    if not len(base):
        raise ValueError("no reachable urgent people in these scenes")
    rows = []
    for k, times in reach.items():
        diff = np.array(times) - base
        rows.append({
            "policy": k,
            "urgent_reached_s": round(float(np.mean(times)), 1),
            "change_vs_nearest_first_s": round(float(diff.mean()), 1),
            "change_95ci": _interval(diff),
            "scenes_faster_same_slower": [int((diff < -1e-6).sum()), int((abs(diff) <= 1e-6).sum()),
                                          int((diff > 1e-6).sum())],
            "urgent_finished_s": round(float(np.mean(finish[k])), 1),
            "total_s": round(float(np.mean(total[k])), 1),
            "measured_fraction": round(measured[k][0] / measured[k][1], 3),
        })  # fmt: skip
    return {
        "source": "simulation", "cue": cue, "scenes": len(list(seeds)), "scenes_with_urgent": len(base),
        "urgent_people": urgent_count, "urgent_reachable": reachable, "policies": rows,
    }  # fmt: skip


def draw(scene, out, path):
    import cv2

    s = 18
    img = np.full((N * s + 60, N * s, 3), 245, np.uint8)
    for r, c in np.argwhere(scene.walls):
        cv2.rectangle(img, (int(c) * s, int(r) * s), (int(c + 1) * s, int(r + 1) * s), (90, 90, 90), -1)

    def px(cell):
        return cell[1] * s + s // 2, cell[0] * s + s // 2

    cv2.polylines(img, [np.array([px(c) for c in out["track"]], np.int32)], False, (210, 130, 40), 2)
    cv2.circle(img, px(scene.start), 7, (210, 130, 40), -1)
    status = {r["person"]: r for r in out["results"]}
    for visit, r in enumerate(out["results"], 1):
        p = scene.people[r["person"]]
        x, y = px(p.known.cell)
        color = (60, 150, 60) if status[p.known.id]["bpm"] is not None else (0, 140, 255)
        cv2.circle(img, (x, y), 9, color, -1)
        if p.urgent:
            cv2.circle(img, (x, y), 14, (40, 40, 200), 2)
        label = f"P{p.known.id}" + (f" #{visit - sum(q['reached_s'] is None for q in out['results'])}"
                                    if r["reached_s"] is not None else " no path")  # fmt: skip
        cv2.putText(img, label, (min(x + 12, N * s - 80), max(y - 10, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    y = N * s + 22
    cv2.putText(img, f"SIMULATED: {out['policy']}, {out['total_s']:.0f} s", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    key = "green: measured  orange: unable to assess  red ring: truly urgent (hidden from robot)  #: visit order"
    cv2.putText(img, key, (8, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)
    if not cv2.imwrite(str(path), img):
        raise OSError(f"could not write {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compare", help="compare policies over many scenes")
    c.add_argument("--scenes", type=int, default=100)
    c.add_argument("--first-seed", type=int, default=0)
    s = sub.add_parser("scene", help="run one scene")
    s.add_argument("seed", type=int)
    s.add_argument("--policy", choices=[BASELINE, "ranked"], default="ranked")
    s.add_argument("--image", help="save a map as a PNG")
    s.add_argument("--html", help="save the offline medic page")
    args = parser.parse_args(argv)
    if args.command == "compare":
        if args.scenes < 1 or args.first_seed < 0:
            parser.error("scenes must be positive and first-seed nonnegative")
        seeds = range(args.first_seed, args.first_seed + args.scenes)
        print(json.dumps([compare(seeds, cue, progress=True) for cue in CUES], indent=2))
        return
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    scene = make_scene(args.seed)
    choice = ("nearest", "nearest") if args.policy == BASELINE else ("ranked", "predicted")
    out = run(scene, *choice)
    if args.image:
        draw(scene, out, args.image)
    if args.html:
        from .medic import render

        Path(args.html).write_text(render(scene, out), encoding="utf-8")
    out.pop("track")
    print(json.dumps({"source": "simulation", **out, "attention_order": rank_for_medics(out["results"])}, indent=2))


if __name__ == "__main__":
    main()
