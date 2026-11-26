import inspect
import math
from dataclasses import replace

import numpy as np
import pytest

from ostinato import simulation as sim
from ostinato.decision import VIEW_PROBLEMS, Status

# Seeds 0-99 are reserved for the reported comparison.
SEEDS = range(1000, 1015)
POLICIES = [(o, v) for o in sim.ORDERS for v in sim.VIEWS]


def test_same_seed_same_scene_and_cue_levels_only_change_cue():
    a, b = sim.make_scene(1000), sim.make_scene(1000)
    assert (a.walls == b.walls).all() and a.people == b.people
    c = sim.make_scene(1000, "useless")
    assert (a.walls == c.walls).all()
    for p, q in zip(a.people, c.people):
        assert replace(p.known, cue=0) == replace(q.known, cue=0)
        assert replace(p, known=q.known) == q


def test_path_goes_around_wall_without_cutting_corners():
    walls = np.zeros((sim.N, sim.N), bool)
    walls[0:20, 10] = True
    cost, prev = sim.paths_from(walls, (0, 0))
    path = sim.trace(prev, (0, 0), (0, 20))
    assert all(not walls[c] for c in path)
    assert cost[(0, 20)] > 2 * 10 * sim.CELL
    assert cost[(0, 5)] == pytest.approx(2.5)
    walls[:, 10] = True
    assert (0, 20) not in sim.paths_from(walls, (0, 0))[0]
    walls[:] = False
    walls[0, 1] = walls[1, 0] = True
    assert len(sim.paths_from(walls, (0, 0))[0]) == 1


def test_wall_removes_view():
    walls = np.zeros((sim.N, sim.N), bool)
    known = sim.Known(0, (20, 20), 0.0, 0.5)
    assert len(sim.mapped_views(walls, walls, known)) == 8
    walls[20, 22] = True
    assert 0 not in sim.mapped_views(walls, walls, known)


@pytest.mark.parametrize("seed", SEEDS)
def test_every_run_is_valid(seed):
    scene = sim.make_scene(seed)
    for policy in POLICIES:
        out = sim.run(scene, *policy)
        assert sorted(r["person"] for r in out["results"]) == [p.known.id for p in scene.people]
        assert not any(scene.walls[c] for c in out["track"])
        # Never drive over, or right next to, a casualty.
        for p in scene.people:
            assert all(max(abs(c[0] - p.known.cell[0]), abs(c[1] - p.known.cell[1])) > 1 for c in out["track"])
        assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 for a, b in zip(out["track"], out["track"][1:]))
        previous = 0
        for r in out["results"]:
            person = scene.people[r["person"]]
            assert len(set(r["views"])) == len(r["views"]) <= 1 + sim.EXTRA_VIEWS
            if r["reached_s"] is not None:
                assert previous <= r["reached_s"] < r["finished_s"]
                assert r["finished_s"] - r["reached_s"] < 182
                previous = r["finished_s"]
            if r["bpm"] is None:
                assert r["status"] == Status.UNABLE.value
            else:
                assert r["bpm"] == person.bpm and not person.weak_signal
                assert r["views"][-1] not in person.blocked


def test_reachability_does_not_depend_on_policy():
    for seed in SEEDS:
        sets = [
            {r["person"] for r in sim.run(sim.make_scene(seed), *p)["results"] if r["reached_s"] is None}
            for p in POLICIES
        ]
        assert all(s == sets[0] for s in sets)


def test_planners_cannot_see_hidden_truth():
    for f in (sim.choose_person, sim.choose_view, sim.predicted_usable, sim.mapped_views):
        assert "person" not in inspect.signature(f).parameters
    assert set(sim.Known.__dataclass_fields__) == {"id", "cell", "facing", "cue"}
    # Changing who is truly urgent must not change where the robot goes.
    scene = sim.make_scene(1001)
    flipped = sim.Scene(scene.walls, scene.start, tuple(replace(p, urgent=not p.urgent) for p in scene.people))
    assert sim.run(scene, "ranked", "predicted") == sim.run(flipped, "ranked", "predicted")


def test_ranked_order_prefers_urgent_cue_at_equal_distance():
    cost = {(0, 1): 1.0, (0, 2): 1.0, (0, 3): 30.0}
    low, high = sim.Known(0, (5, 5), 0, 0.2), sim.Known(1, (6, 6), 0, 0.8)
    items = [(low, {0: (0, 1)}), (high, {0: (0, 2)})]
    assert sim.choose_person(items, cost, "ranked")[0] is high
    assert sim.choose_person(items, cost, "nearest")[0] is low


def test_predicted_view_prefers_the_front():
    known = sim.Known(0, (20, 20), math.pi, 0.5)
    empty = np.zeros((sim.N, sim.N), bool)
    views = sim.mapped_views(empty, empty, known)
    cost = {c: 5.0 for c in views.values()}
    assert sim.choose_view(known, views, cost, "predicted") in (3, 4, 5)
    assert sim.choose_view(known, views, {}, "predicted") is None


def test_unreadable_person_is_never_given_a_rate():
    base = sim.make_scene(1002)
    people = tuple(replace(p, weak_signal=True) for p in base.people)
    out = sim.run(sim.Scene(base.walls, base.start, people), "ranked", "predicted")
    assert all(r["bpm"] is None and r["status"] == Status.UNABLE.value for r in out["results"])


def test_simulated_problems_are_real_decision_reasons():
    p = replace(sim.make_scene(1003).people[0], weak_signal=True, facing=0.0, blocked=frozenset())
    assert sim.observe(p, 0, 30).reason in VIEW_PROBLEMS
    assert sim.observe(p, 4, 30).reason in VIEW_PROBLEMS


def test_medic_order_puts_unassessed_first():
    rows = [{"person": i, "bpm": b, "reason": "x"} for i, b in enumerate([15, 35, None])]
    ranked = sim.rank_for_medics(rows)
    assert [r["person"] for r in ranked] == [2, 1, 0]
    assert "human check" in ranked[0]["why"]


def test_comparison_is_labeled_and_baseline_change_is_zero():
    out = sim.compare(range(1000, 1004))
    assert out["source"] == "simulation"
    row = out["policies"][0]
    assert row["policy"] == sim.BASELINE and row["change_vs_nearest_first_s"] == 0


def test_map_image(tmp_path):
    scene = sim.make_scene(1000)
    sim.draw(scene, sim.run(scene, "ranked", "predicted"), tmp_path / "map.png")
    assert (tmp_path / "map.png").stat().st_size > 1000


def test_worlds_stay_paired_when_one_assumption_changes():
    a, b = sim.make_scene(1000), sim.make_scene(1000, world=sim.World(facing_error=3.0))
    assert (a.walls == b.walls).all()
    for p, q in zip(a.people, b.people):
        assert (p.known.cell, p.facing, p.urgent, p.bpm) == (q.known.cell, q.facing, q.urgent, q.bpm)


def test_extreme_worlds_behave():
    none = sim.compare(range(1000, 1004), world=sim.World(weak=1.0))
    assert all(p["measured_fraction"] == 0 for p in none["policies"])
    hidden = sim.compare(range(1000, 1004), world=sim.World(hidden=1.0))
    assert all(p["measured_fraction"] == 0 for p in hidden["policies"])
    with pytest.raises(ValueError):
        sim.compare(range(1000, 1002), policies=[("ranked", "predicted")])


def test_sweep_covers_every_assumption(monkeypatch):
    monkeypatch.setattr(sim, "SWEEP", {k: v[:1] for k, v in sim.SWEEP.items()})
    out = sim.sweep(range(1000, 1003))
    assert out["source"] == "simulation"
    assert {r["assumption"] for r in out["rows"]} == set(sim.World.__dataclass_fields__)


def test_scene_creates_missing_folders(tmp_path, capsys):
    page, image = tmp_path / "a" / "m.html", tmp_path / "b" / "m.png"
    sim.main(["scene", "1000", "--html", str(page), "--image", str(image)])
    assert page.exists() and image.exists() and "simulation" in capsys.readouterr().out
