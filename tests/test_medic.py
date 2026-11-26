import re
from dataclasses import replace

import pytest

from ostinato import simulation as sim
from ostinato.medic import render


def page(seed):
    scene = sim.make_scene(seed)
    out = sim.run(scene, "ranked", "predicted")
    return scene, out, render(scene, out)


@pytest.mark.parametrize("seed", range(1000, 1006))
def test_everyone_is_listed_and_numbered_once(seed):
    scene, _, html = page(seed)
    assert sorted(map(int, re.findall(r"Person (\d+)", html))) == [p.known.id for p in scene.people]
    assert len(re.findall(r"<circle", html)) == len(scene.people) == len(re.findall(r"<li", html))


def test_labeled_offline_and_safe_wording():
    _, _, html = page(1001)
    assert "SIMULATED DATA" in html and "not a medical device" in html
    assert not re.search(r"<script|https?:|src=|@import|url\(", html)
    assert not re.search(r"not breathing|dead|deceased", html, re.IGNORECASE)
    assert "Needs a human check" in html


def test_hidden_truth_does_not_reach_the_page():
    scene, out, html = page(1001)
    people = tuple(
        replace(p, urgent=not p.urgent, facing=0.0, bpm=99.0, blocked=frozenset()) for p in scene.people
    )
    assert render(sim.Scene(scene.walls, scene.start, people), out) == html


def test_unmeasured_people_have_no_rate_or_score():
    _, out, html = page(1001)
    unmeasured = [r for r in out["results"] if r["bpm"] is None]
    assert unmeasured and all(r["score"] is None for r in unmeasured)
    assert html.count("no signal score") == len(unmeasured)


def test_reasons_are_escaped():
    scene, out, _ = page(1001)
    out["results"][0].update(bpm=None, score=None, reason="<b>x</b>")
    assert "&lt;b&gt;x" in render(scene, out)
