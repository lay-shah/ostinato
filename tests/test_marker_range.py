import numpy as np
import pytest

from ostinato import marker_range
from ostinato.localize import Camera
from ostinato.marker_range import attempt, readable, survey


def test_a_facing_marker_is_read_closely():
    rng = np.random.default_rng(1)
    results = [attempt(Camera(), 2.0, 0, rng, blur=1.0, noise=4) for _ in range(5)]
    assert all(results)
    assert max(abs(r[0]) for r in results) < 0.02
    assert max(abs(r[1]) for r in results) < 0.01


def test_a_pitched_camera_reads_it_too():
    rng = np.random.default_rng(2)
    found = attempt(Camera(pitch=20), 2.0, 30, rng, blur=1.0)
    assert found and abs(found[0]) < 0.03


def test_blur_and_turning_shorten_the_range():
    sharp = survey(Camera(), distances=(2.0, 6.0), turns=(0,), tries=4)
    poor = survey(Camera(), distances=(2.0, 6.0), turns=(65,), tries=4, blur=2.5, noise=10, light=0.3)
    assert readable(sharp, 0) == 6.0
    assert readable(poor, 65) < 6.0


def test_readable_needs_every_nearer_distance():
    empty = np.array([])
    table = {(1.0, 0): (1.0, empty, empty), (2.0, 0): (0.5, empty, empty), (3.0, 0): (1.0, empty, empty)}
    assert readable(table, 0) == 1.0
    table[1.0, 0] = (0.8, empty, empty)
    assert readable(table, 0) == 0.0


def test_main(capsys, monkeypatch):
    monkeypatch.setattr(marker_range, "DISTANCES", (1.0, 5.0))
    monkeypatch.setattr(marker_range, "TURNS", (0, 50))
    marker_range.main(["--tries", "2"])
    out = capsys.readouterr().out
    assert "The setup check counts markers as seen out to 4.9 m" in out
    assert "5.0 m" in out and "real ones read worse" in out


@pytest.mark.parametrize("bad", [["--light", "0"], ["--blur", "-1"], ["--fov", "200"], ["--tries", "0"]])
def test_main_refuses(bad):
    with pytest.raises(SystemExit):
        marker_range.main(bad)
