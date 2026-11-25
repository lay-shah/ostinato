import math

import pytest

from ostinato.tipping import PARTS, Body, G, main, smallest_base


def test_body():
    b = Body.of([("low", 3, 0.1), ("high", 1, 0.9)])
    assert (b.mass, b.height) == (4, pytest.approx(0.3))
    assert b.inertia == pytest.approx(3 * 0.04 + 0.36)
    assert b.slope(0.6) == pytest.approx(45)
    assert b.braking(0.6) == pytest.approx(G)
    assert b.braking(0.6, 45) == pytest.approx(0, abs=1e-9)
    for bad in ([], [("x", 0, 1)], [("x", 1, -1)], [("x", math.nan, 1)], [("x", 1, 0)]):
        with pytest.raises(ValueError):
            Body.of(bad)


@pytest.mark.parametrize("base", [0.15, 0.25, 0.4])
def test_curb_matches_a_step_by_step_fall(base):
    b = Body.of(PARTS)
    half, arm = base / 2, math.hypot(base / 2, b.height)
    inertia = b.inertia + b.mass * arm**2

    def tips(speed):
        angle, spin, dt = 0.0, b.mass * speed * b.height / inertia, 1e-4
        start = math.atan2(half, b.height)
        while spin > 0:
            spin -= b.mass * G * arm * math.sin(start - angle) / inertia * dt
            angle += spin * dt
            if angle >= start:
                return True
        return False

    edge = b.curb(base)
    assert tips(edge * 1.01) and not tips(edge * 0.99)


def test_smallest_base_meets_its_own_limits():
    b = Body.of(PARTS)
    base, _ = smallest_base(b)
    assert b.braking(base, 10) > 3.0 and b.slope(base) > 10 and b.curb(base) >= 0.44
    assert smallest_base(Body.of([*PARTS, ("more camera", 0.5, 0.8)]))[0] > base
    for bad in ({"slope": 60}, {"speed": 0}, {"margin": 0.5}, {"stop": 0}):
        with pytest.raises(ValueError):
            smallest_base(b, **bad)


def test_command(capsys):
    main([])
    out = capsys.readouterr().out
    assert "guesses" in out and "cm apart" in out
    main(["--part", "base, heavy,3,0.05", "--part", "top,0.5,0.8"])
    assert "guesses" not in capsys.readouterr().out
    for bad in (["--part", "x"], ["--part", "x,0,1"], ["--slope", "80"]):
        with pytest.raises(SystemExit):
            main(bad)
