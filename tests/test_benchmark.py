import numpy as np
import pytest

from ostinato.benchmark import NOT_BREATHING_MOTION, PATTERNS, TIMES, main, run
from ostinato.breathing import estimate
from ostinato.synthetic import breaths

RESULT = run(60, 12345)


def test_uneven_breathing_is_usually_measured_and_never_badly_wrong():
    b = RESULT["breathing"]
    assert set(b) == set(PATTERNS)
    assert b["regular"]["reported"] > 0.85 and b["typical"]["reported"] > 0.6
    assert all(row["largest_error"] is None or row["largest_error"] < 4 for row in b.values())


def test_other_motion_is_rarely_given_a_rate():
    o = RESULT["other_motion"]
    assert set(o) == set(NOT_BREATHING_MOTION)
    for name in ("tracking drift", "white noise", "fidgeting"):
        assert o[name]["false_rates"] == 0
    assert all(row["false_rates"] <= 0.12 for row in o.values())


def test_old_rate_does_not_linger():
    kept = RESULT["rate_kept_after_motion_stops"]
    assert all(kept[f"{bpm}/min"]["8 s"] <= 0.05 for bpm in (16, 24, 32))


def test_breaths_have_the_requested_rate():
    for bpm in (8, 15, 30):
        _, truth = breaths(TIMES, np.random.default_rng(0), bpm, 0, 0, 0.2)
        assert truth == pytest.approx(bpm, abs=0.1)


def test_one_missed_breath_is_not_smoothed_over():
    x, _ = breaths(TIMES, np.random.default_rng(1), 15, 0.02, 0.02, 0.1)
    x[(TIMES > 12) & (TIMES < 18)] = x.min()
    assert estimate(TIMES, x).bpm is None


def test_command(capsys):
    main(["--trials", "2"])
    assert "synthetic benchmark" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--trials", "0"])
