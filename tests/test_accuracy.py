import json
import re
import stat

import pytest

from ostinato.accuracy import analyze, compare_lines, main, render, wilson
from ostinato.viewpoints import Observation


def window(bpm, count, view="A", reason="ok"):
    return Observation(view, 30, bpm, 0.9 if bpm else 0.0, reason, count)


PAIRS = [
    (window(16.0, 8), window(15.0, 8, "B")),
    (window(None, 9, reason="chest not visible"), window(21.0, 10, "B")),
    (window(12.0, 6), window(None, 6, "B", "background moving")),
]


def test_known_numbers():
    r = analyze(PAIRS)
    assert (r["windows"], r["reported"]) == (6, 4)
    assert r["mean_absolute_error"] == pytest.approx(0.5)
    assert r["bias"] == pytest.approx(0)
    assert r["largest_error"] == pytest.approx(1)
    assert r["limits_of_agreement"] == pytest.approx([-1.6003, 1.6003], abs=0.001)
    assert r["declines"] == {"chest not visible": 1, "background moving": 1}
    assert r["within_tolerance"] == 1


def test_wilson():
    assert wilson(0, 0) is None
    assert wilson(5, 10) == pytest.approx([0.2366, 0.7634], abs=0.001)
    assert wilson(10, 10)[1] == 1 and wilson(0, 10)[0] == 0


@pytest.mark.parametrize("pairs", [PAIRS, PAIRS[:1], [(window(None, 8), window(None, 8, "B"))]])
def test_page_is_offline_and_cautious(pairs):
    page = render(analyze(pairs))
    assert "Preliminary" in page and "not a medical device" in page
    assert not re.search(r"<script|https?:|src=|nan|inf\b", page)


def test_enough_windows_drops_the_preliminary_note():
    assert "Preliminary" not in render(analyze(PAIRS * 6))


def test_view_names_are_escaped():
    assert "<b>" not in re.findall(r"<title>(.*?)</title></circle>", render(analyze([(window(15, 8, "<b>"), window(15, 8, "B"))])))[0]


def test_bad_input():
    with pytest.raises(ValueError):
        analyze([])
    with pytest.raises(ValueError):
        analyze(PAIRS, float("nan"))


def test_command(tmp_path, capsys):
    rows = [{"view": v, "elapsed_s": 30, "bpm": 16, "signal_quality": 0.9, "reason": "ok", "counted_breaths": 8} for v in "AB"]
    trial = tmp_path / "pair.json"
    trial.write_text(json.dumps({"source": "paired-webcam-v1", "observations": rows}))
    page = tmp_path / "out" / "a.html"
    main([str(trial), "--html", str(page)])
    assert json.loads(capsys.readouterr().out)["reported"] == 2
    assert stat.S_IMODE(page.stat().st_mode) == 0o600
    with pytest.raises(SystemExit):
        main([str(trial), "--html", str(page)])  # never overwrites
    with pytest.raises(SystemExit):
        main([str(trial), str(trial)])


def test_each_view_on_its_own():
    v = analyze(PAIRS)["by_view"]
    assert (v["A"]["reported"], v["A"]["windows"], v["B"]["reported"]) == (2, 3, 2)
    assert v["A"]["mean_absolute_error"] == pytest.approx(0.0)
    assert v["B"]["mean_absolute_error"] == pytest.approx(1.0)
    assert v["B"]["bias"] == pytest.approx(0.0) and v["A"]["within_tolerance"] == 1
    lone = analyze([(window(None, 8), window(15.0, 8, "B"))])["by_view"]["A"]
    assert lone["reported"] == 0 and lone["mean_absolute_error"] is None


def test_views_compared_trial_by_trial():
    pairs = [*PAIRS, (window(None, 8), window(None, 8, "B")), (window(16.2, 8), window(16.0, 8, "B"))]
    (h,) = analyze(pairs)["head_to_head"].values()
    assert h == {
        "trials": 5,
        "both": 2,
        "closer": {"A": 1, "B": 0},
        "tied": 1,
        "only": {"A": 1, "B": 1},
        "neither": 1,
        "closer_chance": 1.0,
        "only_chance": 1.0,
    }
    text = " ".join(compare_lines(analyze(pairs)))
    assert "A vs B, 5 trials: both gave a rate in 2; A was closer in 1, B in 0" in text
    assert "Only A gave a rate in 1. Only B gave a rate in 1. Neither did in 1." in text
    page = render(analyze(pairs))
    assert "By camera position" in page and "A vs B, 5 trials" in page


def test_views_from_different_postures_are_kept_apart():
    lying = (window(15.0, 8, "side-low"), window(12.0, 8, "side-high"))
    result = analyze([*PAIRS, lying])
    assert set(result["by_view"]) == {"A", "B", "side-high", "side-low"}
    assert set(result["head_to_head"]) == {"A vs B", "side-high vs side-low"}
    assert result["head_to_head"]["side-high vs side-low"]["closer"] == {"side-high": 0, "side-low": 1}


def test_chance():
    from ostinato.accuracy import chance

    assert chance(0, 0) is None and chance(2, 2) == 0.5 and chance(1, 2) == 1.0
    assert chance(3, 4) == pytest.approx(0.625) and chance(1, 4) == chance(3, 4)
    assert chance(0, 5) == pytest.approx(0.0625) and chance(1, 10) == pytest.approx(0.0215, abs=1e-4)


def test_few_trials_are_not_called_a_win():
    pairs = [(window(16.0, 8), window(18.0, 8, "B"))] * 3 + [(window(18.0, 8), window(16.0, 8, "B"))]
    (h,) = analyze(pairs)["head_to_head"].values()
    assert h["closer"] == {"A": 3, "B": 1} and h["closer_chance"] == pytest.approx(0.625)
    text = " ".join(compare_lines(analyze(pairs)))
    assert "62% of the time by luck, so this does not yet show one is better." in text
    many = analyze([(window(16.0, 8), window(18.0, 8, "B"))] * 10)
    text = " ".join(compare_lines(many))
    assert "less than 1% of the time by luck." in text and "does not yet" not in text
    even = analyze([(window(16.0, 8), window(18.0, 8, "B")), (window(18.0, 8), window(16.0, 8, "B"))])
    assert "by luck" not in " ".join(compare_lines(even))
