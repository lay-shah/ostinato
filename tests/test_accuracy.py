import json
import re
import stat

import pytest

from ostinato.accuracy import analyze, main, render, wilson
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
