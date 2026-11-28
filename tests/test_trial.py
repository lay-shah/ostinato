import json

import pytest

from ostinato import trial, viewpoints
from ostinato.viewpoints import Observation, load_pair


def fake_capture(calls):
    def capture(camera, view, announce=None):
        calls.append(view)
        return Observation(view, 30.0, 16.0, 0.9, "ok", 8, ((29.0, 16.0, 0.9, "ok"),))

    return capture


def answer(monkeypatch, *answers):
    queue = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt: next(queue))


def test_session_saves_trials_alternates_views_and_reports(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(viewpoints, "capture", fake_capture(calls))
    answer(monkeypatch, "y", "lying", "y", "n")
    trial.main(["--folder", str(tmp_path / "new")])
    assert calls == ["side-low", "side-high", "side-high", "side-low"]
    files = sorted(p.name for p in (tmp_path / "new").glob("*.json"))
    assert files == ["trial-001-lying.json", "trial-002-lying.json"]
    assert len(load_pair(tmp_path / "new" / files[0])) == 2
    out = capsys.readouterr().out
    assert "4 of 4 windows" in out and "Average error 0.0" in out and "Preliminary" in out
    page = next((tmp_path / "new").glob("accuracy-*.html"))
    assert page.stat().st_mode & 0o077 == 0


def test_without_agreement_nothing_is_recorded(monkeypatch, tmp_path):
    monkeypatch.setattr(viewpoints, "capture", lambda *a, **k: pytest.fail("recorded"))
    answer(monkeypatch, "no")
    assert trial.run(tmp_path / "x", 0) == [] and not (tmp_path / "x").exists()


def test_interrupted_view_can_be_repeated_or_dropped(monkeypatch, tmp_path):
    calls = []
    good, failures = fake_capture(calls), [ValueError("Camera stopped"), None, ValueError("Camera stopped")]

    def capture(camera, view, announce=None):
        error = failures.pop(0) if failures else None
        if error:
            raise error
        return good(camera, view)

    monkeypatch.setattr(viewpoints, "capture", capture)
    answer(monkeypatch, "y", "seated", "y", "n", "y", "n")
    saved = trial.run(tmp_path, 0)
    assert [p.name for p in saved] == ["trial-001-seated.json"]  # the dropped one left no file or number
    assert json.loads(saved[0].read_text())["source"] == "paired-webcam-v3"


def test_file_numbers_never_repeat(tmp_path):
    (tmp_path / "trial-007-lying.json").write_text("{}")
    assert trial.next_path(tmp_path, "seated").name == "trial-008-seated.json"


def test_stopping_early_keeps_saved_trials(monkeypatch, tmp_path):
    monkeypatch.setattr(viewpoints, "capture", fake_capture([]))

    queue = iter(["y", "seated"])

    def once(prompt):
        for reply in queue:
            return reply
        raise EOFError

    monkeypatch.setattr("builtins.input", once)
    with pytest.raises(SystemExit):
        trial.main(["--folder", str(tmp_path)])
    assert len(list(tmp_path.glob("trial-*.json"))) == 1


def test_speech_is_optional(monkeypatch):
    monkeypatch.setattr(trial.shutil, "which", lambda c: None)
    trial.say("Start")


def test_lying_views_stand_where_the_rover_would():
    from ostinato.rover import RING
    from ostinato.trial import SETUP, VIEWS

    for view in VIEWS["lying"]:
        assert f"{RING} m from the middle of their body" in SETUP[view]
