"""Integrated checks for completed assessments, paired replay, and search metrics."""

import json

import pytest

from ostinato.breathing import Reading
from ostinato.decision import Assessor
from ostinato.simulation import compare
from ostinato.viewpoints import Observation, load_pair, replay, summarize


def test_completed_time_stays_fixed_when_live_evidence_changes():
    a = Assessor(0, 1)
    a.update(25, Reading(16, 0.9, "ok"))
    done = a.update(28, Reading(16, 0.9, "ok"))
    assert done.completed_at_s == 28
    assert a.update(40, Reading(None, 0, "chest not visible")) == done


def test_paired_orders_use_the_same_recorded_samples():
    hidden = Observation(
        "A",
        30,
        None,
        0,
        "chest not visible",
        8,
        ((0, None, 0, "chest not visible"), (5, None, 0, "chest not visible")),
    )
    clear = Observation("B", 30, 19, 0.9, "ok", 8, ((25, 19, 0.9, "ok"), (28, 19, 0.9, "ok")))
    report = summarize([(hidden, clear)])
    assert report["second_view_reports_after_first_decline"] == 1
    assert report["accurate_second_view_recoveries"] == 0
    assert report["recovered_absolute_errors_bpm"] == [3]
    assert summarize([(hidden, clear)], 3)["accurate_second_view_recoveries"] == 1
    assert [r["time_to_usable_s"] for r in replay([hidden, clear])["runs"]] == [33, 28]


def test_summary_only_trial_does_not_invent_confirmation():
    row = Observation("A", 30, 16, 0.9, "ok", 8)
    with pytest.raises(ValueError, match="timed readings"):
        replay([row, row])


def test_benchmark_accounts_for_failures():
    result = compare(range(1000, 1002))
    for policy in result["policies"]:
        assert policy["unassessed_fraction"] + policy["measured_fraction"] == pytest.approx(1, abs=0.0011)
        assert policy["mean_failed_observation_s_per_scene"] >= 0
        if policy["measured_fraction"]:
            assert policy["mean_time_to_usable_measured_only_s"] > 0


def _pair_file(tmp_path, source, trace):
    rows = [
        {"view": v, "elapsed_s": 30, "bpm": 16, "signal_quality": 0.9, "reason": "ok",
         "counted_breaths": 8, "trace": trace}
        for v in ("A", "B")
    ]  # fmt: skip
    path = tmp_path / f"{source}.json"
    path.write_text(json.dumps({"source": source, "observations": rows}))
    return path


def test_only_per_frame_trials_can_be_replayed(tmp_path):
    trace = [[25, 16, 0.9, "ok"], [28, 16, 0.9, "ok"]]
    old = load_pair(_pair_file(tmp_path, "paired-webcam-v2", trace))
    assert summarize([old])["reported_windows"] == 2
    with pytest.raises(ValueError, match="v3"):
        replay(old)
    new = load_pair(_pair_file(tmp_path, "paired-webcam-v3", trace))
    assert replay(new)["runs"][0]["time_to_usable_s"] == 28


def test_trial_folder_is_ready_before_recording(tmp_path, monkeypatch, capsys):
    from ostinato import viewpoints

    out = tmp_path / "missing" / "pair.json"

    def capture(camera, view):
        assert out.parent.is_dir()
        return Observation(view, 30, 16, 0.9, "ok", 8)

    monkeypatch.setattr(viewpoints, "capture", capture)
    viewpoints.main(["trial", str(out), "--views", "A", "B"])
    assert out.exists() and "coverage" in capsys.readouterr().out


@pytest.mark.parametrize(("reasons", "good"), [(["starting", "ok"], True), (["chest not visible"], False)])
def test_framing_check_advises_and_always_ends(monkeypatch, capsys, reasons, good):
    from types import SimpleNamespace

    from ostinato import viewpoints

    clock, n = [0.0], [0]

    def read():
        clock[0] += 0.1
        return True, "frame"

    def update(_):
        n[0] += 1
        return 0, reasons[min(n[0] - 1, len(reasons) - 1)]

    monkeypatch.setattr(viewpoints, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    result = viewpoints.check_framing(SimpleNamespace(read=read), SimpleNamespace(update=update), lambda f: f)
    assert result == good and clock[0] < 9
    assert ("Recording anyway" in capsys.readouterr().out) != good
