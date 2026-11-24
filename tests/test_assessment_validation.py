"""Integrated checks for completed assessments, paired replay, and search metrics."""
import pytest

from ostinato.breathing import Reading
from ostinato.decision import Assessor
from ostinato.simulation import compare
from ostinato.viewpoints import Observation, replay, summarize


def test_completed_time_stays_fixed_when_live_evidence_changes():
    a = Assessor(0, 1)
    a.update(25, Reading(16, 0.9, "ok"))
    done = a.update(28, Reading(16, 0.9, "ok"))
    assert done.completed_at_s == 28
    assert a.update(40, Reading(None, 0, "chest not visible")) == done


def test_paired_orders_use_the_same_recorded_samples():
    hidden = Observation("A", 30, None, 0, "chest not visible", 8,
                         ((0, None, 0, "chest not visible"), (5, None, 0, "chest not visible")))
    clear = Observation("B", 30, 19, .9, "ok", 8, ((25, 19, .9, "ok"), (28, 19, .9, "ok")))
    report = summarize([(hidden, clear)])
    assert report["second_view_reports_after_first_decline"] == 1
    assert report["accurate_second_view_recoveries"] == 0
    assert report["recovered_absolute_errors_bpm"] == [3]
    assert summarize([(hidden, clear)], 3)["accurate_second_view_recoveries"] == 1
    assert [r["time_to_usable_s"] for r in replay([hidden, clear])["runs"]] == [33, 28]


def test_summary_only_trial_does_not_invent_confirmation():
    row = Observation("A", 30, 16, .9, "ok", 8)
    with pytest.raises(ValueError, match="timed readings"):
        replay([row, row])


def test_benchmark_accounts_for_failures():
    result = compare(range(2))
    for policy in result["policies"]:
        assert policy["unassessed_fraction"] + policy["measured_fraction"] == pytest.approx(1, abs=.0011)
        assert policy["mean_failed_observation_s_per_scene"] >= 0
        if policy["measured_fraction"]:
            assert policy["mean_mission_time_to_usable_s"] > 0
