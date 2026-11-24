import itertools
import re
from pathlib import Path

import numpy as np
import pytest

from ostinato import breathing
from ostinato.breathing import Reading
from ostinato.decision import VIEW_PROBLEMS, Action, Assessor, Status

OK = Reading(15.0, 0.95, "ok")


def bad(reason):
    return Reading(None, 0.0, reason)


def feed(a, reading, start, seconds):
    for t in np.arange(start, start + seconds + 0.5, 0.5):
        d = a.update(t, reading)
        if d.action is not Action.WAIT:
            return d
    return a.decision


def test_reports_only_after_rate_is_confirmed():
    a = Assessor(0, 1)
    assert a.update(1, OK).action is Action.WAIT
    assert a.update(3, OK).action is Action.WAIT
    d = a.update(4, OK)
    assert (d.action, d.status, d.bpm) == (Action.REPORT, Status.MEASURED, 15)


def test_jump_or_dropout_restarts_confirmation():
    a = Assessor(0, 1)
    a.update(0, OK)
    a.update(2, Reading(30.0, 0.9, "ok"))
    assert a.update(4, Reading(30.0, 0.9, "ok")).action is Action.WAIT
    a.update(4.5, bad("too much movement"))
    a.update(5, OK)
    assert a.update(7, OK).action is Action.WAIT
    assert a.update(8, OK).action is Action.REPORT


@pytest.mark.parametrize("reason", sorted(VIEW_PROBLEMS))
def test_bad_view_moves_but_brief_glitch_does_not(reason):
    a = Assessor(0, 1)
    a.update(0, bad(reason))
    assert a.update(4, bad(reason)).action is Action.WAIT
    a.update(4.5, bad("need more data"))
    assert a.update(6, bad(reason)).action is Action.WAIT
    assert feed(a, bad(reason), 6.5, 6).action is Action.MOVE


def test_other_problems_wait_then_move():
    a = Assessor(0, 1)
    assert feed(a, bad("too much movement"), 0, 59).action is Action.WAIT
    assert a.update(60, bad("too much movement")).action is Action.MOVE


def test_moves_then_reports_from_new_view_and_ignores_rates_while_moving():
    a = Assessor(0, 1)
    assert feed(a, bad("chest not visible"), 0, 6).action is Action.MOVE
    assert a.update(7, OK).action is Action.MOVE
    assert a.update(12, OK).action is Action.MOVE
    assert a.arrived(13).action is Action.WAIT
    assert a.update(13, OK).action is Action.WAIT
    assert a.update(16, OK).action is Action.REPORT


def test_no_views_left_gets_help():
    d = feed(Assessor(0, 0), bad("chest not visible"), 0, 6)
    assert (d.action, d.status, d.bpm) == (Action.GET_HELP, Status.UNABLE, None)


def test_failed_moves_try_next_view_then_get_help():
    a = Assessor(0, 2)
    feed(a, bad("chest not visible"), 0, 6)
    assert a.move_failed(7).action is Action.MOVE
    assert a.move_failed(8).action is Action.GET_HELP


def test_stuck_move_gets_help():
    a = Assessor(0, 1)
    feed(a, bad("chest not visible"), 0, 6)
    assert a.update(180, bad("stream interrupted")).action is Action.GET_HELP


def test_response_is_reported_at_any_time():
    a = Assessor(0, 1)
    feed(a, bad("chest not visible"), 0, 6)
    d = a.update(7, bad("chest not visible"), responded=True)
    assert (d.action, d.status, d.bpm) == (Action.REPORT, Status.RESPONDED, None)


def test_final_decision_does_not_change():
    a = Assessor(0, 0)
    first = feed(a, bad("chest not visible"), 0, 6)
    assert a.update(50, OK) is first


def test_rejects_bad_time_and_wrong_calls():
    a = Assessor(5, 1)
    for t in (4, float("nan")):
        with pytest.raises(ValueError):
            a.update(t, OK)
    with pytest.raises(ValueError):
        a.arrived(6)
    with pytest.raises(ValueError):
        Assessor(0, -1)


def known_reasons():
    source = Path(breathing.__file__).parent
    text = "".join((source / f).read_text() for f in ("breathing.py", "chest.py", "app.py"))
    return set(re.findall(r'_decline\("([^"]+)"', text)) | set(re.findall(r'np\.nan, "([^"]+)"', text))


def test_view_problem_names_match_real_reasons():
    assert VIEW_PROBLEMS <= known_reasons()


def test_missing_motion_never_becomes_a_finding():
    """Every unreadable reason, with any number of views, ends in human help."""
    for reason, views in itertools.product(known_reasons() | {"something new"}, range(4)):
        a, t = Assessor(0, views), 0.0
        while not a.decision.final:
            d = a.update(t, bad(reason))
            if d.action is Action.MOVE:
                a.arrived(t)
            t += 1
            assert t <= 181, "must always finish"
        assert a.decision.action is Action.GET_HELP
        assert a.decision.status is Status.UNABLE
        assert a.decision.bpm is None
    words = " ".join([*Status, *Action])
    assert not re.search(r"not breathing|dead|deceased|no breath", words)


def test_random_sequences_always_finish_safely():
    rng = np.random.default_rng(0)
    reasons = sorted(known_reasons())
    for _ in range(200):
        a, t, measured = Assessor(0, int(rng.integers(0, 4))), 0.0, set()
        while not a.decision.final:
            r = Reading(float(rng.uniform(6, 42)), 0.9, "ok") if rng.random() < 0.3 else bad(rng.choice(reasons))
            measured.add(r.bpm)
            d = a.update(t, r)
            if d.action is Action.MOVE and rng.random() < 0.3:
                a.arrived(t) if rng.random() < 0.5 else a.move_failed(t)
            t += float(rng.uniform(0.1, 2))
            assert t < 185
        assert (a.decision.bpm is not None) == (a.decision.status is Status.MEASURED)
        assert a.decision.bpm in measured or a.decision.bpm is None
