"""Decide whether to report, wait, move, or ask for human help.

Safety rule: there is no status meaning "not breathing". If breathing cannot be
measured and the person does not respond, the only outcome is "unable to assess".
"""

import math
from dataclasses import dataclass
from enum import Enum

from .breathing import Reading


class Action(str, Enum):
    REPORT = "report"
    WAIT = "wait"
    MOVE = "move"
    GET_HELP = "get help"


class Status(str, Enum):
    MEASURED = "breathing measured"
    RESPONDED = "responded"
    UNABLE = "unable to assess"


# Problems that a different view may solve. Anything else is worth waiting out first.
VIEW_PROBLEMS = {
    "chest not visible",
    "not enough chest detail",
    "cannot check camera motion",
    "signal too weak",
}


@dataclass(frozen=True)
class Limits:
    confirm_seconds: float = 3.0  # a rate must hold this long before it is reported
    max_rate_change: float = 4.0  # breaths/min, while confirming
    view_problem_seconds: float = 5.0  # patience for a bad view
    view_seconds: float = 60.0  # longest stay at one viewpoint
    total_seconds: float = 180.0  # longest time spent on one person


@dataclass(frozen=True)
class Decision:
    action: Action
    status: Status
    reason: str
    bpm: float | None = None

    @property
    def final(self) -> bool:
        return self.action in (Action.REPORT, Action.GET_HELP)


class Assessor:
    """State machine for one person. Create a new one to reassess.

    After a MOVE decision, call `arrived` or `move_failed`. The caller must also
    discard signal collected while moving.
    """

    def __init__(self, t: float, extra_views: int, limits: Limits = Limits()):
        if extra_views < 0:
            raise ValueError("extra_views cannot be negative")
        self.limits = limits
        self.views_left = extra_views
        self.decision = Decision(Action.WAIT, Status.UNABLE, "starting")
        self._start = self._view_start = self._now = self._check(t)
        self._problem_since = None
        self._confirm_since = None
        self._confirm_bpm = None

    @staticmethod
    def _check(t):
        if not math.isfinite(t):
            raise ValueError("time must be finite")
        return t

    def _tick(self, t):
        if self._check(t) < self._now:
            raise ValueError("time cannot go backwards")
        self._now = t

    def _set(self, action, reason, status=Status.UNABLE, bpm=None):
        self.decision = Decision(action, status, reason, bpm)
        return self.decision

    def _next_view(self, reason):
        if self.views_left:
            self.views_left -= 1
            return self._set(Action.MOVE, reason)
        return self._set(Action.GET_HELP, f"{reason}; no views left")

    def _begin_view(self):
        self._view_start = self._now
        self._problem_since = self._confirm_since = self._confirm_bpm = None

    def arrived(self, t: float) -> Decision:
        if self.decision.action is not Action.MOVE:
            raise ValueError("not moving")
        self._tick(t)
        self._begin_view()
        return self._set(Action.WAIT, "starting new view")

    def move_failed(self, t: float) -> Decision:
        if self.decision.action is not Action.MOVE:
            raise ValueError("not moving")
        self._tick(t)
        return self._next_view("could not reach view")

    def update(self, t: float, reading: Reading, responded: bool = False) -> Decision:
        """`responded=False` means no response was seen, which proves nothing."""
        if self.decision.final:
            return self.decision
        self._tick(t)
        if responded:
            return self._set(Action.REPORT, "response seen", Status.RESPONDED)
        moving = self.decision.action is Action.MOVE
        if not moving and self._confirmed(reading):
            return self._set(Action.REPORT, "rate confirmed", Status.MEASURED, reading.bpm)
        if t - self._start >= self.limits.total_seconds:
            return self._set(Action.GET_HELP, "time limit reached")
        if moving:
            return self.decision

        if reading.bpm is not None:
            self._problem_since = None
            reason = "confirming rate"
        elif reading.reason in VIEW_PROBLEMS:
            reason = reading.reason
            if self._problem_since is None:
                self._problem_since = t
            if t - self._problem_since >= self.limits.view_problem_seconds:
                return self._next_view(reason)
        else:
            self._problem_since = None
            reason = reading.reason
        if t - self._view_start >= self.limits.view_seconds:
            return self._next_view(reason)
        return self._set(Action.WAIT, reason)

    def _confirmed(self, reading):
        bpm = reading.bpm
        if bpm is None or not math.isfinite(bpm):
            self._confirm_since = self._confirm_bpm = None
            return False
        if self._confirm_bpm is None or abs(bpm - self._confirm_bpm) > self.limits.max_rate_change:
            self._confirm_since, self._confirm_bpm = self._now, bpm
            return False
        return self._now - self._confirm_since >= self.limits.confirm_seconds
