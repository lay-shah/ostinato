"""Drive control for the planned rover, with a simulated base: `python -m ostinato.rover`.

No hardware exists yet. `Guard` is the only way to the wheels: whatever the mission
asks for, it enforces the limits and stops when anything goes quiet. It does not
replace a hardware emergency stop that cuts motor power.
"""

import math
from dataclasses import dataclass, field
from enum import Enum

from .decision import Action, Assessor

MAX_SPEED = 0.3  # m/s
MAX_TURN = 1.0  # rad/s
MAX_ACCEL = 0.5  # m/s^2, speeding up; stopping is immediate
MAX_TURN_ACCEL = 2.0  # rad/s^2
QUIET = 0.5  # seconds without news of the link, a sensor, or a command before stopping
OBSTACLE = 0.6  # m, stop driving toward anything closer
KEEP_OUT = 1.0  # m from the person, measured to the rover's center
RING = 1.25  # m, viewpoints and paths lie on this circle around the person
SETTLE = 2.0  # seconds still before measuring, for the mast to stop shaking
ARRIVED = 0.05  # m
BLOCKED = 10.0  # seconds without moving before giving up on a path
LEG = 90.0  # longest time to reach one viewpoint


@dataclass(frozen=True)
class Drive:
    speed: float = 0.0  # m/s forward
    turn: float = 0.0  # rad/s, positive to the left


STOP = Drive()


class Guard:
    """Passes on safe commands only. An emergency stop holds until `clear`."""

    def __init__(self):
        self.stopped = False
        self.reason = "starting"
        self._heard = {}
        self._front = self._rear = self._person = math.nan
        self._last, self._last_t = STOP, None

    def emergency_stop(self):
        self.stopped = True
        self._last = STOP

    def clear(self):
        self.stopped = False
        self._last = STOP

    def link(self, t):
        """The operator's remote was heard from."""
        self._heard["link"] = t

    def sensors(self, t, front, rear):
        self._heard["sensors"] = t
        self._front, self._rear = front, rear

    def person(self, t, distance, approaching):
        """Distance to the person, and whether driving forward would reduce it."""
        self._heard["person"] = t
        self._person, self._approaching = distance, approaching

    def _refuse(self, t, want):
        if self.stopped:
            return "emergency stop"
        for source in ("link", "sensors", "person"):
            heard = self._heard.get(source)
            if heard is None or not 0 <= t - heard <= QUIET:
                return f"no news from {source}"
        if not (math.isfinite(want.speed) and math.isfinite(want.turn)):
            return "invalid command"
        if math.isnan(self._front) or math.isnan(self._rear) or math.isnan(self._person):
            return "invalid sensor"
        if want.speed > 0 and not self._front >= OBSTACLE:
            return "obstacle ahead"
        if want.speed < 0 and not self._rear >= OBSTACLE:
            return "obstacle behind"
        if want.speed and self._person < KEEP_OUT and (want.speed > 0) == self._approaching:
            return "too close to the person"
        return None

    def allow(self, t, want):
        """Return the command to send to the wheels now."""
        step = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t
        refused = self._refuse(t, want)
        self.reason = refused or "ok"
        # A late or repeated call means the control loop itself went wrong.
        if refused or not 0 <= step <= QUIET:
            if not refused:
                self.reason = "control loop late"
            self._last = STOP
            return STOP

        def ramp(value, last, limit, rate):
            value = max(-limit, min(limit, value))
            if abs(value) <= abs(last) and value * last >= 0:
                return value
            start = last if value * last >= 0 else 0.0
            return max(start - rate * step, min(start + rate * step, value))

        self._last = Drive(
            ramp(want.speed, self._last.speed, MAX_SPEED, MAX_ACCEL),
            ramp(want.turn, self._last.turn, MAX_TURN, MAX_TURN_ACCEL),
        )
        return self._last


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading: float


def steer(pose, target):
    """Turn toward a point, then drive to it."""
    dx, dy = target[0] - pose.x, target[1] - pose.y
    distance = math.hypot(dx, dy)
    error = wrap(math.atan2(dy, dx) - pose.heading)
    if abs(error) > 0.35:
        return Drive(0.0, math.copysign(min(MAX_TURN, 2 * abs(error)), error))
    return Drive(min(MAX_SPEED, distance) * math.cos(error), 2 * error)


def ring_path(pose, person, bearing):
    """Points around the person to the viewpoint at `bearing`, never cutting across."""
    here = math.atan2(pose.y - person[1], pose.x - person[0])
    sweep = wrap(bearing - here)
    steps = max(1, math.ceil(abs(sweep) / math.radians(15)))
    # A chord dips inside the circle, so the points sit slightly outside it.
    radius = RING / math.cos(math.radians(7.5))
    angles = [here + sweep * i / steps for i in range(steps + 1)]
    points = [(person[0] + radius * math.cos(a), person[1] + radius * math.sin(a)) for a in angles]
    return [*points[:-1], (person[0] + RING * math.cos(bearing), person[1] + RING * math.sin(bearing))]


class State(Enum):
    DRIVING = "driving"
    FACING = "facing the person"
    SETTLING = "settling"
    MEASURING = "measuring"
    DONE = "done"


@dataclass
class Mission:
    """Visit viewpoints around one person. Measures only when still."""

    person: tuple[float, float]
    bearings: list[float]  # radians, in the order to try them
    state: State = State.DRIVING
    assessor: Assessor | None = None
    log: list = field(default_factory=list)

    def __post_init__(self):
        if not self.bearings:
            raise ValueError("at least one viewpoint is needed")
        self.view = 0
        self._path = None
        self._since = self._progress = None

    @property
    def outcome(self):
        """What to tell the operator. Never a statement about the person's condition."""
        if self.assessor is None:
            return (
                "could not reach any viewpoint: a person must check"
                if self.state is State.DONE
                else "in progress"
            )
        d = self.assessor.decision
        if d.action is Action.REPORT:
            return f"{d.bpm:.1f} breaths/min" if d.bpm is not None else d.reason
        return (
            f"unable to assess ({d.reason}): a person must check"
            if self.state is State.DONE
            else "in progress"
        )

    def _go(self, t, state):
        self.state, self._since = state, t
        self.log.append((t, state.value, self.view))

    def _next(self, t, failed):
        """Leave this viewpoint. The assessor decides whether to try another."""
        if self.assessor is None:
            # Nothing has been measured yet, so there is no assessment to consult.
            if self.view + 1 < len(self.bearings):
                self.view, self._path = self.view + 1, None
                return self._go(t, State.DRIVING)
            return self._go(t, State.DONE)
        decision = self.assessor.move_failed(t) if failed else self.assessor.decision
        if decision.action is Action.MOVE and self.view + 1 < len(self.bearings):
            self.view, self._path = self.view + 1, None
            return self._go(t, State.DRIVING)
        return self._go(t, State.DONE)

    def update(self, t, pose, moving, reading=None, responded=False, asking=False):
        """Return the wanted command. `moving` is whether the wheels turned last time."""
        if self._since is None:
            self._go(t, State.DRIVING)
        if self.state is State.DRIVING:
            return self._drive(t, pose, moving)
        if self.state is State.FACING:
            # The camera looks to the left, so the person must be there.
            toward = math.atan2(self.person[1] - pose.y, self.person[0] - pose.x)
            error = wrap(toward - math.pi / 2 - pose.heading)
            if abs(error) < 0.05:
                self._go(t, State.SETTLING)
                return STOP
            if t - self._since > BLOCKED:
                self._next(t, failed=True)
                return STOP
            return Drive(0.0, math.copysign(min(0.5, max(0.1, 2 * abs(error))), error))
        if self.state is State.SETTLING:
            if moving:
                self._since = t
            elif t - self._since >= SETTLE:
                if self.assessor is None:
                    self.assessor = Assessor(t, len(self.bearings) - 1 - self.view)
                else:
                    self.assessor.arrived(t)
                self._go(t, State.MEASURING)
            return STOP
        if self.state is State.MEASURING and reading is not None:
            decision = self.assessor.update(t, reading, responded, asking=asking)
            if decision.final:
                self._go(t, State.DONE)
            elif decision.action is Action.MOVE:
                self._next(t, failed=False)
        return STOP

    def _drive(self, t, pose, moving):
        if self._path is None:
            self._path = ring_path(pose, self.person, self.bearings[self.view])
            self._progress = t
        if moving:
            self._progress = t
        if t - self._progress > BLOCKED or t - self._since > LEG:
            self._next(t, failed=True)
            return STOP
        while self._path:
            near = ARRIVED if len(self._path) == 1 else 0.12
            if math.dist((pose.x, pose.y), self._path[0]) > near:
                return steer(pose, self._path[0])
            self._path.pop(0)
        self._go(t, State.FACING)
        return STOP


class SimulatedBase:
    """Wheels that slip a little, for trying the controller without hardware."""

    def __init__(self, pose, person, rng, obstacles=(), slip=0.03):
        self.pose, self.person, self.rng, self.obstacles, self.slip = pose, person, rng, obstacles, slip

    def step(self, drive, dt):
        speed = drive.speed * (1 + self.rng.normal(0, self.slip))
        turn = drive.turn * (1 + self.rng.normal(0, self.slip))
        p = self.pose
        self.pose = Pose(
            p.x + speed * math.cos(p.heading) * dt,
            p.y + speed * math.sin(p.heading) * dt,
            wrap(p.heading + turn * dt),
        )

    def ranges(self):
        """Distance to the nearest obstacle ahead and behind, as simple cones."""
        found = [math.inf, math.inf]
        for ox, oy, radius in self.obstacles:
            dx, dy = ox - self.pose.x, oy - self.pose.y
            angle = wrap(math.atan2(dy, dx) - self.pose.heading)
            distance = math.hypot(dx, dy) - radius
            if abs(angle) < 0.6:
                found[0] = min(found[0], distance)
            elif abs(angle) > math.pi - 0.6:
                found[1] = min(found[1], distance)
        return found

    def tell(self, guard, t):
        guard.link(t)
        guard.sensors(t, *self.ranges())
        dx, dy = self.person[0] - self.pose.x, self.person[1] - self.pose.y
        ahead = dx * math.cos(self.pose.heading) + dy * math.sin(self.pose.heading)
        guard.person(t, math.hypot(dx, dy), ahead > 0)


def main():
    import numpy as np

    from .breathing import Reading

    person = (0.0, 0.0)
    base = SimulatedBase(Pose(2.5, 0.5, 2.5), person, np.random.default_rng(1))
    guard, mission, dt, t = Guard(), Mission(person, [0.4, 2.0]), 0.05, 0.0
    drive, closest, measured = STOP, math.inf, None
    print("SIMULATED: no robot exists yet.")
    while mission.state is not State.DONE and t < 300:
        base.tell(guard, t)
        # The first view is blocked, and the second one works.
        if mission.state is not State.MEASURING:
            reading, measured = None, None
        else:
            measured = t if measured is None else measured
            blocked = mission.view == 0 or t - measured < 25
            reason = "chest not visible" if mission.view == 0 else "need more data"
            reading = Reading(None, 0.0, reason) if blocked else Reading(15.0, 0.9, "ok")
        moving = drive != STOP
        before = len(mission.log)
        drive = guard.allow(t, mission.update(t, base.pose, moving, reading))
        for when, state, view in mission.log[before:]:
            print(f"{when:6.1f} s  view {view + 1}: {state}")
        base.step(drive, dt)
        closest = min(closest, math.dist((base.pose.x, base.pose.y), person))
        t += dt
    print(f"Outcome: {mission.outcome}. Never closer than {closest:.2f} m to the person.")


if __name__ == "__main__":
    main()
