"""Drive control for the planned rover, with a simulated base: `python -m ostinato.rover`.

No hardware exists yet. `Guard` is the only way to the wheels: whatever the mission
asks for, it enforces the limits and stops when anything goes quiet. It does not
replace a hardware emergency stop that cuts motor power.
"""

import math
from dataclasses import dataclass, field
from enum import Enum

from .decision import Action, Assessor, Status

MAX_SPEED = 0.3  # m/s
MAX_TURN = 1.0  # rad/s
MAX_ACCEL = 0.5  # m/s^2, speeding up; stopping is immediate
MAX_TURN_ACCEL = 2.0  # rad/s^2
QUIET = 0.5  # seconds without news of the link, a sensor, or a command before stopping
OBSTACLE = 0.6  # m, stop driving toward anything closer
KEEP_OUT = 1.0  # m from the person's body, measured to the rover's center
RING = 1.25  # m from the body: viewpoints and paths lie on this line around the person
# A person lying down, from the chest: to the top of the head, and to the feet. A tall adult.
HEAD = 0.75  # m
FEET = 1.1  # m
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
        # Turning in place cannot bring it nearer the person, so that alone is allowed
        # without knowing where the person is, for example to look for a marker.
        needed = ("link", "sensors", "person") if want.speed else ("link", "sensors")
        for source in needed:
            heard = self._heard.get(source)
            if heard is None or not 0 <= t - heard <= QUIET:
                return f"no news from {source}"
        if not (math.isfinite(want.speed) and math.isfinite(want.turn)):
            return "invalid command"
        if math.isnan(self._front) or math.isnan(self._rear) or (want.speed and math.isnan(self._person)):
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


@dataclass(frozen=True)
class Body:
    """A person: a point at the chest, or, lying down with `head` known, a line from the
    head to the feet. Distances are to the nearest part of it."""

    chest: tuple[float, float]
    head: float | None = None  # radians, the direction from the chest to the head

    @property
    def ends(self):
        x, y = self.chest
        if self.head is None:
            return (x, y), (x, y)
        c, s = math.cos(self.head), math.sin(self.head)
        return (x + HEAD * c, y + HEAD * s), (x - FEET * c, y - FEET * s)

    def nearest(self, x, y):
        """The nearest point of the body to (x, y)."""
        (ax, ay), (bx, by) = self.ends
        dx, dy = bx - ax, by - ay
        length = dx * dx + dy * dy
        k = 0.0 if length == 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length))
        return ax + k * dx, ay + k * dy

    def distance(self, x, y):
        return math.dist((x, y), self.nearest(x, y))

    def around(self, bearing, clearance):
        """The point `clearance` from the body, in direction `bearing` from the chest."""
        c, s = math.cos(bearing), math.sin(bearing)
        low, high = 0.0, clearance + HEAD + FEET
        for _ in range(50):  # moving out from the chest, the distance only grows
            mid = (low + high) / 2
            if self.distance(self.chest[0] + mid * c, self.chest[1] + mid * s) < clearance:
                low = mid
            else:
                high = mid
        return self.chest[0] + high * c, self.chest[1] + high * s


def as_body(person):
    return person if isinstance(person, Body) else Body(tuple(person))


def ring_path(pose, person, bearing):
    """Points around the person to the viewpoint at `bearing`, never cutting across.
    `person` is a Body, or (x, y) for a point."""
    body = as_body(person)
    cx, cy = body.chest
    here = math.atan2(pose.y - cy, pose.x - cx)
    sweep = wrap(bearing - here)
    # Points about 0.33 m apart along the line (15 degrees for a point person), the far
    # ends of a body being up to FEET further out.
    reach = RING + (0.0 if body.head is None else max(HEAD, FEET))
    steps = max(1, math.ceil(abs(sweep) * reach / (RING * math.radians(15))))
    # A chord dips inside the line, so the points sit slightly outside it.
    out = RING / math.cos(math.radians(7.5))
    angles = [here + sweep * i / steps for i in range(steps + 1)]
    return [*(body.around(a, out) for a in angles[:-1]), body.around(bearing, RING)]


class State(Enum):
    DRIVING = "driving"
    FACING = "facing the person"
    SETTLING = "settling"
    MEASURING = "measuring"
    DONE = "done"


@dataclass
class Mission:
    """Visit viewpoints around one person. Measures only when still."""

    person: tuple[float, float] | Body  # a Body for someone lying down whose head is known
    bearings: list[float]  # radians, in the order to try them, seen from the chest
    state: State = State.DRIVING
    assessor: Assessor | None = None
    log: list = field(default_factory=list)
    ask_first: bool = False  # ask the person to raise a hand before leaving each view

    def __post_init__(self):
        if not self.bearings:
            raise ValueError("at least one viewpoint is needed")
        self.body = as_body(self.person)
        self.person = self.body.chest
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
        if d.status is Status.RESPONDED:
            return "hand raised when asked; breathing was not measured"
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

    def update(self, t, pose, moving, reading=None, responded=False, asking=False, shaking=False):
        """Return the wanted command. `moving` is whether the wheels turned last time, and
        `shaking` whether the picture still moves although they did not."""
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
                self._stopped = t
                return STOP
            if t - self._since > BLOCKED:
                self._next(t, failed=True)
                return STOP
            return Drive(0.0, math.copysign(min(0.5, max(0.1, 2 * abs(error))), error))
        if self.state is State.SETTLING:
            if moving or shaking:
                self._since = t
                if t - self._stopped > BLOCKED:
                    # Never still enough to measure here: a bumped mast or a shaking floor.
                    self._next(t, failed=True)
            elif t - self._since >= SETTLE:
                if self.assessor is None:
                    self.assessor = Assessor(t, len(self.bearings) - 1 - self.view, ask_first=self.ask_first)
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
            self._path = ring_path(pose, self.body, self.bearings[self.view])
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
        self.pose, self.rng, self.obstacles, self.slip = pose, rng, obstacles, slip
        self.body = as_body(person)
        self.person = self.body.chest

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
        nx, ny = self.body.nearest(self.pose.x, self.pose.y)
        dx, dy = nx - self.pose.x, ny - self.pose.y
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
