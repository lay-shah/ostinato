import itertools
import math

import numpy as np
import pytest

from ostinato import rover
from ostinato.breathing import Reading
from ostinato.rover import KEEP_OUT, MAX_SPEED, STOP, Drive, Guard, Mission, Pose, SimulatedBase, State

GOOD, NONE = Reading(15.0, 0.9, "ok"), Reading(None, 0.0, "chest not visible")


def ready(t=0.0, front=5.0, rear=5.0, person=3.0, approaching=True):
    guard = Guard()
    tell(guard, t, front, rear, person, approaching)
    return guard


def tell(guard, t, front=5.0, rear=5.0, person=3.0, approaching=True):
    guard.link(t)
    guard.sensors(t, front, rear)
    guard.person(t, person, approaching)


def drive_for(guard, seconds, want, start=0.0, **world):
    for i in range(int(seconds * 20) + 1):
        t = start + i / 20
        tell(guard, t, **world)
        out = guard.allow(t, want)
    return out


def test_speed_is_limited_and_builds_up_but_stops_at_once():
    guard = ready()
    assert guard.allow(0.0, Drive(5, 5)) == STOP
    tell(guard, 0.1)
    first = guard.allow(0.1, Drive(5, 5))
    assert 0 < first.speed <= 0.05 + 1e-9 and 0 < first.turn <= 0.2 + 1e-9
    assert drive_for(guard, 3, Drive(5, 5), 0.2) == Drive(rover.MAX_SPEED, rover.MAX_TURN)
    tell(guard, 3.3)
    assert guard.allow(3.3, STOP) == STOP


@pytest.mark.parametrize("quiet", ["link", "sensors", "person"])
def test_stops_when_anything_goes_quiet(quiet):
    guard = ready()
    drive_for(guard, 1, Drive(0.3))
    for i in range(1, 30):
        t = 1 + i / 20
        if quiet != "link":
            guard.link(t)
        if quiet != "sensors":
            guard.sensors(t, 5, 5)
        if quiet != "person":
            guard.person(t, 3, True)
        out = guard.allow(t, Drive(0.3))
        assert (out == STOP) == (t - 1 > rover.QUIET + 1e-9)
    assert guard.reason == f"no news from {quiet}"


@pytest.mark.parametrize(
    ("want", "world"),
    [
        (Drive(0.2), {"front": 0.5}),
        (Drive(-0.2), {"rear": 0.5}),
        (Drive(0.2), {"person": 0.9}),
        (Drive(-0.2), {"person": 0.9, "approaching": False}),
        (Drive(0.2), {"front": math.nan}),
        (Drive(0.2), {"person": math.nan}),
        (Drive(math.nan), {}),
        (Drive(0.0, math.inf), {}),
    ],
)
def test_unsafe_commands_are_refused(want, world):
    assert drive_for(ready(), 1, want, **world) == STOP


def test_can_back_away_from_the_person_and_from_an_obstacle():
    assert drive_for(ready(), 1, Drive(-0.2), person=0.8).speed < 0
    assert drive_for(ready(), 1, Drive(-0.2), front=0.2).speed < 0


def test_emergency_stop_holds_until_cleared():
    guard = ready()
    drive_for(guard, 1, Drive(0.3))
    guard.emergency_stop()
    assert drive_for(guard, 2, Drive(0.3), 1.05) == STOP and guard.reason == "emergency stop"
    guard.clear()
    assert drive_for(guard, 1, Drive(0.3), 3.1).speed > 0


def test_nothing_moves_before_everything_has_been_heard():
    assert Guard().allow(0.0, Drive(0.3)) == STOP


@pytest.mark.parametrize("gap", [-0.1, 0.7])
def test_late_control_loop_stops(gap):
    guard = ready()
    drive_for(guard, 1, Drive(0.3))
    t = 1 + gap
    tell(guard, t)
    assert guard.allow(t, Drive(0.3)) == STOP


def simulate(
    seed,
    obstacles=(),
    reading=lambda view, t: GOOD,
    cut=None,
    seconds=400,
    bearings=(0.5, 2.2, -2.0),
    shaking=lambda view, t: False,
    body=None,
    truth=None,
    start=None,
    known=(),
):
    """Run a mission and check the rules at every step."""
    rng = np.random.default_rng(seed)
    a, r = start or (rng.uniform(-math.pi, math.pi), rng.uniform(1.6, 4))
    lying = body or rover.Body((0.0, 0.0))
    start = lying.around(a, r)
    base = SimulatedBase(Pose(*start, rng.uniform(-3, 3)), lying, rng, obstacles)
    mission = Mission(body or (0.0, 0.0), list(bearings), obstacles=list(known))
    guard, drive, t = Guard(), STOP, 0.0
    closest = math.inf
    while mission.state is not State.DONE and t < seconds:
        if cut is None or not cut[0] <= t <= cut[1]:
            base.tell(guard, t)
        measuring = mission.state is State.MEASURING
        shakes = mission.state is State.SETTLING and shaking(mission.view, t)
        want = mission.update(
            t, base.pose, drive != STOP, reading(mission.view, t) if measuring else None, shaking=shakes
        )
        drive = guard.allow(t, want)
        assert abs(drive.speed) <= MAX_SPEED and abs(drive.turn) <= rover.MAX_TURN
        if measuring or mission.state is State.SETTLING:
            assert drive == STOP
        if cut and cut[0] + rover.QUIET + 0.06 < t <= cut[1]:
            assert drive == STOP
        before = base.pose
        base.step(drive, 0.05)
        assert math.dist((before.x, before.y), (base.pose.x, base.pose.y)) < 0.02
        # Measured to the person's real body, whatever the rover was told.
        closest = min(closest, (truth or lying).distance(base.pose.x, base.pose.y))
        t += 0.05
    return mission, closest, t


@pytest.mark.parametrize("seed", range(2000, 2020))
def test_reaches_a_view_and_reports_without_coming_close(seed):
    mission, closest, _ = simulate(seed)
    assert mission.outcome == "15.0 breaths/min" and closest >= KEEP_OUT


def test_failed_views_lead_to_the_next_and_then_to_help():
    mission, closest, _ = simulate(2030, reading=lambda view, t: GOOD if view == 2 else NONE)
    assert mission.outcome == "15.0 breaths/min" and mission.view == 2 and closest >= KEEP_OUT
    mission, _, _ = simulate(2031, reading=lambda view, t: NONE)
    assert mission.outcome.startswith("unable to assess") and mission.outcome.endswith("a person must check")


def test_blocked_path_gives_up_and_says_so():
    wall = [(2.0 * math.cos(a), 2.0 * math.sin(a), 0.5) for a in np.linspace(0, 2 * math.pi, 24)]
    for seed in range(2040, 2045):
        rng = np.random.default_rng(seed)
        mission, _, t = simulate(seed, wall + [(0, 0, 0.1)], bearings=(0.5, 2.2))
        assert mission.state is State.DONE and t < 400, rng
        assert "a person must check" in mission.outcome or mission.outcome == "15.0 breaths/min"


def test_lost_link_stops_the_rover_and_it_continues_when_it_returns():
    mission, closest, _ = simulate(2050, cut=(2.0, 6.0))
    assert mission.outcome == "15.0 breaths/min" and closest >= KEEP_OUT


def test_path_never_cuts_inside_the_ring():
    for here in np.linspace(-3, 3, 13):
        for to in np.linspace(-3, 3, 13):
            pose = Pose(3 * math.cos(here), 3 * math.sin(here), 0)
            points = [(3 * math.cos(here), 3 * math.sin(here)), *rover.ring_path(pose, (0, 0), to)]
            for (x0, y0), (x1, y1) in itertools.pairwise(points):
                near = min(math.hypot(x0 + (x1 - x0) * k, y0 + (y1 - y0) * k) for k in np.linspace(0, 1, 30))
                assert near >= rover.RING - 0.02


def test_demo_runs(capsys):
    rover.main()
    out = capsys.readouterr().out
    assert "SIMULATED" in out and "15.0 breaths/min" in out


def test_waits_still_before_measuring():
    mission, _, _ = simulate(2060)
    times = {state: t for t, state, _ in mission.log}
    assert times["measuring"] - times["settling"] >= 2.0


def test_turning_in_place_needs_no_news_of_the_person():
    guard = Guard()
    for t in (0.0, 0.1, 0.2):
        guard.link(t), guard.sensors(t, 5, 5)
        turned = guard.allow(t, Drive(0.0, 0.5))
    assert turned.turn > 0 and guard.reason == "ok"
    assert guard.allow(0.3, Drive(0.1, 0.5)) == STOP and guard.reason == "no news from person"
    guard.link(0.4), guard.sensors(0.4, 5, 5), guard.person(0.4, math.nan, True)
    assert guard.allow(0.4, Drive(0.1, 0.0)) == STOP and guard.reason == "invalid sensor"


def test_waits_for_the_camera_to_stop_shaking():
    plain, _, _ = simulate(2061)
    settled = next(t for t, state, _ in plain.log if state == "settling")
    # Shaking for 3 s after it stops at the first view.
    mission, _, _ = simulate(2061, shaking=lambda view, t: t < settled + 3)
    times = {state: t for t, state, _ in mission.log}
    assert times["measuring"] - times["settling"] >= 3 + rover.SETTLE - 0.06
    assert mission.outcome == "15.0 breaths/min"


def test_a_view_that_never_steadies_is_given_up():
    mission, closest, _ = simulate(2062, shaking=lambda view, t: view == 0)
    assert mission.outcome == "15.0 breaths/min" and mission.view == 1 and closest >= KEEP_OUT
    states = [(state, view) for _, state, view in mission.log]
    assert ("measuring", 0) not in states and ("measuring", 1) in states
    mission, _, _ = simulate(2063, shaking=lambda view, t: True, bearings=(0.5, 2.2))
    assert mission.state is State.DONE and "measuring" not in [s for _, s, _ in mission.log]
    assert "a person must check" in mission.outcome


LYING = rover.Body((0.0, 0.0), math.pi / 2)  # head toward +y, feet toward -y


def test_body_geometry():
    assert LYING.ends[0] == pytest.approx((0, rover.HEAD)) and LYING.ends[1] == pytest.approx(
        (0, -rover.FEET)
    )
    assert LYING.distance(1.0, 0.3) == pytest.approx(1.0) and LYING.distance(0, -2.1) == pytest.approx(1.0)
    assert LYING.around(0.0, 1.25) == pytest.approx((1.25, 0.0))
    assert LYING.around(-math.pi / 2, 1.25) == pytest.approx((0.0, -1.1 - 1.25))
    for bearing in np.linspace(-3, 3, 25):
        assert LYING.distance(*LYING.around(bearing, 1.25)) == pytest.approx(1.25, abs=1e-6)
    point = rover.Body((1.0, 2.0))
    assert point.distance(1.0, 4.0) == pytest.approx(2.0) and point.around(0.0, 1.25) == pytest.approx(
        (2.25, 2.0)
    )


def test_path_around_a_lying_person_keeps_clear_of_head_and_feet():
    for here in np.linspace(-3, 3, 13):
        for to in np.linspace(-3, 3, 13):
            start = LYING.around(here, 3.0)
            points = [start, *rover.ring_path(Pose(*start, 0), LYING, to)]
            for (x0, y0), (x1, y1) in itertools.pairwise(points):
                near = min(
                    LYING.distance(x0 + (x1 - x0) * k, y0 + (y1 - y0) * k) for k in np.linspace(0, 1, 30)
                )
                assert near >= rover.RING - 0.02


def test_knowing_the_head_keeps_it_1_m_from_the_feet():
    # Views beside the chest; the rover must get round the feet to reach the second one.
    views = (0.3, -2.8)
    told, _, _ = simulate(2070, bearings=views, body=LYING)
    assert told.outcome == "15.0 breaths/min"
    for seed in range(2070, 2080):
        _, closest, _ = simulate(seed, bearings=views, body=LYING, reading=lambda view, t: NONE)
        assert closest >= KEEP_OUT
    # Told only where the chest is, the same rover passes within 1 m of the feet.
    worst = min(
        simulate(seed, bearings=views, truth=LYING, reading=lambda view, t: NONE)[1]
        for seed in range(2070, 2080)
    )
    assert worst < 0.5


def test_blocked_the_short_way_it_goes_round_the_other_side():
    # Starting at 1.4 rad, view 1 is at 0: the short way passes a box at 0.7 rad.
    box = [(rover.RING * math.cos(0.7), rover.RING * math.sin(0.7), 0.2)]
    for seed in range(2090, 2095):
        mission, closest, _ = simulate(seed, box, bearings=(0.0,), start=(1.4, 2.2))
        states = [state for _, state, _ in mission.log]
        assert "going round the other way" in states and mission.outcome == "15.0 breaths/min"
        assert closest >= KEEP_OUT
    # Unblocked, it takes the short way.
    mission, _, _ = simulate(2090, bearings=(0.0,), start=(1.4, 2.2))
    assert "going round the other way" not in [state for _, state, _ in mission.log]


def test_a_known_obstacle_is_avoided_from_the_start():
    box = [(rover.RING * math.cos(0.7), rover.RING * math.sin(0.7), 0.2)]
    for seed in range(2090, 2095):
        mission, closest, _t = simulate(seed, box, bearings=(0.0,), start=(1.4, 2.2), known=box)
        states = [state for _, state, _ in mission.log]
        assert "going the long way round, past a known obstacle" in states
        assert "going round the other way" not in states  # it never had to stop for it
        assert mission.outcome == "15.0 breaths/min" and closest >= KEEP_OUT


def test_clearance():
    assert rover.clearance([(0, 0), (2, 0)], []) == math.inf
    assert rover.clearance([(0, 0), (2, 0)], [(1, 1, 0.5)]) == pytest.approx(0.5)
    assert rover.clearance([(0, 0), (2, 0)], [(3, 0, 0.5)]) == pytest.approx(0.5)
    assert rover.clearance([(0, 0), (2, 0)], [(1, 0.2, 0.5)]) < 0


def test_goes_round_clockwise_backwards_to_keep_the_camera_on_the_person():
    # The camera looks left: forwards counter-clockwise, backwards clockwise.
    for views, backwards in (((1.4,), False), ((-1.4,), True)):
        rng = np.random.default_rng(3)
        base = SimulatedBase(Pose(rover.TRAVEL, 0.0, math.pi / 2), (0.0, 0.0), rng)
        guard, mission, drive, t, speeds = Guard(), Mission((0.0, 0.0), list(views)), STOP, 0.0, []
        while mission.state is State.DRIVING and t < 60:
            base.tell(guard, t)
            drive = guard.allow(t, mission.update(t, base.pose, drive != STOP))
            speeds.append(drive.speed)
            base.step(drive, 0.05)
            t += 0.05
        assert mission.state is State.FACING
        # Going round; the last few centimeters straight in may be either way.
        assert (min(speeds) < -0.2) == backwards
        assert (
            sum(v < -0.1 for v in speeds) > sum(v > 0.1 for v in speeds) if backwards else min(speeds) > -0.05
        )
    assert rover.clockwise((0, 0), (1, 0), (1, -1)) and not rover.clockwise((0, 0), (1, 0), (1, 1))
    assert not rover.clockwise((0, 0), (2, 0), (1, -0.1))  # nearly straight in
