"""Surface-following launch and what navigation learns from it (09-12)."""
from types import SimpleNamespace as NS
import pytest
from rclpy.task import Future
from ackermann_adaptive_controller.core import AdaptiveCore
from ackermann_adaptive_controller.response import DeadBand
from ackermann_adaptive_controller.nav2_capability import SurfaceCapability
from plant import Plant, settle_sense


def confirmed_indoor(core, breakaway=0.30, eq=0.23):
    for _ in range(core.policy.deadband_evidence):
        core.deadband.observe(1.0, breakaway)
        core.gain_probe.eq_fwd.add(eq)
    return core


def test_recent_grass_starts_raise_the_launch_floor_and_a_fast_start_lowers_it():
    core = confirmed_indoor(AdaptiveCore())
    p = core.policy
    settle_sense(core, Plant())
    core.v_op = 0.3
    core.v = core.v_fb = 0.0
    core.rolling = False
    core._run(0.0, 0.3, 0.0, 0.1)
    assert core._floor == pytest.approx(0.23)            # indoors: creep from the cruise wire
    for wire in (0.66, 0.72):                              # two grass starts
        core.deadband.observe(1.0, wire)
    assert core.deadband.fwd.value < 0.4                   # the long median has barely moved
    core._run(0.0, 0.3, 0.0, 0.1)
    # the recent window (last four starts) is half grass already
    assert core._floor == pytest.approx(p.deadband_trust * 0.48)
    for wire in (0.70, 0.68):
        core.deadband.observe(1.0, wire)
    core._run(0.0, 0.3, 0.0, 0.1)
    assert core._floor == pytest.approx(p.deadband_trust * 0.69)   # fully on grass after four
    assert core._floor <= p.deadband_max
    core.deadband.observe_upper(1.0, 0.31)                 # back on a hard floor: instant start
    assert core.deadband.recent(1.0) is None                 # grass samples above the bound are gone; one bound alone is not evidence
    core._run(0.0, 0.3, 0.0, 0.1)
    assert core._floor == pytest.approx(0.23)              # never below the cruise wire


def test_a_slow_start_above_an_old_bound_means_the_surface_got_harder():
    band = DeadBand(AdaptiveCore().policy)
    band.observe_upper(1.0, 0.30)
    band.observe_upper(1.0, 0.28)
    assert band.recent(1.0) == pytest.approx(0.28)
    band.observe(1.0, 0.65)
    band.observe(1.0, 0.70)
    assert band.recent(1.0) == pytest.approx(0.675)
    assert band.recent(-1.0) is None                       # per direction


def test_recent_windows_and_surface_measurements_survive_a_restart():
    core = confirmed_indoor(AdaptiveCore())
    core.deadband.observe(1.0, 0.7)
    core.deadband.observe_upper(-1.0, 0.4)
    core.lunge.add(0.45); core.launch_time.add(3.2)
    saved = core.state()
    fresh = AdaptiveCore()
    assert fresh.load_state(saved)
    assert list(fresh.deadband.fwd_recent) == list(core.deadband.fwd_recent)
    assert list(fresh.deadband.rev_recent_ub) == [0.4]
    assert list(fresh.lunge.vals) == [0.45] and list(fresh.launch_time.vals) == [3.2]


def test_learning_resumes_on_the_tick_after_an_interruption():
    core = AdaptiveCore(); settle_sense(core, Plant())
    core.pause_learning(core.now)
    assert core._learning_after == pytest.approx(core.now + core.dt)
    assert core._learning_after - core.now < 3 * core.policy.sensor_tau


def test_lunge_and_launch_time_are_measured_at_breakaway():
    core = AdaptiveCore(); plant = Plant()
    plant.deadband = 0.30; plant.kinetic = 0.2
    t = settle_sense(core, plant)
    from test_core import _start_stop_cycles
    _start_stop_cycles(core, plant, t, cycles=6)
    assert core.creep_speed() is not None and 0.0 < core.creep_speed() < 1.0
    assert core.launch_duration() is not None and 0.0 < core.launch_duration() < 10.0


def pusher(creep=None, launch=None, v_op=0.31):
    calls = []
    def call(req):
        f = Future(); calls.append((req, f)); return f
    client = NS(service_is_ready=lambda: True, call_async=call)
    node = NS(create_client=lambda *_: client,
              get_logger=lambda: NS(info=lambda *_: None, warn=lambda *_: None),
              core=NS(creep_speed=lambda: creep, launch_duration=lambda: launch, v_op=v_op,
                      lon_bank=NS(delay=0.45), policy=NS(tau_d=0.2)))
    cap = SurfaceCapability(node, dict(publish_surface_limits=True, controller_server='/controller',
        approach_speed_param='FollowPath.min_approach_linear_velocity',
        regulated_min_speed_param='FollowPath.regulated_linear_scaling_min_speed',
        stuck_allowance_param='progress_checker.stuck_time_allowance',
        movement_allowance_param='progress_checker.movement_time_allowance'))
    return cap, calls


def answer_reads(calls, values):
    for req, f in calls:
        if hasattr(req, 'names') and not f.done():
            f.set_result(NS(values=[NS(type=3, double_value=values[req.names[0]])]))


BASE = {'FollowPath.min_approach_linear_velocity': 0.1875,
        'FollowPath.regulated_linear_scaling_min_speed': 0.3125,
        'progress_checker.stuck_time_allowance': 9.0,
        'progress_checker.movement_time_allowance': 12.0}


def test_navigation_floors_are_read_back_and_only_ever_raised():
    cap, calls = pusher(creep=0.45, launch=3.0)
    cap.tick()                                           # reads the configured floors first
    assert all(hasattr(r, 'names') for r, _ in calls)
    answer_reads(calls, BASE)
    cap.tick()
    sent = {r.parameters[0].name: r.parameters[0].value.double_value for r, _ in calls if hasattr(r, 'parameters')}
    # a 0.45 m/s lunge is above both speed floors but capped at the operating speed
    assert sent['FollowPath.min_approach_linear_velocity'] == pytest.approx(0.31)
    assert sent['FollowPath.regulated_linear_scaling_min_speed'] == pytest.approx(0.3125)
    # a 3 s launch: two of them plus the sensing delay is still inside the configured 9 s
    assert sent['progress_checker.stuck_time_allowance'] == pytest.approx(9.0)
    for r, f in calls:
        if hasattr(r, 'parameters') and not f.done():
            f.set_result(NS(results=[NS(successful=True)]))
    assert cap.confirmed[('/controller', 'progress_checker.stuck_time_allowance')] == pytest.approx(9.0)
    cap2, calls2 = pusher(creep=0.2, launch=5.0)
    cap2.tick(); answer_reads(calls2, BASE); cap2.tick()
    sent2 = {r.parameters[0].name: r.parameters[0].value.double_value for r, _ in calls2 if hasattr(r, 'parameters')}
    assert sent2['progress_checker.stuck_time_allowance'] == pytest.approx(0.8*12.0)  # 10.65 s asked, capped under the movement window
    assert sent2['FollowPath.min_approach_linear_velocity'] == pytest.approx(0.2)


def test_easier_surface_returns_navigation_to_its_configured_values():
    cap, calls = pusher(creep=0.05, launch=0.4)
    cap.tick(); answer_reads(calls, BASE); cap.tick()
    sent = {r.parameters[0].name: r.parameters[0].value.double_value for r, _ in calls if hasattr(r, 'parameters')}
    assert sent['FollowPath.min_approach_linear_velocity'] == pytest.approx(0.1875)
    assert sent['progress_checker.stuck_time_allowance'] == pytest.approx(9.0)


def test_long_launches_never_push_the_stuck_window_past_the_movement_window():
    cap, calls = pusher(creep=None, launch=20.0)
    cap.tick(); answer_reads(calls, BASE); cap.tick()
    sent = {r.parameters[0].name: r.parameters[0].value.double_value for r, _ in calls if hasattr(r, 'parameters')}
    assert sent['progress_checker.stuck_time_allowance'] == pytest.approx(0.8*12.0)
    assert 'FollowPath.min_approach_linear_velocity' in sent  # unmeasured creep: floor is re-asserted
