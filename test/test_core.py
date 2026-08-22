# Copyright 2026 Lucas Foulkes
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""Tests for the ROS-free controller core."""

import math

import pytest

from ackermann_adaptive_controller.core import (CAL, RUN, SENSE, AdaptiveCore,
                                            Policy, RLS, TwistEstimator, clamp)
from plant import Plant, drive, settle_sense


# -- sensor self-measurement ----------------------------------------------

def test_sense_derives_gates_from_measured_noise():
    core = AdaptiveCore()
    plant = Plant()
    settle_sense(core, plant)
    assert core.phase == RUN            # calibration off by default
    assert core.sigma_v > 0.0
    assert core.gate_d >= 0.15
    assert core.gate_s == pytest.approx(1.6 * core.gate_d)
    # lambda must give the same half-life regardless of sample rate
    assert 0.0 < core.lam < 1.0


def test_sense_restarts_if_the_robot_is_moving():
    core = AdaptiveCore()
    plant = Plant(pose_noise=0.0)
    t = 0.0
    for _ in range(40):
        t += 0.1
        plant.step(0.0, 0.6, 0.1)       # driving during SENSE
        x, y, psi = plant.observe()
        core.step(t, x, y, psi, 0.0, 0.0)
    assert core.phase == SENSE          # never allowed to conclude


def test_lambda_gives_the_same_half_life_at_any_rate():
    half = []
    for dt in (0.05, 0.2):
        core = AdaptiveCore()
        plant = Plant()
        settle_sense(core, plant, dt=dt, seconds=4.0)
        # ticks to decay to 0.5 == t_forget seconds
        half.append(math.log(0.5) / math.log(core.lam) * dt)
    assert half[0] == pytest.approx(half[1], rel=0.05)


# -- identification --------------------------------------------------------

def test_longitudinal_gain_converges():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    _, t = drive(core, plant, lambda s: (0.55, 0.0), 40.0, t0=t)
    assert core.model.b0 == pytest.approx(plant.b0, rel=0.30)


def test_lateral_gain_converges():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    # a slalom keeps the steering axis excited
    _, t = drive(core, plant,
                 lambda s: (0.60, 0.5 * math.sin(2.0 * math.pi * 0.12 * s)),
                 90.0, t0=t)
    assert core.model.n_lat > 100
    assert core.model.a0 == pytest.approx(plant.a0, rel=0.35)


def test_learning_is_gated_below_the_noise_floor():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    before = core.model.n_lon
    # crawl well under gate_d: no samples may be accepted
    drive(core, plant, lambda s: (0.02, 0.0), 5.0, t0=t)
    assert core.model.n_lon == before


# -- control ---------------------------------------------------------------

def test_forward_speed_tracking():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    out, t = drive(core, plant, lambda s: (0.45, 0.0), 30.0, t0=t)
    assert plant.v == pytest.approx(0.45, abs=0.08)


def test_reverse_speed_tracking():
    """The drag term must change sign in reverse or this diverges."""
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    out, t = drive(core, plant, lambda s: (-0.35, 0.0), 30.0, t0=t)
    assert plant.v == pytest.approx(-0.35, abs=0.08)


def test_drag_is_identified_as_opposing_motion_in_reverse():
    """b2*v*|v| and b2*v**2 differ only in reverse; pin the difference.

    A single constant speed cannot identify drag -- the regressor
    [qd, 1, v|v|] is then rank-deficient and any split of the three terms
    fits. The sweep is what makes b2 observable, which is also why passive
    learning needs varied commands and not a cruise.
    """
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    for speed in (-0.25, -0.55, -0.35, -0.70, -0.45):
        _, t = drive(core, plant, lambda s, u=speed: (u, 0.0), 25.0, t0=t)
    assert core.model.b2 < 0.0
    assert core.model.b0 == pytest.approx(plant.b0, rel=0.35)


def test_prediction_is_accurate_in_reverse():
    """Whatever the parameter split, the learned map must predict reverse."""
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    for speed in (-0.3, -0.6, -0.45):
        _, t = drive(core, plant, lambda s, u=speed: (u, 0.0), 25.0, t0=t)
    m = core.model
    v = plant.v
    predicted = m.b0 * core.qd + m.b1 + m.b2 * v * abs(v)
    actual = plant.b0 * plant.qd + plant.b1 + plant.b2 * v * abs(v)
    assert predicted == pytest.approx(actual, abs=0.12)


def test_turning_produces_correctly_signed_steering():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    out, _ = drive(core, plant, lambda s: (0.5, 0.6), 12.0, t0=t)
    assert out.steer > 0.0              # left turn, forward
    assert plant.psi > 0.0


def test_reverse_turn_inverts_steering_sign():
    """Same yaw command while reversing needs the opposite steering angle."""
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    fwd, _ = drive(core, plant, lambda s: (0.5, 0.6), 6.0, t0=t)

    core2 = AdaptiveCore()
    plant2 = Plant()
    t2 = settle_sense(core2, plant2)
    rev, _ = drive(core2, plant2, lambda s: (-0.5, 0.6), 6.0, t0=t2)
    assert fwd.steer * rev.steer < 0.0


def test_zero_command_gives_zero_output():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    out, t = drive(core, plant, lambda s: (0.5, 0.0), 10.0, t0=t)
    out, _ = drive(core, plant, lambda s: (0.0, 0.0), 2.0, t0=t)
    assert out.steer == 0.0 and out.drive == 0.0


# -- safety ----------------------------------------------------------------

def test_outputs_never_leave_the_unit_interval():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    for cmd in ((5.0, 9.0), (-5.0, -9.0), (0.3, 40.0)):
        out, t = drive(core, plant, lambda s, c=cmd: c, 8.0, t0=t)
        assert -1.0 <= out.steer <= 1.0
        assert -1.0 <= out.drive <= 1.0


def test_non_finite_input_fails_to_zero():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    out = core.step(t + 0.1, float('nan'), 0.0, 0.0, 0.5, 0.0)
    assert out.steer == 0.0 and out.drive == 0.0
    out = core.step(t + 0.2, 0.0, 0.0, 0.0, float('inf'), 0.0)
    assert -1.0 <= out.drive <= 1.0


def test_clamp_treats_nan_as_zero():
    # plain min/max would let a NaN through as a full-scale command
    assert clamp(float('nan'), -1.0, 1.0) == 0.0


def test_stall_reflex_escalates_in_the_commanded_direction():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    plant.b0 = 0.0                      # wheels cannot move the vehicle
    plant.b1 = 0.0
    peak = 0.0
    for i in range(30):                 # sample DURING the ramp, before the
        t += 0.1                        # blocked latch (rightly) gives up
        x, y, psi = plant.observe()
        out = core.step(t, x, y, psi, -0.6, 0.0)
        peak = min(peak, out.drive)
        for _ in range(5):
            plant.step(out.steer, out.drive, 0.02)
    assert peak < -0.4                  # reverse, not "escalate forward"


def test_blocked_robot_stops_pushing_into_the_obstacle():
    """Hitting a wall looks exactly like a stall. The reflex must give up
    once the ramp caps out with no motion -- pushing harder is how the robot
    grinds into whatever it just hit at full throttle."""
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    plant.b0 = 0.0                      # against a wall: throttle does nothing
    plant.b1 = 0.0
    last = None
    for i in range(120):                # 12 s of commanding into the wall
        t += 0.1
        x, y, psi = plant.observe()
        last = core.step(t, x, y, psi, 0.5, 0.0)
        for _ in range(5):
            plant.step(last.steer, last.drive, 0.02)
    assert core.blocked
    assert last.drive == 0.0            # gave up, not grinding


def test_blocked_latch_releases_after_the_command_relents():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    plant.b0 = 0.0
    plant.b1 = 0.0
    for i in range(120):
        t += 0.1
        x, y, psi = plant.observe()
        core.step(t, x, y, psi, 0.5, 0.0)
    assert core.blocked
    # commander stops asking (obstacle cleared, new plan) for long enough
    for i in range(30):
        t += 0.1
        x, y, psi = plant.observe()
        core.step(t, x, y, psi, 0.0, 0.0)
    assert not core.blocked
    plant.b0 = 2.2                      # path is clear now
    out, _ = drive(core, plant, lambda s: (0.4, 0.0), 15.0, t0=t)
    assert plant.v > 0.15               # and it drives again


def test_standstill_yaw_command_does_not_divide_by_zero():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    out = core.step(t + 0.1, 0.0, 0.0, 0.0, 0.0001, 0.8)
    assert math.isfinite(out.steer) and math.isfinite(out.drive)


# -- calibration -----------------------------------------------------------

def test_calibration_phase_runs_and_exits():
    core = AdaptiveCore(Policy(enable_calibration=True))
    plant = Plant()
    t = settle_sense(core, plant)
    assert core.phase == CAL
    _, t = drive(core, plant, lambda s: (0.0, 0.0), 9.0, t0=t)
    assert core.phase == RUN
    assert 0.03 <= core.dither <= 0.12


def test_calibration_is_off_by_default():
    core = AdaptiveCore()
    plant = Plant()
    settle_sense(core, plant)
    assert core.phase == RUN
    assert core.dither == 0.0


# -- estimator and learner units ------------------------------------------

def test_twist_estimator_signs_reverse_correctly():
    est = TwistEstimator()
    est.update(0.0, 0.0, 0.0, 0.0)
    out = est.update(0.1, -0.05, 0.0, 0.0)     # moved backwards along heading
    assert out[1] == pytest.approx(-0.5, rel=1e-6)


def test_rls_rejects_non_finite_samples():
    rls = RLS([0.0, 0.0, 0.0], 10.0, 1e4)
    assert not rls.update([1.0, 1.0, 1.0], float('nan'), 0.999)
    assert rls.count == 0
    assert all(v == 0.0 for v in rls.theta)


def test_rls_covariance_stays_bounded_without_excitation():
    rls = RLS([0.0, 0.0, 0.0], 10.0, 1e4)
    for _ in range(20000):
        rls.update([0.0, 1.0, 0.0], 0.0, 0.99)
    trace = sum(rls.P[i][i] for i in range(3))
    assert math.isfinite(trace) and trace <= 1e4 * 1.001


# -- learned steering envelope --------------------------------------------

def test_envelope_starts_derated_and_unconfirmed():
    """With no evidence the radius must be conservative, not optimistic."""
    core = AdaptiveCore()
    plant = Plant()
    settle_sense(core, plant)
    assert not core.envelope.confirmed
    m = core.model
    # prior a0 = 1.25 -> full-lock radius 0.8 m; derated must be larger
    assert core.envelope.min_turning_radius(m) > 1.0 / m.a0


def test_envelope_confirms_from_hard_steering_and_tightens():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    before = core.envelope.min_turning_radius(core.model)
    # alternate full lock both ways: both sides need evidence
    _, t = drive(core, plant,
                 lambda s: (0.6, 4.0 * math.sin(2.0 * math.pi * 0.08 * s)),
                 150.0, t0=t)
    assert core.envelope.confirmed
    after = core.envelope.min_turning_radius(core.model)
    assert after < before                      # evidence beats a derated prior
    # the truth, quoted at the same speed the envelope quotes
    v = core.policy.env_speed
    truth = 1.0 / abs(plant.a0 + plant.a2 * v * v)
    assert after == pytest.approx(truth, rel=0.30)


def test_envelope_is_speed_correct():
    """Evidence gathered at high speed must not flatten the quoted envelope.

    The a2*v^2 term nearly cancels the steering gain at 2 m/s on this plant.
    Measuring raw curvature there and quoting it as the envelope reports a
    radius several times too large; measuring model fidelity does not.
    """
    slow = AdaptiveCore()
    pslow = Plant()
    t = settle_sense(slow, pslow)
    drive(slow, pslow,
          lambda s: (0.5, 4.0 * math.sin(2.0 * math.pi * 0.08 * s)),
          150.0, t0=t)

    # this test's premise is a genuinely FAST run; the production throttle
    # gains are sized for a 0.3 m/s robot and would not reach 1.8 m/s in
    # the window, so give the fast core stronger gains
    fast = AdaptiveCore(Policy(kp_v=1.6, ki_v=0.8))
    pfast = Plant()
    t = settle_sense(fast, pfast)
    # a0 and a2 differ only by v^2: sweep the speed or they are not
    # separable and the quoted radius is arbitrary
    drive(fast, pfast,
          lambda s: (1.4 + 0.4 * math.sin(2.0 * math.pi * 0.03 * s),
                     4.0 * math.sin(2.0 * math.pi * 0.08 * s)),
          150.0, t0=t)

    assert slow.envelope.confirmed and fast.envelope.confirmed
    v = slow.policy.env_speed
    truth = 1.0 / abs(pslow.a0 + pslow.a2 * v * v)
    # both are quoted at env_speed, so both must land near the same truth
    assert fast.envelope.min_turning_radius(fast.model) == \
        pytest.approx(truth, rel=0.40)


def test_envelope_needs_both_directions():
    """One-sided evidence must not confirm; a trim makes the sides differ."""
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    drive(core, plant, lambda s: (0.6, 3.0), 60.0, t0=t)   # left lock only
    assert len(core.envelope.left.vals) > 0
    assert len(core.envelope.right.vals) == 0
    assert not core.envelope.confirmed
    assert core.envelope.fidelity() is None


def test_envelope_rejects_a_single_outlier():
    core = AdaptiveCore()
    plant = Plant()
    settle_sense(core, plant)
    env, m, v = core.envelope, core.model, 0.5
    truth = env.predict(m, 1.0, v)
    for _ in range(env.p.env_evidence + 4):
        env.observe(1.0, truth, v, m)
        env.observe(-1.0, -env.predict(m, -1.0, v), v, m)
    steady = env.max_curvature(m)
    env.observe(1.0, truth * 40.0, v, m)       # one glitched sample
    env.observe(-1.0, -truth * 40.0, v, m)
    assert env.max_curvature(m) == pytest.approx(steady, rel=0.02)


def test_envelope_shrinks_when_the_model_overpromises():
    """A rack that stops short of the linear prediction must be caught."""
    core = AdaptiveCore()
    plant = Plant()
    settle_sense(core, plant)
    env, m, v = core.envelope, core.model, 0.5
    for _ in range(env.p.env_evidence + 4):
        # the vehicle only ever delivers half of what the model predicts
        env.observe(1.0, 0.5 * env.predict(m, 1.0, v), v, m)
        env.observe(-1.0, 0.5 * env.predict(m, -1.0, v), v, m)
    assert env.fidelity() == pytest.approx(0.5, rel=0.05)
    unconstrained = min(abs(m.a1 + m.a0 + m.a2 * env.p.env_speed ** 2),
                        abs(m.a1 - m.a0 - m.a2 * env.p.env_speed ** 2))
    assert env.max_curvature(m) == pytest.approx(0.5 * unconstrained, rel=0.05)


def test_evidence_never_makes_the_envelope_more_optimistic():
    """A ratio above 1 is noise, not a car that out-turns its own model."""
    core = AdaptiveCore()
    settle_sense(core, Plant())
    env, m, v = core.envelope, core.model, 0.5
    for _ in range(env.p.env_evidence + 4):
        env.observe(1.0, 1.4 * env.predict(m, 1.0, v), v, m)
        env.observe(-1.0, 1.4 * env.predict(m, -1.0, v), v, m)
    unconstrained = min(abs(m.a1 + m.a0 + m.a2 * env.p.env_speed ** 2),
                        abs(m.a1 - m.a0 - m.a2 * env.p.env_speed ** 2))
    assert env.max_curvature(m) <= unconstrained * 1.001


def test_envelope_is_bounded_by_the_radius_limits():
    core = AdaptiveCore(Policy(radius_floor=0.5, radius_ceiling=3.0,
                               prior_a0=50.0))
    settle_sense(core, Plant())
    assert core.envelope.min_turning_radius(core.model) >= 0.5
    core2 = AdaptiveCore(Policy(radius_floor=0.5, radius_ceiling=3.0,
                                prior_a0=1e-6))
    settle_sense(core2, Plant())
    assert core2.envelope.min_turning_radius(core2.model) <= 3.0


def test_commanded_curvature_is_clamped_to_the_envelope():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    # an impossible yaw rate must not wind the trim integrator up
    out, _ = drive(core, plant, lambda s: (0.5, 8.0), 15.0, t0=t)
    assert abs(core.iw) <= 0.25 + 1e-9
    assert -1.0 <= out.steer <= 1.0


# -- learning from someone else's commands (PASSIVE) -----------------------

def test_applied_commands_drive_the_learner_not_our_own_output():
    """PASSIVE: the joystick is steering, so regress on the tapped values.

    The plant sits far from the priors (1.25 / 2.0) on purpose: an earlier
    version of this test used a plant within its own tolerance of the priors
    and passed while the applied commands never reached the regressor at all.
    """
    core = AdaptiveCore()
    plant = Plant(a=(0.60, 0.04, -0.30), b=(4.0, 0.0, -0.35))
    t = settle_sense(core, plant)
    # feed the plant a command the core never produced, and tell the core
    for i in range(600):
        t += 0.1
        us = 0.8 * math.sin(2.0 * math.pi * 0.1 * i * 0.1)
        ud = 0.45 + 0.15 * math.sin(2.0 * math.pi * 0.03 * i * 0.1)
        for _ in range(5):
            plant.step(us, ud, 0.02)
        x, y, psi = plant.observe()
        core.step(t, x, y, psi, 0.0, 0.0, applied=(us, ud))
    assert core.model.a0 == pytest.approx(plant.a0, rel=0.35)
    assert core.model.b0 == pytest.approx(plant.b0, rel=0.35)
    assert abs(core.model.a0 - core.policy.prior_a0) > 0.3   # it moved


def test_learning_from_own_output_would_be_wrong_when_someone_else_drives():
    """The same run without `applied` learns nothing and biases what it does.

    PASSIVE publishes nothing, so the core's own output stays zero. Regressing
    on that while the joystick actually steers leaves the steering gain pinned
    at its prior (the regressor carries no steering information at all) and
    dumps the unexplained motion into the bias terms instead.
    """
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    prior_a0 = core.policy.prior_a0
    prior_b0 = core.policy.prior_b0
    for i in range(400):
        t += 0.1
        us = 0.6 * math.sin(2.0 * math.pi * 0.1 * i * 0.1)
        for _ in range(5):
            plant.step(us, 0.55, 0.02)
        x, y, psi = plant.observe()
        core.step(t, x, y, psi, 0.0, 0.0)      # no applied -> wrong regressor
    # samples were accepted, so this is not simply "nothing ran"
    assert core.model.n_lat > 100 and core.model.n_lon > 100
    # ...yet neither gain moved off its prior, because qs and qd stayed zero
    assert core.model.a0 == pytest.approx(prior_a0, abs=1e-9)
    assert core.model.b0 == pytest.approx(prior_b0, abs=1e-9)
    # and the throttle bias absorbed motion it did not cause
    assert abs(core.model.b1 - plant.b1) > 0.5
    assert not core.envelope.confirmed


# -- persistence -----------------------------------------------------------

def test_state_round_trips():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    drive(core, plant,
          lambda s: (0.6, 4.0 * math.sin(2.0 * math.pi * 0.08 * s)),
          150.0, t0=t)
    assert core.envelope.confirmed
    saved = core.state()
    radius = core.envelope.min_turning_radius(core.model)

    fresh = AdaptiveCore()
    assert fresh.load_state(saved)
    assert fresh.envelope.min_turning_radius(fresh.model) == \
        pytest.approx(radius, rel=1e-9)
    # covariance is inflated on restore, not restored
    assert fresh.rls_lat.P[0][0] > core.policy.p0


def test_corrupt_state_is_rejected():
    core = AdaptiveCore()
    good = core.state()
    assert not core.load_state({'version': 99})
    assert not core.load_state({'version': 1, 'lateral': [1.0, 2.0],
                                'longitudinal': [1.0, 2.0, 3.0],
                                'envelope': {'left': [], 'right': []}})
    bad = dict(good, lateral=[float('nan'), 0.0, 0.0])
    assert not core.load_state(bad)
    assert core.load_state(good)


# -- regression: a dead steering actuator must not latch the model off -----

class DeadSteering(Plant):
    """A plant whose steering does nothing: unpowered servo, lost linkage."""

    def step(self, us, ud, dt):
        super().step(0.0, ud, dt)


def test_dead_steering_does_not_latch_the_controller_to_zero():
    """The failure seen on the robot: a0 collapses, steering pins at 0.0.

    Zero steering produces no yaw, and no yaw is exactly the evidence that
    taught the gain to collapse -- so without the prior fallback this is a
    one-way trap the controller can never climb out of.
    """
    core = AdaptiveCore()
    plant = DeadSteering()
    t = settle_sense(core, plant)
    out, t = drive(core, plant,
                   lambda s: (0.5, 0.6 * math.sin(2.0 * math.pi * 0.1 * s)),
                   60.0, t0=t)
    # the learner is entitled to conclude the gain is ~0 -- that is true
    assert abs(core.model.a0) < 0.5
    # ...but the controller must keep steering anyway
    assert out.steering_fault
    assert abs(out.steer) > 0.0


def test_steering_recovers_when_the_actuator_comes_back():
    core = AdaptiveCore()
    plant = DeadSteering()
    t = settle_sense(core, plant)
    _, t = drive(core, plant,
                 lambda s: (0.5, 0.6 * math.sin(2.0 * math.pi * 0.1 * s)),
                 60.0, t0=t)
    assert core.steering_fault
    # servo reconnected: same plant, steering now works
    alive = Plant()
    alive.x, alive.y, alive.psi, alive.v = plant.x, plant.y, plant.psi, plant.v
    out, t = drive(core, alive,
                   lambda s: (0.6, 0.6 * math.sin(2.0 * math.pi * 0.1 * s)),
                   90.0, t0=t)
    assert not core.steering_fault
    # Assert on the span the controller actually inverts, not on a0 alone: at
    # one speed a0 and a2 are not separable (they differ only by v^2), so the
    # split between them is arbitrary while their sum is not.
    m = core.model
    v = 0.6
    span = m.a0 + m.a2 * v * v
    assert span > 0.5
    assert abs(out.steer) > 0.0


def test_broken_model_is_never_persisted():
    core = AdaptiveCore()
    plant = DeadSteering()
    t = settle_sense(core, plant)
    drive(core, plant,
          lambda s: (0.5, 0.6 * math.sin(2.0 * math.pi * 0.1 * s)),
          60.0, t0=t)
    assert not core.plausible()          # must not reach disk
    healthy = AdaptiveCore()
    p2 = Plant()
    t2 = settle_sense(healthy, p2)
    drive(healthy, p2, lambda s: (0.5, 0.3), 30.0, t0=t2)
    assert healthy.plausible()


# -- regression: static friction / breakaway -------------------------------

class Sticky(Plant):
    """Needs a minimum throttle to move at all, like every real vehicle."""

    def __init__(self, breakaway=0.35, **kw):
        super().__init__(**kw)
        self.bk = breakaway

    def step(self, us, ud, dt):
        if abs(self.v) < 1e-3 and abs(ud) < self.bk:
            ud = 0.0                      # static friction wins
        super().step(us, ud, dt)


def test_stall_reflex_fires_at_nav2_speeds():
    """The old threshold was 0.25, exactly the smoother's reverse limit, so
    `abs(cmd_v) > 0.25` was never true and the reflex never ran."""
    core = AdaptiveCore()
    assert core.policy.stall_cmd_min < 0.25
    plant = Sticky(breakaway=0.5)
    t = settle_sense(core, plant)
    peak = 0.0
    engaged = False
    for _ in range(120):
        t += 0.1
        x, y, psi = plant.observe()
        out = core.step(t, x, y, psi, -0.25, 0.0)
        peak = max(peak, abs(out.drive))
        engaged = engaged or out.stalled
        for _ in range(5):
            plant.step(out.steer, out.drive, 0.02)
    # the old 0.25 threshold made `want` false at exactly this command and
    # the reflex NEVER engaged; now it must escalate well past the PI level
    assert engaged
    assert peak > 0.45, peak




def test_normal_cruise_is_not_mistaken_for_a_stall():
    """gate_d*2 was 0.30 m/s -- above Nav2's cruise, so driving normally at
    0.25 m/s counted as stalled and the reflex fought the controller."""
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    out, _ = drive(core, plant, lambda s: (0.25, 0.0), 30.0, t0=t)
    assert plant.v == pytest.approx(0.25, abs=0.10)
    assert not out.stalled


# -- regression: the violent oscillation seen on the robot -----------------

# The model actually learned on the robot, which made it lurch. b2 is
# POSITIVE: the model believed going faster makes you accelerate harder.
ROBOT_BAD_LON = [2.601463431608965, -0.08290769855274661, 4.583415448717575]
ROBOT_BAD_LAT = [0.243002130697819, -0.032426081319866, 0.136622565150797]


def _load_bad(core):
    core.rls_lon.theta = list(ROBOT_BAD_LON)
    core.rls_lat.theta = list(ROBOT_BAD_LAT)


def test_positive_drag_never_reaches_the_actuator():
    """b2 > 0 made the inversion brake while cruising at the target speed."""
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    _load_bad(core)
    plant.v = 0.5
    out, _ = drive(core, plant, lambda s: (0.5, 0.0), 3.0, t0=t)
    assert out.drive > -0.1          # raw model demanded ud ~ -0.41


def test_speed_term_cannot_invert_or_null_the_steering():
    core = AdaptiveCore()
    settle_sense(core, Plant())
    core.rls_lat.theta = [0.8, 0.0, -20.0]
    steers = []
    for v in (0.1, 0.5, 0.9, 1.4):
        core.v = v
        us, _ = core._run(v, 0.6, 0.5, 0.1)
        steers.append(us)
    assert all(x > 0.0 for x in steers), steers


def test_outputs_are_slew_limited():
    """A single sample must not swing an actuator full scale."""
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    prev_s = prev_d = 0.0
    dt = 0.1
    worst_s = worst_d = 0.0
    for i in range(120):
        t += dt
        cv = 0.6 if (i // 10) % 2 == 0 else -0.6
        cw = 3.0 if (i // 7) % 2 == 0 else -3.0
        x, y, psi = plant.observe()
        out = core.step(t, x, y, psi, cv, cw)
        worst_s = max(worst_s, abs(out.steer - prev_s))
        worst_d = max(worst_d, abs(out.drive - prev_d))
        prev_s, prev_d = out.steer, out.drive
        for _ in range(5):
            plant.step(out.steer, out.drive, dt / 5)
    assert worst_s <= core.policy.max_steer_rate * dt + 1e-9
    # throttle may come OFF at twice the rate it goes on
    assert worst_d <= 2.0 * core.policy.max_drive_rate * dt + 1e-9


def test_slew_limit_is_rate_based_not_per_sample():
    core = AdaptiveCore(Policy(max_drive_rate=2.0))
    assert core._slew(0.0, 1.0, 2.0 * 0.10) == pytest.approx(0.20)
    assert core._slew(0.0, 1.0, 2.0 * 0.05) == pytest.approx(0.10)


def test_the_robot_model_no_longer_limit_cycles():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    _load_bad(core)
    drives = []
    dt = 0.1
    for _ in range(150):
        t += dt
        x, y, psi = plant.observe()
        out = core.step(t, x, y, psi, 0.35, 0.0)
        drives.append(out.drive)
        for _ in range(5):
            plant.step(out.steer, out.drive, dt / 5)
    tail = drives[-40:]
    assert max(tail) - min(tail) < 0.5, (min(tail), max(tail))


def test_fault_reopens_the_covariance_so_recovery_is_possible():
    """A confident estimator cannot climb back out on its own.

    Thousands of consistent samples shrink the gain until the estimate is
    effectively frozen. That is right while the samples are trustworthy and
    wrong the instant they are not -- a dead servo teaches "no gain" over and
    over, and the resulting confidence is exactly what would keep the model
    wrong after the servo comes back.
    """
    core = AdaptiveCore()
    settle_sense(core, Plant())
    # well-excited, no forgetting: the covariance genuinely shrinks
    for i in range(3000):
        u = math.sin(i * 0.37)
        v = 0.4 + 0.3 * math.sin(i * 0.11)
        core.rls_lat.update([u, 1.0, u * v * v], 0.0, 1.0)
    shrunk = core.rls_lat.P[0][0]
    assert shrunk < 0.05 * core.policy.p0, shrunk

    core.get_fault_reset()
    assert core.rls_lat.P[0][0] >= core.policy.p0
    # and inflating must not disturb the estimate itself
    assert all(math.isfinite(x) for x in core.rls_lat.theta)


def test_inflate_never_shrinks_an_already_open_covariance():
    r = RLS([0.0, 0.0, 0.0], 10.0, 1e4)
    r.P[0][0] = 500.0
    r.inflate(10.0)
    assert r.P[0][0] == 500.0


# -- regression: sign logic ("left does right", "forward looks backward") --

def test_yaw_trim_winds_the_correct_way_in_reverse():
    """psidot = v * kappa: the trim must divide by SIGNED v. Dividing by |v|
    (as the original spec did) inverts the steering correction whenever the
    robot reverses -- exactly the reported 'left signal does right'."""
    core = AdaptiveCore()
    plant = Plant(a=(1.30, 0.30, -0.30))     # real curvature trim to fight
    t = settle_sense(core, plant)
    # drive straight IN REVERSE; the trim must cancel the pull, not double it
    _, t = drive(core, plant, lambda s: (-0.5, 0.0), 40.0, t0=t)
    assert abs(core.psidot) < 0.25, core.psidot   # not spiralling


def test_inverted_steering_polarity_is_learned():
    """If the servo turns the opposite way from the prior's assumption, the
    learner must cross zero and settle on a negative gain -- and the
    controller must then steer correctly through it."""
    core = AdaptiveCore()
    plant = Plant(a=(-1.30, 0.04, 0.30))     # steering reversed end to end
    t = settle_sense(core, plant)
    out, t = drive(core, plant,
                   lambda s: (0.6, 0.5 * math.sin(2.0 * math.pi * 0.10 * s)),
                   180.0, t0=t)
    assert core.model.a0 < -0.3, core.model.a0
    # commanded left turn now requires the OPPOSITE servo sign
    out, _ = drive(core, plant, lambda s: (0.5, 0.6), 10.0, t0=t)
    assert plant.psi > 0.0                    # the robot actually turns left
    assert out.steer < 0.0                    # via an inverted command


# -- regression: the model may not be inverted before it has evidence ------

def test_unready_model_is_not_inverted():
    """Sample count alone is not evidence. Constant-speed data leaves b1/b2
    unidentifiable, and inverting that split is where the wild throttle came
    from -- so until the spans are covered, throttle must ignore b1/b2."""
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    # plant garbage into the model, as single-speed driving did on the robot
    core.rls_lon.theta = [2.6, -0.08, 4.58]
    core.rls_lon.count = 500                 # plenty of samples...
    assert not core.ready_lon                # ...but no span: not ready
    plant.v = 0.5
    out, _ = drive(core, plant, lambda s: (0.5, 0.0), 3.0, t0=t)
    assert out.drive > -0.05                 # garbage b2 never reached it


def test_readiness_requires_spans_not_just_counts():
    core = AdaptiveCore()
    settle_sense(core, Plant())
    core.rls_lon.count = 10_000
    core.qd_lo, core.qd_hi = 0.30, 0.31      # one operating point
    core.vl_lo, core.vl_hi = 0.40, 0.41
    assert not core.ready_lon
    core.qd_lo, core.qd_hi = 0.10, 0.55      # real coverage
    core.vl_lo, core.vl_hi = 0.15, 0.60
    assert core.ready_lon


def test_varied_driving_reaches_readiness_and_converges():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    for speed in (0.25, 0.55, 0.35, 0.65):
        _, t = drive(core, plant, lambda s, u=speed: (u, 0.0), 20.0, t0=t)
    assert core.ready_lon
    assert core.model.b0 == pytest.approx(plant.b0, rel=0.35)


# -- regression: the stepping gait ------------------------------------------


def test_assist_never_flips_braking_into_throttle():
    """Overspeed + PI braking: the assist must pass the brake through.

    The sgn(cmd)*max(abs(ud), assist) form turned a braking command into
    forward throttle whenever the assist was smaller -- positive feedback
    straight to full throttle at 10x the commanded speed."""
    core = AdaptiveCore()
    settle_sense(core, Plant())
    core.breakaway = 0.4
    core.iv = -0.5                            # PI fully braking
    core.v = 0.9                              # far above the 0.25 command
    _, ud = core._run(0.9, 0.25, 0.0, 0.1)
    assert ud < 0.0, ud                       # still braking, not flipped


def test_cruise_settles_at_the_commanded_speed():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    _, t = drive(core, plant, lambda s: (0.25, 0.0), 40.0, t0=t)
    assert plant.v == pytest.approx(0.25, abs=0.08), plant.v


def test_sticky_vehicle_cruises_without_stepping():
    """End to end on a plant with static AND kinetic friction -- the
    realistic combination, and the one the robot actually is. Once rolling,
    speed must never fall back through the breakaway gate.

    (A plant with static friction but zero kinetic friction -- unsticks at
    0.40 then rolls like glass -- is a physical oddity this controller only
    holds to a wider band; the adaptive feedforward needs a real friction
    level to converge onto.)"""
    core = AdaptiveCore()
    plant = Sticky(breakaway=0.40)
    plant.kinetic = 0.5
    t = settle_sense(core, plant)
    _, t = drive(core, plant, lambda s: (0.32, 0.0), 40.0, t0=t)
    vs = []
    for _ in range(150):                     # 15 s of steady cruise
        t += 0.1
        x, y, psi = plant.observe()
        out = core.step(t, x, y, psi, 0.32, 0.0)
        for _ in range(5):
            plant.step(out.steer, out.drive, 0.02)
        vs.append(plant.v)
    assert min(vs) > 0.12, min(vs)           # never falls back to the gate
    assert max(vs) - min(vs) < 0.15          # tight cruise band


# -- regression: understeer from the v_eff floor, stepping at cruise --------

def test_curvature_is_not_starved_at_nav2_speeds():
    """v_eff_floor = 0.4 delivered 55% of the requested curvature at the
    0.22 m/s Nav2 actually drives -- the follower kept asking harder and the
    robot weaved. The floor must sit below the operating speed."""
    core = AdaptiveCore()
    assert core.policy.v_eff_floor <= 0.15
    plant = Plant()
    t = settle_sense(core, plant)
    # A gentle, ACHIEVABLE arc (kappa = 0.55, well inside the vehicle's
    # envelope). With the old 0.4 floor the controller computed
    # kappa = w/0.4 = 0.30 and delivered barely half the turn; with the
    # floor below the operating speed it computes w/v and delivers it.
    _, t = drive(core, plant, lambda s: (0.22, 0.12), 30.0, t0=t)
    assert core.psidot == pytest.approx(0.12, rel=0.30), core.psidot


def test_high_friction_vehicle_cruises_without_stepping():
    """Kinetic friction persists at cruise; the feedforward, launch latch
    and anti-stall floor together must hold a sticky vehicle rolling with
    no stick-slip. Uses the operating speed Nav2 is configured for."""
    core = AdaptiveCore()
    plant = Sticky(breakaway=0.40)
    plant.kinetic = 0.55          # needs ud ~0.25 just to keep rolling
    t = settle_sense(core, plant)
    _, t = drive(core, plant, lambda s: (0.32, 0.0), 20.0, t0=t)
    vs = []
    for _ in range(150):          # 15 s of steady cruise
        t += 0.1
        x, y, psi = plant.observe()
        out = core.step(t, x, y, psi, 0.32, 0.0)
        for _ in range(5):
            plant.step(out.steer, out.drive, 0.02)
        vs.append(plant.v)
    assert min(vs) > 0.10, min(vs)            # never collapses to a kick
    assert max(vs) - min(vs) < 0.20, (min(vs), max(vs))
    assert sum(vs[-50:]) / 50 == pytest.approx(0.32, abs=0.10)


def test_deep_overspeed_lets_the_pi_brake():
    """Well past the command, the anti-stall floor is gone and the PI's
    braking passes through the (additive) feedforward."""
    core = AdaptiveCore()
    settle_sense(core, Plant())
    core.breakaway = 0.4
    core.kinetic_ud = 0.30
    core.rolling = True
    core.iv = -0.4
    core.v = core.v_fb = 0.60     # 2.4x the command
    _, ud = core._run(0.60, 0.25, 0.0, 0.1)
    assert ud < 0.0, ud


# -- regression: throttle chatter from odometry noise -----------------------

def test_cruise_does_not_chatter_at_robot_noise_levels():
    """The flight recording showed +-0.05 m/s jitter on the differenced
    speed (a quarter of the 0.22 m/s command) and the throttle flipping
    between +0.32 and -0.16 tick to tick chasing it: the stepping gait.
    Same noise in simulation must now produce a calm throttle."""
    core = AdaptiveCore()
    # 4 mm pose jitter at 10 Hz ~= 0.056 m/s speed noise: the robot's level
    plant = Sticky(breakaway=0.40, pose_noise=0.004)
    plant.kinetic = 0.5
    t = settle_sense(core, plant)
    _, t = drive(core, plant, lambda s: (0.32, 0.0), 40.0, t0=t)
    uds, vs = [], []
    for _ in range(200):
        t += 0.1
        x, y, psi = plant.observe()
        out = core.step(t, x, y, psi, 0.32, 0.0)
        for _ in range(5):
            plant.step(out.steer, out.drive, 0.02)
        uds.append(out.drive)
        vs.append(plant.v)
    # recorded on the robot: repeated standstills and rail-to-rail throttle
    stops = sum(1 for v in vs if abs(v) < 0.03)
    assert stops == 0, stops
    assert min(vs) > 0.10, min(vs)
    assert sum(vs) / len(vs) == pytest.approx(0.32, abs=0.10)



# -- regression: "lunges once then doesn't move" ----------------------------




# -- regression: odometry noise must not poison the breakaway estimate -----


# -- regression: breakaway must only come from real probes ------------------


# -- the minimal throttle law: stiction handled by the integrator alone -----

def test_integrator_unsticks_and_cruises_without_stepping():
    """The retired breakaway/feedforward stack measured a biased breakaway
    (probe ramp x detection latency) and over-pushed by that bias on every
    launch: lunge, brake-slam, stall, re-measure higher. The integrator
    ramps slowly and bias-free until the wheels turn, then keeps what it
    needed. Stop-go at robot noise: no stick-slip, no lockouts."""
    core = AdaptiveCore()
    plant = Sticky(breakaway=0.45, pose_noise=0.004)
    plant.kinetic = 0.5
    t = settle_sense(core, plant)
    stops = blocked = 0
    vs = []
    first_move = None
    for cycle in range(4):
        for i in range(120):                     # 12 s go
            t += 0.1
            x, y, psi = plant.observe()
            out = core.step(t, x, y, psi, 0.32, 0.0)
            for _ in range(5):
                plant.step(out.steer, out.drive, 0.02)
            if cycle == 0 and first_move is None and abs(plant.v) > 0.1:
                first_move = i * 0.1
            # gains are sized to the loop delay, so the integrator takes a
            # few seconds to climb the dead band: judge the CRUISE, after 6 s
            if i >= 60:
                vs.append(plant.v)
                stops += abs(plant.v) < 0.03
            blocked += core.blocked
        for _ in range(15):
            t += 0.1
            x, y, psi = plant.observe()
            core.step(t, x, y, psi, 0.0, 0.0)
            plant.v = 0.0
    assert stops == 0 and blocked == 0, (stops, blocked)
    assert first_move is not None and first_move < 7.0, first_move
    assert sum(vs) / len(vs) == pytest.approx(0.32, abs=0.06)
    assert max(vs) < 0.70                       # no full-throttle lunges


def test_reverse_unsticks_too():
    core = AdaptiveCore()
    plant = Sticky(breakaway=0.45, pose_noise=0.004)
    plant.kinetic = 0.5
    t = settle_sense(core, plant)
    _, t = drive(core, plant, lambda s: (-0.25, 0.0), 8.0, t0=t)
    assert plant.v < -0.15, plant.v


def test_blocked_fires_against_an_obstacle_and_releases_on_time():
    """Immovable: the integrator pins at its limit with nothing moving ->
    blocked, zero output, then release and retry -- never a permanent
    lockout while the command persists."""
    core = AdaptiveCore()
    plant = Plant(pose_noise=0.0)
    t = settle_sense(core, plant)
    plant.b0 = 0.0
    plant.b1 = 0.0
    seen = released = False
    blocked_ticks = 0
    for _ in range(150):                         # 15 s, command never lets up
        t += 0.1
        x, y, psi = plant.observe()
        out = core.step(t, x, y, psi, 0.32, 0.0)
        if core.blocked:
            seen = True
            blocked_ticks += 1
            assert out.drive == 0.0
        elif seen:
            released = True
    assert seen and released
    assert blocked_ticks < 100


def test_no_full_lock_while_stationary():
    """A car cannot start with its wheels cranked; the servo may not go to
    lock until the car is rolling. Flight log: launches at lock needed
    0.45-0.70 throttle and lunged; straight-wheel launches needed 0.24."""
    core = AdaptiveCore()
    settle_sense(core, Plant())
    core.v = core.v_fb = 0.0
    us, _ = core._run(0.0, 0.32, 2.0, 0.1)           # hard turn requested, stationary
    assert abs(us) <= core.policy.steer_standstill + 1e-9, us
    core.v = core.v_fb = 0.3                           # rolling: authority back
    us, _ = core._run(0.3, 0.32, 2.0, 0.1)
    # (the learned-envelope clamp still bounds it, so not necessarily 1.0)
    assert abs(us) > core.policy.steer_standstill + 0.2, us


def test_learned_throttle_model_is_not_inverted_by_default():
    """A badly-fitted gain must not reach the wheels. Plant the low gain the
    robot actually learned (b0 ~ 1.2 vs a free-rolling ~6) and mark the
    model ready: the throttle must still come from the prior."""
    core = AdaptiveCore()
    settle_sense(core, Plant())
    core.rls_lon.theta = [1.2, -1.0, 0.0]
    core.rls_lon.count = 10_000
    core.qd_lo, core.qd_hi = 0.0, 0.8
    core.vl_lo, core.vl_hi = 0.0, 0.8
    assert core.ready_lon
    core.v = core.v_fb = 0.1
    core.rolling = True                     # 0.1 m/s is past the latch
    core.iv = 0.0
    _, ud = core._run(0.1, 0.32, 0.0, 0.1)
    expected = core.policy.kp_v * (0.32 - 0.1) / core.policy.prior_b0
    assert ud == pytest.approx(expected, abs=0.02), ud


def test_steering_gain_is_identified_through_a_real_delay():
    """On the robot the steering response arrives ~0.45 s after the command.
    Regressing on the instantaneous command gave r2 ~0.1 and a gain that
    wandered; aligning the regressor to the delay gives r2 ~0.8 and a
    consistent gain. A plant with that delay must be identified correctly."""
    core = AdaptiveCore(Policy(lat_delay=0.40))
    plant = Plant(pose_noise=0.002)
    plant.delay = 0.40
    t = settle_sense(core, plant)
    drive(core, plant,
          lambda s: (0.45 + 0.15 * math.sin(2.0 * math.pi * 0.03 * s),
                     0.6 * math.sin(2.0 * math.pi * 0.10 * s)),
          150.0, t0=t)
    assert core.model.a0 == pytest.approx(plant.a0, rel=0.35), core.model.a0


def test_unaligned_regression_is_worse_on_a_delayed_plant():
    """Control: the same plant with lat_delay = 0 must identify worse."""
    res = {}
    for d in (0.0, 0.40):
        core = AdaptiveCore(Policy(lat_delay=d))
        plant = Plant(pose_noise=0.002)
        plant.delay = 0.40
        t = settle_sense(core, plant)
        drive(core, plant,
              lambda s: (0.45 + 0.15 * math.sin(2.0 * math.pi * 0.03 * s),
                         0.6 * math.sin(2.0 * math.pi * 0.10 * s)),
              150.0, t0=t)
        res[d] = abs(core.model.a0 - plant.a0)
    assert res[0.40] < res[0.0], res


def test_yaw_trim_does_not_wind_up_while_saturated_or_stationary():
    core = AdaptiveCore()
    settle_sense(core, Plant())
    core.prev_us = 1.0                      # servo at lock
    core.v = core.v_fb = 0.3
    for _ in range(50):
        core._run(0.3, 0.32, 3.0, 0.1)      # impossible yaw demand
    assert abs(core.iw) < 1e-9
    core.prev_us = 0.3                      # not saturated, but stationary
    core.v = core.v_fb = 0.0
    for _ in range(50):
        core._run(0.0, 0.32, 1.0, 0.1)
    assert abs(core.iw) < 1e-9
    core.v = core.v_fb = 0.3                # rolling, unsaturated: trims
    for _ in range(50):
        core._run(0.3, 0.32, 0.3, 0.1)
    assert 0.0 < abs(core.iw) <= core.policy.iw_max + 1e-9


# -- regression: direction reversal, gates, launch floor, odometry gaps -----

def test_direction_reversal_resets_the_throttle_integrator():
    """After a cusp the integrator must not keep driving the old way.

    Flight log: 16 of 71 reversals spent more than 2 s (max 4.8 s) with the
    throttle still signed for the previous direction, because ``iv`` only
    reset on an exact (0, 0) command and the smoother's ramp through zero
    rarely produces one.
    """
    core = AdaptiveCore()
    plant = Plant()
    plant.kinetic = 0.4                     # needs real integrator effort
    t = settle_sense(core, plant)
    _, t = drive(core, plant, lambda s: (-0.3, 0.0), 20.0, t0=t)
    assert core.iv < -0.05                  # wound up for reverse
    # command flips without passing through exactly zero
    x, y, psi = plant.observe()
    out = core.step(t + 0.1, x, y, psi, 0.1, 0.0)
    assert abs(core.iv) < 0.05              # reset, then one tick of ramp
    for _ in range(4):                      # within half a second...
        t += 0.1
        for _ in range(5):
            plant.step(out.steer, out.drive, 0.02)
        x, y, psi = plant.observe()
        out = core.step(t + 0.1, x, y, psi, 0.3, 0.0)
    assert out.drive > 0.0                  # ...the throttle is forward


def test_lateral_gate_is_capped_at_the_operating_speed():
    """A noisy SENSE must not put the learning gate above cruise speed."""
    core = AdaptiveCore()
    plant = Plant(pose_noise=0.006)         # ~0.08 m/s of velocity noise
    settle_sense(core, plant, seconds=4.0)
    assert core.phase == RUN
    assert core.sigma_v > 0.03
    assert core.gate_s <= core.policy.gate_s_max + 1e-9
    assert core.gate_s == pytest.approx(1.6 * core.gate_d)


def test_launch_floor_applies_only_before_rolling():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    x, y, psi = plant.observe()
    out = core.step(t + 0.1, x, y, psi, 0.15, 0.0)
    assert out.drive >= core.policy.launch_floor - 1e-9
    for k in range(2, 5):                   # slew-limited sign change
        out = core.step(t + 0.1 * k, x, y, psi, -0.15, 0.0)
    assert out.drive <= -core.policy.launch_floor + 1e-9
    # rolling: the floor is gone and the PI is in charge
    out, t = drive(core, plant, lambda s: (0.5, 0.0), 15.0, t0=t + 0.5)
    assert core.rolling
    core.policy.launch_floor = 0.9
    x, y, psi = plant.observe()
    out = core.step(t + 0.1, x, y, psi, 0.5, 0.0)
    assert out.drive < 0.9


def test_odometry_gap_produces_a_safe_output_not_a_lesson():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    _, t = drive(core, plant, lambda s: (0.5, 0.0), 10.0, t0=t)
    n_lon = core.model.n_lon
    assert core.prev_ud > 0.0
    for _ in range(30):                     # 3 s of motion, no odometry
        plant.step(0.0, 0.3, 0.1)
    x, y, psi = plant.observe()
    out = core.step(t + 3.0, x, y, psi, 0.5, 0.0)
    assert out.drive == 0.0 and out.steer == 0.0
    assert core.model.n_lon == n_lon        # the gap taught nothing
    assert core.prev_ud == 0.0              # slew restarts from rest


# -- regression: persistence carries readiness, and only sane throttle fits --

def test_state_restores_readiness():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    drive(core, plant,
          lambda s: (0.45 + 0.2 * math.sin(2.0 * math.pi * 0.05 * s),
                     0.8 * math.sin(2.0 * math.pi * 0.1 * s)),
          60.0, t0=t)
    assert core.ready_lat and core.ready_lon
    fresh = AdaptiveCore()
    assert fresh.load_state(core.state())
    assert fresh.ready_lat and fresh.ready_lon


def test_v1_state_loads_parameters_but_not_evidence():
    core = AdaptiveCore()
    v1 = {'version': 1, 'lateral': [1.4, 0.01, -0.2],
          'longitudinal': [2.5, 0.0, -0.3],
          'envelope': {'left': [0.3] * 40, 'right': [0.3] * 40}}
    assert core.load_state(v1)
    assert core.model.a0 == pytest.approx(1.4)
    assert not core.envelope.confirmed      # pessimistic v1 evidence dropped
    assert not core.ready_lat and not core.ready_lon


def test_wrong_signed_throttle_model_is_replaced_by_the_prior_on_save():
    core = AdaptiveCore()
    settle_sense(core, Plant())
    core.rls_lon.theta = [-0.9, 0.0, 2.8]   # what the robot actually saved
    core.rls_lon.count = 500
    assert core.plausible()                 # the steering model is fine...
    assert not core.lon_plausible()         # ...the throttle one is not
    saved = core.state()
    assert saved['longitudinal'][0] == core.policy.prior_b0
    assert saved['n_lon'] == 0
    fresh = AdaptiveCore()
    assert fresh.load_state(dict(saved, longitudinal=[-0.9, 0.0, 2.8]))
    assert fresh.model.b0 == core.policy.prior_b0
