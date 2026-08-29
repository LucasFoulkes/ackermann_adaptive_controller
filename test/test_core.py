# Copyright 2026 Lucas Foulkes
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""Tests for the ROS-free controller core."""

import math

import pytest

from ackermann_adaptive_controller.core import (CAL, RUN, SENSE, AdaptiveCore,
                                            Policy, RLS, TwistEstimator, clamp,
                                            lon_sane, sane_gain_cells)
from plant import Plant, drive, settle_sense


# -- sensor self-measurement ----------------------------------------------

def test_sense_derives_gates_from_measured_noise():
    core = AdaptiveCore()
    plant = Plant()
    settle_sense(core, plant)
    assert core.phase == RUN            # calibration off by default
    assert core.sigma_v > 0.0
    # nothing driven yet, so no scale to bound it: the gate IS the noise
    assert core.gate_d == pytest.approx(10.0 * max(core.sigma_v, core.tick))
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


def test_asymmetric_steering_gain_is_learned_per_direction():
    """The 08-23 robot steers left at 1.75 curvature/command, right at 1.20.

    A symmetric model averaged the two and pushed the difference into the
    trim term a1: right turns saturated, and straights carried a phantom
    trim the car did not have. The split model must recover each side's own
    gain and keep the trim near the truth (zero here).
    """
    core = AdaptiveCore()
    plant = Plant(a=(1.75, 0.0, -0.30))
    plant.a0_right = 1.20
    t = settle_sense(core, plant)
    _, t = drive(core, plant,
                 lambda s: (0.60, 0.5 * math.sin(2.0 * math.pi * 0.12 * s)),
                 120.0, t0=t)
    assert core.model.a0l == pytest.approx(1.75, rel=0.30), core.model
    assert core.model.a0r == pytest.approx(1.20, rel=0.30), core.model
    # the asymmetry must not masquerade as trim any more
    assert abs(core.model.a1) < 0.08, core.model.a1


def test_reverse_steering_gain_is_learned_separately():
    """The 08-23 logs measured reverse-left at 0.73x its forward gain.

    Kinematically kappa = tan(delta)/L is direction-invariant, but the
    dynamics are not (caster works against a front-steered car in reverse).
    The forward cells must not be polluted by reverse samples, and the
    reverse cells must converge to the reverse plant, not the forward one.
    """
    core = AdaptiveCore()
    plant = Plant(a=(1.60, 0.0, -0.30))
    plant.rev_gain_scale = 0.7
    t = settle_sense(core, plant)
    slalom = lambda s: 0.5 * math.sin(2.0 * math.pi * 0.12 * s)
    _, t = drive(core, plant, lambda s: (0.60, slalom(s)), 90.0, t0=t)
    _, t = drive(core, plant, lambda s: (-0.60, slalom(s)), 120.0, t0=t)
    m = core.model
    assert m.a0l == pytest.approx(1.60, rel=0.30), m
    assert m.a0l_rev == pytest.approx(0.7 * 1.60, rel=0.30), m
    assert m.a0l_rev < m.a0l, m


def test_single_flipped_gain_cell_is_not_believed():
    """The 08-23 incident: fwd-right taught to -0.09 while the other three
    cells sat at 1.4-1.9. One rack drives all four cells, so a lone flipped
    sign is a poisoned fit, not a vehicle -- yet it steered fwd-right turns
    INVERTED and its near-zero span dragged the worst-cell envelope to a
    7 m planner radius (every path became a looping star)."""
    core = AdaptiveCore()
    settle_sense(core, Plant())
    core.rls_lat.theta = [1.87, -0.09, 1.43, 1.65, -0.016, -0.50]
    core.rls_lat.count = 500
    core.qs_lo, core.qs_hi = -1.0, 1.0
    assert core.ready_lat
    # envelope: the poisoned cell must not shrink the quoted curvature
    # below what the prior would give
    r = core.envelope.min_turning_radius(core.model)
    assert r < 2.0, r
    # inversion: a forward right turn must still steer right (us < 0)
    core.v = core.v_fb = 0.4
    us, _ = core._run(0.4, 0.3, -0.5, 0.1)
    assert us < 0.0, us
    # and a model like this must never reach the disk
    assert not core.plausible()
    # whereas a consistently inverted servo (all four negative) IS a
    # believable vehicle and must persist untouched
    core.rls_lat.theta = [-1.3, -1.3, -1.3, -1.3, 0.0, -0.3]
    assert core.plausible()


def test_learning_is_gated_below_the_noise_floor():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    before = core.model.n_lon
    # someone else creeps the car along under gate_d (a 0.02 m/s command
    # through this controller would lunge to the launch floor, which is
    # real motion): no samples may be accepted
    for _ in range(50):
        t += 0.1
        plant.step(0.0, 0.002, 0.1)
        x, y, psi = plant.observe()
        core.step(t, x, y, psi, 0.0, 0.0, applied=(0.0, 0.002))
    assert abs(core.v) < core.gate_d
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
    v = core.v_op
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
    v = slow.v_op
    truth = 1.0 / abs(pslow.a0 + pslow.a2 * v * v)
    # quoted at the SAME speed, the fast car's evidence must land on the
    # same truth as the slow car's
    assert fast.envelope.min_turning_radius(fast.model, v) == \
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
    unconstrained = min(abs(m.a1 + m.a0 + m.a2 * env.v_op ** 2),
                        abs(m.a1 - m.a0 - m.a2 * env.v_op ** 2))
    assert env.max_curvature(m) == pytest.approx(0.5 * unconstrained, rel=0.05)


def test_evidence_never_makes_the_envelope_more_optimistic():
    """A ratio above 1 is noise, not a car that out-turns its own model."""
    core = AdaptiveCore()
    settle_sense(core, Plant())
    env, m, v = core.envelope, core.model, 0.5
    for _ in range(env.p.env_evidence + 4):
        env.observe(1.0, 1.4 * env.predict(m, 1.0, v), v, m)
        env.observe(-1.0, 1.4 * env.predict(m, -1.0, v), v, m)
    unconstrained = min(abs(m.a1 + m.a0 + m.a2 * env.v_op ** 2),
                        abs(m.a1 - m.a0 - m.a2 * env.v_op ** 2))
    assert env.max_curvature(m) <= unconstrained * 1.001


def test_envelope_is_bounded_by_the_radius_limits():
    """The bounds are ratios of the prior's radius 1/prior_a0."""
    pol = Policy(radius_floor_ratio=0.5, radius_ceiling_ratio=3.0)
    assert pol.radius_floor == pytest.approx(0.5 / pol.prior_a0)
    assert pol.radius_ceiling == pytest.approx(3.0 / pol.prior_a0)
    core = AdaptiveCore(pol)
    settle_sense(core, Plant())
    core.rls_lat.theta = [50.0] * 4 + [0.0, 0.0]        # 2 cm: implausible
    assert core.envelope.min_turning_radius(core.model) == \
        pytest.approx(pol.radius_floor)
    core.rls_lat.theta = [1e-6] * 4 + [0.0, 0.0]        # 1000 km: collapsed
    assert core.envelope.min_turning_radius(core.model) == \
        pytest.approx(pol.radius_ceiling)


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
    # feed the plant a command the core never produced, and tell the core.
    # Throttle sized so the plant cruises near 1 m/s, not 2.2: this test is
    # about PASSIVE tapping, and at 2.2 m/s the shared a2*v^2 column
    # (|a2 v^2| ~ 1.45 vs a0 = 0.6) dominates the per-cell gain columns and
    # the split fit has nothing to pin the cells with -- a regime the real
    # robot (v <= 0.9, speed term <= 4% of gain) never enters.
    for i in range(600):
        t += 0.1
        us = 0.8 * math.sin(2.0 * math.pi * 0.1 * i * 0.1)
        ud = 0.09 + 0.04 * math.sin(2.0 * math.pi * 0.03 * i * 0.1)
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
    # covariance is re-opened to a fresh model's p0 on restore -- readier
    # to move than when saved, no more gullible than knowing nothing
    assert fresh.rls_lat.P[0][0] == pytest.approx(core.policy.p0)


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
    # 180 s: the per-direction split halves each gain's sample rate, so
    # concluding BOTH sides are dead takes twice the evidence of one, and
    # the cells hover at den_min from ~80 s to ~130 s (the fault flag
    # flips, the fallback changes the excitation, the collapse pauses)
    # before settling below it -- at 120 s the verdict depended on the
    # third decimal of the speed (2026-08-29, v_eff in the inversion).
    out, t = drive(core, plant,
                   lambda s: (0.5, 0.6 * math.sin(2.0 * math.pi * 0.1 * s)),
                   180.0, t0=t)
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
                 180.0, t0=t)   # see test_dead_steering_does_not_latch...
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
    assert core.policy.stall_cmd_frac < 1.0   # below any command that sets it
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
ROBOT_BAD_LON = [2.601463431608965, -0.08290769855274661, 4.583415448717575, 0.0]
# Recorded before the per-cell split: the symmetric gain fills all four.
ROBOT_BAD_LAT = [0.243002130697819] * 4 + [-0.032426081319866,
                                           0.136622565150797]


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
    core.rls_lat.theta = [0.8, 0.8, 0.8, 0.8, 0.0, -20.0]
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
        # the launch floor steps the wire to the breakaway by design (it
        # bypasses the slew: see reset()); every other tick is slew-bound
        if core._floor == 0.0:
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
        core.rls_lat.update([max(u, 0.0), min(u, 0.0), 0.0, 0.0,
                             1.0, u * v * v], 0.0, 1.0)
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
    # commanded left turn now requires the OPPOSITE servo sign. Assert on
    # the heading CHANGE during the turn: psi accumulates arbitrary sign
    # while the learner is still converging through the slalom above.
    psi0 = plant.psi
    out, _ = drive(core, plant, lambda s: (0.5, 0.6), 10.0, t0=t)
    assert plant.psi - psi0 > 1.0             # the robot actually turns left
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
    core.rls_lon.theta = [2.6, -0.08, 4.58, 0.0]
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
    plant = Plant()
    t = settle_sense(core, plant)
    # A gentle, ACHIEVABLE arc (kappa = 0.55, well inside the vehicle's
    # envelope). With the old 0.4 floor the controller computed
    # kappa = w/0.4 = 0.30 and delivered barely half the turn; with the
    # floor below the operating speed it computes w/v and delivers it.
    _, t = drive(core, plant, lambda s: (0.22, 0.12), 30.0, t0=t)
    assert core.v_eff_floor < 0.22 * 0.5, core.v_eff_floor
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


def test_ready_throttle_model_is_inverted_and_unready_is_not():
    """The learned longitudinal model drives the wheels once it has EARNED
    it (ready_lon: count and spans), and not a tick before. The old
    permanent distrust existed because the fit was poisoned by the dead
    band; that is compensated now and the fit is sane on hardware."""
    core = AdaptiveCore()
    settle_sense(core, Plant())
    core.rls_lon.theta = [3.0, 0.0, 0.0, -0.4]
    core.rls_lon.count = 10_000
    core.v = core.v_fb = 0.1
    core.rolling = True                     # 0.1 m/s is past the latch
    core.iv = 0.0
    # no spans yet: not ready -> prior gain
    assert not core.ready_lon
    core.now += 10.0
    _, ud = core._run(0.1, 0.32, 0.0, 0.1)
    prior_based = core.policy.kp_v * (0.32 - 0.1) / core.policy.prior_b0
    assert ud == pytest.approx(prior_based, abs=0.02), ud
    # spans covered: ready -> the learned gain takes over
    core.qd_lo, core.qd_hi = 0.0, 0.8
    core.vl_lo, core.vl_hi = -0.4, 0.8      # both directions: b3 usable
    assert core.ready_lon
    core.prev_ud = 0.0
    _, ud = core._run(0.1, 0.32, 0.0, 0.1)
    a_des = core.policy.kp_v * (0.32 - 0.1)
    expected = (a_des - (-0.4) * 1.0) / 3.0  # Coulomb feedforward included
    assert ud == pytest.approx(expected, abs=0.03), ud


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
    # core.now advances so the transport-delay settling window (which also
    # freezes the trim, deliberately) expires and each GUARD is what holds.
    core.prev_us = 1.0                      # servo at lock
    core.v = core.v_fb = 0.3
    for _ in range(50):
        core.now += 0.1
        core._run(0.3, 0.32, 3.0, 0.1)      # impossible yaw demand
    assert abs(core.iw) < 1e-9
    core.prev_us = 0.3                      # not saturated, but stationary
    core.v = core.v_fb = 0.0
    for _ in range(50):
        core.now += 0.1
        core._run(0.0, 0.32, 1.0, 0.1)
    assert abs(core.iw) < 1e-9
    core.v = core.v_fb = 0.3                # rolling, unsaturated: trims
    for _ in range(50):
        core.now += 0.1
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
    plant = Plant(pose_noise=0.002)         # sigma_v ~0.028, the noisy boot
    t = settle_sense(core, plant, seconds=4.0)
    assert core.phase == RUN
    assert core.sigma_v > 0.02
    drive(core, plant, lambda s: (0.35, 0.0), 3.0, t0=t)   # Nav2's regime
    assert core.gate_s <= 1.6 * core.policy.gate_cap_frac * core.v_op + 1e-9
    assert core.gate_s < 0.35
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
    core.rls_lon.theta = [-0.9, 0.0, 2.8, 0.0]  # what the robot once saved
    core.rls_lon.count = 500
    assert core.plausible()                 # the steering model is fine...
    assert not core.lon_plausible()         # ...the throttle one is not
    saved = core.state()
    assert saved['longitudinal'][0] == core.policy.prior_b0
    assert saved['n_lon'] == 0
    fresh = AdaptiveCore()
    assert fresh.load_state(dict(saved, longitudinal=[-0.9, 0.0, 2.8, 0.0]))
    assert fresh.model.b0 == core.policy.prior_b0


# -- regression: the throttle dead band is learned, not configured ---------

def _start_stop_cycles(core, plant, t, cycles, cmd=0.3, on=6.0, off=2.5):
    """Repeated launches from rest. Returns per-cycle time to 0.1 m/s."""
    times = []
    for _ in range(cycles):
        t_on = t
        reached = None
        n = int(on / 0.1)
        for _ in range(n):
            x, y, psi = plant.observe()
            out = core.step(t, x, y, psi, cmd, 0.0)
            for _ in range(5):
                plant.step(out.steer, out.drive, 0.02)
            t += 0.1
            if reached is None and abs(plant.v) > 0.1:
                reached = t - t_on
        times.append(reached if reached is not None else on)
        _, t = drive(core, plant, lambda s: (0.0, 0.0), off, t0=t)
        plant.v = 0.0                       # friction brings it to rest
    return times, t


def test_dead_band_is_learned_from_starts_and_compensated():
    core = AdaptiveCore()
    plant = Plant()
    plant.deadband = 0.30                   # nothing happens below 0.30
    plant.kinetic = 0.2
    t = settle_sense(core, plant)
    times, t = _start_stop_cycles(core, plant, t, cycles=8)
    assert core.deadband.confirmed(1.0)
    learned = core.deadband.fwd.value
    # measured at the wire lon_delay before motion: close to the effective
    # breakaway (dead-zone offset plus stiction over the gain), never above
    # it by more than the ramp x latency bias
    eff = plant.deadband + plant.kinetic / plant.b0
    assert learned == pytest.approx(eff, abs=0.08)
    applied = core.deadband.value(1.0)
    assert 0.0 < applied < eff              # a fraction: cannot move the car alone
    # later starts are faster than the bootstrap ones
    assert max(times[-3:]) < min(times[:2])
    # ...and those fast starts added NO samples: only slow starts measure.
    # Otherwise the estimate feeds on its own output and creeps upward.
    n_slow = sum(x > core.policy.deadband_slow_start for x in times)
    assert len(core.deadband.fwd.vals) == n_slow
    # zero command still means zero on the wire
    x, y, psi = plant.observe()
    out = core.step(t + 0.1, x, y, psi, 0.0, 0.0)
    assert out.drive == 0.0


def test_dead_band_is_per_direction():
    core = AdaptiveCore()
    plant = Plant()
    plant.deadband = 0.25
    t = settle_sense(core, plant)
    # more cycles than evidence needs: once the longitudinal inversion
    # engages, starts speed up and stop qualifying as slow samples
    _start_stop_cycles(core, plant, t, cycles=10, cmd=-0.3)
    assert core.deadband.confirmed(-1.0)
    assert not core.deadband.confirmed(1.0)
    assert core.deadband.value(1.0) == 0.0


def test_dead_band_persists_and_is_optional_in_old_state_files():
    core = AdaptiveCore()
    plant = Plant()
    plant.deadband = 0.25
    t = settle_sense(core, plant)
    _start_stop_cycles(core, plant, t, cycles=5)
    saved = core.state()
    fresh = AdaptiveCore()
    assert fresh.load_state(saved)
    assert fresh.deadband.value(1.0) == pytest.approx(core.deadband.value(1.0))
    without = {k: v for k, v in saved.items() if k != 'deadband'}
    assert AdaptiveCore().load_state(without)
    bad = dict(saved, deadband={'fwd': [1.5], 'rev': []})
    assert not AdaptiveCore().load_state(bad)


# -- regression: Coulomb term, learned delays, learned start feedforward ----

def test_coulomb_friction_is_identified_not_leaked_into_the_gain():
    """Without b3*sgn(v), drivetrain friction leaks into b0/b1 and the fit
    goes to a negative gain on hardware. With it, both come out right."""
    core = AdaptiveCore()
    plant = Plant()
    plant.kinetic = 0.35
    t = settle_sense(core, plant)
    for speed in (0.35, 0.6, -0.35, -0.6, 0.45, -0.45):
        _, t = drive(core, plant, lambda s, u=speed: (u, 0.0), 20.0, t0=t)
    m = core.model
    assert m.b0 == pytest.approx(plant.b0, rel=0.35)
    assert m.b3 < -0.1                       # friction found, right sign
    assert m.b3 == pytest.approx(-plant.kinetic, abs=0.2)


def test_the_command_to_response_delay_is_learned():
    """Two plants, one snappy and one laggy: the delay banks must diverge
    the right way, without either model losing the gain."""
    fast = AdaptiveCore()
    pf = Plant()                             # only the 0.18 s actuator filter
    t = settle_sense(fast, pf)
    drive(fast, pf,
          lambda s: (0.55, 0.7 * math.sin(2.0 * math.pi * 0.15 * s)),
          120.0, t0=t)
    slow = AdaptiveCore()
    ps = Plant()
    ps.delay = 0.4                           # plus a real transport delay
    t = settle_sense(slow, ps)
    drive(slow, ps,
          lambda s: (0.55, 0.7 * math.sin(2.0 * math.pi * 0.15 * s)),
          120.0, t0=t)
    assert fast.lat_bank.delay < slow.lat_bank.delay
    assert fast.model.a0 == pytest.approx(pf.a0, rel=0.4)
    assert slow.model.a0 == pytest.approx(ps.a0, rel=0.4)


def test_start_feedforward_is_the_learned_breakaway_not_a_preset():
    core = AdaptiveCore()
    plant = Plant()
    plant.deadband = 0.30
    plant.kinetic = 0.2
    t = settle_sense(core, plant)
    _, t2 = _start_stop_cycles(core, plant, t, cycles=10)
    assert core.deadband.confirmed(1.0)
    median = core.deadband.fwd.value
    # from rest, the very first commanded tick puts the wire AT LEAST at the
    # measured median (the inversion may reasonably ask for more)
    x, y, psi = plant.observe()
    out = core.step(t2 + 0.1, x, y, psi, 0.3, 0.0)
    assert out.drive >= median - 0.06
    assert out.drive > core.policy.launch_floor  # not the bootstrap preset


def test_trim_holds_during_the_learned_response_delay():
    """A curvature step must not wind the trim while the car cannot yet
    have responded -- that lag is transport delay, not steering error."""
    core = AdaptiveCore()
    plant = Plant()
    plant.delay = 0.4
    t = settle_sense(core, plant)
    _, t = drive(core, plant, lambda s: (0.5, 0.0), 12.0, t0=t)
    iw0 = core.iw                            # steady trim from the cruise
    # step the yaw command; sample iw within the settling window
    for k in range(3):                       # 0.3 s < delay + tau_s
        x, y, psi = plant.observe()
        out = core.step(t + 0.1 * (k + 1), x, y, psi, 0.5, 0.5)
        for _ in range(5):
            plant.step(out.steer, out.drive, 0.02)
    assert core.iw == pytest.approx(iw0, abs=0.01)  # froze through the dead time


def test_pre_coulomb_state_files_still_load():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    drive(core, plant, lambda s: (0.5, 0.3), 30.0, t0=t)
    saved = core.state()
    assert len(saved['longitudinal']) == 4
    legacy = dict(saved, longitudinal=[2.4, 0.1, -0.3])   # 3-param era
    fresh = AdaptiveCore()
    assert fresh.load_state(legacy)
    assert fresh.model.b0 == pytest.approx(2.4)
    assert fresh.model.b3 == 0.0
    assert fresh.lat_bank.delay == pytest.approx(saved['lat_delay'])


# -- regression: lying odometry (the 11:56 LiDAR-USB incident) --------------

def test_a_single_odometry_spike_holds_the_wire():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    out, t = drive(core, plant, lambda s: (0.4, 0.0), 15.0, t0=t)
    held = out.drive
    n = core.model.n_lon
    # one teleported pose (impossible acceleration), then normal again
    x, y, psi = plant.observe()
    out = core.step(t + 0.1, x + 0.4, y, psi, 0.4, 0.0)
    assert out.drive == held                # rode through, no jerk to zero
    assert core.model.n_lon == n            # and learned nothing from it
    assert core.odom_ok
    out, _ = drive(core, plant, lambda s: (0.4, 0.0), 3.0, t0=t + 0.2)
    assert core.odom_ok and abs(plant.v - 0.4) < 0.1


def test_sustained_garbage_odometry_stops_the_robot_and_the_learning():
    import random
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    _, t = drive(core, plant, lambda s: (0.4, 0.0), 15.0, t0=t)
    theta = list(core.rls_lon.theta) + list(core.rls_lat.theta)
    rng = random.Random(3)
    out = None
    for _ in range(30):                     # 3 s of a jumping scan matcher
        t += 0.1
        x, y, psi = plant.observe()
        out = core.step(t, x + rng.uniform(-0.3, 0.3),
                        y + rng.uniform(-0.3, 0.3),
                        psi + rng.uniform(-0.3, 0.3), 0.4, 0.0)
    assert not core.odom_ok
    assert out.drive == 0.0 and out.steer == 0.0
    assert list(core.rls_lon.theta) + list(core.rls_lat.theta) == theta
    # the stream comes back: sane for odom_recover_time, then drives again
    plant.v = 0.0
    for _ in range(25):
        t += 0.1
        x, y, psi = plant.observe()
        core.step(t, x, y, psi, 0.0, 0.0)
    assert core.odom_ok
    out, _ = drive(core, plant, lambda s: (0.4, 0.0), 10.0, t0=t)
    assert plant.v > 0.25


# -- regression: the learned inversion must not double-compensate -----------

def test_learned_inversion_does_not_double_compensate_the_dead_band():
    """On the robot, cruise ran at 1.37x the command the moment ready_lon
    engaged: the inversion's output is a WIRE value (the model is fit on
    the wire), and compensate() then added the dead-band offset again."""
    core = AdaptiveCore()
    plant = Plant()
    plant.deadband = 0.25
    plant.kinetic = 0.3
    t = settle_sense(core, plant)
    # learn everything: dead band, model, spans (forward and reverse)
    times, t = _start_stop_cycles(core, plant, t, cycles=5)
    _, t = drive(core, plant, lambda s: (-0.35, 0.0), 20.0, t0=t)
    _, t = drive(core, plant, lambda s: (0.45, 0.0), 20.0, t0=t)
    assert core.deadband.confirmed(1.0)
    assert core.ready_lon and core.policy.use_learned_lon
    # long steady cruise with the inversion engaged: speed must match
    for cmd in (0.3, 0.45):
        _, t = drive(core, plant, lambda s, c=cmd: (c, 0.0), 25.0, t0=t)
        assert plant.v == pytest.approx(cmd, abs=0.08), (cmd, plant.v)


# -- regression: the inversion must be consistent with its own fit ----------

def test_inversion_uses_the_fitted_sum_even_with_an_ugly_split():
    """At one cruise speed b1/b2/b3 are nearly collinear: the split is
    arbitrary, the sum is not. The old sign clamps zeroed a fitted b2 > 0
    out of the inversion and cruise ran 15%+ over the commanded speed on
    the robot. Plant a weird-but-consistent split of the true map and the
    closed loop must still settle exactly on the command."""
    core = AdaptiveCore()
    plant = Plant()                          # truth: 2.2*w - 0.35*v|v|
    t = settle_sense(core, plant)
    # a split that matches the truth at v = 0.3 but has b2 WRONG-signed:
    # truth at 0.3: bias terms = -0.35*0.09 = -0.0315
    # planted:      b1 + b2*0.09 + b3 = -0.4215 + 1.0*0.09 + 0.3*... 
    core.rls_lon.theta = [2.2, -0.4215, +1.0, 0.3]
    core.rls_lon.count = 10_000
    core.qd_lo, core.qd_hi = 0.0, 0.8
    core.vl_lo, core.vl_hi = 0.20, 0.40      # fitted only around cruise
    for r in core.lon_bank.bank:             # freeze the planted split:
        r.theta = list(core.rls_lon.theta)   # zero covariance = zero gain
        for i in range(r.n):
            for j in range(r.n):
                r.P[i][j] = 1e-12 if i == j else 0.0
    assert core.ready_lon
    _, t = drive(core, plant, lambda s: (0.3, 0.0), 30.0, t0=t)
    assert plant.v == pytest.approx(0.3, abs=0.03), plant.v


# -- regression: the model feedforward must not become feedback -------------

def test_feedforward_at_the_setpoint_does_not_surge_with_a_fat_b2():
    """Evaluated at measured v, the b2 term added gain 2*b2*v/b0 on top of
    kp; with the collinearity-inflated b2 the robot actually fitted (+2)
    and a real actuator delay, the loop limit-cycled at ~1 Hz -- the
    "walks in steps" gait. At the setpoint it is feedforward and the
    cruise must be smooth AND on the commanded speed."""
    core = AdaptiveCore()
    plant = Plant()
    plant.delay = 0.35
    t = settle_sense(core, plant)
    # consistent at v=0.3 with a wildly positive b2 (sum matches truth)
    core.rls_lon.theta = [2.2, -0.2115, +2.0, 0.0]
    core.rls_lon.count = 10_000
    core.qd_lo, core.qd_hi = 0.0, 0.8
    core.vl_lo, core.vl_hi = 0.20, 0.40
    for r in core.lon_bank.bank:
        r.theta = list(core.rls_lon.theta)
        for i in range(r.n):
            for j in range(r.n):
                r.P[i][j] = 1e-12 if i == j else 0.0
    assert core.ready_lon
    _, t = drive(core, plant, lambda s: (0.3, 0.0), 20.0, t0=t)
    vs = []
    for _ in range(100):                     # 10 s of steady cruise
        t += 0.1
        x, y, psi = plant.observe()
        out = core.step(t, x, y, psi, 0.3, 0.0)
        for _ in range(5):
            plant.step(out.steer, out.drive, 0.02)
        vs.append(plant.v)
    mean = sum(vs) / len(vs)
    dev = max(abs(x - mean) for x in vs)
    assert mean == pytest.approx(0.3, abs=0.04), mean
    assert dev < 0.06, dev                   # no surge-stall oscillation


# -- measured twist (EKF) ----------------------------------------------------

def test_measured_twist_replaces_pose_differencing():
    core = AdaptiveCore()
    plant = Plant(pose_noise=0.0)
    t = settle_sense(core, plant)
    # Pose frozen: differencing would say 0. The measured twist says 0.5.
    x, y, psi = plant.observe()
    for _ in range(60):
        t += 0.1
        core.step(t, x, y, psi, 0.0, 0.0, v_meas=0.5, psidot_meas=0.2)
    assert core.odom_ok
    assert core.v == pytest.approx(0.5, abs=0.05)
    assert core.psidot == pytest.approx(0.2, abs=0.05)


def test_measured_twist_survives_fast_jittery_poses():
    """The 08-27 replay: a parked car at 50 Hz with 5 mm of pose jitter.

    Differenced, that is 0.25 m/s of phantom velocity per sample against
    an acceleration bound of a_lim * 0.02 s, and the glitch gate declared
    the odometry failed on 17% of ticks. With the measured twist the same
    poses must never trip it.
    """
    import random
    rng = random.Random(3)
    core = AdaptiveCore()
    plant = Plant(pose_noise=0.0)
    t = settle_sense(core, plant)
    x0, y0, psi0 = plant.observe()
    failed = 0
    for _ in range(50 * 60):
        t += 0.02
        x = x0 + rng.gauss(0.0, 0.005)
        y = y0 + rng.gauss(0.0, 0.005)
        psi = psi0 + rng.gauss(0.0, 0.002)
        core.step(t, x, y, psi, 0.0, 0.0, v_meas=rng.gauss(0.0, 0.01),
                  psidot_meas=rng.gauss(0.0, 0.004))
        failed += not core.odom_ok
    assert failed == 0


def test_differencing_remains_the_fallback():
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    _, t = drive(core, plant, lambda s: (0.55, 0.0), 10.0, t0=t)
    assert core.v > 0.2                 # no twist given: differenced, moving




# -- regression: 08-28 sign poisoning that survived persistence -------------

POISONED_LAT_0828 = [2.17, -4.55, -2.50, 4.32, 0.26, -3.38]   # saved 01:01
POISONED_LON_0828 = [0.39, 0.04, 2.84, -0.04]                 # restored 22:19


def test_two_two_sign_tie_is_poison_not_a_vehicle():
    """The 08-28 file: two cells positive, two negative. Under the old
    'believe a tie' rule it passed plausible(), was persisted, and steered
    every forward right turn LEFT on each launch that restored it."""
    cells = sane_gain_cells(POISONED_LAT_0828[:4], 1.25)
    assert all(c > 0.0 for c in cells), cells
    assert cells[0] == pytest.approx(2.17) and cells[3] == pytest.approx(4.32)
    # an inverted servo still needs a majority, and no evidence stays as is
    assert sane_gain_cells([-1.3, -1.3, -1.3, 1.25], 1.25) == \
        [-1.3, -1.3, -1.3, -1.25]
    assert sane_gain_cells([1.25, 1.25, 1.25, 1.25], 1.25) == [1.25] * 4

    core = AdaptiveCore()
    settle_sense(core, Plant())
    core.rls_lat.theta = list(POISONED_LAT_0828)
    core.rls_lat.count = 500
    core.qs_lo, core.qs_hi = -1.0, 1.0
    assert core.ready_lat
    assert not core.plausible()             # must not reach disk
    core.v = core.v_fb = 0.4
    us_r, _ = core._run(0.4, 0.3, -0.5, 0.1)
    us_l, _ = core._run(0.4, 0.3, +0.5, 0.1)
    assert us_r < 0.0 < us_l, (us_r, us_l)   # right steers right, left left
    # the envelope is quoted from the sanitised cells, never from a
    # negative one (a0r=-4.55 raw would put the right side at |a1-a0r|)
    r = core.envelope.min_turning_radius(core.model)
    assert 0.0 < r < core.policy.radius_ceiling
    assert core.envelope.max_curvature(core.model) < abs(0.26 + 4.55)


def test_positive_drag_is_projected_out_of_the_fit_and_the_car_slows_down():
    """2026-08-29 12:49 run: a collinear cruise fit with b2 = +2.74 made
    the feedforward RISE as the command fell (0.32 wire for a 0.07 m/s
    command vs 0.20 for cruise), so the car ran ~0.30 m/s whatever the
    follower asked below cruise. The constraint lives in the fit, keeps
    the cruise wire (prediction-preserving projection), and a command
    below cruise must now actually slow the car."""
    core = AdaptiveCore()
    plant = Plant()
    plant.delay = 0.35
    t = settle_sense(core, plant)
    # the robot's situation: consistent at v=0.3, wildly positive b2
    seed = [2.2, -0.2115, +2.0, 0.0]
    core.qd_lo, core.qd_hi = 0.0, 0.8
    core.vl_lo, core.vl_hi = 0.20, 0.40
    core.lon_bank.seed(seed, 10_000, 1e-12)
    ff_before = -(seed[1] + seed[2] * 0.09 + seed[3]) / seed[0]
    _, t = drive(core, plant, lambda s: (0.3, 0.0), 20.0, t0=t)
    m = core.model
    assert m.b2 <= 0.0
    # the holding wire at the cruise the data pinned is (nearly) unchanged:
    # the projection preserves the prediction at the sample that fired it,
    # which with this frozen covariance is the first accepted one -- the
    # launch overshoot at ~0.34 m/s -- and nothing re-adapts afterwards.
    # On the robot P is re-opened on load and the excess at a live clip is
    # tiny, so the anchor mismatch is a test artefact: 0.02 wire here.
    ff_after = -(m.b1 + m.b2 * 0.09 + m.b3) / m.b0
    assert ff_after == pytest.approx(ff_before, abs=0.05)
    # and it no longer grows as the command shrinks
    assert -(m.b1 + m.b2 * 0.15 ** 2 + m.b3) / m.b0 <= ff_after + 1e-9
    # follower asks for half speed: the car must come down, not hold 0.3
    vs = []
    for _ in range(80):                       # 8 s
        t += 0.1
        x, y, psi = plant.observe()
        out = core.step(t, x, y, psi, 0.15, 0.0)
        for _ in range(5):
            plant.step(out.steer, out.drive, 0.02)
        vs.append(plant.v)
    assert max(vs) < 0.33
    assert vs[-1] < 0.22, vs[-1]


def test_projection_preserves_the_prediction_at_the_clipping_sample():
    r = RLS([2.0, 0.0, 0.5, 0.0], 10.0, 1e4,
            bounds=[None, None, (None, 0.0), None], absorber=1)
    phi = [0.3, 1.0, 0.09, 1.0]
    before = r.predict(phi)
    r.theta = r.project(r.theta, phi)
    assert r.theta[2] == 0.0
    assert r.predict(phi) == pytest.approx(before)


def test_speed_span_is_of_speeds_the_car_held():
    """A one-tick odometry jump inside the acceleration gate used to
    become the top of the fitted speed range."""
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    _, t = drive(core, plant, lambda s: (0.3, 0.0), 15.0, t0=t)
    hi_before = core.vl_hi
    assert hi_before is not None and hi_before < 0.45
    plant.x += 0.035                # 3.5 cm ICP jump: +0.35 m/s for one tick
    _, t = drive(core, plant, lambda s: (0.3, 0.0), 2.0, t0=t)
    assert core.vl_hi < 0.45, core.vl_hi


def test_persisted_positive_drag_is_projected_on_load():
    core = AdaptiveCore()
    settle_sense(core, Plant())
    d = core.state()
    d['longitudinal'] = [2.194, -0.024, 2.739, -0.700]     # the 12:49 file
    d['n_lon'] = 18237
    d['spans']['qd'] = [-0.42, 0.50]
    d['spans']['vl'] = [-0.6, 0.6]
    assert core.load_state(d)
    m = core.model
    assert core.v_op == 0.0           # no saved v_op: unknown until driven
    v = 0.6                           # judged at the span's top instead
    assert m.b2 == 0.0
    assert -(m.b1 + m.b3) / m.b0 == pytest.approx(
        -(-0.024 + 2.739 * v * v - 0.700) / 2.194, abs=1e-6)


def test_self_accelerating_throttle_model_is_not_a_vehicle():
    """b0 above the floor, so the old b0-only test let this through; it
    claims +0.18 m/s^2 at 0.25 m/s with the throttle at zero."""
    pol, v_op = Policy(), 0.35
    assert not lon_sane(POISONED_LON_0828, -0.4, 0.4, pol, v_op)
    assert lon_sane([1.7, 0.02, -0.4, -0.4], -0.4, 0.4, pol, v_op)  # healthy
    # judged only where the fit has been: no reverse span, no reverse test
    assert lon_sane([1.7, -0.3, -0.1, 0.0], None, 0.4, pol, v_op)
    assert not lon_sane([0.2, -0.3, -0.1, 0.0], None, 0.4, pol, v_op)
    # judged at the operating speed, not the span's lurch peak: the fresh
    # 22:28 fit is -0.005 wire at cruise and would read +1 m/s^2 at 0.89
    assert lon_sane([2.02, -0.07, 1.49, -0.10], -0.5, 0.89, pol, v_op)

    core = AdaptiveCore()
    settle_sense(core, Plant())
    core.rls_lon.theta = list(POISONED_LON_0828)
    core.rls_lon.count = 500
    core.qd_lo, core.qd_hi = -0.5, 0.5
    core.vl_lo, core.vl_hi = -0.4, 0.4
    assert core.ready_lon
    assert not core.lon_plausible()
    saved = core.state()
    assert saved['longitudinal'][0] == core.policy.prior_b0
    # and the inversion does not use it: a forward cruise request must
    # produce forward throttle, not the -0.44 the fit would give
    core.v = core.v_fb = 0.25
    _, ud = core._run(0.25, 0.35, 0.0, 0.1)    # wants to speed up
    assert ud > 0.0, ud


def test_loaded_poison_is_sanitised_before_seeding_the_learner():
    """22:19: a plain-MOLA launch restored the 22:14 file and drove
    inverted from the first tick. Neither half may survive a load."""
    src = AdaptiveCore()
    settle_sense(src, Plant())
    src.rls_lat.theta = list(POISONED_LAT_0828)
    src.rls_lat.count = 500
    src.qs_lo, src.qs_hi = -1.0, 1.0
    src.rls_lon.theta = list(POISONED_LON_0828)
    src.rls_lon.count = 500
    src.qd_lo, src.qd_hi = -0.5, 0.5
    src.vl_lo, src.vl_hi = -0.4, 0.4
    d = src.state()
    d['lateral'] = list(POISONED_LAT_0828)          # as if persisted raw
    d['longitudinal'] = list(POISONED_LON_0828)
    d['spans']['vl'] = [-0.4, 0.4]                  # the span it was fit on
    fresh = AdaptiveCore()
    assert fresh.load_state(d)
    m = fresh.model
    assert min(m.a0l, m.a0r, m.a0l_rev, m.a0r_rev) > 0.0
    assert m.b0 == fresh.policy.prior_b0 and m.b2 == 0.0



def test_braking_a_rolling_car_does_not_cross_the_reverse_dead_band():
    """22:35 relay: with dead-band evidence, a small negative correction
    while rolling forward became reverse torque. Braking is linear."""
    core = AdaptiveCore()
    settle_sense(core, Plant())
    for _ in range(6):
        core.deadband.observe(1.0, 0.19)
        core.deadband.observe(-1.0, 0.27)
    assert core.deadband.confirmed(1.0) and core.deadband.confirmed(-1.0)
    db = core.deadband
    assert db.compensate(-0.05, motion=1.0) == pytest.approx(-0.05)
    assert db.compensate(0.05, motion=-1.0) == pytest.approx(0.05)
    assert db.compensate(-0.05, motion=0.0) < -0.15     # a start: offset
    assert db.compensate(0.05, motion=1.0) > 0.10       # with motion: offset
    # through the controller: rolling forward faster than commanded on the
    # bootstrap path, the wire may retard but not by the reverse offset
    assert not core.ready_lon
    core.v = core.v_fb = core.v_fast = 0.5
    core.rolling = True
    _, ud = core._run(0.5, 0.32, 0.0, 0.1)
    assert -0.12 < ud < 0.0, ud



# -- regression: moving reversal is braking, not a launch (08-28 22:46) -----

def test_moving_reversal_brakes_without_feedforward_or_integrator():
    core = AdaptiveCore()
    settle_sense(core, Plant())
    core.rls_lon.theta = [6.0, 0.02, 0.9, -1.2]    # the 22:46 fit
    core.rls_lon.count = 500
    core.qd_lo, core.qd_hi = -0.5, 0.5
    core.vl_lo, core.vl_hi = -0.6, 0.6
    assert core.ready_lon and core.lon_plausible()
    core.rolling = True
    core.v = core.v_fb = core.v_fast = 0.5
    core.iv = 0.12                                  # wound during the brake
    core._cmd_dir = 1.0
    _, ud = core._run(0.5, -0.30, 0.0, 0.1)
    # brake: retarding, bounded by the proportional term alone
    assert -0.25 < ud < 0.0, ud
    assert core.iv == 0.0
    # the same request from rest IS a launch: floor applies
    core.rolling = False
    core.v = core.v_fb = core.v_fast = 0.0
    _, ud0 = core._run(0.0, -0.30, 0.0, 0.1)
    assert ud0 <= -core.policy.launch_floor, ud0


def test_cusp_reversal_does_not_lunge_on_the_plant():
    """Forward driving at varied speed (so the throttle model is earned),
    then a Reeds-Shepp cusp: the reverse leg must not overshoot the command
    the way the 22:46 log did (0.55 m/s median, 0.90 max, on 0.30)."""
    core = AdaptiveCore()
    plant = Plant()
    plant.delay = 0.3
    plant.deadband = 0.2
    t = settle_sense(core, plant)
    _, t = drive(core, plant,
                 lambda s: (0.40 + 0.25 * math.sin(2.0 * math.pi * 0.1 * s),
                            0.0), 40.0, t0=t)
    assert core.ready_lon and core.lon_plausible()
    # cusp: the smoother ramps the command through zero in ~0.2 s while
    # the car is still rolling forward
    peak_rev = 0.0
    t_flip = t
    for _ in range(40):
        t += 0.1
        cv = max(-0.30, 0.30 - 3.0 * (t - t_flip))
        x, y, psi = plant.observe()
        o = core.step(t, x, y, psi, cv, 0.0)
        for _ in range(5):
            plant.step(o.steer, o.drive, 0.02)
        peak_rev = max(peak_rev, -plant.v)
    assert peak_rev < 0.45, peak_rev


# -- regression: 08-28 23:15 post-restore transient and cusp window ---------

def test_direction_change_rearms_the_settled_gates():
    """A moving reversal never unlatched `rolling`, so the learners took
    the sign-change transient as settled data (16:01: a0lr 2.07 -> 0.04)."""
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    _, t = drive(core, plant, lambda s: (0.30, 0.0), 6.0, t0=t)
    assert core.rolling and core._roll_dir == 1.0
    since = core._dir_since
    roll_since = core._roll_since
    _, t = drive(core, plant, lambda s: (-0.30, 0.0), 6.0, t0=t)
    assert core.rolling and core._roll_dir == -1.0
    assert core._dir_since > since + 1.0       # lateral gate re-armed
    assert core._roll_since == roll_since      # longitudinal gate untouched


def test_restored_model_is_no_more_gullible_than_a_fresh_one():
    src = AdaptiveCore()
    plant = Plant()
    t = settle_sense(src, plant)
    drive(src, plant, lambda s: (0.4 + 0.2 * math.sin(0.6 * s), 0.3), 40.0,
          t0=t)
    fresh = AdaptiveCore()
    assert fresh.load_state(src.state())
    p0 = fresh.policy.p0
    assert fresh.rls_lat.P[0][0] == pytest.approx(p0)
    assert fresh.rls_lon.P[0][0] == pytest.approx(p0)


def test_stop_horizon_is_delay_plus_friction_stopping_time():
    core = AdaptiveCore()
    settle_sense(core, Plant())
    assert core.stop_horizon() is None                 # nothing identified
    core.rls_lon.theta = [6.0, 0.02, 0.9, -0.95]       # the 00:08 fit
    core.rls_lon.count = 500
    core.qd_lo, core.qd_hi = -0.5, 0.5
    core.vl_lo, core.vl_hi = -0.6, 0.6
    core.v_op = 0.35
    assert core.stop_horizon() == pytest.approx(
        core.lon_bank.delay + core.v_op / 0.95)
    core.rls_lon.theta = [6.0, 0.02, 0.9, 0.10]        # friction not found
    assert core.stop_horizon() is None


# -- scale-free: the same code on a different vehicle -----------------------
#
# The goal is a controller that is handed (v*, omega*) on ANY car-steered
# vehicle and learns to deliver them. Everything a vehicle IS is learned;
# the only two things declared are the priors prior_a0 (tan(lock)/wheelbase)
# and prior_b0 (full-throttle acceleration), and those only shorten the
# transient. Every other number in Policy is a fraction of a learned scale
# (v_op, the operating speed) or a ratio of one of those two priors. These
# tests drive three vehicles a decade apart in size and speed through the
# same code path, and pin the derivations to the numbers this robot was
# hand-tuned with so nothing changes here.

def test_derived_values_reproduce_this_robots_hand_tuning():
    """The 08-2x hand values, as functions of this robot's measured scale.

    sigma_v 0.0166 (state file, 08-29 13:45), v_op 0.32 (RPP's
    desired_linear_vel; the smoother allows 0.38), prior_a0 1.25,
    prior_b0 2.0. The hand values these must land on: gate_d 0.15,
    gate_s 0.24, v_eff_floor 0.12, stall_cmd_min 0.05, ready_lon_v_span
    0.15, iw_max 0.12, radius_floor 0.25, radius_ceiling 8, den_min 0.05,
    b0_min 0.3, trim gain 0.6/s (ki_w 0.3 over a 0.5 m/s floor).
    """
    core = AdaptiveCore()
    core.sigma_v, core.tick = 0.0166, 0.0002
    core.v_op = 0.32
    core._update_gates()
    p = core.policy
    assert core.gate_d == pytest.approx(0.15, rel=0.12)
    assert core.gate_s == pytest.approx(0.24, rel=0.12)
    assert core.v_eff_floor == pytest.approx(0.12, rel=0.12)
    assert core.stall_cmd_min == pytest.approx(0.05, rel=0.12)
    assert core.ready_lon_v_span == pytest.approx(0.15, rel=0.12)
    assert p.iw_max == pytest.approx(0.12, rel=0.12)
    assert p.radius_floor == pytest.approx(0.25, rel=0.12)
    assert p.radius_ceiling == pytest.approx(8.0, rel=0.12)
    assert p.den_min == pytest.approx(0.05, rel=0.12)
    assert p.b0_min == pytest.approx(0.30, rel=0.12)
    assert p.ki_w / core.v_op == pytest.approx(0.6, rel=0.12)
    # the delay grid still contains the measured 0.45 s and spans a decade
    # each way of it
    assert 0.45 in core.lat_bank.delays
    assert min(core.lat_bank.delays) <= 0.1
    assert max(core.lat_bank.delays) >= 1.5


def test_operating_speed_is_learned_from_what_is_commanded():
    core = AdaptiveCore()
    plant = Plant()
    plant.kinetic = 0.3                          # so "parked" means stopped
    t = settle_sense(core, plant)
    assert core.v_op == 0.0                      # nothing driven yet
    _, t = drive(core, plant, lambda s: (0.30, 0.0), 10.0, t0=t)
    assert core.v_op == pytest.approx(0.30, abs=0.02)
    _, t = drive(core, plant, lambda s: (0.60, 0.0), 10.0, t0=t)
    assert core.v_op == pytest.approx(0.60, abs=0.02)
    # it forgets at t_forget's half-life while driving, never while parked
    _, t = drive(core, plant, lambda s: (0.0, 0.0), 300.0, t0=t)
    assert core.v_op == pytest.approx(0.60, abs=0.02)
    _, t = drive(core, plant, lambda s: (0.30, 0.0),
                 0.5 * core.policy.t_forget, t0=t)
    assert core.v_op == pytest.approx(0.60 * 0.5 ** 0.5, abs=0.03)
    # and it rides in the state file
    assert core.state()['v_op'] == pytest.approx(core.v_op)
    fresh = AdaptiveCore()
    assert fresh.load_state(core.state())
    assert fresh.v_op == pytest.approx(core.v_op)


def test_gates_follow_the_operating_speed_not_a_constant():
    """A noisy SENSE must not put the learning gate above cruise speed --
    on ANY vehicle. The old cap was a typed 0.24 m/s."""
    for cruise in (0.10, 0.35, 2.0):
        core = AdaptiveCore(Policy(kp_v=1.6, ki_v=0.8))
        # the same sensor-to-speed ratio at every scale: the noisy boot's
        # sigma_v of 0.028 on the 0.35 m/s car (10 sigma above the cap)
        plant = Plant(pose_noise=0.002 * cruise / 0.35)
        t = settle_sense(core, plant, seconds=4.0)
        assert core.sigma_v > 0.02 * cruise / 0.35
        drive(core, plant, lambda s, u=cruise: (u, 0.0), 5.0, t0=t)
        p = core.policy
        assert core.gate_s <= 1.6 * p.gate_cap_frac * core.v_op + 1e-9
        assert core.gate_s < cruise
        assert core.gate_s == pytest.approx(1.6 * core.gate_d)


def _course(speeds, block, amp, hz):
    """A steering slalom over speed STEPS every ``block`` seconds: the
    throttle regressor is rank-deficient at one speed (README, "Drag needs
    varied speeds") and a delay is only visible in a transition, so a fair
    course excites both axes the way Nav2's segment starts and stops do."""
    return lambda s: (speeds[int(s // block) % len(speeds)],
                      amp * math.sin(2.0 * math.pi * hz * s))


def test_slow_small_vehicle_learns_and_tracks():
    """A 10 cm-wheelbase crawler: full lock 5/m, 0.3 m/s^2 at full
    throttle, cruise 0.10 m/s.

    Under the hand-tuned constants this vehicle could never learn: the
    0.15 m/s gate floor sat above its cruise (no sample ever accepted),
    the 0.12 m/s v_eff floor delivered 80% of every turn, and the 0.15 m/s
    speed-span requirement made the throttle model unready forever.

    What it can and cannot learn: the steering converges; the throttle
    GAIN does not, and that is the sensor, not the controller -- the
    whole wire range buys 0.3 m/s^2, so a Nav2-sized speed step is a
    0.01 m/s^2 signal under 0.014 m/s^2 of 10 Hz differentiation noise.
    The PI carries it (tracking is exact), and what the fit does produce
    must stay a vehicle the inversion cannot be hurt by.
    """
    core = AdaptiveCore(Policy(prior_a0=5.0, prior_b0=0.3))
    plant = Plant(a=(5.0, 0.0, -1.0), b=(0.3, 0.0, -3.0), pose_noise=0.0001)
    t = settle_sense(core, plant)
    _, t = drive(core, plant, _course((0.06, 0.10, 0.08, 0.12), 10, 0.35, 0.12),
                 160.0, t0=t)
    m = core.model
    assert m.n_lon > 100 and m.n_lat > 100, m
    assert m.a0 == pytest.approx(plant.a0, rel=0.35), m
    assert core.ready_lat
    assert core.lon_plausible(), m
    assert 0.0 < m.b0 < 2.0 * plant.b0, m
    # and it tracks: a gentle arc at cruise, delivered
    out, t = drive(core, plant, lambda s: (0.10, 0.25), 30.0, t0=t)
    assert core.v == pytest.approx(0.10, rel=0.15), core.v
    assert core.psidot == pytest.approx(0.25, rel=0.15), core.psidot


def test_large_fast_vehicle_learns_and_tracks():
    """A 3 m-wheelbase vehicle: full lock 0.12/m (8 m radius), cruise
    2 m/s, a full second of actuation delay.

    The hand-tuned constants failed it twice over: the 8 m radius ceiling
    clamped its true envelope, and the delay bank searched 0.3-0.68 s
    around a typed 0.45, so the regressors could never align.
    """
    core = AdaptiveCore(Policy(prior_a0=0.12, prior_b0=4.0))
    plant = Plant(a=(0.12, 0.0, -0.002), b=(4.0, 0.0, -0.6),
                  tau_s=0.5, pose_noise=0.002)
    plant.delay = 1.0
    t = settle_sense(core, plant)
    _, t = drive(core, plant, _course((1.2, 2.0, 1.6, 2.4), 10, 0.20, 0.04),
                 240.0, t0=t)
    m = core.model
    assert m.b0 == pytest.approx(plant.b0, rel=0.35), m
    assert m.a0 == pytest.approx(plant.a0, rel=0.35), m
    assert core.ready_lon and core.ready_lat
    # 1.0 s of transport plus the 0.3 s actuator and 0.3 s response
    # filters: the aligned candidate is 1.52 s, and both banks reach it
    assert core.lon_bank.delay >= 1.0, core.lon_bank.delay
    assert core.lat_bank.delay >= 1.0, core.lat_bank.delay
    r = core.envelope.min_turning_radius(m)
    assert r == pytest.approx(1.0 / (plant.a0 + plant.a2 * core.v_op ** 2),
                              rel=0.35), r
    out, t = drive(core, plant, lambda s: (2.0, 0.15), 40.0, t0=t)
    assert core.v == pytest.approx(2.0, rel=0.25), core.v
    assert core.psidot == pytest.approx(0.15, rel=0.30), core.psidot


def test_long_delay_is_found_from_a_wide_bank():
    """This robot's own scale, but a 1.2 s command-to-response delay: the
    bank must reach it without anyone retyping lat_delay."""
    core = AdaptiveCore()
    plant = Plant(pose_noise=0.002)
    plant.delay = 1.2
    t = settle_sense(core, plant)
    drive(core, plant,
          lambda s: (0.45 + 0.15 * math.sin(2.0 * math.pi * 0.03 * s),
                     0.6 * math.sin(2.0 * math.pi * 0.06 * s)),
          200.0, t0=t)
    assert core.lat_bank.delay >= 1.0, core.lat_bank.delay
    assert core.model.a0 == pytest.approx(plant.a0, rel=0.35), core.model.a0


def test_old_state_file_with_a_polluted_span_does_not_set_the_gates():
    """This robot's own state file, 08-29 13:45: no v_op yet, and a held-
    speed span of -1.58..1.81 m/s left by ICP jumps from before
    _held_speed (a span never shrinks). Taking the scale from that span
    put gate_d at 0.63 m/s on a 0.32 m/s car -- nothing would ever have
    been learned again. Unknown must stay unknown until the first command.
    """
    core = AdaptiveCore()
    plant = Plant()
    t = settle_sense(core, plant)
    core.sigma_v, core.tick = 0.0166, 0.0002
    d = core.state()
    del d['v_op']
    d['longitudinal'] = [3.737, 0.026, -0.428, -0.755]
    d['n_lon'] = 18696
    d['spans']['qd'] = [-0.71, 0.61]
    d['spans']['vl'] = [-1.58, 1.81]
    fresh = AdaptiveCore()
    assert fresh.load_state(d)                # as the node does, before SENSE
    t = settle_sense(fresh, plant)
    fresh.sigma_v, fresh.tick = 0.0166, 0.0002   # this robot's measurement
    fresh._update_gates()
    assert fresh.v_op == 0.0
    assert fresh.gate_d == pytest.approx(0.166, abs=0.01)   # 10 sigma
    assert fresh.stop_horizon() is None                    # not yet driven
    assert fresh.ready_lon and fresh.lon_plausible()       # the fit is kept
    # the first Nav2 command sets the scale, and the gates land on the
    # hand-tuned 0.15 within a tick
    fresh.step(t + 0.1, *plant.observe(), 0.32, 0.0)
    assert fresh.v_op == pytest.approx(0.32)
    assert fresh.gate_d == pytest.approx(0.15, rel=0.05)
