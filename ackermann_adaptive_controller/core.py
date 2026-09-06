# Copyright 2026 Lucas Foulkes
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""Odometry-feedback actuator control and lifecycle of the learned vehicle model.

SENSE measures stationary noise, RUN learns from requested driving, and CAL
runs only on an explicit commissioning request. Steering response is learned
per travel direction and steering side; throttle response includes friction
and delay. Production uses timestamped serial-delivery history. The pure
Python plant tests can provide synchronous applied commands directly.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from .model_checks import lon_sane
from .math_utils import finite, clamp, wrap, sgn, sane_gain_cells, _span, _load_span, _stddev
from .policy import Policy
from .identification import RLS, DelayBank
from .motion import TwistEstimator
from .response import Model, CurvatureEnvelope, DeadBand, GainProbe
from .scoring import DriveScore

SENSE = 'SENSE'
CAL = 'CAL'
RUN = 'RUN'


@dataclass
class Output:
    """One control step's result."""

    steer: float = 0.0
    drive: float = 0.0
    phase: str = SENSE
    learning: bool = False
    stalled: bool = False
    v: float = 0.0
    psidot: float = 0.0
    dt: float = 0.1
    model: Model = field(default_factory=Model)
    min_turning_radius: float = 0.0
    max_curvature: float = 0.0
    envelope_confirmed: bool = False
    breakaway: float = 0.0
    steering_fault: bool = False
    drive_fault: str = ''
    deadband_fwd: float = 0.0
    deadband_rev: float = 0.0
    steer_wait: bool = False      # throttle held while the wheels turn


class AdaptiveCore:
    """The phase machine, the learners, and the control law."""

    def __init__(self, policy=None):
        self.policy = policy or Policy()
        self.reset()

    # -- lifecycle ---------------------------------------------------------

    def reset(self):
        p = self.policy
        self.phase = SENSE
        self.phase_t0 = None
        self.now = 0.0
        self.dt = 0.1

        self.twist = TwistEstimator()
        self.v = 0.0
        self.v_fb = 0.0
        self.v_prev = 0.0
        self._v_prev_t = 0.0     # stamp of the sample v_prev came from
        self.vdot = 0.0
        self.psidot = 0.0

        # Sensor character, measured in SENSE.
        self._v_samples = []
        self._psi_diffs = []
        self.sigma_v = 0.0
        self.tick = 0.0
        self.sigma_psi = 0.0

        # Derived, never tuned. The gates are set from the SENSE noise
        # measurement and re-scaled every tick against v_op (below).
        self.gate_d = 0.0
        self.gate_s = 0.0
        self.alpha = 0.33
        self.lam = 0.999
        self.dither = 0.0
        # The operating speed: the one learned SCALE every speed-shaped
        # threshold is a fraction of. A high-water mark of what the vehicle
        # is commanded (ACTIVE) or seen to hold (PASSIVE, CAL), forgotten at
        # t_forget's half-life -- but only while being driven: a car parked
        # for an hour has not become a slower car, and letting the gates
        # decay under a parked car would open them to odometry noise. Zero
        # until the first motion; persisted (see state()).
        self.v_op = 0.0

        def _delays(center):
            """Log-spaced candidates from the prior out to the bounds.

            The prior is always on the grid (so a persisted delay restores
            exactly) and is the starting candidate. A non-positive centre
            means "no alignment": the single candidate 0, which is what a
            test of the alignment itself needs as its control."""
            if center <= 0.0 or p.delay_spread <= 1.0:
                return [max(center, 0.0)]
            below, d = [], center
            while d / p.delay_spread >= p.delay_min:
                d /= p.delay_spread
                below.insert(0, d)
            above, d = [], center
            while d * p.delay_spread <= p.delay_max:
                d *= p.delay_spread
                above.append(d)
            return below + [center] + above
        # All four gain cells start at the same prior; the data splits them.
        # a2 <= 0 is physics (cornering does not improve with speed) and
        # is enforced IN the fit, like b2: at one speed a0 and a2 are
        # collinear, and on a 1.8 m/s vehicle the fit once put the whole
        # gain into a2 = +0.039 (x 3.24 = the true a0) with the cells at
        # -0.1 -- the envelope quoted 9.8 m one run and 22.6 the next
        # (08-29 bench). The absorber is the cell in play (one of the
        # four has a nonzero regressor per sample): it takes the gain a2
        # may not hold, and the sample's prediction is unchanged.
        self.lat_bank = DelayBank([p.prior_a0] * 4 + [0.0, 0.0],
                                  p.p0, p.p_max,
                                  _delays(p.lat_delay), p.delay_ew_tau,
                                  p.delay_switch_margin,
                                  bounds=[None] * 5 + [(None, 0.0)],
                                  absorber=(0, 1, 2, 3),
                                  gain_idx=(0, 1, 2, 3),
                                  min_count=p.ready_lat_samples,
                                  # a2 <= 0 is physics (see _run); an
                                  # unphysical positive a2 may not buy a
                                  # candidate a larger span
                                  gains=lambda th: [
                                      th[k] + min(th[5], 0.0) * self.v_op ** 2
                                      for k in range(4)])
        self.lat_bank.set_delay(p.lat_delay)
        # Drag opposes motion: b2 <= 0 is physics, enforced IN the fit.
        # At one cruise speed b1/b2/b3 are collinear and only their sum is
        # pinned; left free, the split wandered to b2 = +2.74 (2026-08-29
        # 12:49 run, 18k samples at 0.32 m/s), and the inversion evaluated
        # that sum at the COMMANDED speed -- off the cruise point, where
        # the split is everything: the model's holding wire fell with
        # speed (0.33 at 0.05 m/s, 0.20 at 0.32), so a 0.07 command got
        # MORE wire than cruise and the car did 0.30 m/s for any command
        # in 0.05..0.32, arriving at every cusp at 0.36 m/s. The old fix
        # was a post-hoc clamp in the inversion, rejected because it moved
        # the equilibrium; the projection here keeps the prediction at the
        # clipping sample exact by moving the excess onto b1.
        self.lon_bank = DelayBank([p.prior_b0, 0.0, 0.0, 0.0], p.p0, p.p_max,
                                  _delays(p.lon_delay), p.delay_ew_tau,
                                  p.delay_switch_margin,
                                  bounds=[None, None, (None, 0.0), None],
                                  absorber=1, gain_idx=(0,),
                                  min_count=p.ready_lon_samples)
        self.lon_bank.set_delay(p.lon_delay)
        self._kappa_hist = deque()   # (t, kappa_des) for the delayed trim
        self.odom_ok = True
        self._glitch_run = 0
        self._glitch_since = 0.0
        self._sane_run = 0
        # Start floor, re-applied AFTER the slew: stepping the wire straight
        # to the learned breakaway IS the controlled launch; ramping to it
        # through the slew just re-creates the stiction wait.
        self._floor = 0.0
        self._floor_dir = 0.0
        self._cap = 0.0
        self._floor_pinned = False
        self._post_latch_until = 0.0
        # Anti-windup for the throttle integrator: the direction in which
        # the last published throttle fell short of what the loop asked
        # (+1: less than asked, -1: more), because the +-1 clamp or the
        # slew limit shaped it. While that holds, the integrator does not
        # wind further INTO the limit (it may always unwind). The launch
        # floor and cap are deliberately NOT shaping: the reflex reads the
        # integrator pinning under the cap as "against something".
        self._shaped = 0.0
        self._clip = 0.0
        # The feedforward-only throttle for the current command (no PI):
        # what an implausible odometry sample is answered with.
        self._ff_wire = None
        self.envelope = CurvatureEnvelope(p)
        self._roll_dir = 0.0
        self._dir_since = None
        self.deadband = DeadBand(p)
        self.gain_probe = GainProbe(p)
        self.score = DriveScore()
        # odometry-health counters (see DriveScore.count): a glitch hold
        # is a tick answered with the holding wire, an implausible
        # episode a stream distrusted for odom_glitch_hold
        self.n_glitch_holds = 0
        self.n_implausible = 0

        # Launch/rolling latch for the breakaway kick, with hysteresis so
        # odometry noise cannot flicker it.
        self.rolling = False
        self._roll_run = 0
        self._roll_since = None
        self._steer_wait = False
        self._below_gate = False
        self._still_since = None
        self._calm_since = None    # |vdot| within noise since (gain probe eq)
        self.v_fast = 0.0
        self.qs = self.qd = 0.0
        # (stamp, us, ud) history for delay-aligned regression. Deep enough
        # for delay_max at any odometry rate this runs at (2.5 s at 400 Hz);
        # it was 64 entries, i.e. 1.3 s at the EKF's 50 Hz.
        self._cmd_hist = deque(maxlen=1024)
        self.external_history = False
        # Sane raw speeds, for the held-speed span (see _held_speed).
        self._v_hist = deque(maxlen=1024)
        # Steering trim, ONE INTEGRATOR PER CELL (travel direction x
        # steering side) that persists across stops and reversals. A single
        # integrator zeroed at every zero command could never trim a
        # per-cell bias out of a 3 s cusp leg: 09-02 17:33, the fit sat
        # 10-23% above the measured gain in every cell, worst reverse-right
        # (delivered 0.73 of the commanded curvature, at lock 49% of its
        # time), and the trim restarted from nothing on every leg.
        self._iw_cells = {}
        self._iw_key = (1.0, 1.0)
        self.iv = 0.0
        self.prev_us = self.prev_ud = 0.0
        # Last throttle actually put on the wire (after dead-band compensation),
        # its sign, and when that sign last came on (for the slow-start test).
        self.prev_wire = 0.0
        self._wire_sign = 0.0
        self._wire_on_since = None
        # What was actually on the wire this tick, when someone else drives.
        self._wire = None
        self._cmd_dir = 0.0
        self._stall_since = None
        # Smallest throttle that has actually been seen to move this vehicle.
        # Static friction is not expressible in a linear model, so it is
        # measured instead.
        # Excitation ranges actually covered by accepted samples.
        self.qd_lo = self.qd_hi = None
        self.vl_lo = self.vl_hi = None
        self.qs_lo = self.qs_hi = None
        self._capped_since = None
        self._blocked_until = None
        self.blocked = False
        self._failures = 0
        self.steering_fault = False
        # Established signs. steer_sign is None until the learned cells
        # have agreed unanimously on the servo's direction (CAL, or the
        # first time the lateral model is ready); from then on the prior
        # carries it and every tie or minority cell resolves to it, never
        # to the sign the linkage was ASSUMED to have. Persisted. A reset
        # forgets it: that is what a reset is for.
        self.steer_sign = None
        self._b0_ref = None      # last persisted-sane b0 (see _run)
        self._cal_stage = None   # 'signs' | 'gains' | 'reverse'
        self._cal_t0 = None      # when the current stage began
        self._cal_sub = None     # reverse: 'brake' | 'pulse' | 'stop'
        # Sign witness PER candidate delay: sum of qs(t - d) * kappa and
        # of |qs * kappa| for every d on the lateral grid, and the count.
        # Established from the candidate with the largest correlation
        # magnitude, which is the aligned one whatever the prior says: a
        # correctly wired plant with a 1.2 s delay was locked INVERTED by
        # a witness on the 0.45 s prior (150 degrees out of phase with
        # the 0.55 Hz wiggle, 08-29 bench).
        self._sign_num = None
        self._sign_den = None
        self._sign_n = 0
        # The same model-free witness for the throttle: sum of ud(t - d) *
        # vdot per candidate delay. Its peak across the grid is the
        # cross-correlation delay estimate, which is what CAL trusts: the
        # FITTED gains stopped peaking at the aligned delay once a2 was
        # projected (the absorber inflates a misaligned candidate's cell),
        # while the raw sums peaked at 1.01 / 1.52 / 2.28 s for plants at
        # 0.8 / 1.2 / 2.0 s (08-29 bench).
        self._lon_corr = None
        # The throttle sign is the interface contract (positive wire is
        # forward), verified by CAL rather than compensated: a car that
        # backs away from +0.42 wire, or does not move at all, is wiring,
        # and the actuators are held at zero until a reset or another CAL.
        self.drive_fault = ''

    def _enter(self, phase, t):
        self.phase = phase
        self.phase_t0 = t

    def start_cal(self):
        """Begin the scripted calibration wiggle at the next odometry sample.

        ``phase_t0`` is left unset so :meth:`step` stamps it from the next
        odometry stamp — the clock this whole core runs on. Stamping it from
        the caller's clock mixed timebases: under bag replay or sim time the
        wiggle either ended instantly or never.
        """
        self._enter(CAL, None)
        self.drive_fault = ''
        self._cal_stage = 'signs'
        self._cal_t0 = None
        self._cal_sub = None

    @property
    def prior_a0(self):
        """The steering prior with the ESTABLISHED sign, or as declared."""
        p = self.policy
        if self.steer_sign is None:
            return p.prior_a0
        return self.steer_sign * abs(p.prior_a0)

    def witness_delay(self, axis):
        """The cross-correlation delay estimate for 'lat' or 'lon': the
        candidate whose model-free witness sum has the largest magnitude,
        or None before there is any."""
        if axis == 'lat':
            sums, bank = self._sign_num, self.lat_bank
        else:
            sums, bank = self._lon_corr, self.lon_bank
        if not sums or not any(sums):
            return None
        i = max(range(len(sums)), key=lambda k: abs(sums[k]))
        return bank.delays[i]

    def _establish_sign(self):
        """Lock the servo's direction from the command-curvature witness.

        See Policy.sign_evidence. Until it locks, the declared prior's sign
        stands (sane_gain_cells already believes a unanimous inversion in
        the cells; establishment is what makes a LATER tie or minority
        resolve to the real servo instead of the assumed one).
        """
        if self.steer_sign is not None:
            return
        p = self.policy
        if self._sign_n < p.sign_evidence or self._sign_num is None:
            return
        # the correlation peak across the grid: largest |sum|
        i = max(range(len(self._sign_num)), key=lambda k: abs(self._sign_num[k]))
        if self._sign_den[i] <= 0.0:
            return
        r = self._sign_num[i] / self._sign_den[i]
        if abs(r) < p.sign_agreement:
            return
        self.steer_sign = sgn(r)
        self.envelope.prior_a0 = self.prior_a0
        self._reseed_unexcited()

    def _reseed_unexcited(self):
        """Move cells that still sit at the declared seed onto the signed
        prior, in every bank candidate. A cell nobody has driven yet holds
        no evidence; left at +seed on an established-inverted car it read
        as two strong votes for the wrong sign while the correctly learned
        cells, sitting AT the (negative) prior, did not vote at all -- and
        sane_gain_cells flipped the good cells (08-29 bench run, reverse
        cells never driven)."""
        seed = self.policy.prior_a0
        signed = self.prior_a0
        if signed == seed:
            return          # the sign agrees with the seed: nothing to move
        for r in self.lat_bank.bank:
            for k in range(4):
                if abs(r.theta[k] - seed) <= 0.1 * abs(seed):
                    r.theta[k] = signed
        self.lat_bank.prior = [signed] * 4 + list(self.lat_bank.prior[4:])

    def _settled(self):
        """Past the settling window after the last (re-)breakaway."""
        return (self.rolling and self._roll_since is not None
                and self.now - self._roll_since
                >= self.lon_bank.delay + self.policy.tau_d)

    def _speed_up_limit(self, since, v_new=None):
        """Largest plausible gain in |v| since the last accepted sample
        (``since`` seconds ago), or None while the launch transient is
        exempt. margin x b0 (prior-bounded) x the largest effective wire
        in the direction of travel over the settling window, plus five
        sigma. The LARGEST wire over the window, not the wire one learned
        delay ago: the true delay is only known to the bank's grid, and a
        stick-slip release answers a wire 0.3 s old while the 0.45 s-old
        one still read the pre-launch value (bench, 08-29). Scaled by the
        time since the last accepted sample so a held tick cannot chain:
        every later sample is judged against a frozen v_prev."""
        p = self.policy
        if not self._settled():
            return None
        noise = max(self.sigma_v, self.tick)
        if noise <= 0.0:
            return None     # no measured noise floor: no scale for "impossible"
        # Judged in the direction of the NEW sample: the wire that could
        # have produced this speed is the wire pushing that way.
        direction = (sgn(v_new) if v_new is not None else 0.0) \
            or sgn(self.v_prev)
        if not direction:
            return None
        window = self.now - (self.lon_bank.delay + p.tau_d) - since
        u_same = 0.0
        for stamp, _us, ud in reversed(self._cmd_hist):
            if stamp < window:
                break
            u_same = max(u_same, ud * direction)
        reversal = (v_new is not None and sgn(v_new)
                    and sgn(v_new) != sgn(self.v_prev) and self.v_prev != 0.0)
        # The torque-producing wire: beyond the dead band for a speed-up
        # in the direction of travel; the RAW wire for a reversal, because
        # braking on an H-bridge is retarding torque from the first count
        # (see DeadBand.compensate) -- subtracting the band there bounded
        # a braking cusp reversal at five sigma while the reverse wire
        # was still ramping, held honest -0.15 m/s readings, and zeroed
        # the outputs mid-cusp: 3 implausible episodes, 11 holds, 8
        # stalled legs in 45 s (09-02 18:00).
        u_eff = u_same if reversal \
            else max(u_same - self.deadband.value(direction), 0.0)
        # From near rest anything can happen (stiction release): no bound
        # -- but only with SOME wire that way to release it. Without the
        # wire condition one plausible dip below the gate switched the
        # bound off for the rest of the stream, and a +0.18 -> -0.49 m/s
        # phantom on a car driving forward on 0.02 wire passed (bench,
        # 09-02); a MOLA phantom of the "+0.10 while reversing" kind is
        # exactly this shape. Parked with no wire, the bound stays.
        if abs(self.v_prev) < self.gate_d and u_same > 0.0:
            return None
        b0 = clamp(self.rls_lon.theta[0], p.prior_b0, 2.0 * p.prior_b0)
        return p.odom_glitch_margin * b0 * u_eff * since + 5.0 * noise

    def _a_limit(self):
        """Physics bound on plausible acceleration, from the learned model
        but BOUNDED. The unbounded form was circular: the 02:15 session's
        poisoned fit (b0 6.4, b3 -1.4) widened its own acceptance gate to
        15.7 m/s^2 and the gate stopped rejecting anything. The learned
        values may tighten the bound or stretch it moderately, never hold
        the door open for their own poison."""
        p = self.policy
        b = self.rls_lon.theta
        b0 = clamp(b[0], p.prior_b0, 2.0 * p.prior_b0)
        # Friction stronger than the (bounded) drive gain would mean a car
        # that cannot move at all; cap it at the prior.
        b3 = min(abs(min(b[3], 0.0)), p.prior_b0)
        return p.odom_glitch_margin * (b0 + b3)

    @property
    def rls_lat(self):
        return self.lat_bank.rls

    @property
    def rls_lon(self):
        return self.lon_bank.rls

    @property
    def ready_lon(self):
        """Has the longitudinal model earned the right to be inverted?"""
        p = self.policy
        # Both spans checked: a hand-edited state file can restore one span
        # without the other (each degrades to None independently on load).
        # The wire-span bar scales with the vehicle: ready_lon_qd_span was
        # typed for THIS robot (cruise wire ~0.2), and an easy vehicle
        # whose entire operating range lives under 0.1 wire can never
        # span it -- well-behaved control keeps the wire NEAR its
        # equilibrium, so the identifiable range is proportional to the
        # measured cruise wire (the eq itself plus half again: driving
        # that has both braked below and pushed above its equilibrium by
        # half of it has excited the affine fit across the operating
        # point). The typed value stays as the ceiling so a small
        # measured eq cannot lower the bar to nothing on a vehicle where
        # 0.15 is genuinely available.
        eq = self.gain_probe.eq(1.0)
        span_needed = min(p.ready_lon_qd_span,
                          1.5 * eq) if eq else p.ready_lon_qd_span
        return (self.rls_lon.count >= p.ready_lon_samples
                and self.qd_lo is not None and self.vl_lo is not None
                and self.qd_hi - self.qd_lo >= span_needed
                and self.ready_lon_v_span > 0.0
                and self.vl_hi - self.vl_lo >= self.ready_lon_v_span)

    # -- loop gains, from the loop's dead time (Policy.kp_delay_product) --

    @property
    def L_lon(self):
        p = self.policy
        return self.lon_bank.delay + p.tau_d + p.v_fb_tau

    @property
    def L_lat(self):
        p = self.policy
        return self.lat_bank.delay + p.tau_s + p.sensor_tau

    @property
    def kp_v(self):
        return self.policy.kp_delay_product / self.L_lon

    @property
    def ki_v(self):
        return self.kp_v / (self.policy.ti_delay_ratio * self.L_lon)

    @property
    def ki_w(self):
        return self.policy.kw_delay_product / self.L_lat

    @property
    def steer_wait_tol(self):
        """Servo error (fraction of full lock) a launch from rest may start
        with: the curvature band the planner's radius margin leaves below
        the lock (radius_push_margin). Floored at 2% of lock, below any
        servo's resolution, so a margin of 1 cannot wait forever on the
        modeled servo's exponential tail."""
        return max(0.02, 1.0 - 1.0 / max(self.policy.radius_push_margin,
                                          1.0))

    # -- speed-shaped thresholds, all fractions of the learned v_op --------

    @property
    def v_eff_floor(self):
        return self.policy.v_eff_floor_frac * self.v_op

    @property
    def stall_cmd_min(self):
        return self.policy.stall_cmd_frac * self.v_op

    @property
    def ready_lon_v_span(self):
        return self.policy.ready_lon_v_span_frac * self.speed_scale

    @property
    def speed_scale(self):
        """The operating speed, or -- for a fit restored or hand-set
        before anything was driven -- the top of the fit's own speed span.
        ONLY for judging the throttle fit (readiness, lon_sane, the load
        projection), which clamp into that span anyway. Never for the
        gates or the horizon: a persisted span can be polluted (this
        robot's read -1.58..1.81 m/s on 08-29, ICP jumps from before
        _held_speed, and a span never shrinks), and a scale taken from it
        put the gates at 0.63 m/s on a 0.32 m/s car."""
        return self.v_op or max(abs(self.vl_lo or 0.0), abs(self.vl_hi or 0.0))

    @property
    def ready_lat(self):
        p = self.policy
        return (self.rls_lat.count >= p.ready_lat_samples
                and self.qs_lo is not None
                and self.qs_hi - self.qs_lo >= p.ready_lat_qs_span)

    @property
    def model(self):
        a = self.rls_lat.theta
        b = self.rls_lon.theta
        return Model(a0l=a[0], a0r=a[1], a0l_rev=a[2], a0r_rev=a[3],
                     a1=a[4], a2=a[5],
                     b0=b[0], b1=b[1], b2=b[2], b3=b[3],
                     n_lat=self.rls_lat.count, n_lon=self.rls_lon.count)

    # -- the step ----------------------------------------------------------

    def step(self, t, x, y, psi, cmd_v, cmd_w, applied=None,
             v_meas=None, psidot_meas=None, passive=None):
        """One odometry sample in, one actuator command out (see _step).
        Every path through it -- held, zeroed, published -- is scored."""
        out = self._step(t, x, y, psi, cmd_v, cmd_w, applied=applied,
                         v_meas=v_meas, psidot_meas=psidot_meas, passive=passive)
        if self.phase == RUN:
            self.score.observe(
                t, cmd_v, out.v, out.stalled, self.stall_cmd_min,
                self.gate_d,
                # the overshoot has peaked within a few actuation delays
                # of reaching the command; three covers a 1 s vehicle
                3.0 * (self.lon_bank.delay + self.policy.tau_d),
                cmd_w=cmd_w, psidot=out.psidot,
                kappa_max=self.envelope.max_curvature(self.model))
        return out

    def _step(self, t, x, y, psi, cmd_v, cmd_w, applied=None,
             v_meas=None, psidot_meas=None, passive=None):
        """Advance one odometry sample. Returns an :class:`Output`.

        ``t`` is seconds from the odometry stamp; the whole loop is driven by
        measured time so it is correct at any odometry rate.

        ``v_meas`` / ``psidot_meas`` are a MEASURED body twist (signed
        forward speed, yaw rate) when the odometry source provides a real
        one (wheel encoders). Given, they replace the pose-differenced
        values. NOT for a filter's velocity STATE: the robot_localization
        EKF's twist lagged the car by 1-1.5 s and a learner fitting it
        collapsed the steering cells within a minute, four times on
        08-28/29 -- the launch pins use_odom_twist false for it. The rest of
        this note is why differencing is only sound at its native rate:
        replaying a parked 50 Hz EKF session (08-27) through the
        differencer tripped the plausibility gate on 17% of ticks -- 5 mm
        of pose jitter over 20 ms is 0.25 m/s of "velocity" against a
        0.08 m/s bound -- while the same data decimated to 10 Hz tripped
        none. The pose is still consumed for the sample clock and the
        glitch gate; the differencer keeps running as the fallback.

        ``applied`` is the ``(steering, throttle)`` pair actually on the wire,
        when that is not what this controller last emitted -- which is exactly
        the PASSIVE case, where the joystick is driving. Learning from your own
        unpublished outputs while someone else steers teaches the model the
        opposite of the truth, so the caller must pass the real commands.
        """
        if not finite(t, x, y, psi):
            return self._safe_output()
        if not finite(cmd_v, cmd_w):
            cmd_v = cmd_w = 0.0

        step = self.twist.update(t, x, y, psi)
        if step is None:
            return self._safe_output()
        dt, v, psidot_raw = step
        if v_meas is not None and psidot_meas is not None \
                and finite(v_meas, psidot_meas):
            v, psidot_raw = float(v_meas), float(psidot_meas)

        if self.phase != SENSE and dt > 5.0 * self.dt:
            # Odometry gap. The differenced sample is an average over the
            # gap, not a measurement: dropping it and re-seeding the virtual
            # sensors beats teaching the learner from it. The output is zero,
            # so the slew restarts from rest instead of stepping back to the
            # pre-gap command when odometry returns.
            self.now = t
            self.v_prev = self.v_fb = self.v_fast = v
            self._v_prev_t = t
            self._wire = None
            return self._safe_output()

        # ---------- odometry plausibility ---------------------------------
        # A perfectly-timed stream can still be lying (failing LiDAR link,
        # scan matcher jumping). Bounds from the vehicle's LEARNED physics:
        # it cannot accelerate harder than full throttle plus friction says,
        # and cannot yaw faster than its envelope at this speed allows.
        if self.phase != SENSE:
            a_lim = self._a_limit()
            w_lim = max(
                self.policy.odom_glitch_margin
                * self.envelope.max_curvature(self.model, self.v_prev)
                * max(abs(self.v_prev), self.gate_d),
                1.5)
            # Speeding UP is bounded by the wire, not by full throttle: the
            # car cannot gain speed faster than the throttle actually on
            # it (delayed, past the dead band) can push, plus the noise.
            # Slowing down is free (friction, a brake, a wall). Exempt
            # until settled after breakaway: stiction release is real
            # acceleration the wire did not put in. This is the gate the
            # 08-29 drive needed -- 212 ticks read 0.4-0.78 m/s against a
            # 0.32 command, mostly 0.5-0.65 on a 0.11-0.22 wire while
            # coasting into a cusp, and the full-throttle bound (1 m/s per
            # tick) let every one of them through into v_op, the delay
            # banks and the a2 fit.
            # Every bound is per second SINCE THE LAST ACCEPTED SAMPLE, not
            # per tick: a held tick freezes v_prev, and judged per tick every
            # later sample failed against it -- a chain to a fault that only
            # the old 0.3 s trip had hidden (08-29 bench, 2 m/s cruise).
            since = max(t - self._v_prev_t, dt)
            up_lim = self._speed_up_limit(since, v)
            # A sign REVERSAL within one sample is a speed-up from zero in
            # the new direction, not a slow-down: judged as the whole new
            # speed. Without this a +0.26 -> -0.32 m/s reading on a car
            # cruising forward on 0.02 wire passed the gate ("slowing down
            # is free"), the reversing rule then zeroed the feedforward,
            # and every held tick after it held zero (09-02 bench).
            gained = (abs(v) - abs(self.v_prev)
                      if sgn(v) == sgn(self.v_prev) else abs(v))
            if abs(v - self.v_prev) > a_lim * since or abs(psidot_raw) > w_lim \
                    or (up_lim is not None and gained > up_lim):
                self._sane_run = 0
                if self._glitch_run == 0:
                    self._glitch_since = t
                    self.n_glitch_holds += 1
                    self.score.count('glitch', t)
                self._glitch_run += 1
                if self._glitch_run >= self.policy.odom_glitch_trip \
                        and t - self._glitch_since >= self.policy.odom_glitch_hold:
                    if self.odom_ok:
                        self.n_implausible += 1
                        self.score.count('implausible', t)
                    self.odom_ok = False
                if not self.odom_ok:
                    # Once the stream is distrusted, judge it on its own
                    # SELF-consistency: track it, so a stream that settles
                    # can re-earn trust, while one that keeps jumping keeps
                    # failing. (Frozen against the pre-failure speed, a sane
                    # returning stream could never look plausible again.)
                    self.v_prev = v
                    self._v_prev_t = t
                self.now = t
                # One spike: hold the last command rather than jerk to zero.
                # A failed stream: stop, and learn nothing from any of it.
                return self._safe_output() if not self.odom_ok \
                    else self._hold_output()
            self._glitch_run = 0
            if not self.odom_ok:
                self._sane_run += 1
                self.now = t
                self.v_prev = v      # track sane samples so the checks work
                self._v_prev_t = t
                if self._sane_run * self.dt >= self.policy.odom_recover_time:
                    self.odom_ok = True
                    self._sane_run = 0
                else:
                    return self._safe_output()

        self.now = t
        # Smooth the measured sample interval; everything rate-dependent below
        # is derived from this rather than from an assumed rate.
        self.dt = 0.8 * self.dt + 0.2 * dt if self.dt > 0 else dt
        if self.phase_t0 is None:
            self.phase_t0 = t
        elapsed = t - self.phase_t0

        # ---------- PHASE 1: SENSE ----------------------------------------
        if self.phase == SENSE:
            return self._sense(t, v, psidot_raw, dt, elapsed)

        # ---------- virtual sensors ---------------------------------------
        # Fixed bandwidth in SECONDS, so the smoothing is the same at any rate.
        vdot_raw = (v - self.v_prev) / dt
        self.vdot += self.alpha * (vdot_raw - self.vdot)
        self.psidot += self.alpha * (psidot_raw - self.psidot)
        self.v_prev = v
        self._v_prev_t = t
        self.v = v
        self._v_hist.append(v)
        tau = self.policy.v_fb_tau
        if tau > 0.0:
            self.v_fb += (v - self.v_fb) * (1.0 - math.exp(-dt / tau))
        else:
            self.v_fb = v
        # Fast filter for the launch latch only: enough smoothing to keep
        # noise from flickering it, little enough lag to catch breakaway.
        self.v_fast += (v - self.v_fast) * (1.0 - math.exp(-dt / 0.08))
        if not self.rolling:
            # Two consecutive ticks above the threshold, because a single
            # odometry spike through the fast filter (0.71 weight per tick
            # at 10 Hz) latched "rolling" on noise -- which suppressed the
            # launch floor, silenced the stall detector, and left the robot
            # shivering at a standstill it believed was motion.
            # ...and only with a wire on or a command asking: a car does
            # not start rolling by itself. Parked for 20 min on 09-02
            # 17:38 the LiDAR odometry read 0.14-0.23 m/s on 1.6% of ticks
            # (the car moved 6 cm), the latch fired on them, and the
            # throttle fit took 118 "zero wire, half a g" samples: b0
            # 4.67 -> -3.56 before the sanity gate threw it out.
            driven = (abs(cmd_v) > self.stall_cmd_min
                      or (self._cmd_hist and self._cmd_hist[-1][2] != 0.0))
            if abs(self.v_fast) > 0.6 * self.gate_d and driven:
                self._roll_run += 1
            else:
                self._roll_run = 0
            if self._roll_run >= 2:
                self._roll_run = 0
                self.rolling = True
                # For one actuation delay after breakaway, the wire the
                # car is responding to was set BEFORE it moved: pushing
                # harder now cannot reach the wheels until after the
                # overshoot has peaked -- it only deepens it (00:12 run:
                # the wire rose 0.26 -> 0.38 in the 0.4 s after latch and
                # the speed peaked at 2-3x the command 1.3 s in). The cap
                # below holds the wire near the launch level through that
                # window.
                self._post_latch_until = self.now + self.lon_bank.delay
                self._roll_since = self.now
                self._roll_dir = sgn(self.v_fast)
                self._dir_since = self.now
                self._still_since = None
                # The wheels just broke free. Whatever throttle was on the
                # wire lon_delay ago is what did it: a dead-band sample, in
                # the direction the car actually moved. Works in PASSIVE too
                # (the history holds the joystick's commands there).
                d = self._delayed_cmd(self.policy.lon_delay)
                # The stall is over: whatever the wire carried at unstick
                # was a bid against STATIC friction, and carried whole
                # into the rolling regime it was the relaxation oscillator
                # that kept the 08-31 limit cycle going (~0.2 extra wire
                # at breakaway, ~2 s of overspeed to unwind). Once a
                # MEASURED equilibrium wire exists the feedforward carries
                # the cruise and the integrator restarts from zero; before
                # that it restarts from the unstick wire DERATED by
                # deadband_trust -- the same static-to-kinetic derate the
                # dead-band offset uses -- because some feedforward must
                # carry the car until the probe has one (with nothing
                # carrying, every bootstrap launch collapsed back into a
                # stall two ticks after the latch).
                d_dir = sgn(self.v_fast) or 1.0
                if self.gain_probe.eq(d_dir) is not None:
                    self.iv = 0.0
                elif d is not None and sgn(d[1]) == d_dir:
                    cap_iv = self._static_wire_cap(d_dir)
                    self.iv = d_dir * min(
                        self.policy.deadband_trust * abs(d[1]), cap_iv)
                slow = (self._wire_on_since is not None
                        and t - self._wire_on_since
                        > self.policy.deadband_slow_start)
                if d is not None and sgn(d[1]) == sgn(v):
                    if slow:
                        self.deadband.observe(sgn(v), abs(d[1]))
                    else:
                        # a fast start bounds the band from above (see
                        # DeadBand.observe_upper)
                        self.deadband.observe_upper(sgn(v), abs(d[1]))
        else:
            if abs(self.v_fast) < 0.3 * self.gate_d:
                if self._still_since is None:
                    self._still_since = t
                elif t - self._still_since > 0.3:
                    self.rolling = False
            else:
                self._still_since = None
            # A mid-segment stall that never unlatched `rolling` is still
            # a stall: the car came back through the motion gate on a
            # stiction release, and for one settling window that
            # acceleration is energy the wire did not put in. Left
            # "settled", those releases fed the throttle fit as cruise
            # and it became the dead zone -- b0 6.38 with b3/b0 0.216, the
            # breakaway, on the 08-29 16:37 drive, under-driving the car
            # into the next stall (the stepping gait). Any passage back
            # through the gate IN THE SAME DIRECTION re-arms the window,
            # for the learners and for the speed-up gate alike. A passage
            # that comes out the other side is a reversal, which the
            # direction logic above handles and the throttle learner is
            # deliberately NOT re-armed for (a braked reversal is
            # continuous, valid data).
            if abs(self.v_fast) < self.gate_d:
                if not self._below_gate:
                    self._below_gate = sgn(self.v_fast) or self._roll_dir or 1.0
            elif self._below_gate:
                same_way = sgn(self.v_fast) == self._below_gate
                self._below_gate = False
                if same_way:
                    self._roll_since = self.now
            # A direction change is a transient for the LATERAL learner,
            # like breakaway: for one delay plus the servo constant the yaw
            # still belongs to the old direction while the regressors (and
            # the cell, chosen by the sign of v) belong to the new one. A
            # moving reversal never unlatched `rolling`, so that gate never
            # re-armed: 08-28 16:01, the first cusp of the session (0.6 m/s
            # backwards -> forward in 0.6 s) rewrote a0lr 2.07 -> 0.04,
            # a0rr 1.63 -> 3.47, a1 -0.05 -> +1.03 in five ticks, and the
            # planner was quoted 1.75 m for a minute. The longitudinal
            # learner is NOT re-armed: a braked reversal is continuous,
            # valid throttle-vs-acceleration data (no stiction release),
            # and it is where the friction term gets its two-sided samples.
            d = sgn(self.v_fast)
            if d and abs(self.v_fast) > 0.6 * self.gate_d and d != self._roll_dir:
                if self._roll_dir:
                    self._dir_since = self.now
                self._roll_dir = d

        # ---------- command filters: best estimate of actuator state -------
        p = self.policy
        if applied is not None and finite(*applied):
            us_app, ud_app = applied
            self._wire = (float(us_app), float(ud_app))
        else:
            us_app, ud_app = self.prev_us, self.prev_wire
            self._wire = None
        self.qs += (us_app - self.qs) * (1.0 - math.exp(-dt / p.tau_s))
        self.qd += (ud_app - self.qd) * (1.0 - math.exp(-dt / p.tau_d))

        # ---------- the operating speed, and everything scaled by it ------
        self._update_scale(cmd_v, applied is not None if passive is None else passive)

        # ---------- learn, gated by MEASURED noise ------------------------
        learning = self._learn(v)
        if self.steer_sign is None:
            self._establish_sign()

        # ---------- PHASE 2: CAL ------------------------------------------
        if self.phase == CAL:
            us, ud, done = self._cal(t)
            if done:
                self._finish_cal(t)
            return self._publish(us, ud, learning, False, dt)

        # ---------- PHASE 3: RUN ------------------------------------------
        if self.drive_fault:
            return self._safe_output()
        us, ud = self._run(self.v_fb, cmd_v, cmd_w, dt)
        us, ud, stalled = self._reflex(us, ud, self.v_fb, cmd_v, t, dt)
        return self._publish(us, ud, learning, stalled, dt)

    # -- phases ------------------------------------------------------------

    def _sense(self, t, v, psidot_raw, dt, elapsed):
        """Measure the sensor by listening at rest."""
        p = self.policy
        floor = max(0.05, 6.0 * (self._running_sigma() or 0.02))
        if abs(v) > floor:
            # Something moved. The measurement is void; start over.
            self._v_samples.clear()
            self._psi_diffs.clear()
            self.phase_t0 = t
            return self._safe_output()

        self._v_samples.append(v)
        self._psi_diffs.append(psidot_raw * dt)

        if elapsed > p.t_sense and len(self._v_samples) >= 5:
            self.sigma_v = _stddev(self._v_samples)
            nonzero = [abs(s) for s in self._v_samples if s != 0.0]
            self.tick = min(nonzero) if nonzero else 0.0
            self.sigma_psi = _stddev(self._psi_diffs) / math.sqrt(2.0)

            # Everything downstream is derived from the measurement.
            self._update_gates()
            self.alpha = clamp(self.dt / p.sensor_tau, 0.05, 0.6)
            self.lam = math.exp(math.log(0.5) * self.dt / p.t_forget)
            self.v_prev = v
            self._v_prev_t = t
            if p.enable_calibration:
                self.start_cal()
            else:
                self._enter(RUN, t)
        return self._safe_output()

    def _update_scale(self, cmd_v, passive):
        """Track the operating speed and re-derive what depends on it.

        The scale is what the vehicle is being ASKED to do -- Nav2's
        commands, when this controller drives -- with the held speed as
        the fallback ONLY where there is no command to read: PASSIVE (the
        joystick's commands are not speeds) and CAL. Never while this
        controller drives: between two Nav2 segments the car is coasting
        into a cusp, which is exactly where this LiDAR odometry reads
        0.5-0.65 m/s on a sub-dead-band wire (scan matching through the
        rotation), and the held median of that set v_op 0.32 -> 0.53 and
        the planner's radius with it (16:13, 08-29 drive). Forgetting
        runs only while driving, see reset().
        """
        # The command IS the regime when there is one; the held speed
        # stands in only when nobody is commanding through this controller
        # (PASSIVE, CAL, coasting) -- an overshoot is not a faster regime.
        # And the held speed counts only while the rolling latch says the
        # car is moving: at rest the median of the odometry noise is not
        # zero, and a scale of 7 mm/s would put the gates under the noise.
        if cmd_v != 0.0:
            seen = abs(cmd_v)
        elif (passive or self.phase == CAL) and self.rolling:
            seen = abs(self._held_speed())
        else:
            seen = 0.0
        if seen > 0.0 or self.rolling:
            self.v_op = max(seen, self.v_op * self.lam)
        self.envelope.v_op = self.v_op
        self._update_gates()

    def _update_gates(self):
        """Motion gates from the measured noise, bounded by the scale.

        gate_d: below this the speed is noise, not motion. gate_s: the
        lateral learner's stricter gate -- psidot/v amplifies noise.
        Ten sigma of the SENSE measurement, clamped into the floor..cap
        band of v_op once there is a v_op (Policy.gate_floor_frac and
        gate_cap_frac say why both bounds exist).
        """
        p = self.policy
        noise = max(self.sigma_v, self.tick)
        gate = 10.0 * noise
        if self.v_op > 0.0:
            gate = clamp(gate, p.gate_floor_frac * self.v_op,
                         p.gate_cap_frac * self.v_op)
        # ...but the cap may not put the gate INSIDE the noise: a sample
        # under five sigma is not certainly motion, whatever the regime (a
        # 0.02 m/s crawl on a sensor with 0.007 m/s of jitter is not
        # something to learn from). On this robot 5 sigma is 0.08-0.14,
        # under the 0.15 cap on every boot measured so far.
        self.gate_d = max(gate, 5.0 * noise)
        self.gate_s = 1.6 * self.gate_d

    def _held_speed(self):
        """The speed the car has HELD: the median of the raw speed over one
        settling window (learned delay + actuator constant, the same span
        the learner already demands before it believes a sample). The
        speed span exists to say what range the fit was identified over;
        single ticks of pose-differenced odometry are not that range.
        2026-08-29: the persisted span read -1.14..1.15 m/s on a car
        commanded at most 0.32 -- every value past 0.5 was an ICP jump
        (10% of commanded ticks sat 0.2 m/s above the command), so the
        "evaluate inside the fitted range" clamp on the feedforward and
        the identifiability test in ready_lon were both inert."""
        n = max(3, int(round((self.lon_bank.delay + self.policy.tau_d)
                             / max(self.dt, 1e-3))))
        recent = list(self._v_hist)[-n:]
        if not recent:
            return self.v
        return sorted(recent)[len(recent) // 2]

    def _delayed_cmd(self, delay):
        """The (us, ud) that was on the wire `delay` seconds ago."""
        target = self.now - delay
        best = None
        for stamp, us, ud in reversed(self._cmd_hist):
            if stamp <= target:
                best = (us, ud)
                break
        return best

    def _learn(self, v):
        """Feed both learners, but only from samples that carry information."""
        ok = False
        # Longitudinal. Drag is b2*v*|v| (opposes motion, so it changes sign
        # in reverse); b3*sgn(v) is Coulomb friction, which is what used to
        # leak into b0/b1 and drive the fit to a negative gain. Every delay
        # candidate builds its regressor from its own delayed command.
        # Moving samples only: a stuck robot teaches "high throttle, zero
        # acceleration", which is stiction, not gain.
        #
        # The acceleration gate is the same physics bound as the odometry
        # plausibility check: full throttle plus friction, times the risk
        # margin. It used to be a fixed 12 m/s^2, which no wire command can
        # produce -- every ICP jump that slipped past the per-tick glitch
        # gate sailed under it and taught the fit (the 08-23 log has 34
        # such ticks and a b2 of +1.07: drag that ACCELERATES the car).
        a_gate = self._a_limit()
        # `rolling` too, not just instantaneous |v|: a single odometry spike
        # at standstill passes the |v| gate for one tick, and those ticks
        # are what dragged a0l from 1.66 to 0.96 in the 02:04 stuck-shiver
        # (steering was held left the whole time, so every garbage sample
        # landed on one side). The latch demands SUSTAINED motion.
        #
        # And SETTLED motion, not merely sustained: for one learned delay
        # plus the actuator constant after breakaway, the acceleration is
        # stiction release -- energy the current throttle did not put in.
        # Those ticks pair "wire ~0.2, vdot ~1.8" against cruise's "wire
        # ~0.2, vdot ~0", a contradiction the linear fit resolves as a
        # steep line pivoted at the dead-zone edge: the 02:15 session
        # refit b0 1.95 -> 5.7 inside ONE MINUTE that way (b3/b0 landed
        # exactly on the measured breakaway, 0.22 -- the fit had become
        # the dead zone, not the vehicle). Same pattern as the iw freeze.
        settled = (self.rolling and self._roll_since is not None
                   and self.now - self._roll_since
                   >= self.lon_bank.delay + self.policy.tau_d)
        # The speed in the regressor: the raw sample only when it is a
        # TRANSIENT (further from the held median than three sigma of the
        # SENSE noise), the held median otherwise. At cruise the raw
        # sample's deviation from the median IS the odometry noise, and a
        # noisy regressor biases its own coefficient toward zero
        # (errors-in-variables): b2 -> 0 and, b0 and b2 being collinear at
        # one cruise, b0 with it. Measured on the bench (08-29): five
        # minutes at cruise walked b0 3.18 -> 2.16 with the noise on and
        # not at all with it off, forgetting or no forgetting; the median
        # alone held it (1.75 -> 1.74) but its lag in the speed steps cost
        # the transient fit (3.18 -> 1.75). Each regime gets the estimate
        # that is unbiased in it.
        held = self._held_speed()
        v_reg = v if abs(v - held) > 3.0 * self.sigma_v else held
        in_regime = (self.v_op <= 0.0
                     or abs(v) <= self.policy.learn_overspeed_ratio * self.v_op)
        # ---- gain probe: direct measurement, no regressor -----------------
        # Deliberately NOT behind in_regime: the overspeed exclusion
        # protects the RLS from teaching surges as cruise, but the surges
        # are exactly the honest transients that reveal the true wire gain
        # -- excluding them everywhere is how a junk fit and the limit
        # cycle it caused sustained each other for six minutes (08-31
        # 01:08). The physics bound (a_gate) still applies: an ICP jump is
        # not a transient. `settled` too: a stiction release is energy the
        # wire did not put in, for the probe as for the fit.
        if settled and abs(v) > self.gate_d and abs(self.vdot) < a_gate \
                and not (self._stall_since is not None or self.blocked):
            d = self._delayed_cmd(self.lon_bank.delay)
            if d is not None:
                # "Holding a speed" judged against the measurement noise
                # itself: the raw vdot difference carries sqrt(2)*sigma_v
                # per dt, the EMA keeps alpha/(2-alpha) of that variance,
                # and three sigma is the usual glitch line. And SUSTAINED
                # for the actuator's own constant: the reconstructed
                # state q only means "this wire holds this speed" if the
                # wire has actually been holding for tau_d, while a
                # lunge's vdot crossing zero at its peak (wire already
                # cut) spends less than that inside the noise band --
                # those single ticks taught eq ~0.06 on a plant whose
                # true cruise wire is 0.21, and a full plant memory
                # (delay + tau_d) rejected every honest quasi-plateau the
                # bench run had.
                sig_a = (math.sqrt(2.0 * self.alpha / (2.0 - self.alpha))
                         * self.sigma_v / max(self.dt, 1e-3))
                if abs(self.vdot) > 3.0 * sig_a:
                    self._calm_since = None
                elif self._calm_since is None:
                    self._calm_since = self.now
                at_eq = (self._calm_since is not None
                         and self.now - self._calm_since
                         >= self.policy.tau_d)
                # The smallest actuator step a slope may be taken over:
                # the plant's response to it must clear the acceleration
                # noise (three sigma, on the difference of two samples),
                # at the current gain scale. Any smaller step is a step
                # the LOOP made in reaction to a noise wobble in the
                # speed, and the pair is then correlated by the
                # controller, not the plant -- closed-loop bias, which
                # read this car's b0 as 9.9 against 4.6 on the 09-01
                # 21:00 drive (steps >= 3 sigma: 4.7). On a quiet sensor
                # the floor is half the launch floor, the smallest wire
                # that provably moves a car.
                # the SMALLEST available gain scale, never the probe's own
                # reading alone: a probe biased high shrank its own step
                # and kept admitting the noise that biased it (09-01
                # 21:11, 10.8 with the fix live)
                b_ref = min(x for x in (self.gain_probe.b0, self._b0_ref,
                                        2.0 * self.policy.prior_b0) if x)
                step_min = max(0.5 * self.policy.launch_floor,
                               3.0 * math.sqrt(2.0) * sig_a / b_ref)
                self.gain_probe.observe(self.now, sgn(v), d[1], self.vdot,
                                        self.dt, at_eq, step_min)
        # A STUCK car teaches nothing: wire on, no motion is the stall
        # detector's territory (a wall, a low obstacle the LiDAR cannot
        # see, a wheel against a box), and with `rolling` kept latched by
        # the jerks of pushing, those "0.6 wire, zero acceleration" ticks
        # were settled, in-regime samples -- b0 3.1 and a throttle delay
        # of 1.52 s learned in the 09-02 22:50 stuck episode (the
        # regressor reaching 1.5 s back into the push).
        stuck = self._stall_since is not None or self.blocked
        if settled and in_regime and not stuck and abs(self.vdot) < a_gate \
                and abs(v) > self.gate_d:
            def phi_lon(delay):
                # A candidate whose regressor reaches back BEFORE the
                # wheels turned pairs this acceleration with the launch
                # ramp -- wire that produced no motion (stiction), not
                # gain. On a stop-go drive every launch then correlates
                # best at "delay = time since the ramp", and the bank
                # switched 0.30 -> 1.01 s fourteen seconds into the 09-02
                # 01:18 drive (kp halved, the fit inverted to b0 -2.4).
                # The settling gate excludes the SAMPLE times; this
                # excludes the candidate's LOOKBACK.
                if self._roll_since is not None \
                        and self.now - delay < self._roll_since:
                    return None
                d = self._delayed_cmd(delay)
                if d is None:
                    return None
                # A nonzero wire inside the dead zone produces no torque:
                # "throttle but no acceleration" samples are what once
                # taught b0 several times too low. The threshold is the
                # DERATED breakaway (trust * median): the raw median is the
                # static breakaway, which sits above the dead-zone offset by
                # the stiction, and skipping up to it starved the learner of
                # its legitimate cruise samples. Zero-wire coasting stays:
                # it identifies drag and Coulomb. A car with no dead-band
                # evidence is unaffected.
                if d[1] != 0.0:
                    w = self.deadband._win(sgn(d[1]))
                    thr = (self.policy.deadband_trust * w.value
                           if w.vals else 0.0)
                    if 0.0 < abs(d[1]) < thr:
                        return None
                return [d[1], 1.0, v_reg * abs(v_reg), sgn(v)]
            if self.lon_bank.update(phi_lon, self.vdot, self.lam, self.dt):
                ok = True
                if self._lon_corr is None:
                    self._lon_corr = [0.0] * len(self.lon_bank.delays)
                for k, dk in enumerate(self.lon_bank.delays):
                    if self._roll_since is not None \
                            and self.now - dk < self._roll_since:
                        continue            # same lookback rule as phi_lon
                    dd = self._delayed_cmd(dk)
                    if dd is not None:
                        self._lon_corr[k] += dd[1] * self.vdot
                self.qd_lo, self.qd_hi = _span(self.qd_lo, self.qd_hi,
                                               self.qd)
                self.vl_lo, self.vl_hi = _span(self.vl_lo, self.vl_hi,
                                               self._held_speed())
        # Lateral. psidot/v is curvature; dividing by v amplifies noise, which
        # is why the speed gate is 1.6x the longitudinal one. There is
        # deliberately no lower gate on psidot: straight-line samples are what
        # identify the trim term a1, so excluding them would lose the
        # misalignment the trim exists to cancel.
        # Settled, like the longitudinal learner: right after breakaway the
        # yaw response belongs to commands from before the stop, and the
        # 2 s legs of a cusp shuffle are mostly that transient -- which is
        # how fwd-right was taught -0.09 on 08-23 while the robot shuffled
        # with the steering hard over.
        settled_lat = (self.rolling and self._roll_since is not None
                       and self.now - max(self._roll_since,
                                          self._dir_since or self._roll_since)
                       >= self.lat_bank.delay + self.policy.tau_s)
        if settled_lat and in_regime and abs(v) > self.gate_s \
                and abs(self.psidot) < 4.0:
            # Same instrument for the lateral fit: the a2 term's v^2 and
            # the division that makes curvature of the yaw rate both use
            # the held speed (the raw sample's noise would otherwise sit
            # in both the regressor and the target).
            v_k = v_reg if abs(v_reg) > self.gate_s and sgn(v_reg) == sgn(v) \
                else v
            kappa = self.psidot / v_k
            def phi_lat(delay):
                # NO lookback guard here (unlike phi_lon): the lateral
                # transient gate re-arms at every heading reversal, and a
                # guard keyed to it starved the long candidates on the
                # 1 s-delay bench vehicle (peak 1.52 -> 0.68 s). The
                # steering command has no stiction ramp to mis-pair with.
                d = self._delayed_cmd(delay)
                if d is None:
                    return None
                # Split regressor: each (travel direction x steering side)
                # cell sees only its own samples, so neither the linkage
                # asymmetry nor the reverse-dynamics difference can leak
                # into the trim. v is the SIGNED sample speed (|v| > gate_s
                # here, so its sign is meaningful).
                qsp, qsn = max(d[0], 0.0), min(d[0], 0.0)
                if v >= 0.0:
                    return [qsp, qsn, 0.0, 0.0, 1.0, d[0] * v_k * v_k]
                return [0.0, 0.0, qsp, qsn, 1.0, d[0] * v_k * v_k]
            accepted_lat = self.lat_bank.update(phi_lat, kappa, self.lam,
                                                self.dt)
            if accepted_lat:
                ok = True
                self.qs_lo, self.qs_hi = _span(self.qs_lo, self.qs_hi,
                                               self.qs)
            # Same sample, second use: how well the model predicted a
            # near-lock command is the only honest source for the envelope.
            # Same regressor (the ACTIVE delay's command) as the model, or
            # the ratio compares a command the car has not responded to yet
            # against the curvature it is still doing from an earlier one.
            # Third use: the steering-sign witness (Policy.sign_evidence),
            # accumulated at EVERY candidate delay (see reset()).
            d = self._delayed_cmd(self.lat_bank.delay)
            if d is not None:
                self.envelope.observe(d[0], kappa, v, self.model)
            if accepted_lat:
                if self._sign_num is None:
                    self._sign_num = [0.0] * len(self.lat_bank.delays)
                    self._sign_den = [0.0] * len(self.lat_bank.delays)
                counted = False
                for k, dk in enumerate(self.lat_bank.delays):
                    dd = self._delayed_cmd(dk)
                    if dd is None or abs(dd[0]) < 0.5 * self.policy.ready_lat_qs_span:
                        continue
                    w = dd[0] * kappa
                    self._sign_num[k] += w
                    self._sign_den[k] += abs(w)
                    counted = True
                if counted:
                    self._sign_n += 1
        return ok

    def _cal(self, t):
        """Staged wiggle; returns (us, ud, done). Time-based within a stage
        so it is identical at any odometry rate; stages exit on evidence,
        with t_cal as the timeout.

        signs    throttle ramps 0.10 -> 0.60 over 2 s, then holds cal_drive
                 under a cal_steer wiggle until the steering-sign witness
                 locks. On exit the held speed says which way the car
                 answered a positive wire: backwards or not at all is a
                 throttle fault and CAL ends here (RUN, actuators zero).
        gains    the same wiggle at twice the amplitude (at most full
                 lock), until the lateral fit is ready to be inverted.
        reverse  brake to a stop, one cal_reverse-second pulse of
                 -cal_drive straight back, brake again.
        """
        p = self.policy
        if self._cal_t0 is None:
            self._cal_t0 = t
        tc = t - self._cal_t0
        stage = self._cal_stage

        def next_stage(name):
            self._cal_stage, self._cal_t0, self._cal_sub = name, t, None

        def wiggle(amp):
            return (amp * math.sin(2.0 * math.pi * 0.55 * tc)
                    + 0.16 * math.sin(2.0 * math.pi * 0.19 * tc))

        if stage == 'signs':
            if tc < 2.0:
                return 0.0, 0.10 + 0.25 * tc, False
            # the witness needs wiggle samples: at least one settling
            # window past the ramp before its verdict can count
            settled = tc >= 2.0 + self.lat_bank.delay + p.tau_s
            if (settled and self.steer_sign is not None) or tc > p.t_cal:
                v = self._held_speed()
                if abs(v) <= self.gate_d:
                    self.drive_fault = 'no motion from +%.2f wire' % p.cal_drive
                elif v < 0.0:
                    self.drive_fault = ('inverted: +%.2f wire drove %.2f m/s'
                                        % (p.cal_drive, v))
                if self.drive_fault:
                    return 0.0, 0.0, True
                next_stage('gains')
                return wiggle(p.cal_steer), p.cal_drive, False
            return wiggle(p.cal_steer), p.cal_drive, False

        if stage == 'gains':
            if self.ready_lat or tc > p.t_cal:
                next_stage('reverse')
                self._cal_sub = 'brake'
                return 0.0, 0.0, False
            return wiggle(min(1.0, 2.0 * p.cal_steer)), p.cal_drive, False

        # stage == 'reverse'
        if self._cal_sub == 'brake':
            if not self.rolling or tc > p.t_cal:
                self._cal_sub, self._cal_t0 = 'pulse', t
            return 0.0, 0.0, False
        if self._cal_sub == 'pulse':
            if tc > p.cal_reverse:
                self._cal_sub, self._cal_t0 = 'stop', t
                return 0.0, 0.0, False
            return 0.0, -p.cal_drive, False
        # 'stop': done once the car has come to rest (or the timeout)
        return 0.0, 0.0, (not self.rolling) or tc > p.t_cal

    def _finish_cal(self, t):
        """Size the dither from measurements, seen THROUGH the learned
        gain, and enter RUN. The signs were settled by the stages.

        The delay banks jump to their cross-correlation peak here -- the
        model-free witness sums (witness_delay), not the fitted gains: the
        wiggle is a DESIGNED excitation (0.55 Hz, a quarter turn of phase
        per 0.45 s of delay) that separates the candidates as ordinary
        driving does not, and it is free of the rotation artefacts that
        made a Nav2-driven switch untrustworthy (08-29 drive). In RUN the
        banks only follow a sustained lead (DelayBank.update).
        """
        self._establish_sign()
        for axis, bank in (('lat', self.lat_bank), ('lon', self.lon_bank)):
            d = self.witness_delay(axis)
            if d is not None:
                bank.set_delay(d)
                bank._lead = (None, 0)
            else:
                bank.jump_to_peak()
        self._cal_stage = self._cal_sub = None
        b0 = self.rls_lon.theta[0]
        self.dither = clamp(
            5.0 * max(self.tick, self.sigma_v) * 4.4 / max(abs(b0), 1.0),
            0.03, 0.12)
        self._enter(RUN, t)

    def _lon_divisor(self, b0, lon_ok):
        """What one m/s^2 of PI demand costs in wire: demand / this.

        Ranked by how directly the number was measured:

        1. The last fitted b0 that passed every sanity gate (persisted or
           restored) -- vetted.
        2. The gain probe's live wire-to-acceleration slope -- a direct
           measurement, but on this sensor it has twice read 2-3x high
           while its window filled (closed-loop bias, 09-01/02).
        3. Nothing measured yet: TWICE the declared prior, the top of the
           same factor-two band a measured anchor is granted. kp is
           derived to put the loop at half its delay margin when this
           divisor equals the true gain; a divisor N times too SMALL
           multiplies the loop gain N times (the 08-31 01:08 from-zero
           limit cycle: divisor floored at 1.0, plant measuring ~5),
           while one too large merely answers slowly and is corrected by
           the probe within its first steady window. Under total
           ignorance, err on strong.

        A plausible live fit may then scale the anchor by the usual
        factor-two trust band; an implausible one gets no vote (clamping
        junk INTO the band still dragged the divisor to its floor).
        """
        p = self.policy
        # The sanity-gated fit outranks the probe once it exists: on this
        # sensor the probe has read 9.9 and 11.2 against a trace of ~4.6
        # at the start of two drives (closed-loop bias while its window
        # fills), while the vetted fit sat at 4.3-4.7 -- and a divisor
        # twice too large merely answers slowly, but the point of the
        # anchor is to be the number that has passed every gate.
        ref = self._b0_ref or self.gain_probe.b0
        if not ref:
            return 2.0 * p.prior_b0
        if lon_ok:
            return clamp(b0, 0.5 * ref, 2.0 * ref)
        return ref

    def _static_wire_cap(self, s_dir):
        """The wire that HOLDS a speed cannot exceed the wire that
        measurably breaks the car free -- whatever a fit or a probe says.
        Before any dead-band evidence: launch floor plus margin."""
        raw_band = self.deadband.raw(s_dir) if s_dir else None
        return raw_band if raw_band is not None \
            else self.policy.launch_floor + self.policy.launch_cap_margin

    def _run(self, v, cmd_v, cmd_w, dt):
        """Invert the learned model and add integral trim."""
        p = self.policy
        a0l, a0r, a0l_rev, a0r_rev, a1, a2 = self.rls_lat.theta
        b0, b1, b2, b3 = self.rls_lon.theta

        # --- steering -----------------------------------------------------
        # sgn(v) is zero at standstill, which would divide by zero; fall back
        # to the commanded direction, then to forward.
        # The floor is a fraction of the operating speed; the 1 mm/s under
        # it only guards the division for a car that has never moved.
        v_floor = max(self.v_eff_floor, 1e-3)
        if abs(cmd_v) > v_floor:
            v_eff = cmd_v
        else:
            direction = sgn(v) or sgn(cmd_v) or 1.0
            v_eff = direction * max(abs(v), v_floor)

        # Ask for no more curvature than the vehicle is believed to have.
        # Without this the inversion winds the trim integrator up against a
        # saturated steering command it can never satisfy.
        #
        # Evaluated at v_eff, not the raw sample speed, here and in the
        # speed term of the inversion below. Pose-differenced odometry
        # spikes under motion: the 2026-08-29 12:04 run had |v| above the
        # command by 0.2 m/s on 10% of commanded ticks (single-tick peaks to
        # 1.15 m/s against 0.32 commanded; MOLA at rest is quiet, p99 1.5 cm)
        # and with a2 at -3.0 a 0.7 m/s tick took the span from 2.1 to 0.9
        # -- a 1.4x median, 2.1x p90 steering-gain kick on those ticks, and
        # the same tick shrank kappa_max under the demand. The commanded
        # speed is what the car is about to do; the learner keeps the raw
        # speed (filtering a regressor biases RLS).
        kappa_max = self.envelope.max_curvature(self.model, v_eff,
                                                derate=False)
        kappa_des = clamp(cmd_w / v_eff, -kappa_max, kappa_max)
        # The trim compares the yaw the car shows NOW with the curvature
        # that was commanded one learned delay plus the servo constant
        # AGO -- the command that caused it. That replaces the settling
        # freeze (integrate nothing for a delay after a "meaningful" step):
        # the follower's command jitters by more than the freeze
        # threshold on 16% of ticks, which held the trim frozen 41% of
        # the 09-01 21:00 drive, and the rest of the time it integrated
        # an error that was transport delay.
        L_lat = self.lat_bank.delay + p.tau_s
        # the trim cell: travel direction by v_eff, steering side by the
        # commanded curvature (straight keeps the last side)
        self._iw_key = (sgn(v_eff) or 1.0,
                        sgn(kappa_des) or self._iw_key[1])
        self._kappa_hist.append((self.now, kappa_des, self._iw_key))
        while self._kappa_hist \
                and self.now - self._kappa_hist[0][0] > 2.0 * L_lat + 1.0:
            self._kappa_hist.popleft()
        kappa_delayed = None
        for stamp, k, key in reversed(self._kappa_hist):
            if stamp <= self.now - L_lat:
                # ...and from the SAME cell: for one delay after a moving
                # reversal the yaw still answers the previous leg's
                # command, which would be integrated into the new cell
                if key == self._iw_key:
                    kappa_delayed = k
                break
        # ...and it is a CURVATURE error, psidot/v against kappa, not a
        # yaw-rate error. The car ran 1.24x its commanded speed (median)
        # while turning on 09-01: at perfect curvature that is 24% more
        # yaw than commanded, and a yaw-rate trim wound the steering OUT
        # on every overspeed launch -- the car ran wide on the very turns
        # the planner drew at its minimum radius. Speed error is the
        # throttle loop's problem; the steering's is curvature. Signed v,
        # so the correction does not invert in reverse; weighted by
        # |v| / v_op below the operating speed so the gain never exceeds
        # ki_w (as the yaw-rate form's v_op floor did).
        # Anti-windup. Integrating a yaw error the servo cannot act on (at
        # lock, or stationary) only winds the trim to its clamp; the flight
        # log had it pinned 90% of the time, biasing every turn one way and
        # unwinding slowly when the demand flipped -- "all left, all right".
        saturated = abs(self.prev_us) >= 0.95
        if not saturated and kappa_delayed is not None \
                and abs(v) > 0.5 * self.gate_d:
            kappa_meas = self.psidot / (sgn(v_eff) * abs(v))
            weight = min(1.0, abs(v) / max(self.v_op, 1e-3))
            self.iw = clamp(
                self.iw + self.ki_w * weight * (kappa_delayed - kappa_meas)
                * dt, -p.iw_max, p.iw_max)
        # Physically impossible coefficients are clamped HERE rather than in
        # the estimator: projecting every RLS update biases the fit, but a
        # wrong-signed coefficient must never reach an actuator.
        #
        # Longitudinal is different: b1/b2/b3 are nearly collinear at any
        # one cruise speed, so their individual signs are partly arbitrary
        # while their SUM is pinned by the data. Sign-clamping one of them
        # (the old b2/b3 clamps here) made the inversion inconsistent with
        # its own fit and shifted the equilibrium: on the robot a fitted
        # b2 of +0.5..+2.5 was being zeroed and cruise ran 15-37% over the
        # commanded speed. The throttle inversion now uses the fitted sum
        # verbatim, evaluated only inside the speed range it was fitted on.
        a2 = min(a2, 0.0)     # cornering does not improve with speed
        b0 = max(b0, 0.0)     # more throttle cannot mean less acceleration

        # Until the data has genuinely covered a range of steering commands,
        # the learned split of a0/a1/a2 is arbitrary -- steer through the
        # prior gain instead of inverting it. The learner keeps running the
        # whole time; only the INVERSION waits for evidence.
        if not self.ready_lat:
            a0l = a0r = a0l_rev = a0r_rev = self.prior_a0
            a1 = a2 = 0.0
        else:
            # A minority-sign cell is a poisoned fit, not a vehicle (see
            # sane_gain_cells): steer that quadrant through the (signed)
            # prior.
            a0l, a0r, a0l_rev, a0r_rev = sane_gain_cells(
                (a0l, a0r, a0l_rev, a0r_rev), self.prior_a0)

        # The net curvature the wire must produce picks the steering side,
        # and the direction the car is about to travel (v_eff's sign, which
        # already resolved the standstill case above) picks forward or
        # reverse gains: the same wire steers measurably differently
        # backing up (reverse-left 0.73x forward, 08-23 logs).
        knet = kappa_des + self.iw - a1
        if v_eff >= 0.0:
            a0_dir = a0l if knet >= 0.0 else a0r
        else:
            a0_dir = a0l_rev if knet >= 0.0 else a0r_rev
        span = a0_dir + a2 * v_eff * v_eff
        # Collapsed means the EFFECTIVE gain at this speed, not a0 alone:
        # a dead servo's fit lands wherever the collinear a0/a2 split
        # happens to sit (a0l 0.06 with a2 -0.24 cancelling it at 0.5 m/s
        # on the bench, 08-29) -- the same zero, a different address.
        if abs(a0_dir) > p.den_min and abs(span) > p.den_min:
            self.steering_fault = False
            # Understeer may reduce the gain but not erase it, and certainly
            # not invert it -- an inverted denominator steers the wrong way.
            floor = p.span_floor * abs(a0_dir)
            den = span if abs(span) >= floor and span * a0_dir > 0.0 \
                else sgn(a0_dir) * floor
        else:
            # The identified steering gain has collapsed. On a real vehicle
            # that is far more likely a stuck servo, a lost linkage or a dead
            # actuator supply than a genuine property -- and answering with
            # zero steering makes it unrecoverable, because zero steering
            # produces no yaw, which is exactly the evidence that taught the
            # gain to collapse. That is a one-way trap.
            #
            # Fall back to the prior so the robot keeps steering: if the
            # actuator is alive the resulting yaw re-teaches the real gain,
            # and if it is dead nothing is lost.
            den = self.prior_a0 if abs(self.prior_a0) > p.den_min else 1.0
            if not self.steering_fault:
                self.get_fault_reset()
            self.steering_fault = True
        us = clamp(knet / den, -1.0, 1.0)
        # Full steering authority at standstill -- and the THROTTLE waits
        # for it (below). A clamp to 0.45 of lock until rolling used to sit
        # here, from the days the integrator was the stiction prober and a
        # launch at lock lunged (released at 0.45-0.70 wire against 0.24
        # straight); the floor ride replaced that prober. What the clamp
        # cost, 09-02 23:40: every cusp leg left its stop with the wheels
        # 0.6 of the way to the planned arc and the servo still turning
        # through the wire delay, so the car ran ~0.3 m nearly straight
        # (kappa 0.35 measured against 1.5 commanded at 23:43:24), pure
        # pursuit asked 2.9 to recover, the car ran wide into the wall's
        # lethal band and RPP's collision veto ("detected collision
        # ahead", 142 times) failed 15 of the session's legs.

        # --- throttle -----------------------------------------------------
        # Deliberately minimal. Everything that used to sit here -- a
        # measured "breakaway", a kinetic feedforward seeded from it, a
        # launch kick, a standstill cap, an anti-stall floor, a rolling
        # latch -- was built on a breakaway measurement that is biased by
        # construction (the probe ramps faster than motion can be detected,
        # so it always records true breakaway plus ramp x latency, ~0.25),
        # and each layer then over-pushed by that bias: lunge, brake-slam,
        # stall, re-measure higher. The flight logs show the plain PI
        # cruising smoothly whenever that machinery was dormant, and the
        # violent stepping whenever it engaged.
        #
        # Stiction is handled by the integrator: it ramps slowly and
        # bias-free until the wheels turn, and then simply keeps what it
        # needed. That is all a breakaway probe ever should have been.
        # A direction reversal starts the integrator from zero. It is a
        # stiction ramp, and the stiction in the new direction owes nothing
        # to what the old one needed; left alone it held the robot driving
        # the WRONG way for up to 4.8 s after a cusp (16 of 71 reversals in
        # the flight log took more than 2 s to change sign), because the
        # smoother's ramp through zero rarely lands on exactly 0.0.
        direction = sgn(cmd_v) if abs(cmd_v) > self.stall_cmd_min else 0.0
        if direction and self._cmd_dir and direction != self._cmd_dir:
            self.iv = 0.0
        if direction:
            self._cmd_dir = direction

        err = cmd_v - v
        # The feedback gain b0 is used PRIOR-RELATIVE: bounded to half..twice
        # the declared full-throttle acceleration, a risk constant on how
        # far the learned value may scale the loop. A drifted fit dilutes
        # or triples it -- the 02:15 poisoned b0 of 6.4 cut kp's authority
        # 3x and the overspeed ran away uncorrected, while the settling
        # window's under-fit (0.7 in sim) would triple it. The same bounded
        # number is what full throttle can deliver, so the acceleration
        # demand is clamped to it (it was a typed 2.5 m/s^2).
        # The loop-gain divisor is anchored to measurement, never to the
        # junk a cold fit can be: with prior_b0 typed as 2.0 on a car
        # whose measured b0 is ~4.6, a collapsed live fit floored b0_fb
        # at 1.0 and every PI correction ran 4.6x too strong -- the wire
        # slammed 0.44/0.06 at 1 Hz and the car surged 0..0.8 m/s on a
        # 0.32 command (08-30 23:16); the same run repeated from-zero on
        # 08-31 01:08, where no anchor existed at all. See _lon_divisor.
        lon_ok = (self.ready_lon and p.use_learned_lon
                  and self.lon_plausible())
        b0_fb = self._lon_divisor(b0, lon_ok)
        a_des = clamp(self.kp_v * err, -b0_fb, b0_fb)
        # Conditional integration (see _shaped in reset): no winding into
        # a limit the actuator is already at. During a stall the winding
        # is deliberate -- it is what escalates the wire from the
        # bootstrap floor up to a breakaway the dead band has not
        # measured yet -- but everything wound while stuck is a bid
        # against stiction, and it is DISCARDED at the rolling latch
        # (see step): carried into the rolling regime it was the
        # relaxation oscillator that kept the 08-31 limit cycle going
        # even with the probe's correct gain and equilibrium wire (~0.2
        # extra wire at breakaway, ~2 s of overspeed to unwind).
        # The integral path goes through the SAME gain normalisation as
        # the proportional one. It used to add wire directly, which
        # matched a_des/b0_fb only while the divisor sat at ~1: with a
        # measured divisor of ~4 the integrator was 2.5x the proportional
        # term at crossover, an extra -90 degrees of phase where the
        # design ratio ki = kp/(2.8 L) assumed proportionality -- the
        # 4-6 s hunt the bench showed even after the feedforward and the
        # divisor were both measured and right.
        if not (self._shaped and sgn(err) == self._shaped):
            self.iv = clamp(self.iv + self.ki_v * err * dt / b0_fb,
                            -p.iv_max, p.iv_max)
        # A MOVING reversal -- the command points against the car's travel
        # while it is still measurably rolling -- is braking, not a launch.
        # Everything sized for a start from rest is wrong here: the new
        # direction's Coulomb feedforward is a push, and an integrator
        # winding on err ~ -0.6 during the brake adds to it; with the
        # ~0.45 s delay the wire was still 0.29 three ticks after the car
        # had crossed zero, so the new leg peaked at 0.55 m/s median,
        # 0.90 max, on a 0.30 command (08-28 22:46, 45 such reversals).
        # Brake with the proportional term only, integrator held at zero;
        # below gate_d the launch-from-rest logic takes over as usual.
        reversing = bool(direction) and v * direction < -self.gate_d
        if reversing:
            self.iv = 0.0
        # Everything below is in WIRE units -- what the actuator actually
        # receives. The learned model is fit on the wire, so its inversion
        # yields wire directly; the dead-band map only linearizes the
        # BOOTSTRAP prior path. (Two bugs lived here: compensating the
        # model's output again drove cruise 37% over the commanded speed,
        # and flooring every output at the static breakaway stick-slipped
        # any vehicle whose cruise level sits below its breakaway.)
        if not lon_ok:
            # Not yet earned, or not a vehicle (lon_sane: collapsed or
            # wrong-signed gain, or a fit that accelerates with no
            # throttle): inverting a fit that is not physically a car is
            # worse than not having learned yet (08-28 22:19: -0.44 wire
            # of "feedforward" for a +0.25 cruise, the car never left the
            # launch floor).
            s_dir = sgn(v) if abs(v) > self.gate_d \
                else (sgn(cmd_v) or sgn(v))
            eq = self.gain_probe.eq(s_dir) if s_dir else None
            if eq is not None:
                # MEASURED bootstrap: the probe's equilibrium wire -- the
                # wire observed to HOLD a speed -- is the feedforward
                # itself, applied toward the commanded direction only
                # while actually traveling that way (the brake-only-
                # reversal principle, as everywhere). This replaces the
                # dead-band relay below: compensate()'s offset is the
                # derated STATIC breakaway, and on a vehicle whose cruise
                # wire sits below its breakaway (this one: cruise ~0.2,
                # offset ~0.18) the offset IS the surge and its removal on
                # overspeed IS the stall -- the discontinuity that carried
                # the 01:08 limit cycle even at modest loop gain.
                ud = a_des / b0_fb
                # Same principle as the model branch: the wire that holds
                # a speed cannot exceed the wire that measurably breaks
                # the car free.
                ff = clamp(eq, 0.0, self._static_wire_cap(s_dir)) * s_dir
                self._ff_wire = ff if not reversing else 0.0
                if not reversing and v * s_dir >= 0.0:
                    ud += ff
            else:
                # No measurement of any kind yet: prior gain through the
                # dead-band map, as prior_a0 does for steering.
                motion = sgn(v) if abs(v) > self.gate_d else 0.0
                ud = self.deadband.compensate(a_des / b0_fb, motion)
                # no model feedforward yet: an implausible sample holds
                # the last wire (see _hold_output)
                self._ff_wire = None
        else:
            # Invert the model EXACTLY as fitted: whatever the (partly
            # arbitrary) split between b1, b2 and b3, inverting their sum
            # reproduces the equilibrium the data actually showed.
            #
            # Evaluated at the SETPOINT, not the measurement. With measured
            # v in the b2 term, the model became an undelayed feedback path
            # with gain 2*|b2|*v/b0 stacked on kp -- and a collinearity-
            # inflated b2 of +2 tripled the loop gain against ~0.6 s of
            # actuator+odometry delay: the 1 Hz surge-stall "stepping" gait.
            # At the setpoint it is pure feedforward: the equilibrium sits
            # exactly at v = cmd_v, and kp and the trim stay the only
            # feedback. Guards: sane b0 (above), and the quadratic is never
            # evaluated outside the speed range it was fitted on.
            # Coulomb direction from the measurement ONLY when the car is
            # confidently moving. At standstill v is odometry noise, and
            # sgn(noise) flapped the +-|b3|/b0 (~0.2 wire) feedforward at
            # noise rate -- the 02:04 log shows 183 sign flips in 60 s with
            # the wire shivering -0.17..+0.23 against a +0.25 command, and
            # the robot never launching. Static friction opposes the
            # INTENDED motion, so below the gate the command decides.
            s_dir = sgn(v) if abs(v) > self.gate_d else (sgn(cmd_v) or sgn(v))
            v_ff = cmd_v if self.vl_lo is None else clamp(cmd_v, self.vl_lo,
                                                          self.vl_hi)
            # Two inversions, two denominators, deliberately. The STATIC
            # part (-b1 - b2v^2 - b3)/b0 is inverted through the FITTED b0:
            # b0..b3 are collinear at one cruise speed, only their sum is
            # data-pinned, and dividing the fitted numerator by anything
            # but the fitted denominator breaks the equilibrium the data
            # actually showed (the b2-clamp lesson). The FEEDBACK part
            # a_des/b0 goes through the bounded b0_fb (above): there b0 is
            # a loop gain, not a vehicle property.
            ud = a_des / b0_fb
            # The friction part of the feedforward may not exceed the wire
            # that MEASURABLY breaks the car free (the dead-band median):
            # static friction is at least kinetic, so a fit asking more
            # wire for friction than the breakaway is the fit, not the car
            # -- b3/b0 0.46 against a 0.22 breakaway launched a 0.15 m/s
            # command to 1.3 m/s (08-30 11:23). Unjudged without evidence.
            # The whole STATIC part of the feedforward -- bias b1 and
            # Coulomb b3 together -- is capped at the breakaway estimate
            # (or the launch floor plus margin, before any dead-band
            # evidence exists): the wire that holds a speed cannot exceed
            # the wire that measurably breaks the car free, whatever the
            # split of a junk fit says. b3 alone was capped and a junk
            # b1 (+0.09 over b0 0.63) leaked 0.14 wire through the same
            # hole (08-31 bench, the 0.58 m/s reversal).
            static = -(b1 + b3 * s_dir) / b0
            cap_w = self._static_wire_cap(s_dir)
            static = clamp(static, -cap_w, cap_w)
            ff = static - b2 * v_ff * abs(v_ff) / b0
            self._ff_wire = clamp(ff, -1.0, 1.0) if not reversing else 0.0
            if not reversing and v * s_dir >= 0.0:
                # ...and none of it while the car still rolls AGAINST
                # the command at all: braking needs no feedforward (the
                # brake-only-reversal principle applied to the reversal's
                # tail -- at half the gate it still landed 0.25 wire on a
                # car rolling the other way at 0.14 m/s).
                ud += ff
        if p.enable_dither and self.dither > 0.0:
            ud += self.dither * math.sin(2.0 * math.pi * 0.7 * self.now)
        raw = ud + self.iv
        ud = clamp(raw, -1.0, 1.0)
        self._clip = sgn(raw - ud)
        self._floor = 0.0
        self._floor_dir = 0.0
        self._cap = 0.0
        self._floor_pinned = False
        # Wheels first, then throttle. From rest the servo turns while the
        # car sits -- free, the car is stopped at every cusp anyway -- and
        # the throttle is released once the modeled servo (qs: the applied
        # wire through tau_s) is within the planner's margin band of its
        # command. That band is the slack between the planned arc and the
        # lock, the only slack the follower has, so a launch inside it
        # starts on the arc. A dead servo cannot deadlock this: qs models
        # the wire, not the servo's answer.
        self._steer_wait = (bool(direction) and not self.rolling
                            and abs(us - self.qs) > self.steer_wait_tol)
        if self._steer_wait:
            self.iv = 0.0
            return us, 0.0
        if direction and self.rolling and self.now < self._post_latch_until \
                and err * direction > 0.0:
            # post-breakaway: hold near the launch wire for one delay
            raw_band = self.deadband.raw(direction)
            base = raw_band if raw_band is not None else p.launch_floor
            self._cap = min(1.0, base + p.launch_cap_margin)
            self._floor_dir = direction
        if direction and not self.rolling and err * direction > 0.0:
            # Not rolling yet, motion is wanted AND the PI agrees it should
            # speed up that way (the last condition keeps this from ever
            # turning a brake into throttle): the wire goes straight to the
            # LEARNED median breakaway -- applied only until the wheels
            # turn, so it cannot over-floor a cruise. Until there is
            # evidence, launch_floor is the bootstrap, as prior_a0 is for
            # the envelope -- capped, from the first measured start on, by
            # the lowest breakaway seen (Policy.launch_floor).
            w = self.deadband._win(direction)
            if w.confirmed:
                # a slow-start median is the real measurement: the wire
                # goes straight to it
                floor = w.value
            elif self.deadband.value(direction) > 0.0:
                # only fast-start upper bounds: they read HIGH by
                # definition (the wire was already past the band), so the
                # floor takes the derated value -- flooring at min(upper)
                # raised a reverse launch's peak from 0.42 to 0.58 on the
                # bench (08-31)
                floor = self.deadband.value(direction)
            else:
                floor = p.launch_floor
                seen = self.deadband.lowest(direction)
                if seen is not None:
                    floor = min(floor, p.deadband_trust * seen)
            eq = self.gain_probe.eq(direction)
            if eq is not None:
                # With a measured cruise wire, launch by CREEPING from it:
                # the floor starts at the wire known to hold a cruise and
                # the integrator (plus the cap's stuck-time ramp) covers
                # the rest, so the car unsticks at its TRUE breakaway
                # instead of at a median that reads high by the slew past
                # it. The in-flight surplus during the sensing delay is
                # what sizes the launch overshoot: on the bench, flooring
                # at the 0.31 median with true breakaway 0.26 peaked
                # 0.5-0.8 on a 0.30 command. A creeping start is also a
                # SLOW start, so it feeds the dead band its honest median.
                floor = min(floor, eq)
            if floor > 0.0:
                # Stall escalation belongs to the FLOOR, not the
                # integrator (whose rate is now gain-normalised and far
                # too slow to double as a stiction probe): the floor
                # itself rides launch_cap_rate per second stuck, so a
                # creeping start crosses from the cruise wire to the true
                # breakaway in under a second while the wire at the
                # moment of unstick stays minimal.
                stuck = (self.now - self._stall_since
                         if self._stall_since is not None else 0.0)
                # The ride starts only after the floor has been given one
                # full sensing round-trip (transport delay + actuator
                # constant) to produce motion: a correct floor unsticks
                # the car before any ride is added, so the dead-band
                # sample it leaves is the floor itself. Riding from the
                # first stuck tick RATCHETED the breakaway median -- every
                # start's sample included the ride, feeding a higher
                # median, feeding a higher floor (bench: 0.31 -> 0.37
                # over five starts).
                stuck = max(0.0, stuck - (self.lon_bank.delay + p.tau_d))
                # The ride's ceiling is the same escalation authority the
                # integrator used to hold (base floor + iv_max of extra
                # wire): enough to unstick every surface this class of
                # vehicle has, bounded so a wall is not ground into at
                # full throttle. Hitting the ceiling with still no motion
                # is what the blocked reflex now detects (it used to
                # watch the integrator pin, but the integrator is
                # gain-normalised trim now, not the stall prober).
                ceiling = min(1.0, floor + p.iv_max)
                floor = min(ceiling, floor + p.launch_cap_rate * stuck)
                self._floor_pinned = floor >= ceiling - 1e-9
                ud = direction * max(ud * direction, floor)
                self._floor, self._floor_dir = floor, direction
                self._cap = min(1.0, floor + p.launch_cap_margin)

        # A zero speed command means stop, not "servo to zero speed".
        if cmd_v == 0.0 and cmd_w == 0.0:
            self.iv = 0.0
            # the steering trim is NOT zeroed: it is per cell and persists
            # (see reset)
            return 0.0, 0.0
        return us, ud

    def get_fault_reset(self):
        """Entering a steering fault: re-open the lateral covariance.

        The collapsed gain was learned from thousands of consistent samples,
        so the estimator is confident and slow. If the actuator comes back,
        that confidence is exactly what would keep the model wrong.
        """
        self.lat_bank.inflate(self.policy.p0)

    def _reflex(self, us, ud, v, cmd_v, t, dt):
        """Blocked detection. No probe, no escalation, no measurement.

        "Stalled" is reported for diagnostics when motion is asked for and
        the latch says nothing is moving. If that persists with the
        integrator pinned at its limit, the robot is against something:
        output zero for a while rather than grind, then let it try again.
        """
        p = self.policy
        moving = self.rolling
        want = abs(cmd_v) > self.stall_cmd_min

        if self._blocked_until is not None:
            if self._blocked_until <= t or not want:
                self._blocked_until = None
                self.blocked = False
            if self.blocked:
                # Hard stop: bypass the slew so the Pico sees zero NOW, and
                # hold the integrator so it does not wind up against a
                # forced-zero output while blocked.
                self.prev_ud = 0.0
                self.iv = 0.0
                self._floor = 0.0
                self._cap = 0.0
                return us, 0.0, True

        if self._steer_wait:
            # parked on purpose while the wheels turn (see _run): not a
            # stall, so neither the floor ride nor the blocked clock runs
            self._stall_since = None
            self._capped_since = None
            return us, 0.0, False

        if want and not moving:
            if self._stall_since is None:
                self._stall_since = t
            stalled = t - self._stall_since > p.stall_time
            pinned = self._floor_pinned \
                or abs(self.iv) >= p.iv_max - 1e-9
            if stalled and pinned:
                if self._capped_since is None:
                    self._capped_since = t
                elif t - self._capped_since > p.blocked_after:
                    self._failures += 1
                    hold = p.blocked_hold \
                        if self._failures >= p.blocked_retries \
                        else p.blocked_release
                    self.blocked = True
                    self._blocked_until = t + hold
                    self._stall_since = None
                    self._capped_since = None
                    self.iv = 0.0
                    self.prev_ud = 0.0
                    self._floor = 0.0
                    self._cap = 0.0
                    return us, 0.0, True
            return us, ud, stalled
        if moving:
            self._failures = 0
        self._stall_since = None
        self._capped_since = None
        return us, ud, False

    # -- output ------------------------------------------------------------

    def _publish(self, us, ud, learning, stalled, dt):
        us = clamp(us, -1.0, 1.0)
        ud = clamp(ud, -1.0, 1.0)
        # Rate-limit both channels. Nothing upstream bounds how fast a command
        # may change, so a model correction or a breakaway step could swing the
        # throttle full scale between two samples -- which is exactly what
        # "moves violently" is. The limits are in units per SECOND and are
        # applied against measured dt, so they hold at any odometry rate.
        p = self.policy
        us = self._slew(self.prev_us, us, p.max_steer_rate * dt)
        # Throttle may come off twice as fast as it goes on: the lunge after
        # breakaway is bounded by how quickly the controller can back off
        # once the wheels turn, and nothing is gained by ramping DOWN slowly.
        rate = p.max_drive_rate * (2.0 if abs(ud) < abs(self.prev_ud) else 1.0)
        asked = ud
        ud = self._slew(self.prev_ud, ud, rate * dt)
        # What shaped the wire this tick, for the integrator (reset()).
        self._shaped = self._clip or sgn(asked - ud)
        if self._floor > 0.0 and self._floor_dir:
            # the launch floor bypasses the slew: see reset() note
            ud = self._floor_dir * max(ud * self._floor_dir, self._floor)
        if self._cap > 0.0 and self._floor_dir:
            # the launch cap bounds the wire from above -- while stuck AND
            # for one actuation delay after breakaway (the wire at the
            # moment of release is what sizes the lunge)
            ud = clamp(ud, -self._cap, self._cap)
        self.prev_us, self.prev_ud = us, ud
        # The whole throttle path works in wire units (see _run); the dead-
        # band map is applied inside _run to the bootstrap path only.
        wire = ud
        self.prev_wire = wire
        # The history the learner aligns its regressors to is what was ON THE
        # WIRE. In PASSIVE that is the joystick, not this (unpublished) output.
        entry = self._wire or (us, wire)
        if not self.external_history:
            self.record_command(self.now, *entry)
        m = self.model
        return Output(steer=us, drive=wire, phase=self.phase,
                      learning=learning, stalled=stalled, v=self.v,
                      psidot=self.psidot, dt=dt, model=m,
                      min_turning_radius=self.envelope.min_turning_radius(m),
                      max_curvature=self.envelope.max_curvature(m),
                      envelope_confirmed=self.envelope.confirmed,
                      breakaway=self.breakaway,
                      steering_fault=self.steering_fault,
                      drive_fault=self.drive_fault,
                      deadband_fwd=self.deadband.value(1.0),
                      deadband_rev=self.deadband.value(-1.0),
                      steer_wait=self._steer_wait)

    def record_command(self, stamp, steering, throttle):
        """Append a timestamped delivery, or a synchronous test-plant command."""
        if not finite(stamp, steering, throttle):
            return
        if self._cmd_hist and stamp <= self._cmd_hist[-1][0]:
            return
        self._cmd_hist.append((stamp, steering, throttle))
        if sgn(throttle) != self._wire_sign:
            self._wire_sign = sgn(throttle)
            self._wire_on_since = stamp

    @staticmethod
    def _slew(prev, target, limit):
        if limit <= 0.0:
            return target
        return prev + clamp(target - prev, -limit, limit)

    def _hold_output(self):
        """An implausible sample: steer as before, and on the throttle put
        the wire that HOLDS the commanded speed (the model feedforward,
        _ff_wire) -- neither chase the reading nor keep pushing. Holding
        the last wire through a genuine overshoot kept the lunge wire on
        and ran a sticky plant away to 0.9 m/s (08-29 bench)."""
        m = self.model
        drive = self._ff_wire if self._ff_wire is not None else self.prev_wire
        self.prev_ud = self.prev_wire = drive
        return Output(steer=self.prev_us, drive=drive,
                      phase=self.phase, v=self.v, psidot=self.psidot,
                      dt=self.dt, model=m,
                      min_turning_radius=self.envelope.min_turning_radius(m),
                      max_curvature=self.envelope.max_curvature(m),
                      envelope_confirmed=self.envelope.confirmed,
                      breakaway=self.breakaway,
                      steering_fault=self.steering_fault,
                      drive_fault=self.drive_fault,
                      deadband_fwd=self.deadband.value(1.0),
                      deadband_rev=self.deadband.value(-1.0))

    def _safe_output(self):
        self.prev_us = self.prev_ud = self.prev_wire = 0.0
        m = self.model
        return Output(phase=self.phase, v=self.v, psidot=self.psidot,
                      dt=self.dt, model=m,
                      min_turning_radius=self.envelope.min_turning_radius(m),
                      max_curvature=self.envelope.max_curvature(m),
                      envelope_confirmed=self.envelope.confirmed,
                      breakaway=self.breakaway,
                      steering_fault=self.steering_fault,
                      drive_fault=self.drive_fault,
                      deadband_fwd=self.deadband.value(1.0),
                      deadband_rev=self.deadband.value(-1.0))

    # -- persistence -------------------------------------------------------

    def plausible(self):
        """Is the learned STEERING model physically sensible enough to keep?

        A model learned while an actuator was dead is worse than no model: it
        concludes the vehicle cannot steer, and saving it carries that across
        reboots, so a five-minute wiring fault becomes permanent. This is the
        gate on persistence. The throttle model is judged separately by
        :meth:`lon_plausible`: it is not inverted for control, and a
        wrong-signed b0 there must not block saving a good steering model.
        """
        m = self.model
        v2 = self.v_op ** 2
        raw = (m.a0l, m.a0r, m.a0l_rev, m.a0r_rev)
        spans = [abs(gain + m.a2 * v2) for gain in raw]
        # Deliberately does NOT require a2 <= 0. That sign is enforced at the
        # inversion, and a poorly-excited run can leave it wrong while the
        # model is still perfectly usable -- demanding it here would mean
        # almost nothing ever persists. ALL FOUR gain cells must be alive
        # AND sign-consistent: one collapsed or flipped cell saved to disk
        # is a permanent can't-turn-that-way car, and the 08-23 -0.09 cell
        # was being re-saved every 30 s while it steered inverted.
        consistent = list(sane_gain_cells(raw, self.prior_a0)) \
            == list(raw)
        return (finite(*spans)
                and min(spans) > self.policy.den_min
                and consistent
                and not self.steering_fault)

    def authority(self):
        """Fraction of the operating speed the vehicle may be commanded.

        1.0 once both fits have earned inversion and are physically a
        vehicle; Policy.authority_floor while the controller is steering
        or driving on a prior (not ready, not plausible, or a steering
        fault); 0.0 outside RUN and while the odometry or the throttle is
        faulted -- states in which the actuators are already held at zero.
        """
        if self.phase != RUN or not self.odom_ok or self.drive_fault:
            return 0.0
        earned = (self.ready_lat and self.ready_lon and self.plausible()
                  and self.lon_plausible() and not self.steering_fault)
        return 1.0 if earned else self.policy.authority_floor

    def stopping_distance(self):
        """How far the car travels from the approach speed before it is at
        rest: the learned reaction delay at that speed, then coasting on
        the Coulomb term. The follower's goal tolerance is pushed from it
        (the typed 0.20 m ended every cusp leg 0.19 m short, 09-02), and
        leg_end_reachable uses the same number as its miss bound. None
        until the friction term is identified."""
        b3 = self.rls_lon.theta[3]
        v_ap = self.policy.approach_speed_frac * self.v_op
        if not (v_ap > 0.0 and self.ready_lon and self.lon_plausible()
                and finite(b3) and b3 < 0.0):
            return None
        return (v_ap * (self.lon_bank.delay + self.policy.tau_d)
                + v_ap * v_ap / (2.0 * -b3))


    def stop_horizon(self):
        """How long a follower should project a command for collisions.

        The learned reaction delay (nothing the controller does reaches the
        wheels sooner) plus the time the car takes to come to rest from the
        planning speed on its own friction (b3, the Coulomb term). Friction
        alone is the conservative choice -- active braking is shorter. None
        until the friction term is identified (it needs both directions
        fed), so the follower keeps its own default until then.

        Why: RPP's projection horizon is a typed 1.5 s; at the end of a
        segment it projects past the cusp into whatever the planner put the
        cusp next to, and aborts. 08-29 00:08: three such aborts, each a
        clear-costmap + replan costing 25-30 s, on a car that stops in
        ~0.35 s from 0.35 m/s (b3 = -0.95 m/s^2).
        """
        b3 = self.rls_lon.theta[3]
        if not (self.ready_lon and self.lon_plausible() and self.v_op > 0.0
                and finite(b3) and b3 < 0.0):
            return None
        return self.lon_bank.delay + self.v_op / (-b3)

    def lon_plausible(self):
        """More throttle means more acceleration, or the fit is not a vehicle.

        On the robot the longitudinal learner has been seen to settle at
        b0 < 0 (the dead-band actuator defeats the linear fit); an ``abs(b0)``
        test let that reach disk. Such a fit is replaced by the prior on save.
        """
        m = self.model
        return lon_sane((m.b0, m.b1, m.b2, m.b3), self.vl_lo, self.vl_hi,
                        self.policy, self.speed_scale, self._breakaway_median())

    def agreement(self):
        """Do the independent estimators of the throttle plant agree?

        The control law ranks them (probe, then the sane fit, then the
        prior) and switches silently; their DISAGREEMENT is the earliest
        sign that one has gone wrong -- every poisoned-model incident of
        08-28..08-31 would have shown here first. Returns (ok, text):
        ok is False when a plausible fit and the probe differ by more
        than the factor-two trust band the divisor grants a measured
        anchor.
        """
        m = self.model
        pb, eq, brk = self.gain_probe.b0, self.gain_probe.eq(1.0), \
            self.deadband.raw(1.0)
        lon_ok = self.ready_lon and self.lon_plausible()
        parts, ok = [], True
        if pb and m.b0 > 0.0:
            ratio = max(pb / m.b0, m.b0 / pb)
            parts.append(f'b0 probe/fit {pb:.2f}/{m.b0:.2f} ({ratio:.1f}x)')
            if lon_ok and ratio > 2.0:
                ok = False
        if eq is not None and m.b0 > 0.0:
            ff = -(m.b1 + m.b3) / m.b0
            parts.append(f'eq probe/fit {eq:.2f}/{ff:.2f}')
            if self.ready_lon and eq > 0.0 and (ff <= 0.0 or max(eq / ff, ff / eq) > 2.0):
                ok = False
        if brk is not None and m.b0 > 0.0 and m.b3 < 0.0:
            parts.append(f'breakaway/fit {brk:.2f}/{-m.b3 / m.b0:.2f}')
        source = ('learned feedforward' if lon_ok else
                  'bootstrap: measured equilibrium' if eq is not None else
                  'bootstrap: probe gain' if pb else 'bootstrap: prior')
        text = source + (' | ' + ', '.join(parts) if parts else '')
        if not ok:
            text += ' | DISAGREE'
        return ok, text

    @property
    def iw(self):
        """The steering trim of the active cell (see reset)."""
        return self._iw_cells.get(self._iw_key, 0.0)

    @iw.setter
    def iw(self, value):
        self._iw_cells[self._iw_key] = value

    def trim_cells(self):
        """All four trims, keyed 'fwd_left' etc., for diagnostics/state."""
        return {f"{'fwd' if d > 0 else 'rev'}_{'left' if sd > 0 else 'right'}": v
                for (d, sd), v in self._iw_cells.items()}

    def _breakaway_median(self):
        """The forward breakaway estimate (slow-start median, or the
        smallest fast-start upper bound), or None."""
        return self.deadband.raw(1.0)

    @property
    def breakaway(self):
        """The forward breakaway estimate as logged (flight recorder,
        /diagnostics, Output). Until 08-31 this was a vestigial FIELD --
        initialised to 0.0 and never written again -- so the flight log
        showed 0.000 for every run regardless of what the dead-band
        learner actually knew. It is now the live estimate."""
        b = self.deadband.raw(1.0)
        return b if b is not None else 0.0

    def state(self):
        """Everything worth carrying across a reboot.

        Version 2 adds the learner's readiness (sample counts and excitation
        spans). Without it a restored model was never ``ready_*`` until it had
        re-earned 60-80 fresh samples, so persistence changed nothing about
        what the controller inverted after a reboot.
        """
        p = self.policy
        lon_ok = self.lon_plausible()
        if lon_ok:
            # the loop-gain anchor tracks whatever is good enough to keep
            self._b0_ref = self.rls_lon.theta[0]
        return {
            # Version 4: the lateral fit is 6 parameters (a0 per travel
            # direction x steering side, then a1, a2). Version 3 carried 4
            # (left/right only); 1 and 2 a symmetric 3.
            'version': 4,
            'lateral': list(self.rls_lat.theta),
            'longitudinal': (list(self.rls_lon.theta) if lon_ok
                             else [p.prior_b0, 0.0, 0.0, 0.0]),
            'lat_delay': self.lat_bank.delay,
            'lon_delay': self.lon_bank.delay,
            'envelope': self.envelope.state(),
            'n_lat': int(self.rls_lat.count),
            'n_lon': int(self.rls_lon.count) if lon_ok else 0,
            'spans': {
                'qd': [self.qd_lo, self.qd_hi] if lon_ok else [None, None],
                'vl': [self.vl_lo, self.vl_hi] if lon_ok else [None, None],
                'qs': [self.qs_lo, self.qs_hi],
            },
            'deadband': self.deadband.state(),
            'gain_probe': self.gain_probe.state(),
            'trim': self.trim_cells(),
            'sigma_v': self.sigma_v,
            'tick': self.tick,
            # The scale every speed-shaped gate is a fraction of. Restored
            # so a rebooted car does not start with its gates at zero.
            'v_op': self.v_op,
            # The servo's established direction (+1/-1), or None.
            'steer_sign': self.steer_sign,
        }

    def load_state(self, d, inflate=1.0):
        """Restore a saved model. Returns True if the whole thing was valid.

        The covariance is re-opened to a FRESH model's p0 rather than
        restored: the parameters are probably still right, but the tyres,
        the floor and the battery have all had a chance to change while the
        robot was off, so the learner should be readier to move than it was
        when it saved. Not readier than a model that knows nothing, though:
        at 4 x p0 (the old value) five transient ticks at the first cusp of
        08-28 23:15 rewrote three cells of a 3862-sample model (a0lr 2.07
        -> 0.04) and sent the planner a 1.75 m radius for a minute.

        Version 1 files are accepted for their parameters only. Their
        envelope evidence was measured against the lag-filtered command rather
        than the delayed one the model is fit on, which made it systematically
        pessimistic, so it is discarded and re-learned.
        """
        if not isinstance(d, dict) or d.get('version') not in (1, 2, 3, 4):
            return False
        lat, lon = d.get('lateral'), d.get('longitudinal')
        # Older lateral layouts are expanded, never rejected: a symmetric
        # 3-parameter fit seeds all four gain cells, a 4-parameter
        # left/right fit seeds each reverse cell from its forward side. The
        # envelope evidence is KEPT across these upgrades (unlike v1): it
        # was measured against the gains the cells start as, so it stays
        # consistent and simply ages out as the cells diverge.
        if not isinstance(lat, list) or len(lat) not in (3, 4, 6) \
                or not finite(*lat):
            return False
        if len(lat) == 3:
            lat = [lat[0]] * 4 + [lat[1], lat[2]]
        elif len(lat) == 4:
            lat = [lat[0], lat[1], lat[0], lat[1], lat[2], lat[3]]
        # Files from before the Coulomb term carry a 3-parameter fit.
        if not isinstance(lon, list) or len(lon) not in (3, 4) \
                or not finite(*lon):
            return False
        lon = list(lon) + [0.0] * (4 - len(lon))
        p = self.policy
        if d['version'] >= 2:
            if not self.envelope.load(d.get('envelope')):
                return False
            spans = d.get('spans') or {}
            if not isinstance(spans, dict):
                return False
            try:
                n_lat = max(int(d.get('n_lat', 0)), 0)
                n_lon = max(int(d.get('n_lon', 0)), 0)
            except (TypeError, ValueError):
                return False
            qd, vl, qs = (_load_span(spans.get(k)) for k in ('qd', 'vl', 'qs'))
            # Optional: files saved before the dead band was learned lack it.
            if 'deadband' in d and not self.deadband.load(d['deadband']):
                return False
            # Optional likewise: the direct gain measurement.
            if 'gain_probe' in d and not self.gain_probe.load(d['gain_probe']):
                return False
            # Optional: the per-cell steering trims (bounded like live ones)
            trim = d.get('trim')
            if isinstance(trim, dict):
                for key, val in trim.items():
                    try:
                        dname, sname = key.split('_')
                        cell = (1.0 if dname == 'fwd' else -1.0,
                                1.0 if sname == 'left' else -1.0)
                    except ValueError:
                        continue
                    if finite(val):
                        self._iw_cells[cell] = clamp(float(val), -p.iw_max,
                                                     p.iw_max)
        else:
            n_lat = n_lon = 0
            qd = vl = qs = (None, None)

        p0 = min(p.p0 * inflate, p.p_max)
        # The operating speed, when the file has one. A file from before
        # it was learned leaves it unknown: the gates stay at the noise
        # measurement until the first command sets it, which is one tick.
        # NOT inferred from the span (see speed_scale).
        v_op = d.get('v_op')
        self.v_op = float(v_op) if finite(v_op) and v_op > 0.0 else 0.0
        self.envelope.v_op = self.v_op
        self._update_gates()
        # Where the restored throttle fit is judged: the same scale
        # lon_sane and readiness use, clamped into the span as they are.
        judge_v = self.v_op or max(abs(vl[0] or 0.0), abs(vl[1] or 0.0))
        # The established steering sign comes BEFORE the cells are judged:
        # it is what a persisted tie resolves to.
        sign = d.get('steer_sign')
        self.steer_sign = float(sign) if sign in (1, -1, 1.0, -1.0) else None
        self.envelope.prior_a0 = self.prior_a0
        if self.steer_sign is not None and self.prior_a0 != p.prior_a0:
            # Cells the file still carries at the declared seed are
            # unexcited, not evidence (see _reseed_unexcited).
            seed = p.prior_a0
            lat = [self.prior_a0 if abs(c - seed) <= 0.1 * abs(seed) else c
                   for c in lat[:4]] + list(lat[4:])
        # A file can carry poisoned cells (it did: 08-28, a 2-2 sign split
        # saved under the old tie rule and restored on every launch that
        # day). Seed the learner from the sanitised cells so it does not
        # START inverted and have to unlearn its way back through zero.
        lat = list(sane_gain_cells(lat[:4], self.prior_a0)) + list(lat[4:])
        self.lat_bank.seed(lat, n_lat, p0)
        self.qs_lo, self.qs_hi = qs
        if lon[2] > 0.0:
            # Same projection as the fit's, anchored at the speed the fit
            # is judged at so the persisted cruise wire (what lon_sane
            # judges) is kept.
            lon[1] += lon[2] * min(judge_v, vl[1] or judge_v) ** 2
            lon[2] = 0.0
        if lon_sane(lon, vl[0], vl[1], p, judge_v, self._breakaway_median()):
            self.lon_bank.seed(lon, n_lon, p0)
            self._b0_ref = lon[0]
            self.qd_lo, self.qd_hi = qd
            self.vl_lo, self.vl_hi = vl
        else:
            self.lon_bank.seed([p.prior_b0, 0.0, 0.0, 0.0], 0, p0)
            self.qd_lo = self.qd_hi = self.vl_lo = self.vl_hi = None
        for key, bank in (('lat_delay', self.lat_bank),
                          ('lon_delay', self.lon_bank)):
            if finite(d.get(key, float('nan'))):
                bank.set_delay(float(d[key]))
        return True

    def _running_sigma(self):
        return _stddev(self._v_samples) if len(self._v_samples) >= 3 else None
