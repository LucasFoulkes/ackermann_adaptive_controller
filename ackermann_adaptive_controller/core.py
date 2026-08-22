# Copyright 2026 Lucas Foulkes
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""Self-configuring adaptive Twist controller: the ROS-free mathematics.

Turns a desired body twist (v*, omega*) plus an odometry pose series into two
normalized actuator commands in [-1, 1]. Nothing here knows the vehicle or the
sensor in advance; both are identified online by recursive least squares.

Three phases:

* ``SENSE``  sit still and measure the *sensor*: velocity noise floor, encoder
  quantization, heading noise. Every downstream threshold is derived from
  these numbers rather than guessed, which is what makes the same code work on
  a noisy 10 Hz LiDAR odometry and on clean wheel encoders.
* ``CAL``    optional scripted wiggle to excite both axes quickly. Disabled by
  default because it drives the robot autonomously; passive learning from
  ordinary Nav2 commands reaches the same place given priors.
* ``RUN``    invert the learned model, with integral trim riding *through* the
  inversion so the trim stays in physical units as the model changes.

Two learned sub-models, both linear in their parameters so RLS applies:

    longitudinal   vdot      = b0*qd + b1 + b2*v*|v|
    lateral        psidot/v  = a0*qs + a1 + a2*qs*v**2

The regressors are the commands that were on the wire ``lat_delay`` /
``lon_delay`` seconds before the response being explained, not the current
ones: the servo, the motor bridge and the odometry all lag, and regressing the
instantaneous command against a response that belongs to an earlier one is how
the learner once produced confident nonsense. In PASSIVE the wire carries
someone else's commands (the joystick); the caller passes them as ``applied``
and they -- not this controller's unpublished output -- go into the history
the learner aligns to.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

SENSE = 'SENSE'
CAL = 'CAL'
RUN = 'RUN'


def finite(*vals):
    """True only if every value is a real, finite number."""
    return all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) for v in vals)


def clamp(x, lo, hi):
    """Clamp with NaN treated as the low bound rather than passing through."""
    if not math.isfinite(x):
        return 0.0
    return lo if x < lo else (hi if x > hi else x)


def wrap(a):
    """Wrap an angle difference into [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def sgn(x):
    """Sign, with an explicit zero."""
    return 0.0 if x == 0.0 else (1.0 if x > 0.0 else -1.0)


@dataclass
class Policy:
    """The only hand-chosen numbers. These encode risk, not the vehicle."""

    # Phase timing.
    t_sense: float = 2.0
    t_cal: float = 7.0
    cal_steer: float = 0.30
    cal_drive: float = 0.42
    enable_calibration: bool = False

    # Learner.
    p0: float = 10.0          # initial skepticism
    t_forget: float = 170.0   # s half-life: how fast the world changes
    p_max: float = 1.0e4      # covariance windup bound

    # Actuator-class filter constants (tested tau in [0.1, 0.75]).
    tau_s: float = 0.18
    tau_d: float = 0.30

    # Per-second loop gains: rate-invariant, and plant-invariant because the
    # model inversion normalizes the plant away.
    kp_v: float = 0.6  # loop gain sized to ~0.35 s of actuator+odometry delay; at 1.6 the loop limit-cycled 0..0.9 m/s on a 0.32 command
    ki_v: float = 0.25  # same reason; the integrator still unsticks the robot, just without overshooting past the dead band
    # Curvature trim. The effective gain is ki_w / v_signed (at least 2*ki_w
    # per second) against ~0.75 s of steering delay plus yaw filter: at 1.5 a
    # single 0.45 rad/s step wound the trim to its clamp before the car had
    # responded, and the flight log had it pinned 30% of the time with a sign
    # that agreed with the command only half the time -- noise, not trim.
    ki_w: float = 0.3

    # Priors. Without these, steering never excites (no steering -> no yaw ->
    # nothing to learn from), so RUN could never bootstrap without CAL.
    # a0 ~ tan(delta_max)/wheelbase (1.25: wheelbase 0.2775 m, ~18 deg lock);
    # b0 ~ full-throttle acceleration. Only starting points: the steering
    # envelope is learned and supersedes a0; b0 is the throttle gain the
    # controller runs on while use_learned_lon is off. Setting them closer
    # to the truth only shortens the transient.
    prior_a0: float = 1.25
    prior_b0: float = 2.0

    # Evidence required before the learned model may replace the prior in the
    # inversion. Sample count alone is not evidence: a hundred samples at one
    # speed leave b1/b2 (and a1/a2) unidentifiable, and inverting that split
    # is where the wild throttle came from. The spans demand that the data
    # actually covered a range of commands and speeds.
    ready_lon_samples: int = 80
    ready_lon_qd_span: float = 0.15
    ready_lon_v_span: float = 0.15
    ready_lat_samples: int = 60
    ready_lat_qs_span: float = 0.40

    # Inversion guards and limits.
    # Floor on the speed used to convert a yaw-rate command to curvature.
    # MUST sit well below the operating speed: at 0.4 with Nav2 cruising at
    # 0.22 m/s the floor dominated and the controller delivered 55% of the
    # curvature the path follower asked for -- the follower kept asking
    # harder, the trim integrator wound to its clamp and overshot: weaving.
    v_eff_floor: float = 0.12
    den_min: float = 0.05
    b0_min: float = 0.3
    # Understeer cannot cancel the steering entirely: full lock always buys
    # SOME curvature, so the speed term may not null the static gain.
    span_floor: float = 0.25
    # Command-to-response delay used to align the regressors. Measured on
    # the robot by scanning lags against the flight logs: the steering fit
    # goes from r2 ~0.1 at lag 0 to ~0.8 at 0.4-0.5 s in every run, with a
    # consistent gain. Regressing on the instantaneous (or lag-filtered)
    # command against a response that arrives half a second later is how
    # the learner produced confident nonsense.
    lat_delay: float = 0.45
    lon_delay: float = 0.30
    # Ceiling on the lateral learning gate. The gates scale with the noise
    # measured in SENSE (10 sigma), and on LiDAR odometry that measurement
    # varies 2x between boots on the same floor: one session measured
    # sigma_v = 0.028, which put gate_s at 0.45 m/s -- above the 0.38 m/s
    # Nav2 is allowed to command -- and the steering model took 23 samples
    # in ten minutes. The cap ties the gate to the operating speed instead.
    gate_s_max: float = 0.24
    # Invert the learned longitudinal model for control? Off: see _run.
    use_learned_lon: bool = False
    # Steering authority at standstill (fraction of full lock); full
    # authority once rolling at gate_d. See the note in _run.
    steer_standstill: float = 0.45
    # Slew limits, per second. Without these a single sample can swing the
    # throttle full scale, which is what "violent" looks like from outside.
    # Lower them if the robot still feels abrupt; raise them if it feels
    # sluggish to respond.
    max_steer_rate: float = 4.0
    max_drive_rate: float = 2.0
    # Throttle applied whenever motion is commanded and the car is not yet
    # rolling. A CONSTANT, deliberately below the lowest breakaway ever
    # observed (0.24), so it cannot lunge; it only shortens the integrator's
    # climb, which at Nav2's approach speed was 5-8 s per launch and left a
    # third of the flight log "stalled". Zero disables it.
    launch_floor: float = 0.15
    # Only sized (and therefore only active) after the CAL wiggle has run.
    enable_dither: bool = True

    # Stall / blocked detection.
    # "Is motion being asked for at all", not "is a lot of motion asked for".
    # At 0.25 this exactly equalled the velocity smoother's reverse limit, so
    # the strict > test never fired and the robot could stall forever.
    stall_cmd_min: float = 0.05
    stall_time: float = 0.6
    # Integrator authority. Stiction is handled by the integrator ramping
    # until the wheels turn; this is how far it may go, and hitting it while
    # stalled is the cue that the robot is against something.
    iv_max: float = 0.45
    iw_max: float = 0.12           # curvature trim authority (1/m)
    blocked_retries: int = 3       # failed attempts before a long hold
    blocked_hold: float = 5.0      # s, after that many failures
    # Feedback speed filter time constant. The raw pose-differenced speed
    # carries ~0.05 m/s of jitter at 10 Hz -- a quarter of the commanded
    # speed -- and feeding it to the PI unfiltered made the throttle chase
    # noise tick by tick. Learning keeps using the raw signal (filtering the
    # regressor biases RLS); only CONTROL uses the filtered one.
    v_fb_tau: float = 0.10  # every 0.1 s of filter lag here is as costly as the same delay in the actuator
    # If the ramp reaches its cap and the robot STILL is not moving, it is not
    # static friction -- it is a wall. Give up instead of pushing through it.
    blocked_after: float = 1.0     # s at full escalation before declaring it
    blocked_release: float = 1.5   # s of zero/low command before retrying

    # Steering envelope. The turning radius is learned, not configured; these
    # only bound what a learned value is allowed to be, and how much evidence
    # it takes to believe one.
    env_qs_threshold: float = 0.55   # |qs| above which a sample is envelope evidence
    env_evidence: int = 12           # samples needed before observation is trusted
    env_derate: float = 0.6          # trust in raw model extrapolation, pre-evidence
    env_speed: float = 0.35          # speed the planner radius is quoted at
    radius_floor: float = 0.25
    radius_ceiling: float = 8.0

    # Dead-man: no odometry for this many nominal steps stops the actuators.
    odom_timeout_steps: float = 3.0


@dataclass
class Model:
    """A snapshot of what has been learned, for logging and diagnostics."""

    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    b0: float = 0.0
    b1: float = 0.0
    b2: float = 0.0
    n_lat: int = 0
    n_lon: int = 0


class RLS:
    """Exponentially-weighted recursive least squares, 3 parameters.

    Three textbook failure modes are guarded here. Covariance windup is bounded
    by ``p_max`` (the caller also gates uninformative samples out entirely);
    the gain denominator is floored so a zero regressor cannot divide by zero;
    and any non-finite sample is rejected outright rather than poisoning the
    estimate permanently.
    """

    def __init__(self, theta0, p0, p_max):
        self.n = len(theta0)
        self.theta = list(theta0)
        self.P = [[p0 if i == j else 0.0 for j in range(self.n)]
                  for i in range(self.n)]
        self.p_max = p_max
        self.lam = 1.0
        self.count = 0
        self.innovation = 0.0

    def predict(self, phi):
        return sum(t * p for t, p in zip(self.theta, phi))

    def inflate(self, p0):
        """Re-open the covariance so the estimate can move again.

        After a long run of consistent samples the gain is small and the
        estimate is effectively frozen. That is correct while the samples are
        trustworthy, and wrong the moment they stop being -- a dead actuator
        teaches "no gain" thousands of times, and without this the model
        cannot climb back out when the actuator returns.
        """
        for i in range(self.n):
            for j in range(self.n):
                self.P[i][j] = max(self.P[i][j], p0) if i == j else 0.0

    def update(self, phi, y, lam):
        """One measurement. Returns True if it was accepted."""
        if not finite(y, lam, *phi) or lam <= 0.0:
            return False

        # Pphi = P @ phi
        pphi = [sum(self.P[i][j] * phi[j] for j in range(self.n))
                for i in range(self.n)]
        denom = lam + sum(phi[i] * pphi[i] for i in range(self.n))
        if not finite(denom) or abs(denom) < 1e-9:
            return False

        err = y - self.predict(phi)
        if not finite(err):
            return False
        self.innovation = err

        gain = [p / denom for p in pphi]
        theta = [self.theta[i] + gain[i] * err for i in range(self.n)]
        if not finite(*theta):
            return False
        self.theta = theta

        # P = (P - gain outer Pphi) / lam, then symmetrize and bound.
        for i in range(self.n):
            for j in range(self.n):
                self.P[i][j] = (self.P[i][j] - gain[i] * pphi[j]) / lam
        trace = sum(self.P[i][i] for i in range(self.n))
        if not finite(trace) or trace <= 0.0:
            # Numerically dead: restart the covariance rather than propagate.
            for i in range(self.n):
                for j in range(self.n):
                    self.P[i][j] = self.p_max if i == j else 0.0
        elif trace > self.p_max:
            scale = self.p_max / trace
            for i in range(self.n):
                for j in range(self.n):
                    self.P[i][j] *= scale
        for i in range(self.n):
            for j in range(i + 1, self.n):
                avg = 0.5 * (self.P[i][j] + self.P[j][i])
                self.P[i][j] = self.P[j][i] = avg

        self.count += 1
        return True


class TwistEstimator:
    """Signed body twist differenced from a pose series.

    MOLA publishes pose in ``odom``; its twist field is not populated when the
    state estimator is disabled, so velocity is differenced here instead of
    trusted from the message. Forward speed is the displacement *projected onto
    the heading*, which keeps the sign correct in reverse -- a plain
    ``hypot(dx, dy)`` would report reverse as positive and invert every
    downstream control law.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.t = None
        self.x = self.y = self.psi = 0.0
        self.v = 0.0
        self.psidot = 0.0

    def update(self, t, x, y, psi):
        """Returns (dt, v, psidot_raw) or None when no step can be formed."""
        if self.t is None:
            self.t, self.x, self.y, self.psi = t, x, y, psi
            return None
        dt = t - self.t
        if dt <= 1e-6:
            return None
        dx, dy = x - self.x, y - self.y
        v = (dx * math.cos(self.psi) + dy * math.sin(self.psi)) / dt
        psidot = wrap(psi - self.psi) / dt
        self.t, self.x, self.y, self.psi = t, x, y, psi
        if not finite(v, psidot):
            return None
        self.v, self.psidot = v, psidot
        return dt, v, psidot


class _Window:
    """Median of the last N samples.

    A median is what makes one glitched sample unable to move the envelope,
    and the bounded window is the forgetting: evidence from a surface the
    robot has since left simply ages out.
    """

    def __init__(self, evidence, span=3):
        self.evidence = evidence
        self.vals = deque(maxlen=max(evidence * span, 30))

    def add(self, x):
        self.vals.append(x)

    @property
    def confirmed(self):
        return len(self.vals) >= self.evidence

    @property
    def value(self):
        if not self.vals:
            return None
        ordered = sorted(self.vals)
        n = len(ordered)
        mid = n // 2
        return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] +
                                                 ordered[mid])


class CurvatureEnvelope:
    """How tightly this vehicle can actually turn, learned from motion.

    The identified lateral model already gives a curvature for any steering
    command and speed. What it cannot know is whether that *linear* map still
    holds at full lock -- a steering rack has stops, and a tyre has a limit.
    So what is learned here is one number per direction: the ratio of the
    curvature actually achieved to the curvature the model predicted, measured
    only when the steering was near lock.

        ratio ~ 1   the model extrapolates honestly to full lock
        ratio < 1   the model overpromises; the real envelope is smaller

    Measuring a *ratio* rather than raw curvature is what makes this
    speed-correct. Raw curvature at full lock depends strongly on speed
    through the ``a2 * v**2`` term -- on this vehicle that term nearly cancels
    the steering gain at 2 m/s -- so a raw measurement taken while driving
    fast would report a wildly flatter envelope than the robot has at the
    0.35 m/s Nav2 actually plans for.

    Conservative means a *larger* radius: a planner that believes the car
    turns tighter than it does emits paths the car cannot follow.
    """

    def __init__(self, policy):
        self.p = policy
        self.left = _Window(policy.env_evidence)
        self.right = _Window(policy.env_evidence)

    @staticmethod
    def predict(model, qs, v):
        return qs * (model.a0 + model.a2 * v * v) + model.a1

    def observe(self, qs, kappa, v, model):
        """Record how well the model predicted a near-lock steering sample."""
        if not finite(qs, kappa, v) or abs(qs) < self.p.env_qs_threshold:
            return False
        predicted = self.predict(model, qs, v)
        if not finite(predicted) or abs(predicted) < 1e-3:
            return False
        ratio = abs(kappa) / abs(predicted)
        if not finite(ratio):
            return False
        # A ratio far above 1 is noise, not a car that out-turns its own model.
        ratio = min(ratio, 1.5)
        (self.left if qs >= 0.0 else self.right).add(ratio)
        return True

    @property
    def confirmed(self):
        return self.left.confirmed and self.right.confirmed

    def fidelity(self):
        """Worst-direction model fidelity at full lock, or None."""
        if not self.confirmed:
            return None
        return min(self.left.value, self.right.value)

    def max_curvature(self, model, v=None, derate=True):
        """Curvature believed reachable at full lock, at the planning speed.

        ``derate=False`` skips the pre-evidence derating. The controller's own
        clamp must use it: derated, the clamp holds steering below the
        evidence threshold, so the envelope can never confirm and the clamp
        never lifts -- a loop the planner push (which stays derated) is not in.
        """
        v = self.p.env_speed if v is None else v
        span = model.a0 + model.a2 * v * v
        # Worst of the two directions: a trim term makes one side tighter.
        extrapolated = min(abs(model.a1 + span), abs(model.a1 - span))
        if not finite(extrapolated):
            extrapolated = 0.0

        ratio = self.fidelity()
        if ratio is None:
            # No evidence yet: derate the raw extrapolation rather than
            # believe a prior about a vehicle nobody has measured.
            kappa = extrapolated * (self.p.env_derate if derate else 1.0)
        else:
            # Never let evidence make the envelope *more* optimistic than the
            # model; it can only reveal that the model overpromises.
            kappa = extrapolated * min(ratio, 1.0)

        kappa = min(kappa, 1.0 / self.p.radius_floor)
        kappa = max(kappa, 1.0 / self.p.radius_ceiling)
        return kappa

    def min_turning_radius(self, model, v=None):
        return 1.0 / self.max_curvature(model, v)

    def state(self):
        return {'left': list(self.left.vals), 'right': list(self.right.vals)}

    def load(self, d):
        if not isinstance(d, dict):
            return False
        for key, win in (('left', self.left), ('right', self.right)):
            vals = d.get(key)
            if not isinstance(vals, list) or not all(
                    finite(v) and v > 0.0 for v in vals):
                return False
            win.vals = deque(vals[-win.vals.maxlen:], maxlen=win.vals.maxlen)
        return True


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
        self.vdot = 0.0
        self.psidot = 0.0

        # Sensor character, measured in SENSE.
        self._v_samples = []
        self._psi_diffs = []
        self.sigma_v = 0.0
        self.tick = 0.0
        self.sigma_psi = 0.0

        # Derived, never tuned.
        self.gate_d = 0.15
        self.gate_s = 0.24
        self.alpha = 0.33
        self.lam = 0.999
        self.dither = 0.0

        self.rls_lat = RLS([p.prior_a0, 0.0, 0.0], p.p0, p.p_max)
        self.rls_lon = RLS([p.prior_b0, 0.0, 0.0], p.p0, p.p_max)
        self.envelope = CurvatureEnvelope(p)

        # Launch/rolling latch for the breakaway kick, with hysteresis so
        # odometry noise cannot flicker it.
        self.rolling = False
        self._still_since = None
        self.v_fast = 0.0
        self.qs = self.qd = 0.0
        # (stamp, us, ud) history for delay-aligned regression
        self._cmd_hist = deque(maxlen=64)
        self.iw = self.iv = 0.0
        self.prev_us = self.prev_ud = 0.0
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
        self.breakaway = 0.0
        self._capped_since = None
        self._blocked_until = None
        self.blocked = False
        self._failures = 0
        self.steering_fault = False

    def _enter(self, phase, t):
        self.phase = phase
        self.phase_t0 = t

    @property
    def ready_lon(self):
        """Has the longitudinal model earned the right to be inverted?"""
        p = self.policy
        return (self.rls_lon.count >= p.ready_lon_samples
                and self.qd_lo is not None
                and self.qd_hi - self.qd_lo >= p.ready_lon_qd_span
                and self.vl_hi - self.vl_lo >= p.ready_lon_v_span)

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
        return Model(a0=a[0], a1=a[1], a2=a[2], b0=b[0], b1=b[1], b2=b[2],
                     n_lat=self.rls_lat.count, n_lon=self.rls_lon.count)

    # -- the step ----------------------------------------------------------

    def step(self, t, x, y, psi, cmd_v, cmd_w, applied=None):
        """Advance one odometry sample. Returns an :class:`Output`.

        ``t`` is seconds from the odometry stamp; the whole loop is driven by
        measured time so it is correct at any odometry rate.

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

        if self.phase != SENSE and dt > 5.0 * self.dt:
            # Odometry gap. The differenced sample is an average over the
            # gap, not a measurement: dropping it and re-seeding the virtual
            # sensors beats teaching the learner from it. The output is zero,
            # so the slew restarts from rest instead of stepping back to the
            # pre-gap command when odometry returns.
            self.now = t
            self.v_prev = self.v_fb = self.v_fast = v
            self._wire = None
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
        self.v = v
        tau = self.policy.v_fb_tau
        if tau > 0.0:
            self.v_fb += (v - self.v_fb) * (1.0 - math.exp(-dt / tau))
        else:
            self.v_fb = v
        # Fast filter for the launch latch only: enough smoothing to keep
        # noise from flickering it, little enough lag to catch breakaway.
        self.v_fast += (v - self.v_fast) * (1.0 - math.exp(-dt / 0.08))
        if not self.rolling:
            if abs(self.v_fast) > 0.6 * self.gate_d:
                self.rolling = True
                self._still_since = None
        else:
            if abs(self.v_fast) < 0.3 * self.gate_d:
                if self._still_since is None:
                    self._still_since = t
                elif t - self._still_since > 0.3:
                    self.rolling = False
            else:
                self._still_since = None

        # ---------- command filters: best estimate of actuator state -------
        p = self.policy
        if applied is not None and finite(*applied):
            us_app, ud_app = applied
            self._wire = (float(us_app), float(ud_app))
        else:
            us_app, ud_app = self.prev_us, self.prev_ud
            self._wire = None
        self.qs += (us_app - self.qs) * (1.0 - math.exp(-dt / p.tau_s))
        self.qd += (ud_app - self.qd) * (1.0 - math.exp(-dt / p.tau_d))

        # ---------- learn, gated by MEASURED noise ------------------------
        learning = self._learn(v)

        # ---------- PHASE 2: CAL ------------------------------------------
        if self.phase == CAL:
            us, ud = self._cal(elapsed)
            if elapsed > p.t_cal:
                self._finish_cal(t)
            return self._publish(us, ud, learning, False, dt)

        # ---------- PHASE 3: RUN ------------------------------------------
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
            self.gate_d = max(0.15, 10.0 * max(self.sigma_v, self.tick))
            # ...but never so high that the speeds Nav2 actually drives at
            # fall under the lateral gate (Policy.gate_s_max).
            self.gate_d = max(0.15, min(self.gate_d, p.gate_s_max / 1.6))
            self.gate_s = 1.6 * self.gate_d
            self.alpha = clamp(self.dt / 0.30, 0.05, 0.6)
            self.lam = math.exp(math.log(0.5) * self.dt / p.t_forget)
            self.v_prev = v
            self._enter(CAL if p.enable_calibration else RUN, t)
        return self._safe_output()

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
        # Longitudinal. Drag is b2*v*|v|, not b2*v**2: quadratic drag opposes
        # motion, so it must change sign in reverse.
        p = self.policy
        d_lon = self._delayed_cmd(p.lon_delay)
        d_lat = self._delayed_cmd(p.lat_delay)
        # Moving samples only: a stuck robot teaches "high throttle, zero
        # acceleration", which is stiction, not gain.
        if d_lon is not None and abs(self.vdot) < 12.0 and abs(v) > self.gate_d:
            if self.rls_lon.update([d_lon[1], 1.0, v * abs(v)],
                                   self.vdot, self.lam):
                ok = True
                self.qd_lo, self.qd_hi = _span(self.qd_lo, self.qd_hi,
                                               self.qd)
                self.vl_lo, self.vl_hi = _span(self.vl_lo, self.vl_hi, v)
        # Lateral. psidot/v is curvature; dividing by v amplifies noise, which
        # is why the speed gate is 1.6x the longitudinal one. There is
        # deliberately no lower gate on psidot: straight-line samples are what
        # identify the trim term a1, so excluding them would lose the
        # misalignment the trim exists to cancel.
        if d_lat is not None and abs(v) > self.gate_s and abs(self.psidot) < 4.0:
            kappa = self.psidot / v
            qs = d_lat[0]
            if self.rls_lat.update([qs, 1.0, qs * v * v],
                                   kappa, self.lam):
                ok = True
                self.qs_lo, self.qs_hi = _span(self.qs_lo, self.qs_hi,
                                               self.qs)
            # Same sample, second use: how well the model predicted a
            # near-lock command is the only honest source for the envelope.
            # Same regressor as the model, or the ratio compares a command the
            # car has not responded to yet against the curvature it is still
            # doing from an earlier one. With the lag-filtered qs here, every
            # lock transition logged a "ratio" near zero and the envelope
            # quoted 65% of the curvature the car actually has.
            self.envelope.observe(qs, kappa, v, self.model)
        return ok

    def _cal(self, tc):
        """Scripted wiggle. Time-based, so it is identical at any odom rate."""
        p = self.policy
        if tc < 2.0:
            return 0.0, 0.10 + 0.25 * tc
        us = (p.cal_steer * math.sin(2.0 * math.pi * 0.55 * tc)
              + 0.16 * math.sin(2.0 * math.pi * 0.19 * tc))
        return us, p.cal_drive

    def _finish_cal(self, t):
        """Size the dither from measurements, seen THROUGH the learned gain."""
        b0 = self.rls_lon.theta[0]
        self.dither = clamp(
            5.0 * max(self.tick, self.sigma_v) * 4.4 / max(abs(b0), 1.0),
            0.03, 0.12)
        self._enter(RUN, t)

    def _run(self, v, cmd_v, cmd_w, dt):
        """Invert the learned model and add integral trim."""
        p = self.policy
        a0, a1, a2 = self.rls_lat.theta
        b0, b1, b2 = self.rls_lon.theta

        # --- steering -----------------------------------------------------
        # sgn(v) is zero at standstill, which would divide by zero; fall back
        # to the commanded direction, then to forward.
        if abs(cmd_v) > p.v_eff_floor:
            v_eff = cmd_v
        else:
            direction = sgn(v) or sgn(cmd_v) or 1.0
            v_eff = direction * max(abs(v), p.v_eff_floor)

        # Ask for no more curvature than the vehicle is believed to have.
        # Without this the inversion winds the trim integrator up against a
        # saturated steering command it can never satisfy.
        kappa_max = self.envelope.max_curvature(self.model, v, derate=False)
        kappa_des = clamp(cmd_w / v_eff, -kappa_max, kappa_max)
        # psidot = v * kappa, so a yaw error converts to a curvature
        # correction by dividing by SIGNED v. Dividing by |v| (as the original
        # spec did) winds the trim the wrong way whenever the robot reverses:
        # the steering correction inverts exactly while driving backward.
        v_signed = sgn(v_eff) * max(abs(v_eff), 0.5)
        # Standstill steering authority (applied below); computed here because
        # the anti-windup has to know what "saturated" means right now: a
        # command pinned at the standstill clamp is just as unable to act on
        # a yaw error as one pinned at full lock.
        lim = p.steer_standstill + (1.0 - p.steer_standstill) \
            * clamp(abs(v) / max(self.gate_d, 1e-3), 0.0, 1.0)
        # Anti-windup. Integrating a yaw error the servo cannot act on (at
        # lock, or stationary) only winds the trim to its clamp; the flight
        # log had it pinned 90% of the time, biasing every turn one way and
        # unwinding slowly when the demand flipped -- "all left, all right".
        saturated = abs(self.prev_us) >= 0.95 * lim
        if not saturated and abs(v) > 0.5 * self.gate_d:
            self.iw = clamp(
                self.iw + p.ki_w * (cmd_w - self.psidot)
                / v_signed * dt, -p.iw_max, p.iw_max)
        # Physically impossible coefficients are clamped HERE rather than in
        # the estimator: projecting every RLS update biases the fit, but a
        # wrong-signed coefficient must never reach an actuator.
        #
        # A positive b2 means the model believes going faster makes you
        # accelerate harder. The inversion then brakes while cruising, the
        # robot drops below the motion gate, breakaway slams the throttle, and
        # it limit-cycles. That was b2 = +4.58 on the robot.
        a2 = min(a2, 0.0)     # cornering does not improve with speed
        b2 = min(b2, 0.0)     # drag opposes motion
        b0 = max(b0, 0.0)     # more throttle cannot mean less acceleration

        # Until the data has genuinely covered a range of steering commands,
        # the learned split of a0/a1/a2 is arbitrary -- steer through the
        # prior gain instead of inverting it. The learner keeps running the
        # whole time; only the INVERSION waits for evidence.
        if not self.ready_lat:
            a0, a1, a2 = p.prior_a0, 0.0, 0.0

        span = a0 + a2 * v * v
        if abs(a0) > p.den_min:
            self.steering_fault = False
            # Understeer may reduce the gain but not erase it, and certainly
            # not invert it -- an inverted denominator steers the wrong way.
            floor = p.span_floor * abs(a0)
            den = span if abs(span) >= floor and span * a0 > 0.0 \
                else sgn(a0) * floor
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
            den = p.prior_a0 if p.prior_a0 > p.den_min else 1.0
            if not self.steering_fault:
                self.get_fault_reset()
            self.steering_fault = True
        us = clamp((kappa_des + self.iw - a1) / den, -1.0, 1.0)
        # No full lock while stationary. At standstill the curvature request
        # w/v blows up, the servo goes to lock before the car rolls, and a
        # car cannot START with its front wheels cranked: it sits, the
        # integrator winds up, and it breaks free at 3x the throttle it
        # needed -- the lunge. Flight log: stuck-while-pushing ticks were at
        # lock 63% of the time; launches at lock released at 0.45-0.70,
        # straight-wheel launches at 0.24. Authority returns with speed.
        us = clamp(us, -lim, lim)

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
        direction = sgn(cmd_v) if abs(cmd_v) > p.stall_cmd_min else 0.0
        if direction and self._cmd_dir and direction != self._cmd_dir:
            self.iv = 0.0
        if direction:
            self._cmd_dir = direction

        err = cmd_v - v
        a_des = clamp(p.kp_v * err, -2.5, 2.5)
        self.iv = clamp(self.iv + p.ki_v * err * dt, -p.iv_max, p.iv_max)
        if not (self.ready_lon and p.use_learned_lon):
            # The learned longitudinal model is NOT inverted for control.
            # Every flight log shows the same thing: smooth while the throttle
            # runs on the fixed prior, lunging from the tick the learned model
            # takes over. The learner fits a LINEAR gain to a dead-band +
            # stiction actuator and settles at b0 ~ 1.0-1.7 against a real
            # free-rolling gain of ~5-6 -- it averages "high throttle, zero
            # acceleration" samples into a low gain, and the inversion then
            # divides by it. The model keeps being learned (diagnostics,
            # persistence); it just does not drive the wheels.
            ud = a_des / p.prior_b0
        elif abs(b0) > p.b0_min:
            ud = (a_des - b1 - b2 * v * abs(v)) / b0
        else:
            ud = 0.3 * sgn(err)
        if p.enable_dither and self.dither > 0.0:
            ud += self.dither * math.sin(2.0 * math.pi * 0.7 * self.now)
        ud = clamp(ud + self.iv, -1.0, 1.0)
        if (p.launch_floor > 0.0 and direction and not self.rolling
                and err * direction > 0.0):
            # Not rolling yet, motion is wanted AND the PI agrees it should
            # speed up that way: at least the floor, in the commanded
            # direction. The last condition is what keeps this from ever
            # turning a brake into throttle. See Policy.launch_floor.
            ud = direction * max(ud * direction, p.launch_floor)

        # A zero speed command means stop, not "servo to zero speed".
        if cmd_v == 0.0 and cmd_w == 0.0:
            self.iv = 0.0
            self.iw = 0.0
            return 0.0, 0.0
        return us, ud

    def get_fault_reset(self):
        """Entering a steering fault: re-open the lateral covariance.

        The collapsed gain was learned from thousands of consistent samples,
        so the estimator is confident and slow. If the actuator comes back,
        that confidence is exactly what would keep the model wrong.
        """
        self.rls_lat.inflate(self.policy.p0)

    def _reflex(self, us, ud, v, cmd_v, t, dt):
        """Blocked detection. No probe, no escalation, no measurement.

        "Stalled" is reported for diagnostics when motion is asked for and
        the latch says nothing is moving. If that persists with the
        integrator pinned at its limit, the robot is against something:
        output zero for a while rather than grind, then let it try again.
        """
        p = self.policy
        moving = self.rolling
        want = abs(cmd_v) > p.stall_cmd_min

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
                return us, 0.0, True

        if want and not moving:
            if self._stall_since is None:
                self._stall_since = t
            stalled = t - self._stall_since > p.stall_time
            pinned = abs(self.iv) >= p.iv_max - 1e-9
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
        ud = self._slew(self.prev_ud, ud, rate * dt)
        self.prev_us, self.prev_ud = us, ud
        # The history the learner aligns its regressors to is what was ON THE
        # WIRE. In PASSIVE that is the joystick, not this (unpublished) output.
        self._cmd_hist.append((self.now,) + (self._wire or (us, ud)))
        m = self.model
        return Output(steer=us, drive=ud, phase=self.phase, learning=learning,
                      stalled=stalled, v=self.v, psidot=self.psidot, dt=dt,
                      model=m,
                      min_turning_radius=self.envelope.min_turning_radius(m),
                      max_curvature=self.envelope.max_curvature(m),
                      envelope_confirmed=self.envelope.confirmed,
                      breakaway=self.breakaway,
                      steering_fault=self.steering_fault)

    @staticmethod
    def _slew(prev, target, limit):
        if limit <= 0.0:
            return target
        return prev + clamp(target - prev, -limit, limit)

    def _safe_output(self):
        self.prev_us = self.prev_ud = 0.0
        m = self.model
        return Output(phase=self.phase, v=self.v, psidot=self.psidot,
                      dt=self.dt, model=m,
                      min_turning_radius=self.envelope.min_turning_radius(m),
                      max_curvature=self.envelope.max_curvature(m),
                      envelope_confirmed=self.envelope.confirmed,
                      breakaway=self.breakaway,
                      steering_fault=self.steering_fault)

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
        span = abs(m.a0 + m.a2 * self.policy.env_speed ** 2)
        # Deliberately does NOT require a2 <= 0. That sign is enforced at the
        # inversion, and a poorly-excited run can leave it wrong while the
        # model is still perfectly usable -- demanding it here would mean
        # almost nothing ever persists.
        return (finite(span)
                and span > self.policy.den_min
                and not self.steering_fault)

    def lon_plausible(self):
        """More throttle means more acceleration, or the fit is not a vehicle.

        On the robot the longitudinal learner has been seen to settle at
        b0 < 0 (the dead-band actuator defeats the linear fit); an ``abs(b0)``
        test let that reach disk. Such a fit is replaced by the prior on save.
        """
        m = self.model
        return finite(m.b0) and m.b0 > self.policy.b0_min

    def state(self):
        """Everything worth carrying across a reboot.

        Version 2 adds the learner's readiness (sample counts and excitation
        spans). Without it a restored model was never ``ready_*`` until it had
        re-earned 60-80 fresh samples, so persistence changed nothing about
        what the controller inverted after a reboot.
        """
        p = self.policy
        lon_ok = self.lon_plausible()
        return {
            'version': 2,
            'lateral': list(self.rls_lat.theta),
            'longitudinal': (list(self.rls_lon.theta) if lon_ok
                             else [p.prior_b0, 0.0, 0.0]),
            'envelope': self.envelope.state(),
            'n_lat': int(self.rls_lat.count),
            'n_lon': int(self.rls_lon.count) if lon_ok else 0,
            'spans': {
                'qd': [self.qd_lo, self.qd_hi] if lon_ok else [None, None],
                'vl': [self.vl_lo, self.vl_hi] if lon_ok else [None, None],
                'qs': [self.qs_lo, self.qs_hi],
            },
            'sigma_v': self.sigma_v,
            'tick': self.tick,
        }

    def load_state(self, d, inflate=4.0):
        """Restore a saved model. Returns True if the whole thing was valid.

        The covariance is inflated rather than restored: the parameters are
        probably still right, but the tyres, the floor and the battery have all
        had a chance to change while the robot was off, so the learner should
        be readier to move than it was when it saved.

        Version 1 files are accepted for their parameters only. Their
        envelope evidence was measured against the lag-filtered command rather
        than the delayed one the model is fit on, which made it systematically
        pessimistic, so it is discarded and re-learned.
        """
        if not isinstance(d, dict) or d.get('version') not in (1, 2):
            return False
        lat, lon = d.get('lateral'), d.get('longitudinal')
        for vec in (lat, lon):
            if not isinstance(vec, list) or len(vec) != 3 or not finite(*vec):
                return False
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
        else:
            n_lat = n_lon = 0
            qd = vl = qs = (None, None)

        self.rls_lat.theta = list(lat)
        self.rls_lat.count = n_lat
        self.qs_lo, self.qs_hi = qs
        if lon[0] > p.b0_min:
            self.rls_lon.theta = list(lon)
            self.rls_lon.count = n_lon
            self.qd_lo, self.qd_hi = qd
            self.vl_lo, self.vl_hi = vl
        else:
            self.rls_lon.theta = [p.prior_b0, 0.0, 0.0]
            self.rls_lon.count = 0
            self.qd_lo = self.qd_hi = self.vl_lo = self.vl_hi = None
        p0 = min(p.p0 * inflate, p.p_max)
        for rls in (self.rls_lat, self.rls_lon):
            for i in range(rls.n):
                for j in range(rls.n):
                    rls.P[i][j] = p0 if i == j else 0.0
        return True

    def _running_sigma(self):
        return _stddev(self._v_samples) if len(self._v_samples) >= 3 else None


def _span(lo, hi, x):
    """Extend a running [lo, hi] range with a new sample."""
    if lo is None:
        return x, x
    return min(lo, x), max(hi, x)


def _load_span(v):
    """A persisted [lo, hi] range, or (None, None) if absent or malformed."""
    if isinstance(v, list) and len(v) == 2 and finite(*v) and v[0] <= v[1]:
        return float(v[0]), float(v[1])
    return None, None


def _stddev(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return math.sqrt(max(var, 0.0))
