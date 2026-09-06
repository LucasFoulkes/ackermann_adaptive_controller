"""Vehicle response models and observed actuator capabilities."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from collections import deque
from .math_utils import finite, clamp, sgn, _stddev, sane_gain_cells

@dataclass
class Model:
    """A snapshot of what has been learned, for logging and diagnostics."""

    a0l: float = 0.0        # steering gain, forward, left  (qs > 0)
    a0r: float = 0.0        # steering gain, forward, right (qs < 0)
    a0l_rev: float = 0.0    # steering gain, reverse, left
    a0r_rev: float = 0.0    # steering gain, reverse, right
    a1: float = 0.0
    a2: float = 0.0
    b0: float = 0.0
    b1: float = 0.0
    b2: float = 0.0
    b3: float = 0.0
    n_lat: int = 0
    n_lon: int = 0

    @property
    def a0(self):
        """Symmetric-average FORWARD steering gain, for reporting only.
        Control and the envelope always use the per-cell gains."""
        return 0.5 * (self.a0l + self.a0r)

    def a0_cell(self, qs, v):
        """The gain for this steering side and travel direction."""
        if v >= 0.0:
            return self.a0l if qs >= 0.0 else self.a0r
        return self.a0l_rev if qs >= 0.0 else self.a0r_rev


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
        # The speed the envelope is quoted at when none is given: the
        # learned operating speed, kept current by the core. Zero (nothing
        # driven yet) quotes the kinematic lock, which is the right answer
        # for a car that has not moved.
        self.v_op = 0.0
        # The steering prior, SIGNED once the core has established the
        # servo's direction (AdaptiveCore.prior_a0); kept current by it.
        self.prior_a0 = policy.prior_a0

    @staticmethod
    def predict(model, qs, v):
        # v is SIGNED: the cell must match how the sample was driven.
        return qs * (model.a0_cell(qs, v) + model.a2 * v * v) + model.a1

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

    def max_curvature(self, model, v=None, derate=True, forward=False):
        """Curvature believed reachable at full lock, at the planning speed.

        ``derate=False`` skips the pre-evidence derating. The controller's own
        clamp must use it: derated, the clamp holds steering below the
        evidence threshold, so the envelope can never confirm and the clamp
        never lifts -- a loop the planner push (which stays derated) is not in.

        ``forward=True`` takes the worst of the two FORWARD cells only. For
        the follower's lookahead, not the planner: see the node's push.
        """
        v = self.v_op if v is None else v
        # Worst of ALL FOUR cells: Smac plans forward and reverse arcs with
        # ONE minimum_turning_radius, so the quoted envelope must be
        # feasible in whichever cell the maneuver lands in -- and on this
        # car reverse-left is the weakest (0.73x forward, measured).
        # SANITIZED cells: one poisoned near-zero cell once quoted the
        # planner a 7 m radius and every path became a looping star.
        a0l, a0r, a0l_rev, a0r_rev = sane_gain_cells(
            (model.a0l, model.a0r, model.a0l_rev, model.a0r_rev),
            self.prior_a0)
        vv = model.a2 * v * v

        def span(a0c):
            # Understeer may reduce the gain but not erase it (the same
            # floor _run inverts through): an a2 of -2.74 quoted the
            # planner 2.9 m on a car that turns in 0.57 (08-29 drive).
            eff = a0c + vv
            floor = self.p.span_floor * abs(a0c)
            return eff if abs(eff) >= floor and eff * a0c > 0.0 \
                else sgn(a0c) * floor

        cells = [abs(model.a1 + span(a0l)), abs(model.a1 - span(a0r))]
        if not forward:
            cells += [abs(model.a1 + span(a0l_rev)),
                      abs(model.a1 - span(a0r_rev))]
        extrapolated = min(cells)
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

    def min_turning_radius(self, model, v=None, forward=False):
        return 1.0 / self.max_curvature(model, v, forward=forward)

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


class DeadBand:
    """Throttle below which the wheels do not turn, learned from starts.

    One median window per direction (``_Window``: a glitch cannot move it,
    old evidence ages out). ``compensate`` maps a controller command onto
    the live part of the actuator range -- ``d + |u| * (1 - d)`` -- so the
    controller sees a linear motor. The applied offset is a fraction of the
    measured breakaway: it must never move the car by itself.
    """

    def __init__(self, policy):
        self.p = policy
        self.fwd = _Window(policy.deadband_evidence, span=4)
        self.rev = _Window(policy.deadband_evidence, span=4)
        # FAST starts (wheels turning within deadband_slow_start of the
        # wire coming on) are not measurements -- the wire was already
        # above the dead zone -- but each IS an upper bound, and their
        # minimum is the only estimate a vehicle whose launches are all
        # fast ever gets. The 08-31 00:12 from-zero run drove 6 minutes
        # with breakaway 0.00: the learned Coulomb feedforward put ~0.26
        # wire on instantly, every start broke free in 0.2-0.4 s, and the
        # slow-start rule starved the dead-band learner of every sample --
        # which in turn disabled the feedforward cap and the learned
        # launch floor. A slow start remains the real measurement and its
        # median overrides the bound whenever it exists.
        self.fwd_ub = _Window(policy.deadband_evidence, span=4)
        self.rev_ub = _Window(policy.deadband_evidence, span=4)

    def _win(self, direction):
        return self.fwd if direction > 0.0 else self.rev

    def _ub(self, direction):
        return self.fwd_ub if direction > 0.0 else self.rev_ub

    def observe_upper(self, direction, wire):
        """A fast start: the |wire| at breakaway bounds the band above."""
        if direction == 0.0 or not finite(wire) or not 0.0 < wire <= 1.0:
            return False
        self._ub(direction).add(wire)
        return True

    def raw(self, direction):
        """The undereated breakaway estimate: the slow-start median, or
        the smallest fast-start upper bound once there is evidence."""
        w = self._win(direction)
        if w.confirmed:
            return w.value
        ub = self._ub(direction)
        return min(ub.vals) if len(ub.vals) >= self.p.deadband_evidence \
            else None

    def observe(self, direction, wire):
        """The |wire| throttle that was on when the wheels first turned."""
        if direction == 0.0 or not finite(wire) or not 0.0 < wire <= 1.0:
            return False
        self._win(direction).add(wire)
        return True

    def confirmed(self, direction):
        return self._win(direction).confirmed

    def value(self, direction):
        """Offset applied in this direction; 0 until there is evidence."""
        r = self.raw(direction)
        if r is None:
            return 0.0
        return clamp(self.p.deadband_trust * r, 0.0, self.p.deadband_max)

    def lowest(self, direction):
        """The lowest breakaway seen in this direction (slow starts and
        fast-start upper bounds alike), or None. Every sample reads high,
        so this is an UPPER bound -- what a bootstrap floor must stay
        under."""
        vals = list(self._win(direction).vals) + list(self._ub(direction).vals)
        return min(vals) if vals else None

    def compensate(self, ud, motion=0.0):
        """Map a controller command onto the live part of the actuator.

        ``motion`` is the sign of the car's current travel (0 at rest).
        The dead band is a property of STARTING TORQUE: it must be crossed
        to make the wheels turn in the commanded direction. A command
        AGAINST the current motion is braking, and braking a rolling car
        needs no offset -- any opposite-sign PWM on the H-bridge is
        retarding torque from the first count. Applying the reverse
        offset there made the bootstrap path a relay: 08-28 22:35, rolling
        forward at 0.36 m/s on a 0.32 command, a -0.03 PI correction left
        the map as -0.2..-0.36 of reverse torque, the car stopped in 0.4 s
        (-1.0 m/s^2), the PI asked for +0.3, the car surged to 0.6 m/s
        (+1.3 m/s^2) -- a 1.1 s limit cycle that the learner then fit as
        b0 = -5, which disabled the inversion that would have replaced
        the bootstrap. Braking passes through linearly.
        """
        if ud == 0.0:
            return 0.0
        if motion and sgn(ud) != sgn(motion):
            return ud
        d = self.value(sgn(ud))
        return sgn(ud) * (d + abs(ud) * (1.0 - d))

    def state(self):
        return {'fwd': list(self.fwd.vals), 'rev': list(self.rev.vals),
                'fwd_ub': list(self.fwd_ub.vals),
                'rev_ub': list(self.rev_ub.vals)}

    def load(self, d):
        if not isinstance(d, dict):
            return False
        for key, win in (('fwd', self.fwd), ('rev', self.rev)):
            vals = d.get(key)
            if not isinstance(vals, list) or not all(
                    finite(v) and 0.0 < v <= 1.0 for v in vals):
                return False
            win.vals = deque(vals[-win.vals.maxlen:], maxlen=win.vals.maxlen)
        # upper-bound windows: optional, files predate them
        for key, win in (('fwd_ub', self.fwd_ub), ('rev_ub', self.rev_ub)):
            vals = d.get(key)
            if isinstance(vals, list) and all(
                    finite(v) and 0.0 < v <= 1.0 for v in vals):
                win.vals = deque(vals[-win.vals.maxlen:],
                                 maxlen=win.vals.maxlen)
        return True


class GainProbe:
    """Direct wire-to-acceleration measurement, free of the regression.

    The RLS throttle fit can be junk for minutes from cold: its own
    exclusion gates (overspeed, settling) reject exactly the transients a
    misbehaving loop produces, so a wrong fit and a hot loop sustain each
    other. The 08-31 01:08 from-zero run limit-cycled at STEADY command
    for six minutes that way -- surge to 0.5 m/s on a 0.30 command, wire
    cut, friction stall, relaunch, 1.7 s period -- while b0 sat at a junk
    0.63 on a plant whose trace measures ~5 (accel ~1.5 m/s^2 over
    friction at 0.28 wire; coast decel ~1.0 m/s^2).

    So the two numbers the CONTROL LAW actually needs are measured here
    as themselves, by medians (one glitch cannot move them, old surfaces
    age out), from the actuator state reconstructed by passing the
    delayed wire through the declared actuator constant:

      eq   per direction, the wire that HOLDS a speed (acceleration
           within measurement noise while rolling) -- the bootstrap
           feedforward, observed directly instead of reconstructed from
           a fit's b1/b2/b3 split.
      b0   the local wire-to-acceleration slope, from PAIRS of samples a
           few actuator constants apart: differencing cancels friction
           AND any dead-zone offset, the two things a single-sample
           quotient would have to guess at.
    """

    def __init__(self, policy):
        self.p = policy
        # Same evidence count as the dead band, for the same reason: a
        # handful of honest samples before anything downstream leans on
        # the number.
        self.eq_fwd = _Window(policy.deadband_evidence, span=4)
        self.eq_rev = _Window(policy.deadband_evidence, span=4)
        self.gain = _Window(policy.deadband_evidence, span=4)
        self._qd = None            # reconstructed actuator state
        self._t = None
        self._hist = deque(maxlen=64)   # (t, qd, vdot) accepted samples

    def eq(self, direction):
        """The cruise wire toward ``direction``, or None. Coulomb
        friction is near-symmetric, so the other side's measured cruise
        wire is a better bootstrap than nothing when this side has never
        cruised."""
        first = self.eq_fwd if direction > 0.0 else self.eq_rev
        second = self.eq_rev if direction > 0.0 else self.eq_fwd
        for w in (first, second):
            if w.confirmed:
                return w.value
        return None

    @property
    def b0(self):
        """Median measured wire gain (m/s^2 per wire), or None -- also
        None while the median itself is not positive (a car does not
        slow under throttle; a window that says so is noise)."""
        if not self.gain.confirmed:
            return None
        b = self.gain.value
        return b if b > 0.0 else None

    def observe(self, t, direction, ud_delayed, vdot, dt, at_eq, step_min):
        """One rolling, settled, physics-plausible tick.

        ``ud_delayed`` is what the wire carried one estimated transport
        delay ago; the first-order lag below turns it into the actuator
        state the plant is actually responding to, so a mid-ramp tick
        pairs the acceleration with the torque that caused it rather
        than with a command still in flight. ``step_min`` is the
        smallest actuator step a slope may be taken over -- the caller
        sizes it from the measured acceleration noise (see _learn).
        """
        p = self.p
        if self._t is not None and t - self._t > 3.0 * dt:
            # a gap (stall, reversal, implausible ticks): the lag state
            # and any pending pair partner are stale
            self._qd = None
            self._hist.clear()
        self._t = t
        q = self._qd
        q = ud_delayed if q is None \
            else q + (ud_delayed - q) * (1.0 - math.exp(-dt / p.tau_d))
        self._qd = q
        if q * direction > 0.0 and at_eq:
            (self.eq_fwd if direction > 0.0 else self.eq_rev).add(abs(q))
        # Theil-Sen, one pair per tick: the current sample against the
        # oldest one within a few actuator constants. Far enough apart
        # that the actuator state has genuinely moved, near enough that
        # the speed, battery and patch of floor are the same maneuver.
        pair = None
        for pt, pq, pv in self._hist:
            if t - pt <= 4.0 * p.tau_d:
                pair = (pt, pq, pv)
                break
        self._hist.append((t, q, vdot))
        if pair is None:
            return
        _pt, pq, pv = pair
        if abs(q - pq) < step_min:
            return
        b = (vdot - pv) / (q - pq)
        if b > 0.0:
            # a negative slope is noise or a mis-paired transient (a
            # stiction release, an odometry wobble), not a car that slows
            # under throttle. Keeping only the positive half biases the
            # median HIGH under noise -- the safe direction: a divisor
            # too large answers slowly, one too small multiplies the loop
            # (signed medians sat near zero on a noisy bench and ran a
            # 2 m/s vehicle unstable). The noise-scaled step_min is what
            # keeps the bias small (09-01 drive: 5.6 vs 4.7 true).
            self.gain.add(b)

    def state(self):
        return {'eq_fwd': list(self.eq_fwd.vals),
                'eq_rev': list(self.eq_rev.vals),
                'gain': list(self.gain.vals)}

    def load(self, d):
        if not isinstance(d, dict):
            return False
        # File-corruption guard only: live samples are already gated by
        # the acceleration plausibility bound at observation time, so the
        # windows never legitimately hold anything near these limits.
        # The slope window is NOT restored: it re-measures within the
        # first launches, and a persisted one carried a superseded
        # sampling rule's bias straight across a restart (09-01 21:11:
        # 10.8 restored, the fix inert until the window aged out).
        for key, win in (('eq_fwd', self.eq_fwd), ('eq_rev', self.eq_rev)):
            vals = d.get(key)
            if not isinstance(vals, list) or not all(
                    finite(v) and 0.0 < v < 1000.0 for v in vals):
                return False
            win.vals = deque(vals[-win.vals.maxlen:], maxlen=win.vals.maxlen)
        return True
