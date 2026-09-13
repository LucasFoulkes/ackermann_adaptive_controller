"""Bounded recursive identification and delay hypotheses."""
from __future__ import annotations
import math
from collections import deque
from .math_utils import finite, clamp, sgn

class RLS:
    """Exponentially-weighted recursive least squares, 3 parameters.

    Three textbook failure modes are guarded here. Covariance windup is bounded
    by ``p_max`` (the caller also gates uninformative samples out entirely);
    the gain denominator is floored so a zero regressor cannot divide by zero;
    and any non-finite sample is rejected outright rather than poisoning the
    estimate permanently.
    """

    def __init__(self, theta0, p0, p_max, bounds=None, absorber=None, valid_theta=None):
        self.valid_theta = valid_theta
        self.rejected = 0
        self.n = len(theta0)
        self.theta = list(theta0)
        self.P = [[p0 if i == j else 0.0 for j in range(self.n)]
                  for i in range(self.n)]
        self.p_max = p_max
        self.lam = 1.0
        self.count = 0
        self.innovation = 0.0
        # Parameter projection (Ioannou & Sun): ``bounds[i]`` is (lo, hi)
        # or None. A clipped parameter's excess is moved onto ``absorber``
        # (an index, or a tuple of candidate indices of which the first
        # with a nonzero regressor is used -- the steering cell in play,
        # for the lateral fit) scaled by the regressor ratio, so the
        # prediction for the sample that caused the clip is unchanged --
        # the constraint only redistributes a split the data could not
        # pin down. Clipping WITHOUT an absorber diverged the lateral fit
        # to a0l -40 (08-29 bench).
        self.bounds = list(bounds) if bounds else [None] * self.n
        self.absorber = (tuple(absorber) if isinstance(absorber, (tuple, list))
                         else (absorber,) if absorber is not None else ())

    def predict(self, phi):
        return sum(t * p for t, p in zip(self.theta, phi))

    def project(self, theta, phi):
        """Clip ``theta`` into ``bounds``; move each clipped excess onto the
        absorber so ``theta . phi`` is preserved (needs a nonzero absorber
        regressor -- the constant term's is 1)."""
        theta = list(theta)
        for i, b in enumerate(self.bounds):
            if b is None:
                continue
            lo, hi = b
            clipped = theta[i]
            if lo is not None and clipped < lo:
                clipped = lo
            if hi is not None and clipped > hi:
                clipped = hi
            if clipped == theta[i]:
                continue
            excess = theta[i] - clipped
            theta[i] = clipped
            for j in self.absorber:
                if j != i and phi[j] != 0.0 and finite(phi[i], phi[j]):
                    theta[j] += excess * phi[i] / phi[j]
                    break
        return theta

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

        previous_P = [row[:] for row in self.P] if self.valid_theta else None

        # DIRECTIONAL forgetting (Kulhavy & Karny 1984, "restricted
        # exponential forgetting"): discount the information only along
        # the direction this sample excites, then add the sample with no
        # discount. P <- P + ((1-lam)/lam) (P phi)(P phi)^T / (phi^T P phi),
        # which in information terms is R <- R - (1-lam) phi phi^T /
        # (phi^T P phi), followed by the usual R <- R + phi phi^T. Scalar
        # forgetting (R <- lam R + phi phi^T) discounts EVERY direction
        # every sample, so under one cruise speed -- where qd, 1, v|v| and
        # sgn(v) are collinear -- the unexcited directions lose their
        # information (covariance windup), the gain grows there, and every
        # sample's noise walks the fit along the null space: b0 4.29 ->
        # -2.2 with b1 rising to keep b0*qd + b1 fixed, in 60 s of a
        # synthetic 4 m/s^2 vehicle at cruise (08-29 bench), and the
        # b2 = +2.74 split of the 12:49 run. Here a direction that is not
        # being re-measured keeps what it learned, and the information
        # along the excited one settles at phi^T P phi ~ 1-lam instead
        # of growing without bound.
        if lam < 1.0:
            pphi0 = [sum(self.P[i][j] * phi[j] for j in range(self.n))
                     for i in range(self.n)]
            s = sum(a * b for a, b in zip(phi, pphi0))
            if finite(s) and s > 1e-12:
                c = (1.0 - lam) / (lam * s)
                for i in range(self.n):
                    for j in range(self.n):
                        self.P[i][j] += c * pphi0[i] * pphi0[j]
                lam = 1.0
                # (P + c u u^T) phi = u (1 + c s) with u = P phi
                pphi = [u * (1.0 + c * s) for u in pphi0]
            else:
                pphi = pphi0
        else:
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
        theta = self.project(theta, phi)
        if self.valid_theta is not None and not self.valid_theta(theta):
            # Reject the entire inconsistent update, including forgetting.
            # Clipping gain and moving the error into bias hides bad evidence.
            self.P = previous_P
            self.rejected += 1
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


class DelayBank:
    """Identical RLS estimators at candidate delays; the best-aligned drives.

    The command-to-response delay was measured by hand once (scanning lags
    against flight logs until r-squared peaked). This does the same thing
    continuously: every accepted sample updates all candidates, and control
    follows the one whose fitted GAIN is largest -- the cross-correlation
    peak. The regression coefficient of the response on a shifted copy of
    the same command is cov(y, u_d) / var(u), and var(u) is the same for
    every shift, so the largest coefficient is the best-aligned shift: a
    candidate that is early sees the response smeared and fits a smaller
    gain, one that is late sees it inverted and fits a negative one (a
    synthetic 4 m/s^2 vehicle at 1.0 s: 0.5 at 0.06 s, 1.8 at 1.0 s, 3.2
    at 1.5 s, -2.4 at 2.3 s). The gain is taken SIGNED against the
    incumbent's sign, so a late candidate's inverted gain loses rather
    than wins on magnitude, and an inverted servo (all candidates
    negative) is compared consistently.

    Why not the prediction error, the criterion this bank used until
    08-29: on a steady wire every candidate explains the data equally
    well (only the split of the gain against the bias differs), so the
    exponentially-weighted errors sat within 1-5% of each other, the
    hysteresis margin was never met, and whichever candidate the launch
    transient had favoured kept a misaligned fit whose gain had drifted
    toward zero. The measured delay on this robot (0.45 s lat, r-squared
    0.1 at lag 0 to 0.8 at 0.4-0.5 s) is exactly such a peak. A
    challenger must beat the incumbent by the margin (1/margin x its
    gain) and have as many samples as the model needs to be believed at
    all, so noise and a fresh candidate's prior cannot flap the alignment.
    """

    def __init__(self, theta0, p0, p_max, delays, ew_tau, margin,
                 bounds=None, absorber=None, gain_idx=(0,), min_count=0,
                 gains=None, fixed_signs=None, valid_theta=None):
        self.fixed_signs = fixed_signs
        self.delays = list(delays)
        self.prior = list(theta0)
        # ``gains(theta)`` -> per-cell IDENTIFIABLE gains, the veto on a
        # switch: the peak is ranked on the bare coefficients at gain_idx
        # (alignment shows there -- on a slow slalom the identifiable span
        # barely differs between candidates), but a challenger whose
        # identifiable gain is smaller than the incumbent's is a split
        # artefact, not a better alignment: after a fault re-opened every
        # covariance, the 2.28 s candidate once won on a0 alone with a0
        # high and a2 very negative -- the same span, a different address
        # (08-29 bench, recovery test). The lateral bank passes the span
        # a0 + a2 v_op^2 per cell; the default is the bare coefficients.
        self.gains = gains or (lambda th: [th[k] for k in gain_idx])
        self.bank = [RLS(list(theta0), p0, p_max, bounds, absorber, valid_theta)
                     for _ in self.delays]
        self.score = [None] * len(self.delays)
        self.active = len(self.delays) // 2
        self.ew_tau = ew_tau
        self.margin = margin
        self.gain_idx = tuple(gain_idx)
        self.min_count = min_count
        self._lead = (None, 0)     # (candidate, consecutive ticks leading)

    @property
    def rls(self):
        return self.bank[self.active]

    @property
    def delay(self):
        return self.delays[self.active]

    def aligned_gain(self, i, signs=None):
        """The candidate's fitted gain, signed against the incumbent."""
        if signs is None:
            signs = self._signs()
        th = self.bank[i].theta
        return sum(th[k] * s for k, s in zip(self.gain_idx, signs))

    def identifiable_gain(self, i, signs=None):
        if signs is None:
            signs = self._signs()
        return sum(g * s for g, s in zip(self.gains(self.bank[i].theta),
                                         signs))

    def _signs(self):
        if self.fixed_signs is not None:
            return self.fixed_signs
        # From the BARE cells, not the span: a span's sign flips when a2
        # is large, and signs taken from it hopped the bank between
        # candidates with inverted cells (08-29 bench).
        act = self.bank[self.active]
        return [sgn(act.theta[k]) or sgn(self.prior[k]) or 1.0
                for k in self.gain_idx]

    def update(self, phi_fn, y, lam, dt):
        """One sample for every candidate. Returns True if the active
        estimator accepted it. ``phi_fn(delay)`` builds the regressor from
        that candidate's delayed command, or None if history is too short.
        The exponentially-weighted prediction error is kept per candidate
        for diagnostics."""
        accepted = False
        a = clamp(dt / self.ew_tau, 0.0, 1.0)
        for i, (d, r) in enumerate(zip(self.delays, self.bank)):
            phi = phi_fn(d)
            if phi is None:
                continue
            inn = y - r.predict(phi)
            if r.update(phi, y, lam) and finite(inn):
                if i == self.active:
                    accepted = True
                e = inn * inn
                self.score[i] = e if self.score[i] is None \
                    else self.score[i] + a * (e - self.score[i])
        signs = self._signs()
        cur = self.aligned_gain(self.active, signs)
        best, best_gain = self.active, cur
        for i, r in enumerate(self.bank):
            if i == self.active or r.count < self.min_count:
                continue
            g = self.aligned_gain(i, signs)
            if finite(g) and g > best_gain:
                best, best_gain = i, g
        if best != self.active and finite(cur) \
                and best_gain * self.margin > cur:
            # The challenger must identify at least as much gain at the
            # operating speed as the incumbent (0.8x let the 0.06 s
            # candidate in with a0 inflated and a2 at -2.74, 08-29 drive),
            # and must have led for a full readiness window of ticks: a
            # lead born in one rotation artefact is not an alignment.
            span_cur = self.identifiable_gain(self.active, signs)
            span_best = self.identifiable_gain(best, signs)
            if finite(span_cur, span_best) and span_best >= span_cur:
                # A leaky net count: +1 per tick this candidate leads, -1
                # per tick it does not (never below zero), the switch at
                # min_count net ticks. Noise that flips the leader every
                # other tick never accumulates; a real alignment that
                # leads most ticks takes over within a readiness window.
                who, n = self._lead
                if who == best:
                    self._lead = (best, n + 1)
                elif n <= 1:
                    self._lead = (best, 1)
                else:
                    self._lead = (who, n - 1)
                if self._lead[0] == best and self._lead[1] >= max(self.min_count, 1):
                    self.active = best
                    self._lead = (None, 0)
                return accepted
        who, n = self._lead
        self._lead = (who, n - 1) if n > 1 else (None, 0)
        return accepted

    def inflate(self, p0):
        for r in self.bank:
            r.inflate(p0)

    def seed(self, theta, count, p0):
        """Restore a saved model into every candidate (they re-diverge)."""
        for r in self.bank:
            r.theta = list(theta)
            r.count = count
            for i in range(r.n):
                for j in range(r.n):
                    r.P[i][j] = p0 if i == j else 0.0

    def set_delay(self, delay):
        self.active = min(range(len(self.delays)),
                          key=lambda i: abs(self.delays[i] - delay))

    def peak(self):
        """The candidate at the cross-correlation peak, span-vetoed: the
        largest aligned gain among candidates with enough samples whose
        identifiable gain is not below the incumbent's. What CAL trusts at
        once (a designed excitation); RUN only follows a sustained lead."""
        signs = self._signs()
        cur_span = self.identifiable_gain(self.active, signs)
        best, best_gain = self.active, self.aligned_gain(self.active, signs)
        for i, r in enumerate(self.bank):
            if r.count < self.min_count:
                continue
            g = self.aligned_gain(i, signs)
            sp = self.identifiable_gain(i, signs)
            # Loose veto here (a tenth): at one speed every candidate's
            # a0/a2 split is arbitrary within noise, and the strict veto
            # RUN uses for a switch hid a genuine 1.0 s peak behind a
            # 3% smaller span. What it still blocks is the gross artefact
            # (a0 up 30%, a2 -2.74, span down: 08-29 16:16).
            if finite(g, sp) and g > best_gain and sp >= 0.9 * cur_span:
                best, best_gain = i, g
        return best

    def jump_to_peak(self):
        self.active = self.peak()
        self._lead = (None, 0)
