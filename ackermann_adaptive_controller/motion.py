"""Signed motion estimation from timestamped body poses."""
from __future__ import annotations
import math
from .math_utils import finite, wrap

class TwistEstimator:
    """Estimate signed body speed and yaw rate from timestamped rear-axle poses."""

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
        # Project the chord on the interval's midpoint heading. The old
        # start-heading projection underestimated speed during a turn; wrap
        # the increment first so crossing +/-pi does not flip direction.
        dpsi = wrap(psi - self.psi)
        heading = self.psi + .5 * dpsi
        v = (dx * math.cos(heading) + dy * math.sin(heading)) / dt
        psidot = dpsi / dt
        self.t, self.x, self.y, self.psi = t, x, y, psi
        if not finite(v, psidot):
            return None
        self.v, self.psidot = v, psidot
        return dt, v, psidot
