# Copyright 2026 Lucas Foulkes
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""A synthetic vehicle mirroring the model structure the learner assumes.

Written to be obviously correct by reading rather than shared with
:mod:`core`: if the two ever shared a bug, the convergence tests would pass on
a model that does not describe the robot.
"""

import math
import random


class Plant:
    """Bicycle-ish plant with first-order actuators and quadratic drag."""

    def __init__(self, a=(1.30, 0.04, -0.30), b=(2.20, 0.0, -0.35),
                 tau_s=0.18, tau_d=0.30, pose_noise=0.0005, seed=7):
        self.a0, self.a1, self.a2 = a
        self.b0, self.b1, self.b2 = b
        self.tau_s, self.tau_d = tau_s, tau_d
        self.qs = self.qd = 0.0
        self.x = self.y = self.psi = self.v = 0.0
        self.rng = random.Random(seed)
        self.pose_noise = pose_noise

    kinetic = 0.0     # constant decel opposing motion while rolling
    delay = 0.0       # pure command-to-actuator delay, seconds
    deadband = 0.0    # dead zone OFFSET: torque ~ (|qd| - deadband), 0 below

    def step(self, us, ud, dt):
        if self.delay > 0.0:
            if not hasattr(self, '_dq'):
                self._dq = []
            self._dq.append((us, ud))
            n = max(1, int(round(self.delay / dt)))
            if len(self._dq) > n:
                us, ud = self._dq.pop(0)
            else:
                us, ud = 0.0, 0.0
        self.qs += (us - self.qs) * (1.0 - math.exp(-dt / self.tau_s))
        self.qd += (ud - self.qd) * (1.0 - math.exp(-dt / self.tau_d))

        # Offset-type dead zone, the physical form: the first `deadband` of
        # the command range produces nothing, the rest is linear from zero.
        qd_eff = math.copysign(max(abs(self.qd) - self.deadband, 0.0),
                               self.qd)
        vdot = self.b0 * qd_eff + self.b1 + self.b2 * self.v * abs(self.v)
        if self.v != 0.0 and self.kinetic:
            vdot -= math.copysign(self.kinetic, self.v)
        v_new = self.v + vdot * dt
        if self.kinetic and v_new * self.v < 0.0 and abs(ud) < 0.05:
            v_new = 0.0            # friction stops it, does not reverse it
        self.v = v_new

        kappa = self.a0 * self.qs + self.a1 + self.a2 * self.qs * self.v ** 2
        psidot = self.v * kappa
        self.psi += psidot * dt
        self.x += self.v * math.cos(self.psi) * dt
        self.y += self.v * math.sin(self.psi) * dt

    def observe(self):
        """Pose as the odometry source would report it: noisy, no twist."""
        n = self.pose_noise
        return (self.x + self.rng.gauss(0.0, n),
                self.y + self.rng.gauss(0.0, n),
                self.psi + self.rng.gauss(0.0, n))


def settle_sense(core, plant, dt=0.1, seconds=3.0):
    """Run the stationary SENSE phase to completion. Returns the end time."""
    t = 0.0
    n = int(seconds / dt)
    for _ in range(n):
        t += dt
        x, y, psi = plant.observe()
        core.step(t, x, y, psi, 0.0, 0.0)
    return t


def drive(core, plant, cmd, seconds, t0=0.0, dt=0.1, substeps=5):
    """Closed-loop run. ``cmd`` is a callable t -> (v_star, omega_star)."""
    t = t0
    out = None
    n = int(seconds / dt)
    for _ in range(n):
        cv, cw = cmd(t - t0)
        x, y, psi = plant.observe()
        out = core.step(t, x, y, psi, cv, cw)
        # The Pico holds the command between odometry samples.
        for _ in range(substeps):
            plant.step(out.steer, out.drive, dt / substeps)
        t += dt
    return out, t
