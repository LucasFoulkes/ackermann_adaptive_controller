"""Pure physical plausibility checks for learned throttle models."""
import math


def finite(*values):
    return all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) for value in values)


def lon_sane(b, vl_lo, vl_hi, policy, v_op, breakaway=None):
    """Is a longitudinal fit ``[b0, b1, b2, b3]`` physically a vehicle?

    Two signs are physics, not tuning: more throttle means more
    acceleration (``b0 > b0_min``), and holding a cruise takes throttle,
    not brake -- drag and friction oppose motion, a flat floor does not
    push. The second is judged where the inversion is actually evaluated:
    the feedforward wire ``-(b1 + b2*v|v| + b3*sgn(v)) / b0`` at the
    operating speed (``v_op``, clamped into the fitted span, per
    direction with evidence only -- a span that never reversed says
    nothing about reverse). It is judged on the SUM, never on b2 or b3
    alone (at one cruise speed only their sum is data-pinned: the
    b2-clamp lesson in _run), and NOT at the span's extreme: a lurch peak
    of 0.89 m/s once sat in the span while the data was pinned at 0.3,
    and a fit that is -0.005 wire at cruise read +1.0 m/s^2 there.

    Tolerance in wire units: ``launch_floor``, the constant already
    justified as below the lowest breakaway ever observed. A fit asking
    for less than -launch_floor of throttle to hold cruise is not fit
    noise. Why it exists: the model restored on 08-28 22:19 was
    ``[0.39, 0.04, +2.84, -0.04]`` -- b0 above the floor, so the old
    b0-only test let it through -- and its cruise feedforward was -0.44
    wire; the car sat at the launch floor and stalled 31% of the session.
    Flight-log audit (sum at 0.35 m/s): the 08-23 sessions violate the
    sign on 0-3% of ticks, the lagged-odometry sessions on 27-98%.
    """
    b0, b1, b2, b3 = b
    if not finite(b0, b1, b2, b3) or b0 <= policy.b0_min:
        return False
    # The wire that breaks the car free (the measured breakaway) must
    # produce at least the friction the fit claims, or the car could not
    # have started: b0 x breakaway >= |b3|, with a factor of two for the
    # breakaway sample reading high. A fit of b0 0.78 with b3 -0.53 says
    # full throttle is 0.78 m/s^2 -- it put the feedback gain at its floor
    # and every correction 4x too strong (08-29 17:29: rail-to-rail wire,
    # six stalls in 34 s). The 13:45 model: 4.64 x 0.22 = 1.02 >= 0.48.
    if breakaway is not None and finite(breakaway) and breakaway > 0.0 \
            and b0 * breakaway < 0.5 * abs(b3):
        return False
    tol = policy.launch_floor
    if not finite(v_op) or v_op < 0.0:
        return False
    if vl_hi is not None and finite(vl_hi) and vl_hi > 0.0:
        v = min(v_op, vl_hi)
        if -(b1 + b2 * v * v + b3) / b0 < -tol:
            return False
    if vl_lo is not None and finite(vl_lo) and vl_lo < 0.0:
        v = -min(v_op, -vl_lo)
        # reverse: the wire needed is negative; "less than -tol" mirrors
        if -(b1 + b2 * v * abs(v) - b3) / b0 > tol:
            return False
    return True
