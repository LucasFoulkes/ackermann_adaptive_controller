"""Numerical validation and shared scalar operations."""
from __future__ import annotations
import math

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


def sane_gain_cells(cells, prior):
    """Steering gain cells with any minority-SIGN cell replaced.

    One rack and one servo drive all four (direction x side) cells, so a
    single cell with the opposite sign is not a vehicle property -- it is a
    poisoned fit (08-23: shuffle transients taught fwd-right -0.09 while
    the other three sat at 1.4-1.9; the worst-cell envelope then quoted the
    planner a 7 m turning radius and fwd-right steering INVERTED). The vote
    is taken only among cells that have moved off the prior -- cells still
    at their seed carry no evidence and must not outvote a learner that
    legitimately discovered an inverted servo (all-negative is consistent
    and believed).

    A TIE is poison too, not a vehicle: two cells cannot say the servo is
    wired one way while the other two say the opposite. Under the old
    "believe a tie as it stands" rule a 2-2 split (a0l +2.17, a0r -4.55,
    a0lr -2.50, a0rr +4.32, learned from lagged odometry on 08-28 01:01)
    passed plausible(), was persisted, was restored on every launch that
    day and steered every forward right turn LEFT; it recurred at 22:14
    (+0.73, -3.54, -2.79, +2.33) and was restored again at 22:19. A tie
    resolves to the sign the linkage was installed with (the prior's):
    an inverted servo has to win a majority, as it always did.
    """
    voters = [c for c in cells if abs(c - prior) > 0.1 * abs(prior)]
    pos = sum(1 for c in voters if c > 0.0)
    neg = sum(1 for c in voters if c < 0.0)
    if pos == neg:
        if pos == 0:
            return list(cells)          # no evidence either way
        s = 1.0 if prior > 0.0 else -1.0
    else:
        s = 1.0 if pos > neg else -1.0
    out = [c if c * s > 0.0 else s * abs(prior) for c in cells]
    # One rack, one servo: the four cells differ by geometry and caster
    # (0.73x measured), not by an order of magnitude. A cell under a
    # quarter of the median of the other three is a collapsed fit, not
    # a vehicle (a0r 0.15 against 2.43 / 2.18 / 1.66 after 34 s on a
    # poisoned model, 08-29 17:29 -- and the envelope quotes the WORST
    # cell to the planner). It is replaced by that median; the fraction
    # is Policy.span_floor's, the same "reduced but not erased" floor.
    # Siblings are EXCITED cells only (off the prior, as for the vote): a
    # dead servo collapses the driven cells while the undriven ones sit
    # at the prior, and those must not rescue it (the fault must fire).
    for i, c in enumerate(out):
        others = sorted(abs(out[j]) for j in range(4)
                        if j != i and abs(out[j] - s * abs(prior)) > 0.1 * abs(prior))
        if not others:
            continue
        med = others[len(others) // 2]
        if med > 0.0 and abs(c) < 0.25 * med:
            out[i] = s * med
    return out


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
