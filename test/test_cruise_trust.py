"""Recorded 2026-09-07 cruise disagreement, exercised through the control law."""
import pytest
from ackermann_adaptive_controller.core import AdaptiveCore
from plant import Plant, settle_sense


def recorded():
    c = AdaptiveCore()
    settle_sense(c, Plant())
    c.rls_lon.theta = [2.963, -.387, 0., -1.011]
    c.rls_lon.count = 810
    c.qd_lo, c.qd_hi = -.24, .28
    c.vl_lo, c.vl_hi = -.55, .49
    c.v_op = .309
    for _ in range(c.policy.deadband_evidence):
        c.gain_probe.eq_fwd.add(.23)
        c.gain_probe.eq_rev.add(.22)
        c.deadband.observe(1., .256)
        c.deadband.observe(-1., .206)
    c.rolling = True
    assert c.ready_lon and c.lon_plausible()
    return c


def test_recorded_oversized_fit_uses_measured_cruise():
    c = recorded()
    assert not c._lon_feedforward_trusted(1., .309)
    c._run(.309, .309, 0., .1)
    assert c._ff_wire == pytest.approx(.23)
    assert 'bootstrap: measured equilibrium' in c.agreement()[1]


def test_reverse_fit_is_checked_independently():
    c = recorded()
    assert c._lon_feedforward_trusted(-1., -.309)
    c._run(-.309, -.309, 0., .1)
    assert c._ff_wire < 0.


@pytest.mark.parametrize('ratio', [2.01, 3.0])
def test_oversized_static_effort_rejected(ratio):
    c = recorded()
    c.rls_lon.theta = [2.963, 0., 0., -2.963*.23*ratio]
    assert not c._lon_feedforward_trusted(1., .309)


def test_restored_fit_does_not_step_cruise_output():
    c = recorded()
    c._roll_since = c.now - 5.
    c._run(.309, .309, 0., .1)
    before = c._ff_wire + c.iv
    c.rls_lon.theta = [2.963, 0., 0., -2.963*.25]
    c._run(.309, .309, 0., .1)
    assert c._lon_feedforward_trusted(1., .309)
    assert c._ff_wire + c.iv == pytest.approx(before, abs=1e-9)


def test_speed_dependent_drag_is_not_compared_to_cruise_median():
    c = recorded()
    c.rls_lon.theta = [2.963, 0., -4., -.1]
    assert c._lon_feedforward_trusted(1., .49)


def test_opposite_direction_evidence_does_not_veto_asymmetry():
    c = recorded()
    c.gain_probe.eq_fwd.vals.clear()
    assert c._lon_feedforward_trusted(1., .309)


def test_collinear_terms_with_correct_total_remain_trusted():
    c = recorded()
    # Large static term canceled by fitted quadratic: total is measured cruise.
    c.rls_lon.theta = [2.963, 0., (1.365 - 2.963*.23)/(.309**2), -1.365]
    assert c._lon_feedforward_trusted(1., .309)
