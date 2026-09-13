"""Identification failures from the interrupted September 8 drive."""
import copy
import pytest
from ackermann_adaptive_controller.core import AdaptiveCore
from plant import Plant, settle_sense


def test_throttle_update_cannot_invert_gain_or_change_covariance_on_rejection():
    c = AdaptiveCore()
    r = c.lon_bank.rls
    theta, covariance, count = r.theta[:], copy.deepcopy(r.P), r.count
    assert not r.update([1., 1., 0., 1.], -100., .99)
    assert r.theta == theta and r.P == covariance and r.count == count
    assert r.rejected == 1


def test_throttle_delay_does_not_learn_polarity_from_bad_incumbent():
    c = AdaptiveCore()
    c.lon_bank.rls.theta[0] = -.669  # Recorded failed fit.
    assert c.lon_bank._signs() == (1.,)


def test_derivative_uses_time_since_last_accepted_speed():
    c = AdaptiveCore(); settle_sense(c, Plant())
    t = c.now + .1
    c.v_prev = .2; c._v_prev_t = t - .3; c.vdot = 0.
    c.twist.update = lambda *args: (.1, .23, 0.)
    c._speed_up_limit = lambda *args: None
    c._a_limit = lambda: 100.
    c.step(t, 0., 0., 0., 0., 0.)
    assert c.vdot == pytest.approx(c.alpha * .1)


def test_interruption_preserves_models_but_discards_transient_learning():
    c = AdaptiveCore(); settle_sense(c, Plant())
    theta = c.rls_lon.theta[:]
    c.gain_probe.eq_fwd.add(.21)
    c.gain_probe._hist.append((c.now, .2, 1.))
    c.pause_learning(c.now)
    assert not c._learn(.3)
    assert c.rls_lon.theta == theta
    assert list(c.gain_probe.eq_fwd.vals) == [.21]
    assert not c.gain_probe._hist


def test_rejected_saved_throttle_fit_does_not_restore_its_delay():
    c = AdaptiveCore(); settle_sense(c, Plant())
    saved = c.state()
    saved['longitudinal'] = [-.669, -.375, -.008, -.218]
    saved['lon_delay'] = 1.01
    saved['lat_delay'] = .45
    fresh = AdaptiveCore()
    assert fresh.load_state(saved)
    assert fresh.lon_bank.delay == pytest.approx(fresh.policy.lon_delay)
    assert fresh.lat_bank.delay == pytest.approx(.45)


def test_sanitized_saved_fit_drops_unearned_delay():
    c = AdaptiveCore(); settle_sense(c, Plant())
    c.lon_bank.set_delay(1.01)
    c.rls_lon.theta = [-.669, -.375, -.008, -.218]
    saved = c.state()
    assert saved['n_lon'] == 0
    assert saved['lon_delay'] == pytest.approx(c.policy.lon_delay)
    # Older versions saved the bad delay alongside the sanitized prior.
    saved['lon_delay'] = 1.01
    fresh = AdaptiveCore()
    assert fresh.load_state(saved)
    assert fresh.lon_bank.delay == pytest.approx(fresh.policy.lon_delay)
