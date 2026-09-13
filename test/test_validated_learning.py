"""Normal-driving validation: later measurements, not fit self-confidence."""
import math
import pytest
from ackermann_adaptive_controller.core import AdaptiveCore
from ackermann_adaptive_controller.policy import Policy
from ackermann_adaptive_controller.validated_identification import ValidatedDelayBank
from plant import Plant, settle_sense, drive


def bank():
    b = ValidatedDelayBank([2., 0., 0., 0.], 10., 1e4, [.1, .3], 30., .8,
                           min_count=5, fixed_signs=(1.,))
    b.published_delay = .3
    return b


def samples(b, actual, candidate, varying=True, reverse=True):
    b.pending = (tuple(candidate), .1, 100)
    for i in range(60):
        d = -1 if reverse and i % 2 else 1
        wire = d * (.2 + (.15*math.sin(i*.8) if varying else 0.))
        phi = [wire, 1., d*.3**2, d]
        y = sum(a*x for a,x in zip(actual,phi))
        b._evaluate(lambda delay: phi, y)


def test_good_frozen_fit_promotes_after_later_excited_data():
    b = bank();samples(b, [4.,0.,0.,-.8], [4.,0.,0.,-.8])
    assert b.promotions == 1 and b.delay == .1
    assert b.rls.theta == [4.,0.,0.,-.8]


def test_bad_reverse_cannot_hide_behind_good_forward():
    b = bank();samples(b, [4.,0.,0.,-.8], [4.,.7,0.,-1.5])
    assert b.promotions == 0 and b.rejections == 1
    assert b.delay == .3 and b.rls.theta == [2.,0.,0.,0.]


def test_cruise_only_or_one_direction_cannot_validate_global_gain():
    for kwargs in [{'varying':False}, {'reverse':False}]:
        b = bank();samples(b,[4.,0.,0.,-.8],[4.,0.,0.,-.8],**kwargs)
        assert b.promotions == 0


def test_training_does_not_mutate_frozen_control_or_pending_model():
    b = bank(); b.pending=((4.,0.,0.,-.8),.1,100)
    before=b.pending
    for _ in range(4):b.update(lambda d:[.4,1.,.09,1.], .8, .99, .1)
    assert b.pending == before and b.rls.count == 0 and b.delay == .3


def test_interruption_discards_trial_preserves_approved_model():
    b=bank();samples(b,[4.,0.,0.,-.8],[4.,0.,0.,-.8])
    b.pending=((10.,0.,0.,-2.),.3,200)
    b.interrupt_validation()
    assert b.pending is None and b.rls.theta[0] == 4. and b.delay == .1


def test_live_bad_cruise_fit_is_not_promoted():
    c=AdaptiveCore(Policy(validate_lon=True));settle_sense(c,Plant())
    for _ in range(30):
        c.gain_probe.eq_fwd.add(.21);c.gain_probe.eq_rev.add(.22)
    # Recorded run 70e5: positive gain but more than double forward cruise.
    assert not c._candidate_cruise_valid([2.421,-.291,-.074,-.863])
    assert c._candidate_cruise_valid([4.,0.,0.,-.86])


def test_old_model_is_training_only_and_survives_save_without_promotion():
    c=AdaptiveCore();settle_sense(c,Plant());s=c.state()
    s.update(longitudinal=[2.421,-.291,-.074,-.863], n_lon=470, lon_delay=.3)
    s['spans'].update(qd=[-.25,.27],vl=[-.36,.47])
    new=AdaptiveCore(Policy(validate_lon=True));assert new.load_state(s)
    assert new.rls_lon.count==0 and new.lon_bank.delay==.45
    assert new.lon_bank.bank[new.lon_bank.active].count==470
    saved=new.state();again=AdaptiveCore(Policy(validate_lon=True));assert again.load_state(saved)
    assert again.rls_lon.count==0
    assert again.lon_bank.bank[again.lon_bank.active].count==470
    assert again.rls_lat.count==new.rls_lat.count


def test_normal_driving_collects_candidates_without_calibration():
    c=AdaptiveCore(Policy(validate_lon=True));p=Plant()
    t=settle_sense(c,p)
    _,t=drive(c,p,lambda s: (.3+.12*math.sin(s*.5),.05*math.sin(s*.3)),60,t0=t)
    assert c.phase=='RUN' and not c.policy.enable_calibration
    assert max(r.count for r in c.lon_bank.bank)>c.policy.ready_lon_samples
    assert c.gain_probe.eq(1.) is not None
    assert abs(p.v)<.8 and c.odom_ok
    assert c.lon_bank.promotions==0  # No reverse evidence in this course.


def test_ordinary_bidirectional_driving_can_earn_a_model():
    c=AdaptiveCore(Policy(validate_lon=True));p=Plant();p.delay=.2;p.kinetic=.35
    t=settle_sense(c,p)
    _,t=drive(c,p,lambda s: ((1 if int(s//25)%2==0 else -1)*
             (.35+.12*math.sin(s*.65)),.06*math.sin(s*.3)),240,t0=t)
    assert c.lon_bank.promotions>=1 and c.ready_lon
    assert c.rls_lon.theta[0] == pytest.approx(p.b0,rel=.25)
    assert c.lon_plausible() and c.odom_ok and not c.policy.enable_calibration
    saved=c.state();assert saved['lon_validated']
    again=AdaptiveCore(Policy(validate_lon=True));assert again.load_state(saved)
    assert again.rls_lon.theta==c.rls_lon.theta
    assert again.lon_bank.delay==c.lon_bank.delay
    assert again.lon_bank.pending is None  # Never resume a partially scored trial.
