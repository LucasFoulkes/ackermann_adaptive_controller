"""Regressions from the recorded grass -> indoor transition on September 11."""
import math
import pytest
from ackermann_adaptive_controller.core import AdaptiveCore
from ackermann_adaptive_controller.policy import Policy
from ackermann_adaptive_controller.validated_identification import ValidatedSteeringBank
from ackermann_adaptive_controller.validated_identification import ValidatedDelayBank
from plant import Plant,settle_sense,drive


def test_measured_cruise_below_static_start_threshold_teaches_throttle():
    c=AdaptiveCore(Policy(validate_lon=True));settle_sense(c,Plant())
    c.now=20.;c._roll_since=5.;c.rolling=True;c._learning_after=0.
    c._dir_since=5.;c.v_op=.32;c.vdot=.02;c.psidot=.05;c._stall_since=None
    for _ in range(30):
        c.deadband.observe(1.,.342)
        c.gain_probe.eq_fwd.add(.217)
    c._delayed_cmd=lambda delay:(.3,.212)
    c._learn(.32)
    assert max(r.count for r in c.lon_bank.bank)>0
    # Stuck output with the same wire must still not train the moving model.
    before=[r.count for r in c.lon_bank.bank];c.rolling=False;c._stall_since=c.now
    c._learn(0.)
    assert [r.count for r in c.lon_bank.bank]==before


def test_recorded_wrong_sign_reverse_lock_cannot_replace_working_steering():
    c=AdaptiveCore(Policy(validate_lat=True));c.v_op=.32;c.steer_sign=1.
    b=c.lat_bank;before=b.rls.theta[:]
    bad=(1.925,2.513,2.006,.808,.226,-6.064)
    b.pending=(bad,.45,44579)
    b._evaluate(lambda delay:[0.,0.,0.,-.8,1.,-.8*.32**2],-.7)
    assert b.rejections==1 and b.rls.theta==before
    assert b.pending is None


def test_later_directional_measurements_promote_real_steering_change():
    b=ValidatedSteeringBank([1.3]*4+[0.,0.],10.,1e4,[.1,.3],30.,.8,min_count=5)
    candidate=[1.0,1.3,1.3,1.3,0.,0.]
    b.pending=(tuple(candidate),.1,100)
    for i in range(40):
        q=.4+.3*math.sin(i*.6)
        b._evaluate(lambda d:[q,0.,0.,0.,1.,q*.32**2],q)
    assert b.promotions==1 and b.rls.theta==candidate and b.delay==.1


def test_forward_evidence_cannot_rewrite_unobserved_reverse_cell():
    b=ValidatedSteeringBank([1.3]*4+[0.,0.],10.,1e4,[.1],30.,.8,min_count=5)
    b.pending=((1.,1.3,1.3,.5,0.,0.),.1,100)
    for _ in range(120):b._evaluate(lambda d:[.6,0.,0.,0.,1.,.06],.6)
    assert b.promotions==0 and b.rejections==1
    assert b.rls.theta[3]==1.3


def test_restart_keeps_working_steering_without_promoting_pending_fit():
    c=AdaptiveCore(Policy(validate_lat=True));settle_sense(c,Plant())
    c.lat_bank.restore_working([2.4,2.,2.2,1.6,-.01,-.5],45000,.3)
    c.lat_bank.pending=((1.,1.,1.,.5,.2,-5.),.1,99999)
    saved=c.state();new=AdaptiveCore(Policy(validate_lat=True))
    assert new.load_state(saved)
    assert new.rls_lat.theta==c.rls_lat.theta and new.lat_bank.delay==.3
    assert new.lat_bank.pending is None


def test_validated_throttle_keeps_short_hold_evidence_but_expires_old_surface():
    b=ValidatedDelayBank([2.,0.,0.,0.],10.,1e4,[.1],30.,.8,min_count=5)
    now=[0.];b.clock=lambda:now[0]
    candidate=(4.,0.,0.,-.8);b.pending=(candidate,.1,100)
    def sample(direction,i):
        wire=direction*(.25+.15*math.sin(i*.8));phi=[wire,1.,direction*.09,direction]
        b._evaluate(lambda delay:phi,sum(a*x for a,x in zip(candidate,phi)))
        now[0]+=.1
    for i in range(75):sample(1.,i)
    assert len(b.rows)==60 and b.promotions==0
    b.interrupt_validation(preserve=True);now[0]+=3.
    assert b.pending is not None
    for i in range(12):sample(-1.,i)
    assert b.promotions==1
    # An old surface cannot help a later one pass directional validation.
    b.pending=(candidate,.1,200)
    for i in range(30):sample(1.,i)
    now[0]+=61.
    for i in range(30):sample(-1.,i)
    assert b.promotions==1 and all(r[0]==-1 for r in b.rows)


def test_production_validation_learns_steering_without_radius_explosion():
    c=AdaptiveCore(Policy(validate_lon=True,validate_lat=True,launch_effort_limit=.85))
    p=Plant(pose_noise=.0001);p.delay=.3;p.deadband=.1;p.kinetic=.2
    t=settle_sense(c,p);radii=[]
    for k in range(8):
        direction=1 if k%4<2 else -1
        turn=1 if k%2 else -1
        _,t=drive(c,p,lambda s:(direction*(.3+.1*math.sin(s*.4)),turn*.23),15,t0=t)
        radii.append(c.envelope.min_turning_radius(c.model))
    assert c.lat_bank.promotions>=1 and max(radii)<1.5
    assert max(r.count for r in c.lon_bank.bank)>100
    assert c.odom_ok and not c.steering_fault


@pytest.mark.parametrize('direction',[-1.,1.])
def test_heavy_start_can_cross_old_ceiling_then_stop_normally(direction):
    c=AdaptiveCore(Policy(launch_effort_limit=.85,enable_dither=False));dt=.1;t=0.;x=v=q=0.
    for _ in range(30):t+=dt;c.step(t,x,0.,0.,0.,0.)
    for _ in range(30):
        c.deadband.observe(direction,.34)
        (c.gain_probe.eq_fwd if direction>0 else c.gain_probe.eq_rev).add(.227)
    peak=0.
    for _ in range(90):
        t+=dt;out=c.step(t,x,0.,0.,direction*.4,0.)
        for _ in range(10):
            q+=(out.drive-q)*(1-math.exp(-.01/.12))
            if abs(v)<.001 and abs(q)<.74:v=0.
            else:
                nv=v+.01*(4*(q-math.copysign(.45,v or q))-.3*v)
                v=0. if nv*v<0 and abs(q)<.74 else nv
            x+=v*.01
        peak=max(peak,abs(out.drive))
    assert abs(x)>.1 and peak>.7 and peak<=.85
    t+=dt;out=c.step(t,x,0.,0.,0.,0.,passive=True)
    assert out.drive==0.


def test_immovable_load_gets_bounded_effort_and_a_hold_not_infinite_push():
    c=AdaptiveCore(Policy(launch_effort_limit=.85));settle_sense(c,Plant(pose_noise=0.))
    for _ in range(30):
        c.deadband.observe(1.,.34);c.gain_probe.eq_fwd.add(.227)
    outputs=[]
    for _ in range(140):
        out=c.step(c.now+.1,0.,0.,0.,.4,0.);outputs.append((out.drive,c.blocked))
    assert max(d for d,b in outputs)<=.85
    assert any(b and d==0. for d,b in outputs)
    assert not c.ready_lon
