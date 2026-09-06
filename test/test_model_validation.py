"""A predictor must earn its score on separate informative observations."""
import numpy as np
from ackermann_adaptive_controller.model_validation import compare,_fit


def test_stationary_log_cannot_claim_an_adaptive_improvement():
    result=compare([dict(stamp=i*.1,v=0.,cmd_v=0.) for i in range(100)])
    assert result['status']=='no commanded motion'


def test_constant_inputs_do_not_identify_gain_and_bias():
    x=np.tile([.3,.2,1.],(100,1))
    assert _fit(x,np.ones(100)) is None


def test_timestamp_resets_require_a_separate_session():
    assert compare([dict(stamp=t) for t in [0.,1.,.5]])['status'].startswith('non-monotonic')


def test_validation_selects_delay_without_training_on_final_observations():
    rows=[];v=.3
    for i in range(1600):
        t=i*.1;q=.25+.12*np.sin(.21*t)+.06*np.sin(.83*t)
        s=.5*np.sin(.47*t)
        delayed=rows[max(0,i-4)]['qs'] if rows else 0.
        rows.append(dict(stamp=t,v=v,cmd_v=.3,qd=q,qs=s,
                         psidot=v*(1.8*delayed+.03),stalled=0,blocked=0,fault=0))
        v+=.1*(-.8*v+1.2*q)
    result=compare(rows,delays=[0.,.4,.8])
    model=result['models']['forward/yaw_fit']
    assert model['status']=='scored' and model['delay']==.4
    changed=[dict(r) for r in rows]
    for row in changed:
        if row['stamp']>=result['split_stamps'][1]:row['psidot']+=1.
    altered=compare(changed,delays=[0.,.4,.8])['models']['forward/yaw_fit']
    assert altered['delay']==model['delay']
    assert altered['coefficients']==model['coefficients']
    assert altered['test_rmse']>model['test_rmse']+.5
