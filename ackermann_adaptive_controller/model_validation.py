"""Offline, chronological model comparison. Never publishes or changes a profile.

Fits on the first 60% of usable observations in a recorded drive, chooses delay on the next 20%,
and reports prediction errors on the untouched final 20%. These are prediction
checks on noisy, closed-loop data, not evidence of stable vehicle control.
Logged qs/qd are reconstructed actuator states, not measured wheel positions.
"""
import argparse
import json
import math
import numpy as np
from .flight_report import DEFAULT_LOG, read_log, sessions
from .core import Policy


def _fit(x, y):
    if len(x) < 10*x.shape[1]:  # ten observations per coefficient, not readiness by sample count alone
        return None
    scale=np.sqrt(np.mean(x*x,axis=0))
    if np.any(scale == 0):
        return None
    a=x/scale
    coef,_,rank,sv=np.linalg.lstsq(a,y,rcond=None)
    # Reject effectively indistinguishable coefficients (condition > 1e4).
    if rank < x.shape[1] or sv[-1] < sv[0]*1e-4:
        return None
    return coef/scale


def compare(rows, delays=None):
    if len(rows) < 3:
        return {'status':'insufficient observations'}
    p=Policy()
    if delays is None:
        delays=sorted(set([p.lat_delay,p.lon_delay,*np.geomspace(p.delay_min,p.delay_max,9)]))
    delays=np.asarray(delays,dtype=float)
    if not len(delays) or not np.isfinite(delays).all() or np.any(delays<0):
        raise ValueError('delays must be finite and nonnegative')
    keys=('stamp','v','psidot','qd','qs','cmd_v','stalled','blocked','fault')
    a={k:np.array([r.get(k,0.) for r in rows],dtype=float) for k in keys}
    t=a['stamp']; dt=np.diff(t)
    if not np.isfinite(t).all() or np.any(dt<=0):
        return {'status':'non-monotonic timestamps; select a single run'}
    median_dt=float(np.median(dt))
    commanded=np.abs(a['cmd_v']); moving=commanded[commanded>0]
    if not len(moving):
        return {'status':'no commanded motion','run_id':rows[0].get('_run_id','')}
    # Same vehicle-relative principle as the live learner: avoid near-rest
    # ratios and separate startup from the moving-response models.
    gate=p.gate_floor_frac*float(np.median(moving))
    v=a['v']; direction=np.sign(v)
    valid=np.ones(len(t),dtype=bool)
    for values in a.values(): valid &= np.isfinite(values)
    valid &= (np.abs(v)>gate)&(commanded>gate)
    valid &= (a['stalled']==0)&(a['blocked']==0)&(a['fault']==0)
    segment=np.cumsum(np.r_[True,(dt>2*median_dt)|(direction[1:]!=direction[:-1])|(~valid[:-1])])
    # Common observations for all delays: no candidate receives easier data.
    lookbacks=[np.searchsorted(t,t-delay,side='right')-1 for delay in delays]
    usable=valid.copy();usable[-1]=False
    usable[:-1] &= valid[1:]&(dt<=2*median_dt)&(direction[:-1]==direction[1:])
    for index in lookbacks:
        safe=np.clip(index,0,len(t)-1)
        usable &= (index>=0)&(segment[safe]==segment)&valid[safe]
    indices=np.flatnonzero(usable)
    if len(indices)<3:
        return {'status':'insufficient uninterrupted motion across candidate delays'}
    # Split by eligible observations in time order; a long parked tail must
    # not consume the whole validation/test portion of an otherwise useful run.
    cut1=t[indices[int(.6*len(indices))]];cut2=t[indices[int(.8*len(indices))]]
    result={'status':'evaluated','run_id':rows[0].get('_run_id',''),
            'split_stamps':[float(cut1),float(cut2)],'speed_gate':gate,
            'measurement':'logged estimated v/yaw and reconstructed qs/qd',
            'limitations':['prediction comparison, not control validation',
                           'rank/sign screening does not certify physical coefficients',
                           'terrain and driver-applied commands are not recorded here'],
            'models':{}}
    for sign,label in [(1,'forward'),(-1,'reverse')]:
        ii=indices[direction[indices]==sign]
        # Purge delay history across split boundaries to keep held-out inputs
        # entirely within their own time block.
        train=t[ii+1]<cut1
        validation=(t[ii]>=cut1+max(delays))&(t[ii+1]<cut2)
        test=t[ii]>=cut2+max(delays)
        for kind in ('curvature_fit','yaw_fit','acceleration_fit','velocity_arx'):
            name=label+'/'+kind
            output={'counts':dict(train=int(train.sum()),validation=int(validation.sum()),test=int(test.sum()))}
            result['models'][name]=output
            if min(train.sum(),validation.sum(),test.sum())<10:
                output['status']='insufficient motion in separate time blocks';continue
            best=None
            for delay,lookback in zip(delays,lookbacks):
                jj=lookback[ii];speed=v[ii];step=t[ii+1]-t[ii]
                if kind in ('curvature_fit','yaw_fit'):
                    q=a['qs'][jj]
                    phi=np.column_stack((np.maximum(q,0),np.minimum(q,0),np.ones(len(ii)),q*speed**2))
                    target=a['psidot'][ii]
                    x=phi if kind=='curvature_fit' else phi*speed[:,None]
                    y=target/speed if kind=='curvature_fit' else target
                else:
                    q=a['qd'][jj];target=v[ii+1]
                    if kind=='acceleration_fit':
                        x=np.column_stack((q,np.ones(len(ii)),speed*np.abs(speed)))
                        y=(target-speed)/step
                    else:
                        # Fixed-interval ARX is inappropriate across gaps. Use
                        # its variable-dt form with linear drag: v+=dt*(a*v+b*q+c).
                        x=step[:,None]*np.column_stack((speed,q,np.ones(len(ii))))
                        y=target-speed
                coef=_fit(x[train],y[train])
                if coef is None: continue
                if kind=='acceleration_fit' and (coef[0]<=0 or coef[2]>0): continue
                if kind=='velocity_arx' and (coef[1]<=0 or coef[0]>0): continue
                if kind in ('curvature_fit','yaw_fit') and coef[0]*coef[1]<=0: continue
                prediction=x@coef
                if kind=='curvature_fit':prediction*=speed
                if kind=='acceleration_fit':prediction=speed+step*prediction
                if kind=='velocity_arx':prediction+=speed
                errors=(prediction-target)**2
                score=float(np.mean(errors[validation]))
                if best is None or score<best[0]:best=(score,float(delay),coef,errors)
            if best is None:
                output['status']='unidentifiable or implausible fit';continue
            score,delay,coef,errors=best
            baseline=(a['psidot'][ii]**2 if kind in ('curvature_fit','yaw_fit')
                      else (v[ii+1]-v[ii])**2)
            output.update(status='scored',delay=delay,coefficients=coef.tolist(),
                          validation_rmse=math.sqrt(score),test_rmse=math.sqrt(float(np.mean(errors[test]))),
                          baseline_test_rmse=math.sqrt(float(np.mean(baseline[test]))),
                          units='rad/s' if kind in ('curvature_fit','yaw_fit') else 'm/s')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',nargs='?',default=DEFAULT_LOG)
    parser.add_argument('--last-drive',action='store_true',help='explicitly select the last run with commanded motion')
    args=parser.parse_args()
    _,rows=read_log(args.path);runs=sessions(rows)
    if args.last_drive:runs=[r for r in runs if any(abs(x.get('cmd_v',0))>0 for x in r)]
    result=compare(runs[-1]) if runs else {'status':'no matching run'}
    if runs:
        result.setdefault('run_id',runs[-1][0].get('_run_id',''))
        result.update(run_start=runs[-1][0]['stamp'],run_end=runs[-1][-1]['stamp'])
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
