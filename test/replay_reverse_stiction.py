import json, math
from ackermann_adaptive_controller.core import AdaptiveCore, Policy

def trial(direction=-1., kinetic=.30, static=.45):
    core=AdaptiveCore(Policy(enable_dither=False,validate_lon=True))
    dt=1/7.5;t=0.;x=0.;v=0.;q=0.;rows=[]
    for _ in range(30):
        t+=dt;core.step(t,x,0.,0.,0.,0.)
    core.v_op=.3
    for _ in range(core.policy.deadband_evidence):
        core.gain_probe.eq_fwd.add(.225);core.gain_probe.eq_rev.add(.225)
        core.deadband.observe(direction,.30)
    for i in range(225):
        t+=dt;out=core.step(t,x,0.,0.,direction*.25,0.)
        for _ in range(10):
            h=dt/10;q+=(out.drive-q)*(1-math.exp(-h/.12))
            if abs(v)<.001 and abs(q)<static:v=0.
            else:
                force=4*(q-math.copysign(kinetic,v or q))-.3*v
                nv=v+h*force
                v=0. if nv*v<0 and abs(q)<static else nv
            x+=v*h
        rows.append([i*dt,v,out.drive,core.iv,int(out.stalled)])
    cruising=rows[75:]
    return dict(moving_fraction=sum(abs(r[1])>.03 for r in cruising)/len(cruising),
                mean_speed=sum(abs(r[1]) for r in cruising)/len(cruising),
                peak_speed=max(abs(r[1]) for r in rows))

if __name__=="__main__":
    print(json.dumps({str((d,k)):trial(d,k) for k in [.225,.26,.30] for d in [-1.,1.]},indent=2))
