import math,json
from test_cruise_trust import recorded
from plant import Plant

def original(self,direction,speed):
    if not (self.policy.use_learned_lon and self.ready_lon and self.lon_plausible()):return False
    b0,b1,b2,b3=self.rls_lon.theta
    v=speed if self.vl_lo is None else min(max(speed,self.vl_lo),self.vl_hi)
    ff=-(b1+b2*v*abs(v)+b3*direction)/b0*direction
    eq=self.gain_probe.eq(direction)
    return math.isfinite(ff) and (eq is None or eq<=0 or ff>0)

for baseline in [True,False]:
 for direction in [1.,-1.]:
    c=recorded()
    if baseline:c._lon_feedforward_trusted=original.__get__(c)
    plant=Plant(b=(4.71,0.,-.35),tau_d=.3,pose_noise=0.)
    plant.kinetic=4.71*(.23 if direction>0 else .22)-.35*.309**2
    plant.delay=.2;plant.v=direction*.309;plant.qd=direction*(.23 if direction>0 else .22)
    c._roll_since=c.now-5.
    rows=[]
    for i in range(400):
        c.now+=.1
        _,wire=c._run(plant.v,direction*.309,0.,.1)
        for _ in range(5):plant.step(0.,wire,.02)
        rows.append((plant.v,wire))
    print(json.dumps({'baseline':baseline,'direction':direction,'rmse':math.sqrt(sum((v-direction*.309)**2 for v,w in rows)/len(rows)), 'peak_speed':max(abs(v) for v,w in rows),'final_speed':rows[-1][0],'zero_throttle':sum(abs(w)<.001 for v,w in rows)}))
