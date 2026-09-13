"""Replay the incident's acceleration gate only, not a physical closed loop.
Initial accepted speed, noise and deadband come from incident diagnostics.
"""
import json,sys
from ackermann_adaptive_controller.core import AdaptiveCore
def replay(path):
    c=AdaptiveCore();c.now=75.708;c.v_prev=.247;c.gate_d=.1;c.sigma_v=.0052
    c.rolling=True;c._roll_since=74.8;c.lon_bank.set_delay(.20)
    for _ in range(30):c.deadband.observe(1., .252/c.policy.deadband_trust)
    accepted=75.708;held_since=None;held=0;longest=0
    for row in json.load(open(path)):
     t,kind,*v=row
     if kind=='wire':c.record_command(t,*v);continue
     limit=c._speed_up_limit(t-accepted,v[0]);c.now=t
     if limit is not None and v[0]-abs(c.v_prev)>limit:
      held+=1;held_since=t if held_since is None else held_since
      longest=max(longest,t-held_since)
     else:
      c.v_prev=v[0];accepted=t;held_since=None
    return {'held_samples':held,'longest_hold_s':round(longest,3),'final_accepted_speed':round(c.v_prev,3)}

if __name__ == "__main__":
    print(json.dumps(replay(sys.argv[1])))
