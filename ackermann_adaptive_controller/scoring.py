"""Driving quality metrics independent of ROS transport."""
from __future__ import annotations
import math
from collections import deque
from .response import _Window
from .math_utils import finite, clamp, sgn

class DriveScore:
    """How well the car follows its commands -- the score a self-tuning
    controller keeps of itself. A MONITOR only: nothing here feeds back
    into control. Until 08-31 none of these numbers existed anywhere but
    in offline reads of the flight log, so the node could limit-cycle
    for six minutes without anything on /diagnostics changing.

    Tracked at odometry rate over commanded ticks:

      err_rms   RMS speed error over the last WINDOW seconds
      launch    per start from rest: overshoot ratio (peak |v| over the
                command within a few actuation delays of reaching it) and
                time to reach the command; medians kept in a window
      cycles    surge-stall events: |v| went beyond SURGE x the command
                and then fell under STALL_FRAC x the command while the
                command persisted -- the lunge signature, counted
      stalls    stall-detector rising edges
      events    named counters the node feeds (direction holds, glitch
                holds, implausible episodes, dead-man trips, command
                timeouts), each with a per-minute rate
    """

    # Risk constants, all about what counts as an event -- none shape
    # the wire.
    # One cusp segment on the 08-31 drives lasts 6-11 s: the tracking
    # window is about one segment, so a bad segment shows and a good one
    # clears it.
    WINDOW = 10.0
    RATE_WINDOW = 60.0      # per-minute rates
    # 30% over the command is beyond anything a converged loop overshoots
    # by; a third of the command is not following, it is stalling.
    SURGE = 1.3
    STALL_FRAC = 0.3
    # A launch has "reached" the command at 80% of it: the last 20% is
    # where the loop hands over from feedforward to trim.
    REACHED = 0.8

    def __init__(self):
        self._errs = deque()          # (t, err^2) over commanded ticks
        self._cmds = deque()          # (t, |cmd|) same ticks
        self._curv = deque()          # (t, achieved/commanded curvature)
        self._seg_dir = 0.0
        self._seg_t0 = None
        self._seg_peak = 0.0
        self._surged = False
        self._launch = None           # dict while a launch is being scored
        self.launches = 0
        self.last_launch = None       # (overshoot, t_reach)
        self.launch_over = _Window(3, span=4)
        self.launch_time = _Window(3, span=4)
        self.cycles = 0
        self.stalls = 0
        self._was_stalled = False
        self._stamps = {'cycle': deque(), 'stall': deque()}
        self.events = {}

    def count(self, name, t):
        """An externally observed event (see class note)."""
        self.events[name] = self.events.get(name, 0) + 1
        self._stamps.setdefault(name, deque()).append(t)

    def _prune(self, t):
        for q in (self._errs, self._cmds, self._curv):
            while q and t - q[0][0] > self.WINDOW:
                q.popleft()
        for q in self._stamps.values():
            while q and t - q[0] > self.RATE_WINDOW:
                q.popleft()

    def observe(self, t, cmd_v, v, stalled, cmd_min, gate, horizon,
                cmd_w=0.0, psidot=0.0, kappa_max=0.0):
        """One tick. ``cmd_min`` is the command below which nothing is
        asked (stall_cmd_min), ``gate`` the motion gate (gate_d),
        ``horizon`` how long after reaching the command the launch
        overshoot may still peak (a few actuation delays)."""
        self._prune(t)
        d = sgn(cmd_v) if abs(cmd_v) > cmd_min else 0.0
        a, c = abs(v), abs(cmd_v)
        if d != self._seg_dir:
            # a new commanded segment (from rest, or a reversal)
            self._seg_dir = d
            self._seg_t0 = t
            self._seg_peak = 0.0
            self._surged = False
            self._close_launch()
            if d and a < gate:
                self._launch = {'t0': t, 'peak': 0.0, 'reached': None}
        if stalled and not self._was_stalled:
            self.stalls += 1
            self._stamps['stall'].append(t)
        self._was_stalled = bool(stalled)
        if not d:
            return
        err = cmd_v - v
        self._errs.append((t, err * err))
        self._cmds.append((t, c))
        # steering: curvature achieved over curvature commanded, on real
        # turns (a fifth of the envelope or more) while moving
        if kappa_max > 0.0 and a > gate:
            k_cmd = cmd_w / cmd_v
            if abs(k_cmd) >= 0.2 * kappa_max:
                self._curv.append((t, (psidot / v) / k_cmd))
        self._seg_peak = max(self._seg_peak, a)
        if self._launch is not None:
            L = self._launch
            L['peak'] = max(L['peak'], a)
            # the follower ramps its command up from a fraction; the
            # overshoot is judged against the LARGEST command the launch
            # window saw, not the one first reached (a 0.6 m/s peak on a
            # 0.07 opening command scored 8x on the 09-01 drive)
            L['cmd'] = max(L.get('cmd', 0.0), c)
            if L['reached'] is None and a >= self.REACHED * c:
                L['reached'] = t - L['t0']
                L['until'] = t + horizon
            elif L['reached'] is not None and t >= L['until']:
                self._close_launch()
        if a >= self.SURGE * c:
            self._surged = True
        elif self._surged and a < self.STALL_FRAC * c:
            self._surged = False
            self.cycles += 1
            self._stamps['cycle'].append(t)

    def _close_launch(self):
        L, self._launch = self._launch, None
        if L is None or L['reached'] is None:
            return
        over = L['peak'] / L['cmd'] if L['cmd'] > 0.0 else 0.0
        self.launches += 1
        self.last_launch = (over, L['reached'])
        self.launch_over.add(over)
        self.launch_time.add(L['reached'])

    @property
    def err_rms(self):
        if not self._errs:
            return None
        return math.sqrt(sum(e for _, e in self._errs) / len(self._errs))

    @property
    def err_rel(self):
        """RMS error as a fraction of the mean commanded speed."""
        r = self.err_rms
        if r is None or not self._cmds:
            return None
        mean = sum(c for _, c in self._cmds) / len(self._cmds)
        return r / mean if mean > 0.0 else None

    @property
    def curv_ratio(self):
        """Median achieved/commanded curvature over the window, or None."""
        if not self._curv:
            return None
        vals = sorted(r for _, r in self._curv)
        return vals[len(vals) // 2]

    def per_minute(self, name):
        q = self._stamps.get(name)
        return len(q) if q else 0

    def summary(self):
        """Plain-text pairs for /diagnostics."""
        r, rel = self.err_rms, self.err_rel
        track = ('idle' if r is None
                 else f'err_rms={r:.3f} ({rel * 100:.0f}% of cmd)')
        cr = self.curv_ratio
        if cr is not None:
            track += f' curv={cr:.2f}'
        if self.last_launch:
            over, t_r = self.last_launch
            launch = (f'last over={over:.2f}x reach={t_r:.1f}s '
                      f'median over={self.launch_over.value:.2f}x '
                      f'reach={self.launch_time.value:.1f}s n={self.launches}')
        else:
            launch = 'none yet'
        ev = ' '.join(f'{k}={v}({self.per_minute(k)}/min)'
                      for k, v in sorted(self.events.items()))
        return {
            'drive_score': (f'{track} cycles={self.cycles}'
                            f'({self.per_minute("cycle")}/min) '
                            f'stalls={self.stalls}'
                            f'({self.per_minute("stall")}/min)'),
            'launch': launch,
            'events': ev or 'none',
        }
