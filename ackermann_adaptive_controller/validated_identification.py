"""Frozen throttle hypotheses tested on later, ordinary driving samples."""
from collections import deque
import math

from .identification import DelayBank, RLS


class ValidatedSteeringBank(DelayBank):
    """Keep a working steering map while scoring a frozen candidate later.

    Shared bias and speed terms affect every steering cell. A candidate must
    therefore demonstrate improvements in observed cells and earn evidence in
    any other cell whose full-lock prediction it would materially change.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.published = RLS(self.prior, 1., 1.)
        self.published_delay = self.delays[self.active]
        self.pending = None
        self.rows = deque(maxlen=120)
        self.promotions = self.rejections = 0
        self.status = 'collecting steering evidence'
        self.model_valid = lambda theta: True
        self.operating_speed = lambda: .32

    @property
    def rls(self):
        return self.published

    @property
    def delay(self):
        return self.published_delay

    def interrupt_validation(self):
        self.pending = None
        self.rows.clear()
        self.status = 'waiting for fresh steering measurements'

    def restore_working(self, theta, count, delay):
        self.published.theta = list(theta)
        self.published.count = count
        self.published_delay = self.delays[min(range(len(self.delays)),
                                              key=lambda i: abs(self.delays[i]-delay))]
        self.interrupt_validation()

    def _evaluate(self, phi_fn, y):
        if self.pending is None:
            return
        theta, delay, count = self.pending
        if not self.model_valid(theta):
            self._finish(False, 'rejected invalid steering geometry')
            return
        phi, base = phi_fn(delay), phi_fn(self.published_delay)
        if phi is None or base is None:
            return
        # Do not compare different travel/steering cells across a reversal.
        cell = max(range(4), key=lambda i: abs(phi[i]))
        if abs(phi[cell]) < .15 or base[cell]*phi[cell] <= 0:
            return
        pred = sum(a*b for a,b in zip(theta,phi))
        old = self.published.predict(base)
        if not all(math.isfinite(x) for x in (y,pred,old,*phi)):
            return
        self.rows.append((cell, (y-pred)**2, (y-old)**2))
        if len(self.rows) < 24:
            self.status = 'testing steering on later driving'
            return
        groups = [[r for r in self.rows if r[0]==i] for i in range(4)]
        # Speed-zero and operating-speed endpoints catch shared-term changes
        # that fit one reverse turn but break forward geometry.
        affected = set()
        for i in range(4):
            sign = 1 if i%2==0 else -1
            for speed in (0., self.operating_speed()):
                old_k = sign*(self.published.theta[i]+self.published.theta[5]*speed**2)+self.published.theta[4]
                new_k = sign*(theta[i]+theta[5]*speed**2)+theta[4]
                if abs(new_k-old_k) > .15*max(abs(old_k),.1):
                    affected.add(i)
        tested = {i for i,g in enumerate(groups) if len(g)>=8}
        if not tested or not affected.issubset(tested):
            self.status = 'waiting for affected steering directions'
            if len(self.rows)==self.rows.maxlen:
                self._finish(False, 'insufficient independent directional evidence')
            return
        # Every tested direction must improve; many forward samples cannot
        # hide a regression in reverse.
        good = all(sum(r[1] for r in groups[i]) <
                   .8*sum(r[2] for r in groups[i]) for i in tested)
        self._finish(good, 'validated steering promoted' if good else
                     'steering candidate rejected; retaining working map')

    def _finish(self, good, status):
        if good:
            theta, delay, count = self.pending
            self.published.theta = list(theta)
            self.published.count = count
            self.published_delay = delay
            self.promotions += 1
        else:
            self.rejections += 1
        self.pending = None
        self.rows.clear()
        self.status = status

    def update(self, phi_fn, y, lam, dt):
        self._evaluate(phi_fn,y)
        accepted = super().update(phi_fn,y,lam,dt)
        candidate = self.bank[self.active]
        if self.pending is None and candidate.count >= self.min_count:
            self.pending = (tuple(candidate.theta),self.delays[self.active],candidate.count)
        return accepted


class ValidatedDelayBank(DelayBank):
    """Training never directly changes the coefficients or delay used by control.

    The pending model is copied before its evaluation samples arrive. Candidate
    and incumbent predict the same later outputs. Both travel directions and
    changes of delivered throttle must be observed before a global fit is used.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.published = RLS(self.prior, 1., 1.)
        self.published_delay = self.delays[self.active]
        self.pending = None
        self.rows = deque(maxlen=120)
        self.promotions = self.rejections = 0
        self.status = 'collecting normal driving'
        self.last_errors = None
        self.cruise_check = lambda theta: True
        self.clock = lambda: 0.

    @property
    def rls(self):
        return self.published

    @property
    def delay(self):
        return self.published_delay

    def interrupt_validation(self, preserve=False):
        # A brief tracking hold invalidates new measurements, not the already
        # scored, independent samples. Expire those by time when driving resumes.
        if not preserve:
            self.pending = None
            self.rows.clear()
        self._lead = (None, 0)
        self.status = 'waiting for fresh measurements'

    def _evaluate(self, phi_fn, y):
        if self.pending is None:
            return
        theta, delay, count = self.pending
        phi = phi_fn(delay)
        base_phi = phi_fn(self.published_delay)
        if phi is None or base_phi is None:
            return
        pred = sum(a*b for a, b in zip(theta, phi))
        base = self.published.predict(base_phi)
        if not all(math.isfinite(x) for x in (y, pred, base, *phi)):
            return
        direction = 1 if phi[3] > 0 else -1
        now = self.clock()
        self.rows = deque((r for r in self.rows if 0 <= now-r[5] <= 60.), maxlen=120)
        # A long forward leg must not evict all reverse evidence. Each direction
        # keeps its latest 60 measurements; both still have to pass independently.
        same = [i for i,r in enumerate(self.rows) if r[0] == direction]
        if len(same) >= 60:
            del self.rows[same[0]]
        self.rows.append((direction, phi[0], (y-pred)**2, (y-base)**2, y*y, now))
        if len(self.rows) < 40:
            self.status = 'testing on later driving'
            return
        groups = [[r for r in self.rows if r[0] == d] for d in (1, -1)]
        # Steady cruise can identify equilibrium, but cannot validate motor gain
        # or delay. Require a modest relative wire span in EACH direction.
        if any(len(g) < 6 or max(r[1] for r in g)-min(r[1] for r in g)
               < max(.03, .25*max(abs(r[1]) for r in g)) for g in groups):
            self.status = 'waiting for ordinary speed changes / reverse evidence'
            return
        ratios = []
        for g in groups:
            e, old, zero = (sum(r[k] for r in g)/len(g) for k in (2, 3, 4))
            ratios.append((e, old, zero))
        self.last_errors = ratios
        # Beat both the working model and the zero-acceleration cruise predictor.
        # Per-direction tests prevent abundant forward samples hiding bad reverse.
        good = all(e < .8*max(old, 1e-12) and e < .8*max(zero, 1e-12)
                   for e, old, zero in ratios) and self.cruise_check(theta)
        if good:
            self.published.theta = list(theta)
            self.published.count = count
            self.published_delay = delay
            self.promotions += 1
            self.status = 'validated model promoted'
        else:
            self.rejections += 1
            self.status = 'candidate rejected; retaining working model'
        self.pending = None
        self.rows.clear()

    def update(self, phi_fn, y, lam, dt):
        # Score BEFORE training on this sample. The pending copy never mutates.
        self._evaluate(phi_fn, y)
        accepted = super().update(phi_fn, y, lam, dt)
        candidate = self.bank[self.active]
        if self.pending is None and candidate.count >= self.min_count:
            self.pending = (tuple(candidate.theta), self.delays[self.active], candidate.count)
        return accepted
