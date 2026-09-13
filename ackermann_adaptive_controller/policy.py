"""Vehicle learning and control policy defaults."""
from __future__ import annotations
from dataclasses import dataclass, field
import math

@dataclass
class Policy:
    """The only hand-chosen numbers. These encode risk, not the vehicle.

    Two kinds live here, and the distinction is what lets the same code
    drive a different vehicle. RISK constants (watchdogs, evidence counts,
    margins, per-second loop gains) are dimensionless or in seconds and
    carry over unchanged. SCALE-RELATIVE constants are fractions of a
    learned quantity -- ``v_op``, the operating speed the controller sees
    itself being driven at (see :meth:`AdaptiveCore._update_scale`) -- or
    ratios of one of the two declared priors, ``prior_a0`` and
    ``prior_b0``. Nothing here is in m/s, 1/m or m/s^2 any more: the
    2026-08-29 audit found ten such numbers (a 0.15 m/s learning-gate
    floor, a 0.24 m/s cap, a 0.12 m/s curvature floor, 0.25/8 m radius
    bounds, a 0.12/m trim clamp, a 0.05 m/s "motion wanted" threshold, a
    0.15 m/s speed-span requirement, gain floors of 0.05 and 0.3) that
    together pinned the controller to a 0.3 m/s, 0.28 m-wheelbase car: a
    0.1 m/s crawler could never accept a single learning sample, and an
    8 m-radius vehicle had its true envelope clamped. Each was replaced by
    the fraction that reproduces this robot's hand value at its measured
    scale (test_derived_values_reproduce_this_robots_hand_tuning).
    """

    # Phase timing.
    t_sense: float = 2.0
    # CAL is staged, each stage exiting on its own evidence (AdaptiveCore
    # _cal): signs (the steering-sign witness locks), gains (the lateral
    # fit is ready to be inverted), then a brake and one bounded reverse
    # pulse so the Coulomb term b3 and the bias b1 are separable at all
    # (one-sided data cannot split them) and the reverse cells and dead
    # band get their first samples. t_cal is the TIMEOUT per stage, the
    # 7 s the whole one-shot wiggle used to run for; cal_reverse the
    # length of the reverse pulse (about a metre on this robot).
    t_cal: float = 7.0
    cal_reverse: float = 2.0
    cal_steer: float = 0.30
    cal_drive: float = 0.42
    enable_calibration: bool = False

    # Enable staged throttle-model validation; robot bringup opts in.
    validate_lon: bool = False
    validate_lat: bool = False

    # Learner.
    p0: float = 10.0          # initial skepticism
    t_forget: float = 170.0   # s half-life: how fast the world changes
    p_max: float = 1.0e4      # covariance windup bound

    # Actuator time constants: DECLARED priors of the actuator class (a
    # hobby servo, a brushed motor behind an H-bridge; tested tau in
    # [0.1, 0.75]), alongside prior_a0 / prior_b0. Not learned: the delay
    # bank's cross-correlation peak already lands on the EFFECTIVE delay
    # (transport plus the first-order smear), so the fits are aligned
    # regardless; these only size the settling windows and the command
    # filters, and learning them would mean a second bank dimension for a
    # second-order effect. A much slower actuator shortens those windows
    # relative to the truth -- a degradation, not a failure.
    tau_s: float = 0.18
    tau_d: float = 0.30
    # Loop gains, DERIVED from the loop's dead time (AdaptiveCore.kp_v,
    # ki_v, ki_w): the learned delay plus the actuator constant plus the
    # feedback filter. A P loop on an integrating plant with dead time L
    # is well damped at kp = 1/(2L) (IMC with the closure time equal to
    # the dead time) and limit-cycles past kp*L ~ 1.57. This robot: L_lon
    # 0.45 + 0.30 + 0.10 = 0.85 s, kp 0.59 -- the 0.6 typed here until
    # 08-29, which was sized by hand to "~0.35 s of delay" and had
    # limit-cycled 0..0.9 m/s on a 0.32 command at 1.6 (kp*L 1.4).
    kp_delay_product: float = 0.5
    # Integral time as a multiple of L. SIMC puts it at 8L for an
    # integrating process; this integrator doubles as the stiction ramp
    # (see _run) and at 8L a launch took three times longer, so 2.8L -- the
    # 2.4 s (ki 0.25) that has been quiet since 08-23.
    ti_delay_ratio: float = 2.8
    # Curvature trim: an integral-only loop, yaw-rate error converted to a
    # curvature correction by dividing by the speed floored at v_op (so the
    # per-second loop gain is ki_w * v / v_op, at most ki_w). Against
    # L_lat = delay + servo constant + yaw filter (0.45 + 0.18 + 0.30 =
    # 0.93 s here): at an effective 0.9/s a single 0.45 rad/s step wound
    # the trim to its clamp before the car had responded (08-23 log, pinned
    # 30% of the time with a sign agreeing with the command only half the
    # time -- noise, not trim); 0.2/s has been quiet since. 0.19 / 0.93.
    kw_delay_product: float = 0.19
    # Smoothing of the differenced sensor signals (vdot, psidot), in
    # seconds: fixed bandwidth so it is the same at any odometry rate. It
    # is the "yaw filter" in L_lat.
    sensor_tau: float = 0.30

    # Priors. Without these, steering never excites (no steering -> no yaw ->
    # nothing to learn from), so RUN could never bootstrap without CAL.
    # a0 ~ tan(delta_max)/wheelbase (1.25: wheelbase 0.2775 m, ~18 deg lock);
    # b0 ~ full-throttle acceleration. Only starting points: the steering
    # envelope is learned and supersedes a0; b0 is the throttle gain the
    # controller runs on while use_learned_lon is off. Setting them closer
    # to the truth only shortens the transient.
    prior_a0: float = 1.25
    prior_b0: float = 2.0

    # Evidence required before the learned model may replace the prior in the
    # inversion. Sample count alone is not evidence: a hundred samples at one
    # speed leave b1/b2 (and a1/a2) unidentifiable, and inverting that split
    # is where the wild throttle came from. The spans demand that the data
    # actually covered a range of commands and speeds.
    ready_lon_samples: int = 80
    ready_lon_qd_span: float = 0.15
    # Speed coverage, as a fraction of the operating speed (0.15 m/s of
    # this robot's 0.32): the fit must have seen the car at speeds spread
    # over nearly half its range before b1/b2 are separable at all.
    ready_lon_v_span_frac: float = 0.45
    ready_lat_samples: int = 60
    ready_lat_qs_span: float = 0.40
    # Steering-sign establishment. The witness is the sign of the running
    # sum of (delayed steering command x measured curvature) over accepted
    # lateral samples with SUBSTANTIAL steering (|qs| of at least half
    # ready_lat_qs_span: at trim-level wire the curvature is the trim a1,
    # not the gain, and a straight drive with -0.02 of wire once witnessed
    # an inverted servo on a correctly wired plant). The raw
    # cross-correlation never passes through a prior; the fitted cells
    # do -- on an inverted servo they travel from +prior through zero, and
    # a vote taken on the way locked the wrong sign in the 08-29 bench
    # run. Locked once sign_evidence such samples agree by at least
    # sign_agreement of their total magnitude: 0.5 is three parts agreeing
    # to one dissenting, which a shuffle transient (one poisoned cell of
    # four) cannot reach. The CAL wiggle yields ~28 such samples at 10 Hz
    # (5 s of wiggle, the settling window and the |qs| gate taken out), so
    # 20 lets a fresh vehicle leave CAL with its sign known.
    sign_evidence: int = 20
    sign_agreement: float = 0.5

    # Inversion guards and limits.
    # Floor on the speed used to convert a yaw-rate command to curvature,
    # as a fraction of the operating speed. MUST sit well below it: at a
    # typed 0.4 m/s with Nav2 cruising at 0.22 the floor dominated and the
    # controller delivered 55% of the curvature the path follower asked
    # for -- the follower kept asking harder, the trim integrator wound to
    # its clamp and overshot: weaving. 0.35 x 0.32 m/s is the 0.12 that
    # has been quiet since.
    v_eff_floor_frac: float = 0.35
    # Gain floors, as fractions of the declared priors: a steering gain
    # below den_min_frac x prior_a0 is a dead actuator, not a vehicle
    # (0.04 x 1.25 = 0.05/m per unit command); a throttle gain below
    # b0_min_frac x prior_b0 likewise (0.15 x 2.0 = 0.3 m/s^2).
    den_min_frac: float = 0.04
    b0_min_frac: float = 0.15
    # Understeer cannot cancel the steering entirely: full lock always buys
    # SOME curvature, so the speed term may not null the static gain.
    span_floor: float = 0.25
    # Command-to-response delay used to align the regressors. Measured on
    # the robot by scanning lags against the flight logs: the steering fit
    # goes from r2 ~0.1 at lag 0 to ~0.8 at 0.4-0.5 s in every run, with a
    # consistent gain. Regressing on the instantaneous (or lag-filtered)
    # command against a response that arrives half a second later is how
    # the learner produced confident nonsense.
    lat_delay: float = 0.45
    # 0.30 until 08-29; the bank has sat at 0.45 on every session since
    # 08-28 (state file), and the derived loop gains start from this.
    lon_delay: float = 0.45
    # Learning gates, as fractions of the operating speed. gate_d is 10 x
    # the velocity noise measured in SENSE, clamped into
    # [gate_floor_frac, gate_cap_frac] x v_op. The CAP: on LiDAR odometry
    # the rest measurement varies 2x between boots on the same floor; one
    # session measured sigma_v = 0.028, which put gate_s at 0.45 m/s --
    # above the 0.38 m/s Nav2 is allowed to command -- and the steering
    # model took 23 samples in ten minutes. The FLOOR: the at-rest sigma
    # under-reads the moving noise (MOLA at rest p99 1.5 cm; under motion
    # 10% of commanded ticks read 0.2 m/s above the command, 08-29 12:04),
    # so a quiet boot must not open the gate to spikes. On this robot both
    # land on the 0.15 m/s that was typed here until 08-29 (0.35..0.45 x
    # 0.32 m/s); the noise measurement only matters inside that band. The
    # gates are recomputed every tick as v_op moves.
    gate_floor_frac: float = 0.35
    gate_cap_frac: float = 0.45
    # Invert the learned longitudinal model once ready_lon holds. This was
    # OFF for a long time because the fit itself was poisoned: a linear gain
    # regressed against a dead-band + friction actuator averages "high
    # throttle, zero acceleration" into b0 several times too low, and the
    # inversion divided by it -- the lunge. Both causes are gone: the dead
    # band is measured and compensated at the output (the learner sees a
    # linearized motor), and Coulomb friction has its own term (b3) instead
    # of leaking into the gain. First hardware session on that model fit
    # b0 = 2.03, b3 = -0.52: sane. The inversion still falls back to the
    # prior until ready_lon (count AND spans), and wrong-signed terms are
    # clamped before they can reach the wheels.
    use_learned_lon: bool = True
    # The planner is quoted the learned radius x this margin (the node
    # pushes it): planned arcs then sit inside the car's lock by 1 - 1/margin
    # of the envelope, and that band is all the slack the follower has for
    # tracking error. Quoted exactly the learned radius, any tracking error
    # made RPP's recovery chord tighter than the car could do (08-23: 35% of
    # turning ticks past the envelope, steering clamp saturated 17%). The
    # same band sizes the launch wait (steer_wait_tol, _run): from rest the
    # throttle is released once the modeled servo is within 1 - 1/margin of
    # full lock from its command, so a leg starts ON its planned arc instead
    # of 0.3 m straight ahead of it.
    radius_push_margin: float = 1.2
    # Slew limits, per second. Without these a single sample can swing the
    # throttle full scale, which is what "violent" looks like from outside.
    # Lower them if the robot still feels abrupt; raise them if it feels
    # sluggish to respond.
    max_steer_rate: float = 4.0
    max_drive_rate: float = 2.0
    # Throttle applied whenever motion is commanded and the car is not yet
    # rolling. A PRIOR in wire units, deliberately below the lowest
    # breakaway this robot had shown (0.24), so it cannot lunge; it only
    # shortens the integrator's climb, which at Nav2's approach speed was
    # 5-8 s per launch and left a third of the flight log "stalled". Zero
    # disables it. On another vehicle it holds only until the first start
    # has been measured: from then on the bootstrap is capped at
    # deadband_trust x the LOWEST breakaway seen (DeadBand.lowest), so a
    # car whose motor breaks free at 0.05 lunges once, not four times
    # (the confirmed median takes over at deadband_evidence starts).
    launch_floor: float = 0.15
    # Launch wire CAP, the floor's counterpart. While motion is commanded
    # and the car has not broken free, the wire may not exceed the launch
    # floor (the learned median breakaway) by more than the margin: the
    # 08-23 flight log shows cusp launches climbing to 0.43-0.48 wire
    # (integrator ramp + Coulomb feedforward, often with the wheels cranked
    # at the standstill clamp) and then surging to 0.6-0.9 m/s against a
    # 0.25-0.32 command when static friction let go -- the wire at release
    # bounds the lunge, so bound the wire. The cap GROWS with time stuck, so
    # a genuinely harder start (slope, carpet, cranked wheels) still
    # escalates to full wire, and the blocked reflex -- which needs the
    # integrator to pin -- is never starved. Both are risk constants, not
    # vehicle properties: the base the cap rides on is learned.
    launch_cap_margin: float = 0.10
    launch_cap_rate: float = 0.10   # cap growth per second stuck
    # Explicit bounded start effort, independent of the PI trim budget.
    # Zero retains the legacy floor + iv_max ceiling for existing profiles.
    launch_effort_limit: float = 0.0
    # Only sized (and therefore only active) after the CAL wiggle has run.
    enable_dither: bool = True

    # Stall / blocked detection.
    # "Is motion being asked for at all", not "is a lot of motion asked for":
    # a fraction of the operating speed (0.15 x 0.32 = 0.05 m/s). At a typed
    # 0.25 this exactly equalled the velocity smoother's reverse limit, so
    # the strict > test never fired and the robot could stall forever.
    stall_cmd_frac: float = 0.15
    stall_time: float = 0.6
    # Integrator authority. Stiction is handled by the integrator ramping
    # until the wheels turn; this is how far it may go, and hitting it while
    # stalled is the cue that the robot is against something.
    iv_max: float = 0.45
    # Curvature trim authority as a fraction of the prior full-lock
    # curvature (0.1 x 1.25 = 0.125/m): the trim may correct a tenth of
    # the lock, never steer the car by itself.
    iw_max_frac: float = 0.10
    blocked_retries: int = 3       # failed attempts before a long hold
    blocked_hold: float = 5.0      # s, after that many failures
    # Feedback speed filter time constant. The raw pose-differenced speed
    # carries ~0.05 m/s of jitter at 10 Hz -- a quarter of the commanded
    # speed -- and feeding it to the PI unfiltered made the throttle chase
    # noise tick by tick. Learning keeps using the raw signal (filtering the
    # regressor biases RLS); only CONTROL uses the filtered one.
    v_fb_tau: float = 0.10  # every 0.1 s of filter lag here is as costly as the same delay in the actuator
    # If the ramp reaches its cap and the robot STILL is not moving, it is not
    # static friction -- it is a wall. Give up instead of pushing through it.
    blocked_after: float = 1.0     # s at full escalation before declaring it
    blocked_release: float = 1.5   # s of zero/low command before retrying

    # Steering envelope. The turning radius is learned, not configured; these
    # only bound what a learned value is allowed to be, and how much evidence
    # it takes to believe one.
    env_qs_threshold: float = 0.55   # |qs| above which a sample is envelope evidence
    env_evidence: int = 12           # samples needed before observation is trusted
    env_derate: float = 0.6          # trust in raw model extrapolation, pre-evidence
    # The radius is quoted at the learned operating speed v_op (it was a
    # typed 0.35 m/s until 08-29). Its bounds are ratios of the prior's
    # radius 1/prior_a0: the learned envelope may sit anywhere from a
    # third of the declared lock to ten times it (0.24..8 m here) --
    # tighter is an implausible fit, wider is a collapsed cell.
    radius_floor_ratio: float = 0.3
    radius_ceiling_ratio: float = 10.0

    # Throttle dead band -- LEARNED, from every start. The wire command that
    # was applied lon_delay seconds before the wheels first turned is the one
    # that broke them free; a median of those per direction is the dead
    # band, and a fraction of it is applied as a static offset at the output
    # so the PI no longer has to climb it on every launch. The earlier
    # breakaway machinery failed because it measured the instantaneous
    # command during a fast probe ramp (true value + ramp x latency, ~0.25
    # high) and then pushed through the integrator on that number.
    deadband_evidence: int = 4      # starts per direction before it is trusted
    # Fraction applied. The sample reads high by ramp x detection latency
    # (~+0.05 in simulation, a slow ramp against ~0.5 s of actuator + odometry
    # lag), so 0.7 keeps the applied offset below the true breakaway: it must
    # never move the car on its own.
    deadband_trust: float = 0.7
    deadband_max: float = 0.5       # sanity bound on what may be applied
    # Only a SLOW start is a measurement: if the wheels turned within this
    # many seconds of the throttle coming on, whatever was on the wire was
    # already above the dead band -- an upper bound, not a sample. Without
    # this, once compensation works every start is instant and the median
    # creeps up on its own output.
    deadband_slow_start: float = 0.8

    # Delay learning. lat_delay / lon_delay above are only the STARTING
    # candidates: each axis runs a bank of estimators at delays spaced by
    # delay_spread from the prior out to [delay_min, delay_max], and
    # control follows the one with the largest aligned gain -- the
    # cross-correlation peak (see DelayBank). A challenger must beat the
    # incumbent's gain by 1/margin to take over, so noise cannot flap the
    # alignment; delay_ew_tau times the per-candidate prediction-error
    # score kept for diagnostics. The bounds are what any ground vehicle's actuator +
    # odometry chain can plausibly span; before 08-29 the bank was three
    # candidates within x1.5 of the typed prior, so a 1 s vehicle could
    # never be aligned and its steering fit was confident nonsense (the
    # same failure as regressing on the instantaneous command). Estimation
    # constants, not vehicle properties -- the delay itself is measured.
    delay_spread: float = 1.5
    delay_min: float = 0.05
    delay_max: float = 2.5
    delay_ew_tau: float = 30.0
    delay_switch_margin: float = 0.8

    # After the commanded curvature changes by more than this fraction of
    # the envelope, the yaw-rate error is pure transport delay for one
    # (learned) delay + servo constant -- integrating it is what kept the
    # trim pinned on every transient. Risk constant; the window is learned.

    # Odometry plausibility. The bounds come from the vehicle's own learned
    # physics -- acceleration from b0 + |b3|, yaw rate from the envelope --
    # scaled by a risk margin; only the margin and the trip counts are
    # hand-set. A step the car could not physically have produced is an
    # odometry glitch: one is ridden through on the previous command,
    # odom_glitch_trip in a row declare the odometry failed (zero output,
    # no learning) until it has been sane for odom_recover_time. Learned
    # from the 11:56 incident: a failing LiDAR USB link fed physically
    # impossible poses at a perfectly healthy 10 Hz for 40 s -- the
    # controller chased them and the learner ate them.
    odom_glitch_margin: float = 2.0
    odom_glitch_trip: int = 3
    odom_recover_time: float = 1.0
    # An implausible stream is answered with the neutral wire for this
    # long before it is declared failed (zero output). odom_glitch_trip
    # samples in a row was 0.3 s: right for the 11:56 dead LiDAR link,
    # wrong for a scan-matcher excursion, which is over in a second --
    # zeroing the wire inside it is itself a step in the motion. What is
    # NOT gated, deliberately: this LiDAR odometry reads 0.5-1.1 m/s
    # against a 0.32 command on 7-16% of commanded ticks in every session
    # since 08-23 -- sustained excursions, not spikes -- and a ratio gate
    # on them (1.5 x v_op) could not be told from a real launch lunge,
    # which peaks at 2.3x the command 1.2 s after breakaway on the sticky
    # plant, or from the controller's own push after a phantom dip; it
    # turned lunges into stops (08-29 bench). The excursions are a sensor
    # problem: the gyro-fused odometry (use_ekf) showed 0-4% in the 08-28
    # sessions against 13-24% for raw MOLA.
    odom_glitch_hold: float = 1.0

    # Dead-man: no odometry for this many nominal steps stops the actuators.
    odom_timeout_steps: float = 3.0

    # Authority earned by confidence (AdaptiveCore.authority): the fraction
    # of its operating speed the vehicle may be commanded while the map is
    # still a prior -- either fit not ready to be inverted, or a steering
    # fault holding the fallback. Published to Nav2 as a speed limit by
    # the node. ONE on this robot, deliberately: slower is not safer on a
    # stiction-dominated car. At 0.5 the 08-29 19:25 drive ran at exactly
    # 0.16 m/s (0.32 x 50%), inside the stall-attraction zone below
    # ~0.3 m/s where cruise degenerates into stick-slip (nav2_params.yaml)
    # -- stuck on 28% of commanded ticks with the wire at the dead band,
    # 46 stall ticks, "moves in steps". A vehicle without stiction may
    # lower this; it stays a risk constant, not a vehicle property.
    authority_floor: float = 1.0

    # The follower's approach speed for the last stretch of every leg,
    # pushed to Nav2 as a fraction of the learned operating speed. It was
    # typed (min_approach_linear_velocity 0.15) and sat in this car's
    # stall-attraction zone: 5 stalls in 9 min on 09-02 00:54, all at leg
    # ends. Below ~0.6 of v_op the car sits between breakaway and cruise
    # (08-29: stall attraction under ~0.3 m/s at v_op 0.32); a fraction
    # rather than a speed so a crawler and a fast car get the same rule.
    approach_speed_frac: float = 0.6

    # Samples faster than this multiple of the operating speed do not
    # teach: they are lunges or scan-matcher excursions, not the regime
    # the model is inverted in. (Control is deliberately NOT gated on it;
    # a lunge is real and must be braked, see odom_glitch_hold.) The 08-30
    # 11:23 drive: six launches on a 0.15 m/s command peaked at 0.9-1.4
    # m/s, and their decays -- 1.3 -> 0.3 m/s on a low wire -- taught the
    # fit "friction is enormous" (b3 -0.96 -> -1.47 in 60 s), which raised
    # the next launch's friction feedforward to 0.38 wire, which lunged
    # harder: a feedback loop between the lunge and the Coulomb term.
    learn_overspeed_ratio: float = 1.5

    # -- derived from the two declared priors (not fields, not parameters) --

    @property
    def radius_floor(self):
        return self.radius_floor_ratio / abs(self.prior_a0)

    @property
    def radius_ceiling(self):
        return self.radius_ceiling_ratio / abs(self.prior_a0)

    @property
    def iw_max(self):
        return self.iw_max_frac * abs(self.prior_a0)

    @property
    def den_min(self):
        return self.den_min_frac * abs(self.prior_a0)

    @property
    def b0_min(self):
        return self.b0_min_frac * abs(self.prior_b0)
