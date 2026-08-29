# ackermann_adaptive_controller

Learning Twist-to-actuator controller for an Ackermann vehicle. Every
hand-chosen number lives in `Policy` (`core.py`) with its rationale next to
it; the robot's parameter file only carries integration (topics, arming,
e-stop, Nav2 push targets) and whatever it genuinely needs to override.

Assumptions baked into the defaults, all overridable by parameter: odometry
is `nav_msgs/Odometry` pose in `odom` (MOLA, twist field unused); actuators
take normalized `[-1, 1]` `std_msgs/Float32` on `/actuators/{steering,throttle}/command`
with a ~0.25 s downstream watchdog; `prior_a0` is for a 0.28 m wheelbase.

Turns a desired body twist into the two normalized actuator commands the Pico
expects, learning the vehicle online. It replaces the DS4 mappers as the thing
that drives the robot; Nav2 becomes the operator.

```
/cmd_vel (geometry_msgs/Twist)  ─┐
                                 ├─> ackermann_adaptive_controller ─┬─> /actuators/steering/command
/odometry (nav_msgs/Odometry)   ─┘                              └─> /actuators/throttle/command
```

Nothing about the vehicle or the sensor is configured in advance. Both are
identified online by recursive least squares.

## Phases

| Phase | What happens |
|---|---|
| `SENSE` | Sit still for `t_sense` and measure the *sensor*: velocity noise floor, quantization, heading noise. Every downstream gate is derived from these numbers. Restarts if the robot moves. |
| `CAL` | Optional scripted wiggle to excite both axes fast. **Off by default** — it drives the robot autonomously. |
| `RUN` | Invert the learned model, with integral trim riding through the inversion. |

Two learned sub-models, both linear in their parameters:

```
longitudinal   vdot     = b0*qd + b1 + b2*v*|v| + b3*sgn(v)
lateral        psidot/v = a0*qs + a1 + a2*qs*v^2
```

The regressors are the commands that were on the wire `lat_delay` /
`lon_delay` seconds before the response being explained (measured from the
flight logs: the steering fit goes from r² ≈ 0.1 at lag 0 to ≈ 0.8 at
0.45 s). In PASSIVE that is what the joystick sent, tapped from the actuator
topics — the node's own unpublished output is never used as a regressor.

## Modes and safety

The node starts **PASSIVE**: it learns from motion but publishes nothing, so
the teleop mappers keep sole ownership of the actuator topics. A car that can
drive away should not start doing so because a launch file came up.

```bash
ros2 service call /ackermann_adaptive_controller/set_active std_srvs/srv/SetBool "{data: true}"
ros2 service call /ackermann_adaptive_controller/set_active std_srvs/srv/SetBool "{data: false}"
ros2 service call /ackermann_adaptive_controller/calibrate  std_srvs/srv/Trigger   # keep the area clear
ros2 service call /ackermann_adaptive_controller/reset      std_srvs/srv/Trigger   # clear e-stop, forget the model
```

Layered stops, outermost first:

1. **E-stop** — DS4 circle (`estop_button: 1`) latches PASSIVE and sends an
   immediate zero burst. Clear it with `~/reset`.
2. **Odometry dead-man** — no odometry for ~3 sample intervals means blind;
   actuators are zeroed rather than left on the last command.
3. **cmd_vel timeout** — 0.5 s without a Nav2 command zeroes the output.
4. **Pico watchdogs** — 250 ms in the driver, ~520 ms in firmware, unchanged.

Commands are republished at 50 Hz even though decisions are made at the
odometry rate, because a 10 Hz odometry stream would otherwise let the Pico's
250 ms deadman expire between samples.

## Velocity comes from pose, not from twist

MOLA publishes pose in `odom`; with `use_state_estimator: False` its twist
field is not populated. Velocity is therefore differenced from consecutive
poses, projected onto the heading:

```
v = (dx*cos(psi) + dy*sin(psi)) / dt
```

The projection is what keeps the sign right in reverse — `hypot(dx, dy)` would
report reverse as positive and invert every control law downstream.

## The turning radius is learned, not configured

Nav2 needs a `minimum_turning_radius`, and that number is a property of the
vehicle — so it is identified rather than typed into a YAML.

The lateral model already predicts curvature for any steering command and
speed. What it cannot know is whether that *linear* map still holds at full
lock; a steering rack has stops and a tyre has a limit. So what is learned is
one number per direction — the ratio of curvature actually achieved to
curvature the model predicted, measured only when steering was near lock:

```
ratio ~ 1   the model extrapolates honestly to full lock
ratio < 1   the model overpromises; the real envelope is smaller
```

Measuring a *ratio* is what makes this speed-correct. Raw curvature at full
lock depends heavily on speed through `a2*v^2` — on this vehicle that term
nearly cancels the steering gain around 2 m/s — so a raw measurement taken
while driving fast reports a far flatter envelope than the robot has at the
0.35 m/s Nav2 plans for. `env_speed` is the speed the radius is quoted at.

Robustness comes from a median over a bounded window: `env_evidence` samples
per side are needed before the value is believed at all, one glitched sample
cannot move it, and old evidence ages out when the surface changes. Evidence
can only ever make the envelope *smaller* — a ratio above 1 is noise, not a
car that out-turns its own model. Before evidence exists the raw extrapolation
is derated by `env_derate`, because conservative means a **larger** radius: a
planner that thinks the car turns tighter than it does emits paths the car
cannot follow.

Once both directions are confirmed the value is pushed live into Nav2. The
car has four minimum turning radii (direction × side) and the controller
steers with all four; Nav2 takes one number, so it gets the weakest cell ×
`radius_push_margin`:

- `planner_server` → `GridBased.minimum_turning_radius` (quoted radius)
- `controller_server` → `FollowPath.max_robot_pose_search_dist` (½ quoted:
  RPP's closest-pose search must not reach across a cusp)
- `controller_server` → `FollowPath.min/max_lookahead_dist` (raw radius, ×2:
  pure pursuit's curvature demand 2e/L² stays inside the car's lock for any
  lateral error up to R/2)
- `controller_server` → `FollowPath.max_allowed_time_to_collision_up_to_carrot`
  (learned throttle delay + stopping time on friction)

The `FollowPath.regulated_linear_scaling_min_radius` push exists but is off
by configuration (`controller_server: ""`): RPP has no curvature limit, only
a slow-down threshold, and pushing a radius into it made the follower crawl.

Verified against Nav2 Jazzy 1.3.12: Smac accepts the change while active and
regenerates its motion primitives — the same goal replans from 2.22 m at
0.9 m radius to 4.75 m at 3.0 m. Pushes are low-pass filtered and hysteretic,
because each one costs a primitive-table rebuild.

It is also persisted to `state_file` and restored on the next boot, with the
covariance inflated rather than restored: the parameters are probably still
right, but the tyres, floor and battery all had a chance to change while the
robot was off. The learner's readiness (sample counts and excitation spans)
travels with it, so a restored steering model is inverted immediately rather
than after 60 fresh samples. A throttle fit with the wrong sign (`b0 < 0`,
which the dead-band actuator can produce) is replaced by the prior on save;
at the time it did not drive the wheels (`use_learned_lon` was off until
the dead band and Coulomb term made the fit trustworthy).

`~/min_turning_radius` (`std_msgs/Float32`) carries the current value, and
`/diagnostics` reports `envelope`, `envelope_evidence` and
`radius_pushed_to_nav2`.

### Bootstrapping it

There is a chicken-and-egg here worth knowing: the planner uses a conservative
radius, so it emits gentle paths, so the steering never approaches lock, so
the envelope never confirms. Two ways out, and the first is better:

1. **Drive it with the DS4 in PASSIVE mode**, using full lock both ways. The
   node taps the actuator topics and learns from what is really on the wire,
   publishing nothing. Vary the speed while you do it — see below.
2. `~/calibrate`, which runs the scripted wiggle under its own power.

**Vary the speed.** `a0` and `a2` both multiply the steering command and
differ only by `v^2`, so a constant-speed run cannot separate them and the
extrapolation to planning speed drifts. In a bench run against a synthetic
plant, a fixed-speed slalom recovered `a0 = 0.92` against a true `1.30`; the
same slalom with the speed swept recovered `1.27`, and the learned radius
settled at 0.89 m against a true 0.79 m.

## The delays and the friction are learned too

`b3*sgn(v)` is Coulomb friction. Without it, drivetrain friction leaks into
`b0`/`b1` and the longitudinal fit converges to nonsense (a negative gain was
observed on hardware); with it, the gain and the friction come out separately.
The Coulomb feedforward is only used in the inversion once both directions
have fed the fit — one-sided data cannot separate `b3` from the bias `b1`.

The command-to-response delay each model aligns to is not configured either:
each axis runs a small bank of identical estimators at candidate delays
(`lat_delay`/`lon_delay` are just the initial centers) and control follows
whichever currently predicts best, scored by exponentially-weighted
prediction error with switching hysteresis. The learned lateral delay also
sizes the trim freeze window: after the commanded curvature changes, the yaw
error is pure transport delay for one delay + servo constant, and the trim
integrator holds instead of winding up on it. `/diagnostics` reports
`learned_delays`.

## The throttle dead band is learned too

A brushed motor behind an H-bridge does nothing for the first part of its
command range, and the flight logs put this robot's breakaway at 0.18–0.38
with a median of 0.24. Rather than configure that, every start measures it:
the throttle that was on the wire `lon_delay` seconds before the wheels first
turned is a dead-band sample in the direction the car moved. A median window
per direction (`deadband_evidence` starts before it counts) is the estimate,
and `deadband_trust` of it is applied as a **static** offset at the output —
`d + |u|·(1 − d)` — so the PI sees a linear motor and no longer climbs the
dead zone on each launch. Being a fraction, the offset can never move a
stopped car by itself; being static, there is nothing to unwind after
breakaway (the old "launch kick" lived in the integrator and lunged).
Starts get the **full** learned median as feedforward — the wire goes
straight to the measured breakaway when motion is commanded and the car is
not yet rolling, so the integrator only tops up the spread. Until there is
evidence, `launch_floor` is the bootstrap, exactly as `prior_a0` is for the
envelope. `/diagnostics` reports `deadband`.

## Things to measure before trusting it

- **`prior_a0`** defaults to `1.25`, from wheelbase 0.2775 m and an *assumed*
  ~18° max steering angle. It is only a bootstrap so `RUN` works before
  anything is learned; the envelope above supersedes it once there is
  evidence. Setting it closer to the truth just shortens the transient.
- **`v_eff_floor`** defaults to `0.12`. Curvature is `omega/v`, and this
  floors the divisor. It must sit well below the speed Nav2 commands: at 0.4
  against a 0.22 m/s cruise the floor dominated and only ~55% of every
  requested turn was delivered (weaving).
- **Learning gates vs. top speed.** `gate_d = max(0.15, 10*sigma_v)` and
  `gate_s = 1.6*gate_d`, capped so that `gate_s <= gate_s_max` (0.24 m/s).
  Without the cap a `sigma_v` of 0.028 — measured on one boot, 0.015 on the
  next, same floor — put `gate_s` at 0.45 m/s, above the 0.38 m/s the
  velocity smoother allows, and the steering model took 23 samples in ten
  minutes. If you raise Nav2's speeds, `gate_s_max` can go up with them.
  Check `sigma_v` and `gate_d` in `/diagnostics` after `SENSE`.
- **`launch_floor`** (0.15) is a constant throttle applied while motion is
  commanded and the car is not rolling. It is below the lowest breakaway ever
  observed (0.24) so it cannot lunge; it only shortens the integrator's climb,
  which at Nav2's approach speed was 5–8 s per launch.
- **Drag needs varied speeds.** At one constant speed the regressor
  `[qd, 1, v|v|]` is rank-deficient and `b0`/`b1`/`b2` cannot be separated —
  the prediction stays right but the split is arbitrary. A cruise teaches the
  learner much less than a varied run. The same applies to `a0`/`a2`.

## Tests

The mathematics is ROS-free and tested against a synthetic plant:

```bash
cd src/ackermann_adaptive_controller && python3 -m pytest test/ -q
```
