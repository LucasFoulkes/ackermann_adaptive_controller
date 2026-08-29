# Copyright 2026 Lucas Foulkes
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""Learning controller between Nav2 ``/cmd_vel`` and the steering/throttle actuators.

This file is deliberately only plumbing: subscriptions, timers, modes and
safety gates. Every line of mathematics lives in :mod:`core`, which is what
makes the tests runnable without a robot.

Two rates, on purpose. Decisions are made at the odometry rate, because that
is the rate at which new information exists. Commands are republished at 50 Hz
because the Pico's watchdog expires after 0.25 s, and a 10 Hz odometry stream
would otherwise leave the actuators to time out between samples.

Modes (service ``~/set_active``, ``std_srvs/SetBool``):

* PASSIVE: learn from motion but publish nothing, so the DS4 teleop
  mappers keep sole ownership of the actuator topics.
* ACTIVE: turn Nav2 velocity commands into actuator commands.

The PARAMETER default is PASSIVE, but this robot arms itself at launch:
config/robot.yaml sets ``start_active: true``, which is safe on its own only
because Nav2 emits no cmd_vel until a goal is accepted (see robot.launch.py's
docstring for the reasoning and the stop procedure).
"""

import math
import os
import tempfile

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import (Twist, TwistStamped,
                               TwistWithCovarianceStamped)
from nav_msgs.msg import Odometry
from nav2_msgs.msg import SpeedLimit
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, Float32, Int8
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import SetBool, Trigger

import yaml

from .core import CAL, RUN, AdaptiveCore, Policy, finite

PUBLISH_HZ = 50.0

# Policy fields exposed as ROS parameters. Defaults come from core.Policy so
# the two cannot drift apart: this node once declared v_eff_floor=0.4 against
# a core default of 0.12 -- the value the comments blame for weaving.
POLICY_PARAMS = (
    'enable_calibration', 'enable_dither', 't_sense', 't_cal', 't_forget',
    'tau_s', 'tau_d', 'kp_delay_product', 'ti_delay_ratio',
    'kw_delay_product', 'sensor_tau', 'prior_a0', 'prior_b0',
    'v_eff_floor_frac', 'stall_cmd_frac', 'iv_max', 'iw_max_frac',
    'v_fb_tau', 'span_floor', 'steer_standstill', 'use_learned_lon',
    'lat_delay', 'lon_delay', 'gate_floor_frac', 'gate_cap_frac',
    'launch_floor', 'launch_cap_margin', 'launch_cap_rate',
    'max_steer_rate', 'max_drive_rate', 'blocked_after', 'blocked_release',
    'env_qs_threshold', 'env_evidence', 'env_derate',
    'radius_floor_ratio', 'radius_ceiling_ratio',
    'deadband_evidence', 'deadband_trust', 'deadband_max',
    'deadband_slow_start', 'b0_min_frac', 'den_min_frac', 'p0', 'p_max',
    'ready_lat_samples', 'ready_lat_qs_span', 'sign_evidence',
    'sign_agreement', 'ready_lon_samples',
    'ready_lon_qd_span', 'ready_lon_v_span_frac', 'stall_time',
    'blocked_retries', 'blocked_hold', 'odom_timeout_steps',
    'cal_steer', 'cal_drive', 'cal_reverse',
    'delay_spread', 'delay_min', 'delay_max', 'delay_ew_tau',
    'delay_switch_margin', 'iw_freeze_frac',
    'odom_glitch_margin', 'odom_glitch_trip', 'odom_recover_time',
    'odom_glitch_hold',
    'authority_floor',
)
_DEFAULTS = Policy()


def _spans(qd_lo, qd_hi, v_lo, v_hi):
    if qd_lo is None:
        return 'none'
    return (f'qd {qd_lo:+.2f}..{qd_hi:+.2f}  v {v_lo:+.2f}..{v_hi:+.2f}')


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class AckermannAdaptiveController(Node):
    """Twist in, two normalized actuator commands out."""

    def __init__(self):
        super().__init__('ackermann_adaptive_controller')

        p = self.declare_parameters('', [
            # MOLA's nav_msgs/Odometry topic; NOT /odometry.
            ('odom_topic', '/lidar_odometry/pose'),
            # Use the message's twist (signed vx, yaw rate) instead of
            # differencing the pose. Only for a source that fills it in
            # (robot_localization EKF); MOLA's twist is all zeros.
            ('use_odom_twist', False),
            # Independent LiDAR-odometry liveness watchdog. The dead-man
            # below keys on message arrival; with the EKF fused
            # (odom_topic=/odometry/filtered) that topic keeps ticking at
            # 50 Hz off the gyro even if the LiDAR dies, so the dead-man
            # would never fire and the car would drive on a dead-reckoned,
            # drifting pose. Watching the RAW LiDAR odometry (MOLA still
            # publishes it under use_ekf -- only its TF was turned off)
            # restores the pre-EKF semantics: no scans -> MOLA stops
            # publishing this -> stale -> stop. Empty, or equal to
            # odom_topic (the plain-LO case, where the dead-man already
            # covers it), disables the extra watchdog.
            ('lidar_odom_topic', '/lidar_odometry/pose'),
            ('lidar_odom_timeout', 0.5),
            ('cmd_vel_topic', '/cmd_vel'),
            ('steering_topic', '/actuators/steering/command'),
            ('throttle_topic', '/actuators/throttle/command'),
            ('use_stamped_cmd_vel', False),
            ('cmd_vel_timeout', 0.5),
            ('estop_joy_topic', '/joy'),
            ('estop_button', 1),
            ('start_active', False),
            # A vehicle with no learned state (no state file, or just
            # reset) runs the CAL wiggle once, as soon as it is ACTIVE and
            # SENSE has measured the noise floor. The wiggle drives the
            # robot on its own; arming the controller is the consent. It
            # is what establishes the signs on an unknown vehicle. This
            # robot restores its state file at every boot, so it never
            # fires here except after ~/reset.
            ('calibrate_on_fresh_start', True),
            # Policy (see core.Policy for what each one means).
            *[(name, getattr(_DEFAULTS, name)) for name in POLICY_PARAMS],
            # Learned turning radius -> Nav2. Both plugins accept it live and
            # Smac regenerates its motion primitives when it changes. An empty
            # server or parameter name disables that push (NavFn has no
            # turning radius to set).
            ('publish_turning_radius', True),
            ('planner_server', '/planner_server'),
            ('planner_radius_param', 'GridBased.minimum_turning_radius'),
            ('controller_server', '/controller_server'),
            ('controller_radius_param',
             'FollowPath.regulated_linear_scaling_min_radius'),
            # RPP's path-localization window, kept at HALF a cusp-leg of
            # path (0.5 x the quoted turning radius): any window that
            # reaches across a cusp lets the nearest-pose search hop
            # between the overlapping fwd/rev legs, flipping the carrot's
            # direction -- the forward/reverse shuffle. A FULL radius was
            # still too long (16:19 log: legs ~0.8 m at radius 0.91). Half
            # a radius stays several times the observed cross-track error.
            # Pushed with the planner radius so it tracks the LEARNED leg
            # scale. Empty server name disables.
            ('search_dist_server', '/controller_server'),
            ('search_dist_param', 'FollowPath.max_robot_pose_search_dist'),
            # The follower's own geometry, tied to the LEARNED car. Pure
            # pursuit asks curvature 2e/L^2 for a lateral error e at
            # lookahead L; with L below the turning radius R the demand
            # exceeds the car's lock for errors under R/2 (08-29 00:08:
            # min_lookahead 0.4 m, a 0.2 m error asked 2.5/m of a 1.4/m
            # car, the car fell off the arc, the carrot ended up behind it
            # and RPP reversed -- 25 s of fwd/rev inside a single-direction
            # segment). L = R keeps the demand inside the envelope for any
            # error up to R/2; the maximum lookahead is a ratio of the same
            # radius (a carrot never needs to sit beyond the diameter of the
            # tightest circle the car drives). The collision projection
            # horizon is core.stop_horizon (delay + stopping time). Empty
            # server disables all three.
            ('follower_server', '/controller_server'),
            ('lookahead_min_param', 'FollowPath.min_lookahead_dist'),
            ('lookahead_max_param', 'FollowPath.max_lookahead_dist'),
            ('lookahead_max_ratio', 2.0),
            ('collision_horizon_param',
             'FollowPath.max_allowed_time_to_collision_up_to_carrot'),
            # Direction of the single-direction segment the navigator handed
            # the follower (+1/-1, 0 between segments). A cmd_vel against it
            # is not a maneuver the plan contains, it is the follower's
            # carrot having fallen behind the car: executed, it is the
            # shuffle; held as a stop, the progress checker fails the leg in
            # 8 s and the navigator replans from where the car is.
            ('segment_direction_topic', '/cusp_navigator/segment_direction'),
            # Hysteresis: republishing on every wobble would make Smac rebuild
            # its primitive table continuously.
            ('radius_rel_change', 0.10),
            ('radius_abs_change', 0.05),
            ('radius_push_period', 5.0),
            ('radius_filter_alpha', 0.25),
            # The planner is quoted radius * margin, not the raw learned
            # limit. Pushed exactly the learned radius, planned arcs sit AT
            # the car's limit, so any tracking error makes RPP's recovery
            # chord tighter than the car can do -- the 08-23 log had 35% of
            # turning ticks demanding curvature beyond the envelope, with the
            # steering clamp saturated 17% of the time.
            ('radius_push_margin', 1.2),
            ('state_file',
             os.path.expanduser('~/.ros/ackermann_adaptive_controller.yaml')),
            ('save_period', 30.0),
            # Flight recorder: one CSV row per odometry tick -- every input,
            # every internal state, every output. This is the ground truth
            # for "what did it see, what did it try, what did it learn".
            # Empty string disables. ~180 bytes * 10 Hz: ~150 MB/day if left
            # running, so mind the SD card on long soak tests. The file is
            # rotated automatically when the column set changes.
            ('flight_log', os.path.expanduser('~/.ros/ackermann_flight.csv')),
            # Authority earned by confidence -> Nav2. controller_server
            # subscribes to this topic by default (speed_limit_topic) and
            # scales its speeds by the percentage; empty disables. The
            # limit reflects the LEARNING state only (core.authority):
            # faults that hold the actuators at zero need no limit.
            ('speed_limit_topic', '/speed_limit'),
        ])
        g = {d.name: d.value for d in p}

        self.cmd_vel_timeout = float(g['cmd_vel_timeout'])
        self.estop_button = int(g['estop_button'])
        self.steering_topic = g['steering_topic']
        self.throttle_topic = g['throttle_topic']

        self.core = AdaptiveCore(Policy(**{
            name: type(getattr(_DEFAULTS, name))(g[name])
            for name in POLICY_PARAMS}))

        self.state_file = str(g['state_file'])
        self.publish_radius = bool(g['publish_turning_radius'])
        self.planner_server = str(g['planner_server'])
        self.planner_param = str(g['planner_radius_param'])
        self.controller_server = str(g['controller_server'])
        self.controller_param = str(g['controller_radius_param'])
        self.search_server = str(g['search_dist_server'])
        self.follower_server = str(g['follower_server'])
        self.lookahead_min_param = str(g['lookahead_min_param'])
        self.lookahead_max_param = str(g['lookahead_max_param'])
        self.lookahead_max_ratio = float(g['lookahead_max_ratio'])
        self.collision_horizon_param = str(g['collision_horizon_param'])
        self.pushed_lookahead = None
        self.pushed_horizon = None
        self.segment_dir = 0
        self._dir_held = False
        self.search_param = str(g['search_dist_param'])
        self.radius_rel = float(g['radius_rel_change'])
        self.radius_abs = float(g['radius_abs_change'])
        self.radius_margin = float(g['radius_push_margin'])
        self.pushed_radius = None
        self._param_clients = {}
        self._push_warned = False
        self._warned_implausible = False
        self._warned_fault = False
        self._warned_drive_fault = False
        self.radius_filt = None
        self.lookahead_filt = None
        self.radius_alpha = float(g['radius_filter_alpha'])
        self.tap_steer = 0.0
        self.tap_drive = 0.0
        self._flight = None
        path = str(g['flight_log'])
        if path:
            header = ('stamp,phase,active,cmd_v,cmd_w,v,vdot,psidot,'
                      'qs,qd,us,ud,iw,iv,a0l,a0r,a0lr,a0rr,a1,a2,'
                      'b0,b1,b2,b3,breakaway,'
                      'ready_lon,ready_lat,stalled,blocked,fault,'
                      'x,y,yaw\n')
            try:
                os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
                # Rotate a log whose columns no longer match, so one file
                # never mixes two schemas (analysis reads the header once).
                if os.path.exists(path):
                    with open(path, errors='replace') as fh:
                        old_header = fh.readline()
                    if old_header != header:
                        stamp = int(os.path.getmtime(path))
                        rotated = f'{path}.{stamp}'
                        os.replace(path, rotated)
                        self.get_logger().info(
                            f'flight log columns changed; '
                            f'rotated old log to {rotated}')
                new_file = not os.path.exists(path)
                self._flight = open(path, 'a', buffering=1)
                if new_file:
                    self._flight.write(header)
                self.get_logger().info(f'flight log: {path}')
            except OSError as exc:
                self.get_logger().warn(f'no flight log: {exc}')
        self._cal_on_fresh = bool(g['calibrate_on_fresh_start'])
        self._cal_pending = self._cal_on_fresh and not self._load_state()

        self.active = bool(g['start_active'])
        self.estopped = False
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        self.last_cmd_t = None
        self.last_odom_t = None
        self.out_steer = 0.0
        self.out_drive = 0.0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        self.pub_steer = self.create_publisher(Float32, self.steering_topic, 1)
        self.pub_drive = self.create_publisher(Float32, self.throttle_topic, 1)
        # Zero-velocity witness for the EKF, published only while the
        # dead-man is holding the actuators at zero (odometry or LiDAR
        # stale). During a LiDAR outage the filter has NO velocity
        # measurement, so its velocity state latches at the last value and
        # the pose dead-reckons (08-28: 0.825 m/s held for a 40 s outage
        # = 33 m of phantom). But the halt is a FACT this node created:
        # the wheels are stopped because it stopped them (and the Pico
        # dead-man backs it). Telling the filter "v = 0" while halted is
        # a measurement, not a guess.
        self.pub_halt = self.create_publisher(
            TwistWithCovarianceStamped, '~/halted_twist', 1)
        self.pub_diag = self.create_publisher(
            DiagnosticArray, '/diagnostics', 1)
        self.pub_radius = self.create_publisher(
            Float32, '~/min_turning_radius', 1)
        self.speed_limit_topic = str(g['speed_limit_topic'])
        self.pub_speed_limit = (
            self.create_publisher(SpeedLimit, self.speed_limit_topic, 1)
            if self.speed_limit_topic else None)
        self._speed_limit_sent = None

        # Actuator taps. In PASSIVE the joystick owns these topics, and the
        # tapped values are what the learner must regress on.
        self.create_subscription(
            Float32, self.steering_topic,
            lambda m: setattr(self, 'tap_steer', float(m.data)), 1)
        self.create_subscription(
            Float32, self.throttle_topic,
            lambda m: setattr(self, 'tap_drive', float(m.data)), 1)

        self.use_odom_twist = bool(g['use_odom_twist'])
        self.create_subscription(
            Odometry, g['odom_topic'], self.on_odom, sensor_qos)

        # LiDAR-odometry liveness, independent of the (possibly fused)
        # control odometry above. See the parameter comment.
        self.last_lidar_odom_t = None
        self._lidar_odom_timeout = float(g['lidar_odom_timeout'])
        lidar_topic = str(g['lidar_odom_topic'])
        self._lidar_watchdog = bool(lidar_topic) and \
            lidar_topic != str(g['odom_topic'])
        if self._lidar_watchdog:
            self.create_subscription(
                Odometry, lidar_topic,
                lambda m: setattr(self, 'last_lidar_odom_t', self._now()),
                sensor_qos)
        if bool(g['use_stamped_cmd_vel']):
            self.create_subscription(
                TwistStamped, g['cmd_vel_topic'],
                lambda m: self.on_cmd(m.twist), 1)
        else:
            self.create_subscription(
                Twist, g['cmd_vel_topic'], self.on_cmd, 1)
        if self.estop_button >= 0:
            self.create_subscription(
                Joy, g['estop_joy_topic'], self.on_joy, sensor_qos)
        if str(g['segment_direction_topic']):
            latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(
                Int8, str(g['segment_direction_topic']),
                lambda m: setattr(self, 'segment_dir', int(m.data)), latched)
            # Tell the navigator when a hold is in force, so it can replan
            # instead of waiting for Nav2's progress checker.
            self.pub_held = self.create_publisher(Bool, '~/direction_held',
                                                  latched)
            self.pub_held.publish(Bool(data=False))
        else:
            self.pub_held = None

        self.create_service(SetBool, '~/set_active', self.srv_set_active)
        self.create_service(Trigger, '~/calibrate', self.srv_calibrate)
        self.create_service(Trigger, '~/reset', self.srv_reset)

        self.create_timer(1.0 / PUBLISH_HZ, self.on_publish_tick)
        self.create_timer(1.0, self.on_diagnostics)
        self.create_timer(1.0, self.on_authority_tick)
        self.create_timer(float(g['radius_push_period']), self.on_radius_tick)
        self.create_timer(float(g['save_period']), self.save_state)

        self.get_logger().info(
            f"odom={g['odom_topic']}"
            f"{' (twist from message)' if self.use_odom_twist else ''} "
            f"cmd_vel={g['cmd_vel_topic']} "
            f"-> {self.steering_topic}, {self.throttle_topic} | "
            f"mode={'ACTIVE' if self.active else 'PASSIVE'} "
            f"calibration={'on' if g['enable_calibration'] else 'off'}")

    # -- clock -------------------------------------------------------------

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # -- inputs ------------------------------------------------------------

    def on_cmd(self, msg):
        v, w = msg.linear.x, msg.angular.z
        if not finite(v, w):
            self.get_logger().warn('non-finite cmd_vel rejected')
            return
        if self.segment_dir and v * self.segment_dir < 0.0:
            # Against the segment's direction: hold, do not reverse (see the
            # segment_direction_topic parameter).
            if not self._dir_held:
                self.get_logger().warn(
                    'follower asked to drive against the current segment '
                    f'direction ({self.segment_dir:+d}); holding instead',
                    throttle_duration_sec=5.0)
                self._dir_held = True
                if self.pub_held is not None:
                    self.pub_held.publish(Bool(data=True))
            v, w = 0.0, 0.0
        elif self._dir_held:
            self._dir_held = False
            if self.pub_held is not None:
                self.pub_held.publish(Bool(data=False))
        self.cmd_v, self.cmd_w = v, w
        self.last_cmd_t = self._now()

    def on_joy(self, msg):
        if self.estop_button < len(msg.buttons) and \
                msg.buttons[self.estop_button]:
            if not self.estopped:
                self.get_logger().error('E-STOP pressed: latching PASSIVE')
            self.estopped = True
            self.active = False
            self._zero_burst()

    def on_odom(self, msg):
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        pose = msg.pose.pose
        psi = yaw_from_quat(pose.orientation)
        if not finite(t, pose.position.x, pose.position.y, psi):
            self.get_logger().warn('non-finite odometry rejected')
            return
        self.last_odom_t = self._now()

        # LiDAR watchdog gates LEARNING too, not just the actuators: on
        # 08-27 the fused odometry kept sliding after the LiDAR died and
        # 97 s of phantom motion collapsed the learned steering gains
        # (a0l 2.21 -> 0.16, "radius 9.6 m"). Odometry that no live scan
        # is vouching for teaches nothing; the core's gap handling drops
        # the first sample on resume, so skipping here is seamless.
        if not self._lidar_fresh():
            return

        cmd_v, cmd_w = self.cmd_v, self.cmd_w
        if not self._cmd_fresh():
            cmd_v = cmd_w = 0.0

        # ACTIVE: our own last output is what the Pico is holding. PASSIVE:
        # someone else is driving, so learn from what is actually on the wire.
        applied = None if self.active else (self.tap_steer, self.tap_drive)
        v_meas = psidot_meas = None
        if self.use_odom_twist:
            tw = msg.twist.twist
            v_meas, psidot_meas = tw.linear.x, tw.angular.z
        out = self.core.step(
            t, pose.position.x, pose.position.y, psi, cmd_v, cmd_w,
            applied=applied, v_meas=v_meas, psidot_meas=psidot_meas)
        self.out_steer, self.out_drive = out.steer, out.drive

        # Fresh vehicle, armed, noise floor measured: calibrate once.
        if self._cal_pending and self.active and not self.estopped \
                and self.core.phase == RUN:
            self._cal_pending = False
            self.core.start_cal()
            self.get_logger().warn(
                'no learned state: calibration wiggle starting now to '
                'establish the signs and seed the model; keep the area clear')
        if out.drive_fault and not self._warned_drive_fault:
            self.get_logger().error(
                f'THROTTLE FAULT from calibration: {out.drive_fault}; '
                'actuators held at zero (check wiring; ~/reset or '
                '~/calibrate to try again)')
            self._warned_drive_fault = True
        elif not out.drive_fault:
            self._warned_drive_fault = False

        if self._flight is not None:
            c = self.core
            m = c.model
            self._flight.write(
                f'{t:.3f},{c.phase},{int(self.active)},'
                f'{cmd_v:.3f},{cmd_w:.3f},{out.v:.3f},{c.vdot:.3f},'
                f'{out.psidot:.3f},{c.qs:.3f},{c.qd:.3f},'
                f'{out.steer:.3f},{out.drive:.3f},{c.iw:.3f},{c.iv:.3f},'
                f'{m.a0l:.3f},{m.a0r:.3f},'
                f'{m.a0l_rev:.3f},{m.a0r_rev:.3f},'
                f'{m.a1:.3f},{m.a2:.3f},'
                f'{m.b0:.3f},{m.b1:.3f},{m.b2:.3f},{m.b3:.3f},'
                f'{c.breakaway:.3f},'
                f'{int(c.ready_lon)},{int(c.ready_lat)},'
                f'{int(out.stalled)},{int(c.blocked)},'
                f'{int(out.steering_fault)},'
                f'{pose.position.x:.4f},{pose.position.y:.4f},{psi:.4f}\n')

        if out.steering_fault and not self._warned_fault:
            self.get_logger().error(
                'identified steering gain has collapsed - steering appears '
                'dead (unpowered servo? lost linkage?). Falling back to the '
                'prior gain so the robot keeps steering and can recover.')
            self._warned_fault = True
        elif not out.steering_fault and self._warned_fault:
            self.get_logger().info('steering gain recovered')
            self._warned_fault = False

    # -- outputs -----------------------------------------------------------

    def _cmd_fresh(self):
        return (self.last_cmd_t is not None
                and self._now() - self.last_cmd_t < self.cmd_vel_timeout)

    def _odom_fresh(self):
        if self.last_odom_t is None:
            return False
        timeout = max(self.core.policy.odom_timeout_steps * self.core.dt, 0.3)
        return self._now() - self.last_odom_t < timeout

    def _lidar_fresh(self):
        """Is the RAW LiDAR odometry still arriving? Independent of the
        fused control odometry, which the EKF keeps alive off the gyro even
        when the LiDAR is blind. No watchdog configured -> always fresh."""
        if not self._lidar_watchdog:
            return True
        if self.last_lidar_odom_t is None:
            return False
        return self._now() - self.last_lidar_odom_t < self._lidar_odom_timeout

    def on_publish_tick(self):
        """Republish at 50 Hz so the Pico watchdog never expires mid-drive."""
        if not self.active or self.estopped:
            return
        if not self._odom_fresh() or not self._lidar_fresh():
            # Blind: stop. Either the control odometry stalled, or -- with
            # the EKF masking a LiDAR blackout by dead-reckoning off the
            # gyro -- the raw LiDAR odometry went stale while the fused
            # topic kept ticking. When odometry returns the core sees the
            # gap and drops that sample, so the slew restarts from rest.
            if not self._lidar_fresh():
                self.get_logger().warn(
                    'LiDAR odometry stale; stopping (EKF may still be '
                    'publishing a dead-reckoned pose)',
                    throttle_duration_sec=2.0)
            self._zero_burst()
            self.out_steer = self.out_drive = 0.0
            m = TwistWithCovarianceStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = 'base_link'
            # Twist is all zeros by construction. Variances: the wheels
            # are commanded stopped and the Pico dead-man enforces it
            # within 0.25 s; 0.02 m/s / 0.05 rad/s std covers rolling to
            # a stop on the slew.
            m.twist.covariance[0] = 4e-4    # vx
            m.twist.covariance[7] = 4e-4    # vy
            m.twist.covariance[35] = 2.5e-3  # vyaw
            self.pub_halt.publish(m)
            return
        if self.core.phase != CAL and not self._cmd_fresh():
            self._zero_burst()
            return
        self.pub_steer.publish(Float32(data=float(self.out_steer)))
        self.pub_drive.publish(Float32(data=float(self.out_drive)))

    def _zero_burst(self):
        """Explicit zero rather than waiting for the Pico's 0.25 s deadman."""
        self.pub_steer.publish(Float32(data=0.0))
        self.pub_drive.publish(Float32(data=0.0))

    # -- services ----------------------------------------------------------

    def srv_set_active(self, req, resp):
        if req.data and self.estopped:
            resp.success = False
            resp.message = 'E-stop latched; call ~/reset first'
            return resp
        self.active = bool(req.data)
        if not self.active:
            self._zero_burst()
        resp.success = True
        resp.message = 'ACTIVE' if self.active else 'PASSIVE'
        self.get_logger().info(f'mode -> {resp.message}')
        return resp

    def srv_calibrate(self, _req, resp):
        if not self.active:
            resp.success = False
            resp.message = 'must be ACTIVE to calibrate'
            return resp
        if self.core.phase != RUN:
            # SENSE has not measured the noise floor yet (or CAL is already
            # running); calibrating now would size every gate from zeros.
            resp.success = False
            resp.message = f'phase is {self.core.phase}; wait for RUN'
            return resp
        # One-shot on purpose: this must NOT set policy.enable_calibration,
        # which would silently re-enter CAL (and self-drive the wiggle) after
        # every later ~/reset once armed.
        self.core.start_cal()
        resp.success = True
        resp.message = 'calibration started; keep the area clear'
        self.get_logger().warn(resp.message)
        return resp

    def srv_reset(self, _req, resp):
        self.core.reset()
        self.estopped = False
        self.active = False
        self._cal_pending = self._cal_on_fresh
        self._zero_burst()
        resp.success = True
        resp.message = 'learner reset; mode PASSIVE; e-stop cleared'
        if self._cal_pending:
            resp.message += ('; the next set_active true runs the '
                             'calibration wiggle (fresh vehicle)')
        self.get_logger().info(resp.message)
        return resp

    # -- learned turning radius -> Nav2 -----------------------------------

    def on_authority_tick(self):
        """Tell Nav2 how fast the map has earned the right to go."""
        if self.pub_speed_limit is None:
            return
        a = self.core.authority()
        # 0 means "no limit" to Nav2, and the controller's zero-output
        # faults need none: publish the floor instead, and full when earned.
        pct = 100.0 * max(a, self.core.policy.authority_floor)
        if self._speed_limit_sent is not None \
                and abs(pct - self._speed_limit_sent) < 0.5:
            return
        m = SpeedLimit()
        m.header.stamp = self.get_clock().now().to_msg()
        m.percentage = True
        m.speed_limit = pct
        self.pub_speed_limit.publish(m)
        if self._speed_limit_sent is not None or pct < 100.0:
            self.get_logger().info(
                f'authority {pct:.0f}% of Nav2 speed '
                f'({"earned" if a >= 1.0 else "map is a prior"})')
        self._speed_limit_sent = pct

    def on_radius_tick(self):
        """Push the learned turning radius to Nav2 when it has moved enough.

        Verified against Nav2 Jazzy 1.3.12: SmacPlannerHybrid accepts
        ``minimum_turning_radius`` while active and regenerates its motion
        primitives, so this really does change the paths that come back. It is
        rate-limited because that regeneration is not free.
        """
        raw = self.core.envelope.min_turning_radius(self.core.model)
        if not finite(raw) or raw <= 0.0:
            return
        # Smooth before comparing. The raw estimate breathes with whatever the
        # robot happens to be doing, and every push makes Smac rebuild its
        # motion primitive table, so an unfiltered value flaps the planner.
        if self.radius_filt is None:
            self.radius_filt = raw
        else:
            self.radius_filt += self.radius_alpha * (raw - self.radius_filt)
        r = self.radius_filt
        self.pub_radius.publish(Float32(data=float(r)))
        # The follower's lookahead is sized from the FORWARD cells only,
        # filtered the same way. The planner's quote must be the weakest of
        # all four cells (a wider radius is the safe error for a planner);
        # the follower's risk runs the other way -- a longer lookahead puts
        # the carrot at the end of every leg shorter than it and the arc is
        # never followed -- and the legs with room to pursue are the
        # forward ones (2026-08-29 12:04 run: 57 reverse legs median
        # 0.63 m, forward median ~1.3 m). In that run the rev-R cell,
        # 295 samples all session, wandered 0.60 -> 1.08 -> 0.83 m and
        # dragged min_lookahead 0.68 -> 1.00 m while 49% of legs were
        # under 1.0 m.
        raw_fwd = self.core.envelope.min_turning_radius(
            self.core.model, forward=True)
        if finite(raw_fwd) and raw_fwd > 0.0:
            if self.lookahead_filt is None:
                self.lookahead_filt = raw_fwd
            else:
                self.lookahead_filt += self.radius_alpha * (
                    raw_fwd - self.lookahead_filt)
        # Collision projection horizon: learned, independent of the envelope.
        h = self.core.stop_horizon()
        if h is not None and self.follower_server and (
                self.pushed_horizon is None
                or abs(h - self.pushed_horizon) >= self.radius_rel * self.pushed_horizon):
            self.pushed_horizon = h
            self.get_logger().info(
                f'collision horizon -> {h:.2f} s ({self.follower_server})')
            self._set_remote(self.follower_server, self.collision_horizon_param, h)
        # Only trust the envelope enough to steer Nav2 once the model has
        # some evidence behind it; before that the value is a derated
        # extrapolation of a prior and should not override the launch default.
        if not self.core.envelope.confirmed:
            return
        if not (self.publish_radius or self.search_server
                or self.follower_server):
            return
        # Quote the planner a LARGER radius than the car's true limit (see
        # the radius_push_margin declaration): paths must leave RPP headroom
        # to cut a tighter recovery chord without saturating the clamp.
        quoted = r * self.radius_margin
        # The car has FOUR minimum turning radii (direction x side) and the
        # controller steers with all four; Nav2 takes one number, so the
        # planner is quoted the weakest cell x margin. Say which is which.
        m = self.core.model
        v2 = self.core.speed_scale ** 2
        cells = {'fwd-L': abs(m.a1 + m.a0l + m.a2 * v2),
                 'fwd-R': abs(m.a1 - m.a0r - m.a2 * v2),
                 'rev-L': abs(m.a1 + m.a0l_rev + m.a2 * v2),
                 'rev-R': abs(m.a1 - m.a0r_rev - m.a2 * v2)}
        self.get_logger().info(
            'learned turning radii at %.2f m/s: %s -> planner gets %.2f m '
            '(weakest x %.1f)' % (
                self.core.speed_scale,
                ', '.join(f'{k} {1.0 / c if c > 1e-3 else float("inf"):.2f}'
                          for k, c in cells.items()),
                quoted, self.radius_margin))
        self._push_lookahead()
        prev = self.pushed_radius
        if prev is not None:
            if abs(quoted - prev) < self.radius_abs or \
                    abs(quoted - prev) < self.radius_rel * prev:
                return
        # Recorded optimistically; a rejection (typically Nav2 still
        # configuring, so the parameter is not declared yet) clears it again
        # so the next tick retries instead of waiting for a 10% change.
        self.pushed_radius = quoted
        if self.publish_radius:
            self._set_remote(self.planner_server, self.planner_param, quoted)
            self._set_remote(self.controller_server, self.controller_param,
                             quoted)
        # Third consumer, at half scale: a cusp leg is about one quoted
        # radius of path, and RPP's nearest-pose search must not be able to
        # reach across the cusp -- half a leg keeps it on the current one.
        # Gated by its OWN server name, not by publish_turning_radius.
        # TRIED AND REVERTED 08-28 23:40: sizing this from a learned cusp
        # overshoot (0.17-0.22 m; measurement since removed) instead. The
        # follower needs more than the overshoot: after a cusp the nearest
        # path point sits further along than the slide, the closest-pose
        # search stuck, the carrot fell behind the robot and it reversed to
        # it -- 15 direction flips/min against 6.7 with the half-radius
        # rule, legs 0.36 m, 1 of 8 goals reached.
        self._set_remote(self.search_server, self.search_param, 0.5 * quoted)

    def _push_lookahead(self):
        """Follower lookahead from the RAW learned FORWARD radius (see the
        parameter comment and on_radius_tick), on its own change gate so a
        wandering reverse cell moving the planner quote does not re-push
        it. Order the two writes so min never exceeds max between them:
        max first when growing, min first when shrinking."""
        r = self.lookahead_filt
        if not self.follower_server or r is None:
            return
        prev = self.pushed_lookahead
        if prev is not None and (abs(r - prev) < self.radius_abs
                                 or abs(r - prev) < self.radius_rel * prev):
            return
        lo, hi = r, self.lookahead_max_ratio * r
        if prev is None or r > prev:
            self._set_remote(self.follower_server, self.lookahead_max_param, hi)
            self._set_remote(self.follower_server, self.lookahead_min_param, lo)
        else:
            self._set_remote(self.follower_server, self.lookahead_min_param, lo)
            self._set_remote(self.follower_server, self.lookahead_max_param, hi)
        self.pushed_lookahead = r
        self.get_logger().info(
            f'follower lookahead -> {lo:.2f}..{hi:.2f} m (forward cells, '
            f'{self.follower_server})')

    def _set_remote(self, server, name, value):
        """Fire-and-forget remote parameter set; never block the executor."""
        if not server or not name:
            return
        # One persistent client per server. A client created fresh for each
        # push raced DDS endpoint matching: service_is_ready() checked
        # microseconds after creation reports False even against a healthy
        # server, so every push depended on matching latency staying small.
        cli = self._param_clients.get(server)
        if cli is None:
            cli = self.create_client(SetParameters, f'{server}/set_parameters')
            self._param_clients[server] = cli
        if not cli.service_is_ready():
            self._push_failed(f'{server} parameters not available yet')
            return
        req = SetParameters.Request()
        req.parameters = [Parameter(name, Parameter.Type.DOUBLE,
                                    float(value)).to_parameter_msg()]
        future = cli.call_async(req)

        def _done(fut, _server=server, _name=name, _value=value):
            try:
                res = fut.result()
                if res and res.results and not res.results[0].successful:
                    self._push_failed(
                        f'{_server} rejected {_name}: {res.results[0].reason}')
                else:
                    if self._push_warned:
                        self.get_logger().info(
                            f'{_server} accepted {_name}')
                    self._push_warned = False
                    self.get_logger().info(
                        f'pushed {_name} -> {_value:.3f} ({_server})')
            except Exception as exc:
                self._push_failed(f'{_server} set {_name} failed: {exc}')

        future.add_done_callback(_done)

    def _push_failed(self, why):
        """Forget the push so the next tick retries; complain once."""
        self.pushed_radius = None
        self.pushed_lookahead = None
        self.pushed_horizon = None
        if not self._push_warned:
            self.get_logger().warn(f'{why}; will keep retrying')
            self._push_warned = True

    # -- persistence -------------------------------------------------------

    def _load_state(self):
        """Warm start, so the radius is not relearned from scratch each boot.
        Returns True if a model was restored (a fresh vehicle otherwise)."""
        try:
            with open(self.state_file) as fh:
                data = yaml.safe_load(fh)
        except FileNotFoundError:
            return False
        except Exception as exc:
            self.get_logger().warn(f'ignoring unreadable state file: {exc}')
            return False
        if self.core.load_state(data):
            r = self.core.envelope.min_turning_radius(self.core.model)
            self.get_logger().info(
                f'restored model from {self.state_file} '
                f'(turning radius {r:.3f} m, '
                f'envelope {"confirmed" if self.core.envelope.confirmed else "unconfirmed"})')
            return True
        self.get_logger().warn(
            f'state file {self.state_file} failed validation; ignoring')
        return False

    def save_state(self):
        """Atomic write: a half-written state file must never load."""
        if self.core.phase != RUN:
            return
        if not self.core.plausible():
            # Refuse to persist a model learned while an actuator was dead.
            # Otherwise a wiring fault becomes permanent: the saved model says
            # the vehicle cannot steer, and every future boot starts there.
            if not self._warned_implausible:
                self.get_logger().warn(
                    'steering model is not physically plausible '
                    f'(steering_fault={self.core.steering_fault}); '
                    'refusing to save it')
                self._warned_implausible = True
            return
        self._warned_implausible = False
        try:
            path = self.state_file
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or '.')
            with os.fdopen(fd, 'w') as fh:
                yaml.safe_dump(self.core.state(), fh)
            os.replace(tmp, path)
        except Exception as exc:
            self.get_logger().warn(f'could not save state: {exc}')

    # -- diagnostics -------------------------------------------------------

    def on_diagnostics(self):
        m = self.core.model
        status = DiagnosticStatus(
            name='ackermann_adaptive_controller', hardware_id='ackermann')
        if self.estopped:
            status.level, status.message = DiagnosticStatus.ERROR, 'e-stopped'
        elif not self._odom_fresh():
            status.level, status.message = DiagnosticStatus.ERROR, 'no odometry'
        elif not self.core.odom_ok:
            status.level = DiagnosticStatus.ERROR
            status.message = ('odometry implausible - outputs zeroed, '
                              'learning suspended')
        elif self.core.drive_fault:
            status.level = DiagnosticStatus.ERROR
            status.message = f'throttle fault: {self.core.drive_fault}'
        elif self.core.steering_fault:
            status.level = DiagnosticStatus.ERROR
            status.message = 'steering gain collapsed - check the servo'
        elif self.core.phase != RUN:
            status.level = DiagnosticStatus.WARN
            status.message = f'phase {self.core.phase}'
        else:
            status.level = DiagnosticStatus.OK
            status.message = 'ACTIVE' if self.active else 'PASSIVE'
        pairs = {
            'phase': self.core.phase,
            'active': str(self.active),
            'v': f'{self.core.v:.3f}',
            'psidot': f'{self.core.psidot:.3f}',
            'dt': f'{self.core.dt:.4f}',
            'sigma_v': f'{self.core.sigma_v:.4f}',
            # the learned operating speed every speed-shaped gate is a
            # fraction of, and the gates it currently sets
            'v_op': f'{self.core.v_op:.3f}',
            'gate_d': f'{self.core.gate_d:.3f}',
            'gate_s': f'{self.core.gate_s:.3f}',
            # gains per (travel direction x steering side) -- unequal is
            # real: linkage geometry left/right, caster dynamics fwd/rev
            'lateral_a': (f'fwd L{m.a0l:.3f} R{m.a0r:.3f}  '
                          f'rev L{m.a0l_rev:.3f} R{m.a0r_rev:.3f}  '
                          f'{m.a1:.3f} {m.a2:.3f}'),
            'longitudinal_b':
                f'{m.b0:.3f} {m.b1:.3f} {m.b2:.3f} coulomb={m.b3:.3f}',
            # command-to-response delays, learned (bank winner per axis)
            'learned_delays':
                f'lat={self.core.lat_bank.delay:.2f}s '
                f'lon={self.core.lon_bank.delay:.2f}s',
            # loop gains derived from those delays (Policy.kp_delay_product)
            'gains': (f'kp_v={self.core.kp_v:.2f} ki_v={self.core.ki_v:.3f} '
                      f'ki_w={self.core.ki_w:.3f} '
                      f'(L_lon={self.core.L_lon:.2f}s L_lat={self.core.L_lat:.2f}s)'),
            'samples': f'lat={m.n_lat} lon={m.n_lon}',
            'min_turning_radius':
                f'{self.core.envelope.min_turning_radius(m):.3f}',
            'max_curvature': f'{self.core.envelope.max_curvature(m):.3f}',
            'envelope': ('confirmed' if self.core.envelope.confirmed
                         else 'extrapolated (derated)'),
            'envelope_evidence':
                f'left={len(self.core.envelope.left.vals)} '
                f'right={len(self.core.envelope.right.vals)}',
            'breakaway': f'{self.core.breakaway:.3f}',
            # learned throttle dead band: offset applied per direction and
            # how many starts it rests on (0 until deadband_evidence)
            'deadband':
                f'fwd={self.core.deadband.value(1.0):.3f} '
                f'rev={self.core.deadband.value(-1.0):.3f} '
                f'starts fwd={len(self.core.deadband.fwd.vals)} '
                f'rev={len(self.core.deadband.rev.vals)}',
            # what the learner has actually covered -- and whether that has
            # earned the model the right to be inverted
            'ready_lon': str(self.core.ready_lon),
            'ready_lat': str(self.core.ready_lat),
            'lon_spans': _spans(self.core.qd_lo, self.core.qd_hi,
                                self.core.vl_lo, self.core.vl_hi),
            'lat_span': ('none' if self.core.qs_lo is None else
                         f'qs {self.core.qs_lo:+.2f}..{self.core.qs_hi:+.2f}'),
            'blocked': str(self.core.blocked),
            'odom_plausible': str(self.core.odom_ok),
            'lidar_odom': ('n/a' if not self._lidar_watchdog
                           else 'live' if self._lidar_fresh() else 'STALE'),
            'steering': ('FAULT - gain collapsed'
                         if self.core.steering_fault else 'ok'),
            # servo direction: established from evidence, or still the
            # declared prior's
            'steer_sign': ('assumed +' if self.core.steer_sign is None
                           else f'established {self.core.steer_sign:+.0f}'),
            'throttle': self.core.drive_fault or 'ok',
            'calibration': ('pending (fresh vehicle)' if self._cal_pending
                            else f'stage {self.core._cal_stage}'
                            if self.core.phase == CAL
                            else 'done' if self.core.dither > 0.0
                            else 'not run'),
            'model_plausible': str(self.core.plausible()),
            'authority': f'{100.0 * self.core.authority():.0f}%',
            'radius_pushed_to_nav2':
                'none' if self.pushed_radius is None
                else f'{self.pushed_radius:.3f}',
        }
        status.values = [KeyValue(key=k, value=v) for k, v in pairs.items()]
        arr = DiagnosticArray(status=[status])
        arr.header.stamp = self.get_clock().now().to_msg()
        self.pub_diag.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = AckermannAdaptiveController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.save_state()
            node._zero_burst()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
