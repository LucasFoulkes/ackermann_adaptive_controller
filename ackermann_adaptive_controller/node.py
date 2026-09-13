# Copyright 2026 Lucas Foulkes
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE file or at https://opensource.org/licenses/MIT.

"""ROS transport for odometry-feedback steering and throttle control.

The controller reports turning capability to Nav2. Navigation policy stays in
Nav2 configuration. Learning/control run on new odometry; outputs refresh at 50 Hz.
"""

import math
import os
import uuid
import time
from .telemetry import BackgroundIO, save_model
from .nav2_capability import TurningCapability, SurfaceCapability
from .input_health import OdomHealth
from .delivery import DeliveryHistory
from ackermann_interfaces.msg import ActuatorState

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from nav2_msgs.msg import SpeedLimit
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, Float32

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
    'v_fb_tau', 'span_floor', 'use_learned_lon',
    'lat_delay', 'lon_delay', 'gate_floor_frac', 'gate_cap_frac',
    'launch_floor', 'launch_cap_margin', 'launch_cap_rate', 'launch_effort_limit',
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
    'delay_switch_margin', 'validate_lon', 'validate_lat',
    'odom_glitch_margin', 'odom_glitch_trip', 'odom_recover_time',
    'odom_glitch_hold',
    'authority_floor', 'learn_overspeed_ratio', 'approach_speed_frac',
    'radius_push_margin',
)
_DEFAULTS = Policy()


def _fmt_span(lo, hi):
    return 'none' if lo is None or hi is None else f'{lo:+.2f}..{hi:+.2f}'


def _spans(qd_lo, qd_hi, v_lo, v_hi):
    """Each span may be absent on its own: a state file can carry the wire
    span with the speed span cleared (the 08-29 restore did), and a None
    reaching a float format killed the node from the diagnostics timer one
    second after launch (08-30 10:53, 11:00)."""
    return f'qd {_fmt_span(qd_lo, qd_hi)}  v {_fmt_span(v_lo, v_hi)}'


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class AckermannAdaptiveController(Node):
    """Twist in, two normalized actuator commands out."""

    def __init__(self):
        super().__init__('ackermann_adaptive_controller')

        p = self.declare_parameters('', [
            ('odom_topic', '/lidar_odometry/pose'),
            ('use_odom_twist', False),
            ('odom_frame', 'odom'),
            ('base_frame', 'base_link'),
            ('odom_max_age', 0.5),
            ('motion_speed_limit', 0.38),
            ('motion_yaw_rate_limit', 0.45),
            ('cmd_vel_topic', '/cmd_vel'),
            ('steering_topic', '/actuators/steering/command'),
            ('throttle_topic', '/actuators/throttle/command'),
            ('use_stamped_cmd_vel', False),
            ('cmd_vel_timeout', 0.5),
            ('estop_joy_topic', '/joy'),
            ('estop_button', 1),
            ('start_active', False),
            # Policy (see core.Policy for what each one means).
            *[(name, getattr(_DEFAULTS, name)) for name in POLICY_PARAMS],
            ('publish_turning_radius', True),
            ('planner_server', '/planner_server'),
            ('planner_radius_param', 'GridBased.minimum_turning_radius'),
            ('controller_server', ''),
            ('controller_radius_param', ''),
            ('radius_rel_change', 0.10),
            ('radius_abs_change', 0.05),
            ('radius_push_period', 5.0),
            ('radius_filter_alpha', 0.25),
            # Surface capability -> Nav2 (learned floors for the follower's
            # minimum speeds and the progress checker's stuck window).
            ('publish_surface_limits', True),
            ('approach_speed_param', 'FollowPath.min_approach_linear_velocity'),
            ('regulated_min_speed_param', 'FollowPath.regulated_linear_scaling_min_speed'),
            ('stuck_allowance_param', 'progress_checker.stuck_time_allowance'),
            ('movement_allowance_param', 'progress_checker.movement_time_allowance'),
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
        self.capability = TurningCapability(self, g)
        self.surface = SurfaceCapability(self, g)
        self.health = OdomHealth(float(g['odom_max_age']), str(g['odom_frame']), str(g['base_frame']))
        self.motion_speed_limit = float(g['motion_speed_limit'])
        self.motion_yaw_rate_limit = float(g['motion_yaw_rate_limit'])
        if not finite(self.motion_speed_limit, self.motion_yaw_rate_limit) or min(self.motion_speed_limit, self.motion_yaw_rate_limit) <= 0:
            raise ValueError('motion limits must be positive and finite')
        self.io = BackgroundIO()
        self._warned_implausible = False
        self._warned_fault = False
        self._warned_drive_fault = False
        self.delivery = DeliveryHistory()
        self._flight = None
        path = str(g['flight_log'])
        if path:
            header = ('stamp,phase,active,cmd_v,cmd_w,v,vdot,psidot,'
                      'qs,qd,us,ud,iw,iv,a0l,a0r,a0lr,a0rr,a1,a2,'
                      'b0,b1,b2,b3,breakaway,'
                      'ready_lon,ready_lat,stalled,blocked,fault,'
                      'x,y,yaw,probe_b0,probe_eq,err_rms,cycles,learn,wait,unreach\n')
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
                # A controller restart within one launch also begins a new session.
                run_id = os.environ.get('LUKUMA_RUN_ID', 'standalone') + ':' + uuid.uuid4().hex
                self._flight.write(f'# run_id {run_id}\n')
                self.get_logger().info(f'flight log: {path}')
            except OSError as exc:
                self.get_logger().warn(f'no flight log: {exc}')
        # No autonomous calibration, by the operator's explicit decision
        # (2026-08-30): a robot must not drive itself unprompted, state
        # file or none. A fresh vehicle learns from the goals it is given
        # -- the priors carry the first legs -- and the staged CAL wiggle
        # exists ONLY behind the ~/calibrate service, where invoking it IS
        # the consent.
        self._load_state()

        self.active = bool(g['start_active'])
        self.estopped = False
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        self.last_cmd_t = None
        self.last_odom_t = None
        self._deadman_tripped = False
        self._cmd_timed_out = False
        self.out_steer = 0.0
        self.out_drive = 0.0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        self.pub_steer = self.create_publisher(Float32, self.steering_topic, 1)
        self.pub_drive = self.create_publisher(Float32, self.throttle_topic, 1)
        self.pub_diag = self.create_publisher(
            DiagnosticArray, '/diagnostics', 1)
        self.pub_radius = self.create_publisher(
            Float32, '~/min_turning_radius', 1)
        self.pub_stop_horizon = self.create_publisher(Float32, '~/stop_horizon', 1)
        self.speed_limit_topic = str(g['speed_limit_topic'])
        self.pub_speed_limit = (
            self.create_publisher(SpeedLimit, self.speed_limit_topic, 1)
            if self.speed_limit_topic else None)
        self._speed_limit_sent = None

        self.create_subscription(ActuatorState, '/actuators/state', self.on_delivery, 20)

        self.use_odom_twist = bool(g['use_odom_twist'])
        self.create_subscription(
            Odometry, g['odom_topic'], self.on_odom, sensor_qos)

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
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_stalled = self.create_publisher(Bool, '~/stalled', latched)
        self.pub_stalled.publish(Bool(data=False))
        self._stalled_flag = False

        self.create_service(SetBool, '~/set_active', self.srv_set_active)
        self.create_service(Trigger, '~/calibrate', self.srv_calibrate)
        self.create_service(Trigger, '~/reset', self.srv_reset)

        self.create_timer(1.0 / PUBLISH_HZ, self.on_publish_tick)
        self.create_timer(1.0, self.on_diagnostics)
        self.create_timer(1.0, self.on_authority_tick)
        self.create_timer(float(g['radius_push_period']), self.capability.tick)
        self.create_timer(float(g['radius_push_period']), self.surface.tick)
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
            self.get_logger().warn('non-finite cmd_vel revoked')
            self.cmd_v = self.cmd_w = self.out_steer = self.out_drive = 0.0
            self.last_cmd_t = None
            if self.active:
                self._zero_burst()
            return
        scale = max(1.0, abs(v) / self.motion_speed_limit,
                    abs(w) / self.motion_yaw_rate_limit)
        v, w = v / scale, w / scale
        self.cmd_v, self.cmd_w = v, w
        self.last_cmd_t = time.monotonic()


    def on_joy(self, msg):
        if self.estop_button < len(msg.buttons) and \
                msg.buttons[self.estop_button]:
            if not self.estopped:
                self.get_logger().error('E-STOP pressed: latching PASSIVE')
            self.estopped = True
            self.active = False
            self._zero_burst()

    def on_delivery(self, msg):
        sample = self.delivery.accept(msg, self._now())
        if sample is not None:
            self.core.external_history = True
            # Transferred at acquisition time in on_odom; never record future inputs.
        if not self.delivery.fresh(self._now()):
            self.out_steer = self.out_drive = 0.0

    def on_odom(self, msg):
        now = self._now()
        if not self.health.accept(msg, now):
            self.core.pause_learning(now)
            self.out_steer = self.out_drive = 0.0
            return
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        pose = msg.pose.pose
        psi = yaw_from_quat(pose.orientation)
        self.last_odom_t = time.monotonic()

        cmd_v, cmd_w = self.cmd_v, self.cmd_w
        if not self._cmd_fresh():
            cmd_v = cmd_w = 0.0

        applied = self.delivery.at(t)
        if not self.delivery.fresh(now) or applied is None:
            self.core.pause_learning(now)
            self.out_steer = self.out_drive = 0.0
            return
        self.core.external_history = True
        previous = self.core._cmd_hist[-1][0] if self.core._cmd_hist else float('-inf')
        for sample in self.delivery.samples:
            if previous < sample[0] <= t:
                self.core.record_command(*sample)
        v_meas = psidot_meas = None
        if self.use_odom_twist:
            tw = msg.twist.twist
            v_meas, psidot_meas = tw.linear.x, tw.angular.z
        out = self.core.step(
            t, pose.position.x, pose.position.y, psi, cmd_v, cmd_w,
            applied=applied, v_meas=v_meas, psidot_meas=psidot_meas, passive=not self.active)
        self.out_steer, self.out_drive = out.steer, out.drive
        stalled_now = bool(out.stalled or self.core.blocked)
        if stalled_now != self._stalled_flag:
            self._stalled_flag = stalled_now
            self.pub_stalled.publish(Bool(data=stalled_now))

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
            self.io.submit(self._flight.write,
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
                f'{pose.position.x:.4f},{pose.position.y:.4f},{psi:.4f},'
                f'{c.gain_probe.b0 or 0.0:.3f},'
                f'{c.gain_probe.eq(1.0) or 0.0:.3f},'
                f'{c.score.err_rms or 0.0:.3f},{c.score.cycles},'
                f'{int(out.learning)},{int(out.steer_wait)},'
                '0\n')

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
                and time.monotonic() - self.last_cmd_t < self.cmd_vel_timeout)

    def _odom_fresh(self):
        if self.last_odom_t is None:
            return False
        timeout = max(self.core.policy.odom_timeout_steps * self.core.dt, 0.3)
        return (time.monotonic() - self.last_odom_t < timeout
                and self.health.fresh(self._now()))


    def on_publish_tick(self):
        """Republish at 50 Hz so the Pico watchdog never expires mid-drive."""
        if not self.active or self.estopped:
            return
        if not self._odom_fresh() or not self.delivery.fresh(self._now()):
            self.core.pause_learning(self._now())
            if not self._deadman_tripped:
                self._deadman_tripped = True
                self.core.score.count('deadman', self._now())
            self._zero_burst()
            self.out_steer = self.out_drive = 0.0
            return
        self._deadman_tripped = False
        if self.core.phase != CAL and not self._cmd_fresh():
            if not self._cmd_timed_out and self.last_cmd_t is not None \
                    and (self.cmd_v != 0.0 or self.cmd_w != 0.0):
                # a command stream that died MID-MOTION (its last message
                # was still asking for speed) -- not the normal end of a
                # segment, where the follower stops publishing after a
                # zero (21 of those per 4 min on the 09-01 drive)
                self._cmd_timed_out = True
                self.core.score.count('cmd_timeout', self._now())
            self._zero_burst()
            self.out_steer = self.out_drive = 0.0
            return
        self._cmd_timed_out = False
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
        self.health = OdomHealth(self.health.max_age, self.health.odom_frame, self.health.base_frame)
        self.delivery = DeliveryHistory()
        self.last_cmd_t = self.last_odom_t = None
        self.cmd_v = self.cmd_w = self.out_steer = self.out_drive = 0.0
        self.estopped = False
        self.active = False
        self._zero_burst()
        resp.success = True
        resp.message = ('learner reset; mode PASSIVE; e-stop cleared '
                        '(no automatic calibration: the robot learns from '
                        'the goals it is given, or ~/calibrate on request)')
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
        m = SpeedLimit()
        m.header.stamp = self.get_clock().now().to_msg()
        m.percentage = True
        m.speed_limit = pct
        self.pub_speed_limit.publish(m)
        if self._speed_limit_sent is None or abs(pct-self._speed_limit_sent) >= 0.5:
            self.get_logger().info(
                f'authority {pct:.0f}% of Nav2 speed '
                f'({"earned" if a >= 1.0 else "map is a prior"})')
        self._speed_limit_sent = pct


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
        self.io.submit(save_model, self.state_file, self.core.state())

    # -- diagnostics -------------------------------------------------------

    def on_diagnostics(self):
        m = self.core.model
        status = DiagnosticStatus(
            name='ackermann_adaptive_controller', hardware_id='ackermann')
        if self.estopped:
            status.level, status.message = DiagnosticStatus.ERROR, 'e-stopped'
        elif not self._odom_fresh():
            status.level, status.message = DiagnosticStatus.ERROR, 'no odometry'
        elif not self.delivery.fresh(self._now()):
            status.level, status.message = DiagnosticStatus.ERROR, 'actuator delivery unavailable'
        elif not self.core.odom_ok:
            status.level = DiagnosticStatus.ERROR
            status.message = ('odometry implausible - propulsion stopped, '
                              'learning suspended')
        elif self.core.drive_fault:
            status.level = DiagnosticStatus.ERROR
            status.message = f'throttle fault: {self.core.drive_fault}'
        elif self.core.steering_fault:
            status.level = DiagnosticStatus.ERROR
            status.message = 'steering gain collapsed - check the servo'
        elif not self.active:
            status.level, status.message = DiagnosticStatus.OK, 'PASSIVE'
        elif not self._cmd_fresh():
            status.level, status.message = DiagnosticStatus.WARN, 'waiting for velocity command'
        elif self.core.blocked:
            status.level, status.message = DiagnosticStatus.WARN, 'blocked: actuator stall hold'
        elif self.core._reversal_target:
            status.level, status.message = DiagnosticStatus.WARN, 'stopping before changing direction'
        elif self.core._steer_wait:
            status.level, status.message = DiagnosticStatus.WARN, 'waiting for steering response'
        elif self.core.phase != RUN:
            status.level = DiagnosticStatus.WARN
            status.message = f'phase {self.core.phase}'
        else:
            status.level = DiagnosticStatus.OK
            status.message = 'ACTIVE' if self.active else 'PASSIVE'
        agree_ok, agree_text = self.core.agreement()
        if status.level == DiagnosticStatus.OK and not agree_ok:
            # the estimators of the throttle plant disagree beyond the
            # trust band: the earliest sign of a poisoned model
            status.level = DiagnosticStatus.WARN
            status.message += ' - throttle estimators disagree'
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
            # per-cell steering trims (curvature, 1/m), persisting across legs
            'trim': ' '.join(f'{k}={v:+.3f}'
                             for k, v in sorted(self.core.trim_cells().items()))
                    or 'none',
            # how well the car is following (DriveScore): tracking error,
            # launch overshoot, surge-stall cycles, stalls, and the event
            # counters with per-minute rates
            **self.core.score.summary(),
            # which estimator the throttle law is running on, and whether
            # the independent ones agree (core.agreement)
            'learning': agree_text,
            'odom_health': (f'glitch_holds={self.core.n_glitch_holds} '
                            f'implausible={self.core.n_implausible}'),
            # direct wire-to-acceleration measurement (GainProbe): the
            # loop divisor and measured-bootstrap source
            'gain_probe':
                f'b0={self.core.gain_probe.b0 or 0.0:.2f} '
                f'eq_fwd={self.core.gain_probe.eq(1.0) or 0.0:.2f} '
                f'eq_rev={self.core.gain_probe.eq(-1.0) or 0.0:.2f} '
                f'samples gain={len(self.core.gain_probe.gain.vals)} '
                f'eq={len(self.core.gain_probe.eq_fwd.vals)}/'
                f'{len(self.core.gain_probe.eq_rev.vals)}',
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
            'longitudinal_fit_valid': str(self.core.lon_plausible()),
            'throttle_validation': getattr(self.core.lon_bank, 'status', 'legacy'),
            'steering_validation': getattr(self.core.lat_bank, 'status', 'legacy'),
            'steering_promotions': str(getattr(self.core.lat_bank, 'promotions', 0)),
            'steering_validation_rejections': str(getattr(self.core.lat_bank, 'rejections', 0)),
            'steering_candidate_samples': str(self.core.lat_bank.bank[self.core.lat_bank.active].count),
            'throttle_promotions': str(getattr(self.core.lon_bank, 'promotions', 0)),
            'throttle_validation_rejections': str(getattr(self.core.lon_bank, 'rejections', 0)),
            'throttle_candidate': str(self.core.lon_bank.bank[self.core.lon_bank.active].theta),
            'throttle_candidate_delay': str(self.core.lon_bank.delays[self.core.lon_bank.active]),
            'throttle_candidate_samples': str(self.core.lon_bank.bank[self.core.lon_bank.active].count),
            'throttle_validation_samples': str(len(getattr(self.core.lon_bank, 'rows', ()))),
            'longitudinal_rejected_updates': str(sum(r.rejected for r in self.core.lon_bank.bank)),
            'learning_settling': str(self.core.now < self.core._learning_after),
            'surface_creep_speed': ('%.3f' % self.core.creep_speed()) if self.core.creep_speed() is not None else 'unmeasured',
            'surface_launch_time': ('%.2f' % self.core.launch_duration()) if self.core.launch_duration() is not None else 'unmeasured',
            'surface_recent_breakaway': 'fwd=%s rev=%s' % tuple(
                ('%.2f' % v) if v is not None else '-' for v in (self.core.deadband.recent(1.0), self.core.deadband.recent(-1.0))),
            'surface_limits_confirmed': str({k[1]: round(v, 3) for k, v in self.surface.confirmed.items()}),
            'input_status': self.health.reason,
            'actuator_delivery': str(bool(self.delivery.fresh(self._now()))),
            'telemetry_error': self.io.error or 'none',
            'telemetry_dropped': str(self.io.dropped),
            'steering': ('FAULT - gain collapsed'
                         if self.core.steering_fault else 'ok'),
            # servo direction: established from evidence, or still the
            # declared prior's
            'steer_sign': ('assumed +' if self.core.steer_sign is None
                           else f'established {self.core.steer_sign:+.0f}'),
            'throttle': self.core.drive_fault or 'ok',
            'calibration': (f'stage {self.core._cal_stage}'
                            if self.core.phase == CAL
                            else 'done' if self.core.dither > 0.0
                            else 'not run (on request: ~/calibrate)'),
            'model_plausible': str(self.core.plausible()),
            'authority': f'{100.0 * self.core.authority():.0f}%',
            'radius_pushed_to_nav2':
                str(self.capability.confirmed),
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
        node.io.close()
        if node._flight is not None and not node.io.thread.is_alive():
            node._flight.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
