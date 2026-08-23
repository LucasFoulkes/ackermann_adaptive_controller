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

* PASSIVE (default): learn from motion but publish nothing, so the DS4 teleop
  mappers keep sole ownership of the actuator topics.
* ACTIVE: turn Nav2 velocity commands into actuator commands.

The node starts PASSIVE. A car that can drive away should not do so because a
launch file came up.
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
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32
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
    'tau_s', 'tau_d', 'kp_v', 'ki_v', 'ki_w', 'prior_a0', 'prior_b0',
    'v_eff_floor', 'stall_cmd_min', 'iv_max', 'iw_max', 'v_fb_tau',
    'span_floor', 'steer_standstill', 'use_learned_lon', 'lat_delay',
    'lon_delay', 'gate_s_max', 'launch_floor', 'max_steer_rate',
    'max_drive_rate', 'blocked_after', 'blocked_release',
    'env_qs_threshold', 'env_evidence', 'env_derate', 'env_speed',
    'radius_floor', 'radius_ceiling',
    'deadband_evidence', 'deadband_trust', 'deadband_max',
    'delay_spread', 'delay_ew_tau', 'delay_switch_margin', 'iw_freeze_frac',
    'odom_glitch_margin', 'odom_glitch_trip', 'odom_recover_time',
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
            # Hysteresis: republishing on every wobble would make Smac rebuild
            # its primitive table continuously.
            ('radius_rel_change', 0.10),
            ('radius_abs_change', 0.05),
            ('radius_push_period', 5.0),
            ('radius_filter_alpha', 0.25),
            ('state_file',
             os.path.expanduser('~/.ros/ackermann_adaptive_controller.yaml')),
            ('save_period', 30.0),
            # Flight recorder: one CSV row per odometry tick -- every input,
            # every internal state, every output. This is the ground truth
            # for "what did it see, what did it try, what did it learn".
            # Empty string disables. ~40 bytes * 10 Hz: trivial.
            ('flight_log', os.path.expanduser('~/.ros/ackermann_flight.csv')),
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
        self.radius_rel = float(g['radius_rel_change'])
        self.radius_abs = float(g['radius_abs_change'])
        self.pushed_radius = None
        self._push_warned = False
        self._warned_implausible = False
        self._warned_fault = False
        self.radius_filt = None
        self.radius_alpha = float(g['radius_filter_alpha'])
        self.tap_steer = 0.0
        self.tap_drive = 0.0
        self._flight = None
        path = str(g['flight_log'])
        if path:
            try:
                os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
                new_file = not os.path.exists(path)
                self._flight = open(path, 'a', buffering=1)
                if new_file:
                    self._flight.write(
                        'stamp,phase,active,cmd_v,cmd_w,v,vdot,psidot,'
                        'qs,qd,us,ud,iw,iv,a0,a1,a2,b0,b1,b2,breakaway,'
                        'ready_lon,ready_lat,stalled,blocked,fault,'
                        'x,y,yaw\n')
                self.get_logger().info(f'flight log: {path}')
            except OSError as exc:
                self.get_logger().warn(f'no flight log: {exc}')
        self._load_state()

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
        self.pub_diag = self.create_publisher(
            DiagnosticArray, '/diagnostics', 1)
        self.pub_radius = self.create_publisher(
            Float32, '~/min_turning_radius', 1)

        # Actuator taps. In PASSIVE the joystick owns these topics, and the
        # tapped values are what the learner must regress on.
        self.create_subscription(
            Float32, self.steering_topic,
            lambda m: setattr(self, 'tap_steer', float(m.data)), 1)
        self.create_subscription(
            Float32, self.throttle_topic,
            lambda m: setattr(self, 'tap_drive', float(m.data)), 1)

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

        self.create_service(SetBool, '~/set_active', self.srv_set_active)
        self.create_service(Trigger, '~/calibrate', self.srv_calibrate)
        self.create_service(Trigger, '~/reset', self.srv_reset)

        self.create_timer(1.0 / PUBLISH_HZ, self.on_publish_tick)
        self.create_timer(1.0, self.on_diagnostics)
        self.create_timer(float(g['radius_push_period']), self.on_radius_tick)
        self.create_timer(float(g['save_period']), self.save_state)

        self.get_logger().info(
            f"odom={g['odom_topic']} cmd_vel={g['cmd_vel_topic']} "
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

        cmd_v, cmd_w = self.cmd_v, self.cmd_w
        if not self._cmd_fresh():
            cmd_v = cmd_w = 0.0

        # ACTIVE: our own last output is what the Pico is holding. PASSIVE:
        # someone else is driving, so learn from what is actually on the wire.
        applied = None if self.active else (self.tap_steer, self.tap_drive)
        out = self.core.step(
            t, pose.position.x, pose.position.y, psi, cmd_v, cmd_w,
            applied=applied)
        self.out_steer, self.out_drive = out.steer, out.drive

        if self._flight is not None:
            c = self.core
            m = c.model
            self._flight.write(
                f'{t:.3f},{c.phase},{int(self.active)},'
                f'{cmd_v:.3f},{cmd_w:.3f},{out.v:.3f},{c.vdot:.3f},'
                f'{out.psidot:.3f},{c.qs:.3f},{c.qd:.3f},'
                f'{out.steer:.3f},{out.drive:.3f},{c.iw:.3f},{c.iv:.3f},'
                f'{m.a0:.3f},{m.a1:.3f},{m.a2:.3f},'
                f'{m.b0:.3f},{m.b1:.3f},{m.b2:.3f},{c.breakaway:.3f},'
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

    def on_publish_tick(self):
        """Republish at 50 Hz so the Pico watchdog never expires mid-drive."""
        if not self.active or self.estopped:
            return
        if not self._odom_fresh():
            # Blind: stop. When odometry returns the core sees the gap and
            # drops that sample, so the slew restarts from rest as well.
            self._zero_burst()
            self.out_steer = self.out_drive = 0.0
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
        self.core.policy.enable_calibration = True
        self.core._enter(CAL, self._now())
        resp.success = True
        resp.message = 'calibration started; keep the area clear'
        self.get_logger().warn(resp.message)
        return resp

    def srv_reset(self, _req, resp):
        self.core.reset()
        self.estopped = False
        self.active = False
        self._zero_burst()
        resp.success = True
        resp.message = 'learner reset; mode PASSIVE; e-stop cleared'
        self.get_logger().info(resp.message)
        return resp

    # -- learned turning radius -> Nav2 -----------------------------------

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
        if not self.publish_radius:
            return
        # Only trust the envelope enough to steer the planner once the model
        # has some evidence behind it; before that the value is a derated
        # extrapolation of a prior and should not override the launch default.
        if not self.core.envelope.confirmed:
            return
        prev = self.pushed_radius
        if prev is not None:
            if abs(r - prev) < self.radius_abs or \
                    abs(r - prev) < self.radius_rel * prev:
                return
        # Recorded optimistically; a rejection (typically Nav2 still
        # configuring, so the parameter is not declared yet) clears it again
        # so the next tick retries instead of waiting for a 10% change.
        self.pushed_radius = r
        self._set_remote(self.planner_server, self.planner_param, r)
        self._set_remote(self.controller_server, self.controller_param, r)

    def _set_remote(self, server, name, value):
        """Fire-and-forget remote parameter set; never block the executor."""
        if not server or not name:
            return
        cli = self.create_client(SetParameters, f'{server}/set_parameters')
        if not cli.service_is_ready():
            self._push_failed(f'{server} parameters not available yet')
            cli.destroy()
            return
        req = SetParameters.Request()
        req.parameters = [Parameter(name, Parameter.Type.DOUBLE,
                                    float(value)).to_parameter_msg()]
        future = cli.call_async(req)

        def _done(fut, _cli=cli, _server=server, _name=name, _value=value):
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
                        f'learned minimum turning radius -> {_value:.3f} m '
                        f'({_server})')
            except Exception as exc:
                self._push_failed(f'{_server} set {_name} failed: {exc}')
            _cli.destroy()

        future.add_done_callback(_done)

    def _push_failed(self, why):
        """Forget the push so the next tick retries; complain once."""
        self.pushed_radius = None
        if not self._push_warned:
            self.get_logger().warn(f'{why}; will keep retrying')
            self._push_warned = True

    # -- persistence -------------------------------------------------------

    def _load_state(self):
        """Warm start, so the radius is not relearned from scratch each boot."""
        try:
            with open(self.state_file) as fh:
                data = yaml.safe_load(fh)
        except FileNotFoundError:
            return
        except Exception as exc:
            self.get_logger().warn(f'ignoring unreadable state file: {exc}')
            return
        if self.core.load_state(data):
            r = self.core.envelope.min_turning_radius(self.core.model)
            self.get_logger().info(
                f'restored model from {self.state_file} '
                f'(turning radius {r:.3f} m, '
                f'envelope {"confirmed" if self.core.envelope.confirmed else "unconfirmed"})')
        else:
            self.get_logger().warn(
                f'state file {self.state_file} failed validation; ignoring')

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
            'gate_d': f'{self.core.gate_d:.3f}',
            'lateral_a': f'{m.a0:.3f} {m.a1:.3f} {m.a2:.3f}',
            'longitudinal_b':
                f'{m.b0:.3f} {m.b1:.3f} {m.b2:.3f} coulomb={m.b3:.3f}',
            # command-to-response delays, learned (bank winner per axis)
            'learned_delays':
                f'lat={self.core.lat_bank.delay:.2f}s '
                f'lon={self.core.lon_bank.delay:.2f}s',
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
            'steering': ('FAULT - gain collapsed'
                         if self.core.steering_fault else 'ok'),
            'model_plausible': str(self.core.plausible()),
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
