"""Report learned capability to Nav2: turning limits, and what the current
surface allows (slowest holdable speed, launch time). Navigation keeps its
configured values as floors; learned values only raise them."""
import math
import time
from rcl_interfaces.srv import SetParameters, GetParameters
from rclpy.parameter import Parameter
from std_msgs.msg import Float32


class _ParameterPusher:
    """Send, confirm and periodically verify parameters on a Nav2 server."""

    def _init_pusher(self, node):
        self.node = node
        self.clients = {}
        self.read_clients = {}
        self.confirmed = {}
        self.pending = {}
        self.sent_at = {}
        self.confirmed_at = {}

    def _read(self, server, name, on_value):
        """Read one double parameter; on_value(value) once it arrives."""
        key = ('read', server, name)
        if key in self.pending:
            if time.monotonic()-self.sent_at[key] < 2.0:
                return
            self.pending.pop(key).cancel()
        client = self.read_clients.get(server)
        if client is None:
            client = self.node.create_client(GetParameters, f'{server}/get_parameters')
            self.read_clients[server] = client
        if not client.service_is_ready():
            return
        future = client.call_async(GetParameters.Request(names=[name]))
        self.pending[key] = future
        self.sent_at[key] = time.monotonic()
        def done(result):
            if self.pending.get(key) is not future:
                return
            self.pending.pop(key, None)
            try:
                values = result.result().values
                if len(values) != 1 or values[0].type != 3 or not math.isfinite(values[0].double_value):
                    raise RuntimeError(f'{name} is not a finite double')
                on_value(values[0].double_value)
            except Exception as exc:
                self.node.get_logger().warn(f'parameter read failed: {exc}')
        future.add_done_callback(done)

    def _verify(self, target):
        # Check lifecycle resets without rebuilding Smac's A* on every poll.
        if target in self.pending:
            if time.monotonic()-self.sent_at[target] < 2.0:
                return
            self.pending.pop(target).cancel()
        server, name = target
        client = self.read_clients.get(server)
        if client is None:
            client = self.node.create_client(GetParameters, f'{server}/get_parameters')
            self.read_clients[server] = client
        if not client.service_is_ready():
            return
        future = client.call_async(GetParameters.Request(names=[name]))
        self.pending[target] = future
        self.sent_at[target] = time.monotonic()
        def done(result):
            if self.pending.get(target) is not future:
                return
            self.pending.pop(target, None)
            try:
                values = result.result().values
                if len(values) != 1 or values[0].type != 3:
                    raise RuntimeError(f'{name} is not a double')
                actual = values[0].double_value
                if not math.isfinite(actual) or not math.isclose(
                        actual, self.confirmed[target], rel_tol=1e-6, abs_tol=1e-6):
                    self.confirmed.pop(target, None)  # next tick reapplies after reset
                else:
                    self.confirmed_at[target] = time.monotonic()
            except Exception as exc:
                self.node.get_logger().warn(f'{name} verification failed: {exc}')
        future.add_done_callback(done)

    def _send(self, target, value, unit='m'):
        server, name = target
        client = self.clients.get(server)
        if client is None:
            client = self.node.create_client(SetParameters, f'{server}/set_parameters')
            self.clients[server] = client
        if target in self.pending:
            if time.monotonic()-self.sent_at[target] < 2.0:
                return
            expired = self.pending.pop(target)
            expired.cancel()
        if not client.service_is_ready():
            return
        request = SetParameters.Request(parameters=[Parameter(name, Parameter.Type.DOUBLE, float(value)).to_parameter_msg()])
        future = client.call_async(request)
        self.pending[target] = future
        self.sent_at[target] = time.monotonic()
        def done(result):
            if self.pending.get(target) is not future:
                return
            self.pending.pop(target, None)
            try:
                response = result.result()
                if not response or len(response.results) != 1 or not response.results[0].successful:
                    raise RuntimeError(response.results[0].reason if response and response.results else 'empty response')
                self.confirmed[target] = value
                self.confirmed_at[target] = time.monotonic()
                self.node.get_logger().info(f'confirmed learned limit {server} {name}={value:.3f} {unit}')
            except Exception as exc:
                self.node.get_logger().warn(f'{name} update failed; will retry: {exc}')
        future.add_done_callback(done)


class SurfaceCapability(_ParameterPusher):
    """Tell navigation what the surface under the wheels allows.

    Two learned numbers, both measured at every start (core.lunge and
    core.launch_time): the slowest speed the car can hold here, and how
    long a start takes. Nav2's configured values are read back once and
    kept as floors; a learned value only raises them and returns to the
    floor as the surface gets easier, so navigation never asks for a creep
    the car answers with stall-lunge-stall, nor calls a normal grass start
    "stuck".
    """

    def __init__(self, node, parameters):
        self._init_pusher(node)
        self.enabled = bool(parameters['publish_surface_limits'])
        server = str(parameters['controller_server'])
        self.speed_targets = [(server, str(parameters[k])) for k in
                              ('approach_speed_param', 'regulated_min_speed_param') if parameters[k]]
        self.stuck_target = (server, str(parameters['stuck_allowance_param'])) if parameters['stuck_allowance_param'] else None
        self.movement_target = (server, str(parameters['movement_allowance_param'])) if parameters['movement_allowance_param'] else None
        self.enabled = self.enabled and bool(server)
        self.baseline = {}
        self.values = {}

    def _ensure_baseline(self, target):
        if target in self.baseline:
            return True
        self._read(target[0], target[1], lambda v, t=target: self.baseline.__setitem__(t, v))
        return False

    def tick(self):
        if not self.enabled:
            return
        core = self.node.core
        creep = core.creep_speed()
        launch = core.launch_duration()
        # Read every configured floor first; they arrive asynchronously.
        for target in self.speed_targets + [t for t in (self.stuck_target, self.movement_target) if t]:
            self._ensure_baseline(target)
        for target in self.speed_targets:
            if target not in self.baseline:
                continue
            floor = self.baseline[target]
            # Never ask navigation to go faster than the learned operating
            # speed just because a launch lunged; that is the cruise itself.
            value = floor if creep is None else min(max(floor, creep), max(floor, core.v_op or floor))
            self._push(target, value, 0.02, 'm/s')
        if self.stuck_target and self.stuck_target in self.baseline:
            floor = self.baseline[self.stuck_target]
            cap = None
            if self.movement_target and self.movement_target in self.baseline:
                cap = 0.8*self.baseline[self.movement_target]
            value = floor
            if launch is not None:
                # Two learned launches plus the sensing delay: a start that
                # takes twice the usual is not stuck yet.
                value = max(floor, 2.0*launch + core.lon_bank.delay + core.policy.tau_d)
                if cap is not None:
                    value = min(value, max(floor, cap))
            self._push(self.stuck_target, value, 0.5, 's')

    def _push(self, target, value, tolerance, unit):
        self.values[target] = value
        previous = self.confirmed.get(target)
        if previous is None or abs(value-previous) >= tolerance:
            self._send(target, value, unit)
        elif time.monotonic()-self.confirmed_at.get(target, 0) >= 30.0:
            self._verify(target)


class TurningCapability(_ParameterPusher):
    def __init__(self, node, parameters):
        self._init_pusher(node)
        self.enabled = bool(parameters['publish_turning_radius'])
        self.targets = [(str(parameters[s]), str(parameters[p])) for s,p in (
            ('planner_server','planner_radius_param'),
            ('controller_server','controller_radius_param')) if parameters[s] and parameters[p]]
        self.controller_target = (str(parameters['controller_server']), str(parameters['controller_radius_param']))
        self.alpha = float(parameters['radius_filter_alpha'])
        self.relative = float(parameters['radius_rel_change'])
        self.absolute = float(parameters['radius_abs_change'])
        self.filtered = None
        if not 0 < self.alpha <= 1 or min(self.relative,self.absolute) < 0:
            raise ValueError('invalid turning capability filter/change thresholds')

    def tick(self):
        core = self.node.core
        raw = core.envelope.min_turning_radius(core.model)
        if not math.isfinite(raw) or raw <= 0:
            return
        self.filtered = raw if self.filtered is None else self.filtered+self.alpha*(raw-self.filtered)
        self.node.pub_radius.publish(Float32(data=float(self.filtered)))
        horizon = core.stop_horizon()
        if horizon is not None and math.isfinite(horizon) and horizon > 0:
            self.node.pub_stop_horizon.publish(Float32(data=float(horizon)))
        if not self.enabled or not core.envelope.confirmed:
            return
        quoted = self.filtered * core.policy.radius_push_margin
        for target in self.targets:
            value = self.filtered if target == self.controller_target else quoted
            previous = self.confirmed.get(target)
            if previous is None or (abs(value-previous) >= self.absolute and
                                    abs(value-previous) >= self.relative*previous):
                self._send(target, value)
            elif time.monotonic()-self.confirmed_at.get(target,0) >= 30.0:
                self._verify(target)

