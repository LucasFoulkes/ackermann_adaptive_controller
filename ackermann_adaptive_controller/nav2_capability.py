"""Report learned turning limits; leave navigation policy in Nav2 configuration."""
import math
import time
from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter
from std_msgs.msg import Float32


class TurningCapability:
    def __init__(self, node, parameters):
        self.node = node
        self.enabled = bool(parameters['publish_turning_radius'])
        self.targets = [(str(parameters[s]), str(parameters[p])) for s,p in (
            ('planner_server','planner_radius_param'),
            ('controller_server','controller_radius_param')) if parameters[s] and parameters[p]]
        self.alpha = float(parameters['radius_filter_alpha'])
        self.relative = float(parameters['radius_rel_change'])
        self.absolute = float(parameters['radius_abs_change'])
        self.filtered = None
        self.clients = {}
        self.confirmed = {}
        self.pending = {}
        self.sent_at = {}
        self.confirmed_at = {}
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
            previous = self.confirmed.get(target)
            if previous is None or time.monotonic()-self.confirmed_at.get(target,0) >= 30.0 or (abs(quoted-previous) >= self.absolute and
                                    abs(quoted-previous) >= self.relative*previous):
                self._send(target, quoted)

    def _send(self, target, value):
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
                self.node.get_logger().info(f'confirmed turning limit {server} {name}={value:.3f} m')
            except Exception as exc:
                self.node.get_logger().warn(f'turning limit update failed; will retry: {exc}')
        future.add_done_callback(done)
