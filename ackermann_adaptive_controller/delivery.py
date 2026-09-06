"""Serial-delivery history; acknowledgments describe writes, not wheel motion."""
from collections import deque
import math
import time


class DeliveryHistory:
    def __init__(self, timeout=.3):
        self.timeout = timeout
        self.samples = deque(maxlen=1024)
        self.arrival = None
        self.connected = False

    def accept(self, msg, now):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if not all(math.isfinite(v) for v in (stamp,msg.steering,msg.throttle,now)) or not 0 <= now-stamp <= self.timeout:
            return None
        if self.samples and stamp <= self.samples[-1][0]:
            return None
        if abs(msg.steering) > 2 or abs(msg.throttle) > 1:
            return None
        steering = msg.steering if msg.connected and msg.steering_active else 0.0
        throttle = msg.throttle if msg.connected and msg.throttle_active else 0.0
        sample = (stamp, float(steering), float(throttle))
        self.samples.append(sample)
        self.arrival, self.connected = time.monotonic(), msg.connected and not msg.command_conflict
        return sample

    def fresh(self, now):
        return (self.connected and self.arrival is not None and self.samples
                and 0 <= now-self.samples[-1][0] <= self.timeout
                and time.monotonic()-self.arrival <= self.timeout)

    def at(self, stamp):
        for t, steer, drive in reversed(self.samples):
            if t <= stamp:
                return (steer,drive) if stamp-t <= self.timeout else None
        return None
