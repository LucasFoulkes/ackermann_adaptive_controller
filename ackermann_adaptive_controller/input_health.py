"""Acquisition-time and frame checks for control odometry."""
import math
import time


class OdomHealth:
    def __init__(self, max_age, odom_frame, base_frame):
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError('odom_max_age must be positive and finite')
        self.max_age = max_age
        self.odom_frame, self.base_frame = odom_frame, base_frame
        self.stamp = self.arrival = self.last_clock = None
        self.reason = 'waiting for odometry'
        self.clock_fault = False

    def accept(self, msg, now):
        if self.clock_fault:
            return False
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pose = msg.pose.pose
        q = pose.orientation
        values = (stamp, now, pose.position.x, pose.position.y, pose.position.z, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(v) for v in values):
            self.reason = 'non-finite odometry'
            return False
        if self.last_clock is not None and now < self.last_clock:
            self.stamp = self.arrival = None
            self.last_clock = now
            self.clock_fault = True
            self.reason = 'clock reset; reset controller before resuming'
            return False
        self.last_clock = now
        if not 0 <= now-stamp <= self.max_age:
            self.reason = 'odometry acquisition stale or future'
            return False
        if msg.header.frame_id != self.odom_frame or msg.child_frame_id != self.base_frame:
            self.reason = 'odometry frame mismatch'
            return False
        if abs(sum(v*v for v in (q.x,q.y,q.z,q.w))-1.0) > .05:
            self.reason = 'invalid odometry orientation'
            return False
        if self.stamp is not None and stamp <= self.stamp:
            self.reason = 'odometry timestamp did not advance'
            return False
        self.stamp, self.arrival = stamp, time.monotonic()
        self.reason = 'ok'
        return True

    def fresh(self, now):
        return (self.reason == 'ok' and self.stamp is not None
                and 0 <= now-self.stamp <= self.max_age
                and 0 <= time.monotonic()-self.arrival <= self.max_age)
