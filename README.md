# Adaptive Ackermann actuator controller

This node tracks requested linear velocity and yaw rate using odometry
feedback. It learns steering response, throttle response, friction/deadband and
delay; it converts requested motion to normalized steering/throttle commands.

## Interfaces

- `/cmd_vel` (`Twist`): body motion request; stale or invalid commands stop output.
- `/odom` in the robot launch (`Odometry`): rear-axle pose in `odom`, child `base_link`.
  Acquisition age, frame, orientation and timestamp progression are checked.
- `/actuators/state` (`robot_interfaces/ActuatorState`): timestamped commands
  successfully written by the Pico driver. Missing/disconnected delivery stops
  actuation and suspends learning. It is not confirmation of physical motion.
- `/actuators/steering/command` and `/actuators/throttle/command` (`Float32`):
  normalized outputs, refreshed at 50 Hz; control updates on new odometry.
- `/diagnostics`: mode, input failures, stall/steering hold, tracking and model health.
- `~/min_turning_radius`, `~/stop_horizon`: estimates for inspection.
- `/speed_limit`: periodic confidence-based Nav2 speed limit.

Confirmed turning capability updates Nav2's planner radius (and MPPI's radius
constraint when selected). Updates are acknowledged and retried. The controller
**does not change** goal tolerance, lookahead, approach speed or collision
horizon. These belong to Nav2 configuration. `stop_horizon` is informational.
Reverse turns retain separate limit evidence so later forward driving cannot
erase a measured reverse limitation. Existing saved profiles remain compatible;
new reverse evidence accumulates during normal driving.

## Modes and files

`robot.launch.py` defaults active, but no movement is initiated without a motion
request. Standalone node defaults passive. Calibration runs only when requested:

```sh
ros2 service call /ackermann_adaptive_controller/set_active std_srvs/srv/SetBool '{data: false}'
ros2 service call /ackermann_adaptive_controller/calibrate std_srvs/srv/Trigger '{}'
```

Calibration requires active mode and completed stationary sensing. `/reset`
forgets the model, clears faults, and returns passive. A ROS clock reset requires
an explicit controller reset or a fresh launch.

The learned profile is `~/.ros/ackermann_adaptive_controller.yaml`; normal code
updates preserve it. Flight data is `~/.ros/ackermann_flight.csv`. Both writes
use bounded background I/O. Logging failure is diagnostic, not a control fault.

```sh
ros2 run ackermann_adaptive_controller ackermann_flight_report --help
```

## Code layout

- `node.py`: ROS inputs, modes, outputs and diagnostics.
- `core.py`: control execution, feedback and reversal/stall handling.
- `policy.py`: tunable defaults; `math_utils.py`: scalar validation/helpers.
- `motion.py`: signed pose-to-twist estimation.
- `identification.py`: constrained RLS and delay candidates.
- `response.py`: learned response and observed capabilities.
- `scoring.py`: tracking metrics.
- `input_health.py`, `delivery.py`: acquisition and serial-delivery contracts.
- `nav2_capability.py`: confirmed turning-limit updates only.
- `telemetry.py`, `flight_report.py`: recording and analysis.

Tests use a small numerical vehicle model and isolated ROS initialization.
They are regression evidence, not proof of real-world navigation performance.

Startup-to-cruise control preserves the throttle correction missing from the
active feedforward model. If a fitted cruising command opposes observed motion,
the controller uses measured cruising throttle with PI feedback while learning
continues. It bounds the combined fitted compensation rather than independently
clipping terms that may cancel each other.
During isolated implausible odometry samples, settled cruise trim is retained
with feedforward, bounded by the previous throttle. Acceleration feedback and
launch windup are excluded. Stop or direction-change requests override the hold;
a sustained implausible stream still stops output.

A requested direction change while moving brakes toward zero speed, waits for
stationary odometry across the estimated response delay, then launches the new
leg. Diagnostics report `stopping before changing direction`. Steering trim is
not updated using measurements from the opposite travel direction.
