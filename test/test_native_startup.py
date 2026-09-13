"""The production controller starts passive without legacy navigation hooks."""
import pytest


@pytest.mark.parametrize('validate', [False, True])
def test_native_startup(tmp_path, monkeypatch, validate):
    rclpy = pytest.importorskip('rclpy')
    monkeypatch.setenv('ROS_DOMAIN_ID', '175')
    monkeypatch.setenv('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST')
    from ackermann_adaptive_controller.node import AckermannAdaptiveController
    rclpy.init(args=['--ros-args', '-p', 'start_active:=false', '-p',
                    'state_file:='+str(tmp_path/'state.yaml'), '-p', "flight_log:=''",
                    '-p', 'validate_lon:='+str(validate).lower() ])
    node = None
    try:
        node = AckermannAdaptiveController()
        assert node.pub_stalled is not None
        assert not node.active
        assert node.core.policy.validate_lon is validate
        assert not node.has_parameter('goal_tolerance_param')
        assert not node.has_parameter('segment_direction_topic')
        assert node.capability.targets == [('/planner_server', 'GridBased.minimum_turning_radius')]
    finally:
        if node is not None:
            node.io.close()
            node.destroy_node()
        rclpy.shutdown()
