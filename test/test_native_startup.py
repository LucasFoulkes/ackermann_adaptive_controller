"""Native and legacy navigation must both initialize the actuator node."""
import pytest


@pytest.mark.parametrize('topic', ['', '/cusp_navigator/segment_direction'])
def test_startup_without_cusp_handshake_keeps_stall_status(topic,tmp_path,monkeypatch):
    rclpy=pytest.importorskip('rclpy')
    monkeypatch.setenv('ROS_DOMAIN_ID','175')
    monkeypatch.setenv('ROS_AUTOMATIC_DISCOVERY_RANGE','LOCALHOST')
    from ackermann_adaptive_controller.node import AckermannAdaptiveController
    rclpy.init(args=['--ros-args','-p','start_active:=false','-p',
                    'segment_direction_topic:='+repr(topic),'-p',
                    'state_file:='+str(tmp_path/'state.yaml'),'-p',"flight_log:=''" ])
    node=None
    try:
        node=AckermannAdaptiveController()
        assert node.pub_stalled is not None
        assert (node.pub_held is not None)==bool(topic)
        assert not node.active
    finally:
        if node is not None:node.destroy_node()
        rclpy.shutdown()
