"""Control/ROS boundary failures, independent of physical motor hardware."""
from concurrent.futures import Future
from types import SimpleNamespace as NS
import threading
import pytest

from ackermann_adaptive_controller.input_health import OdomHealth
from ackermann_adaptive_controller.delivery import DeliveryHistory
from ackermann_adaptive_controller.telemetry import BackgroundIO
from ackermann_adaptive_controller.core import AdaptiveCore


def odom(stamp=10., frame='odom'):
    return NS(header=NS(stamp=NS(sec=int(stamp),nanosec=int(round((stamp%1)*1e9))),frame_id=frame),
              child_frame_id='base_link',pose=NS(pose=NS(position=NS(x=0.,y=0.,z=0.),orientation=NS(x=0.,y=0.,z=0.,w=1.))))


def test_measurement_age_frames_and_progress_are_required():
    h=OdomHealth(.5,'odom','base_link')
    assert not h.accept(odom(8.),10.)
    assert not h.accept(odom(10.,'map'),10.)
    assert h.accept(odom(10.),10.1)
    assert h.fresh(10.2)
    assert not h.accept(odom(10.),10.2)
    assert not h.fresh(10.2)
    assert h.accept(odom(10.2),10.3)
    assert not h.fresh(11.)


def test_invalid_command_revokes_previous_output_immediately():
    from ackermann_adaptive_controller.node import AckermannAdaptiveController
    zero=[]
    n=NS(active=True,cmd_v=.3,cmd_w=0.,out_steer=.2,out_drive=.4,last_cmd_t=1.,
         get_logger=lambda:NS(warn=lambda *_:None),_zero_burst=lambda:zero.append(True))
    AckermannAdaptiveController.on_cmd(n,NS(linear=NS(x=float('nan')),angular=NS(z=0.)))
    assert (n.cmd_v,n.out_drive,n.out_steer,n.last_cmd_t)==(0.,0.,0.,None)
    assert zero


def test_input_limits_preserve_requested_curvature():
    from ackermann_adaptive_controller.node import AckermannAdaptiveController
    n=NS(motion_speed_limit=.38,motion_yaw_rate_limit=.45)
    AckermannAdaptiveController.on_cmd(n,NS(linear=NS(x=1.),angular=NS(z=.5)))
    assert n.cmd_v==pytest.approx(.38)
    assert n.cmd_w/n.cmd_v==pytest.approx(.5)


def test_writer_failure_does_not_escape_or_prevent_next_record():
    writer=BackgroundIO(); seen=[]
    def fail():raise OSError('disk full')
    writer.submit(fail)
    writer.submit(seen.append,'next')
    writer.close()
    assert 'disk full' in writer.error
    assert seen==['next']


def test_writer_queue_is_bounded_and_submit_never_waits_for_disk():
    writer=BackgroundIO(capacity=1); started=threading.Event(); release=threading.Event()
    def block():started.set();release.wait(2.)
    writer.submit(block);assert started.wait(1.)
    assert writer.submit(lambda:None)
    assert not writer.submit(lambda:None)
    assert writer.dropped==1
    release.set();writer.close()


def test_failed_radius_set_retries_same_value_then_confirms():
    from ackermann_adaptive_controller.nav2_capability import TurningCapability
    calls=[]
    def call(req):
        f=Future();calls.append(f);return f
    client=NS(service_is_ready=lambda:True,call_async=call)
    n=NS(create_client=lambda *_:client,get_logger=lambda:NS(info=lambda *_:None,warn=lambda *_:None))
    cap=TurningCapability(n,dict(publish_turning_radius=True,planner_server='/planner',planner_radius_param='radius',
        controller_server='',controller_radius_param='',radius_filter_alpha=.25,radius_rel_change=.1,radius_abs_change=.05))
    target=('/planner','radius')
    cap._send(target,.9);assert target not in cap.confirmed
    calls[-1].set_result(NS(results=[NS(successful=False,reason='configuring')]))
    cap._send(target,.9);assert len(calls)==2
    calls[-1].set_result(NS(results=[NS(successful=True)]))
    assert cap.confirmed[target]==.9


def test_delivery_history_uses_successful_send_time_and_disconnect():
    hist=DeliveryHistory()
    def msg(t,connected=True):
        return NS(header=odom(t).header,connected=connected,command_conflict=False,steering_active=True,throttle_active=True,steering=.2,throttle=.3)
    assert hist.accept(msg(10.),10.1)==(10.,.2,.3)
    assert hist.at(9.9) is None
    assert hist.at(10.1)==(.2,.3)
    assert hist.accept(msg(10.1,False),10.2)==(10.1,0.,0.)
    assert not hist.fresh(10.2)
    assert hist.at(10.1)==(0.,0.)


def test_external_history_is_not_overwritten_by_generated_output():
    core=AdaptiveCore();core.external_history=True
    core.record_command(1.,.2,.3)
    core.now=1.1
    core._publish(.8,.9,False,False,.1)
    assert list(core._cmd_hist)==[(1.,.2,.3)]


def test_clock_reset_is_latched_until_explicit_reset():
    h=OdomHealth(.5,'odom','base_link')
    assert h.accept(odom(10.),10.)
    assert not h.accept(odom(5.),5.)
    assert not h.accept(odom(5.1),5.1)
    assert h.clock_fault
