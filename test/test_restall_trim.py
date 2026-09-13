"""Regression for repeated reverse stalls with an underestimated cruise effort."""
import pytest
from ackermann_adaptive_controller.core import AdaptiveCore
from replay_reverse_stiction import trial

@pytest.mark.parametrize('direction', [-1., 1.])
def test_stale_cruise_estimate_recovers_without_repeated_stalls(direction):
    result = trial(direction, kinetic=.30)
    assert result['moving_fraction'] > .95
    assert .20 < result['mean_speed'] < .30
    assert result['peak_speed'] < .40

@pytest.mark.parametrize('direction', [-1., 1.])
def test_correct_cruise_estimate_keeps_original_launch(direction):
    result = trial(direction, kinetic=.225)
    assert result['moving_fraction'] == 1.
    assert result['peak_speed'] == pytest.approx(.3401342696, abs=.002)

@pytest.mark.parametrize('command,passive', [(0.,False),(-.25,False),(.25,True)])
def test_command_boundary_discards_restall_state(command, passive):
    core=AdaptiveCore();core._cmd_dir=1.;core._leg_rolled=True
    core.step(1.,0.,0.,0.,command,0.,passive=passive)
    assert not core._leg_rolled
