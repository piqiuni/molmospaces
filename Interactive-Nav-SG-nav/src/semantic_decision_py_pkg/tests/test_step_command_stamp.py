import pytest

from semantic_decision_py_pkg.step_command_gate import StepCommandGate


@pytest.mark.parametrize('image_first', [True, False])
def test_extra_macro_images_do_not_change_logical_command_steps(image_first):
    gate = StepCommandGate()
    for step in range(100):
        stamp = 1789412000.0 + step
        # Extra RGB can depict open/close views without authorizing navigation.
        gate.record_rgb_stamp(stamp + .01, now=step)
        gate.record_rgb_stamp(stamp + .02, now=step)
        if image_first:
            assert gate.record_rgb_stamp(stamp, now=step) is None
            assert gate.record_fresh_gate(step, stamp_sec=stamp, now=step)
        else:
            assert not gate.record_fresh_gate(step, stamp_sec=stamp, now=step)
            assert gate.consume_step(now=step) is None
            assert gate.record_rgb_stamp(stamp, now=step) == step
        assert gate.consume_step(now=step) == step
        assert gate.consume_step(now=step) is None
        gate.record_step_sync(step, action_source='cmd_vel')
        assert gate.take_acks()[0].command_applied


def test_different_source_stamp_cannot_authorize_command():
    gate = StepCommandGate()
    gate.record_rgb_stamp(20.0, now=1.)
    gate.record_fresh_gate(7, stamp_sec=20.01, now=1.)
    assert gate.consume_step(now=1.) is None
    gate.record_rgb_stamp(20.01, now=1.)
    assert gate.consume_step(now=1.) == 7
    gate.record_step_sync(7, action_source='cmd_vel')
    gate.record_rgb_stamp(20.01, now=1.)
    gate.record_fresh_gate(7, stamp_sec=20.01, now=1.)
    assert gate.consume_step(now=1.) is None
    gate.reset()
    gate.record_fresh_gate(0, stamp_sec=20.01, now=2.)
    assert gate.consume_step(now=2.) is None
