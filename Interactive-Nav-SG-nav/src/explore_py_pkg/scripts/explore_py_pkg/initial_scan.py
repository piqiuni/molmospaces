"""Evaluator-step gating for the ExplorePy startup scan."""

from __future__ import annotations


def initial_scan_rgb_step_gate(
    *,
    last_sent_rgb_step_seq: int | None,
    current_rgb_step_seq: int | None,
    latest_step_sync_index: int | None,
    nonzero_commands_sent: int,
    max_control_steps: int,
) -> str:
    """Return whether a 360-degree startup scan may send one command.

    RosBridge publishes RGB for observation ``N`` before waiting for the
    command that produces action ``N``; its ``step_sync(N)`` arrives only
    after that action.  Therefore the first command only requires RGB, while
    every later command requires both a newer RGB frame and the step-sync
    acknowledgement for the preceding command.  This prevents a timer from
    injecting several commands into one evaluator action wait.
    """

    if current_rgb_step_seq is None:
        return "wait"
    # Do not declare the cap until the final command has been acknowledged by
    # the following evaluator observation.  Otherwise the 10 Hz housekeeping
    # timer can stop the scan before the last cmd_vel has affected odometry.
    if int(nonzero_commands_sent) >= max(0, int(max_control_steps)):
        if (
            last_sent_rgb_step_seq is not None
            and int(current_rgb_step_seq) > int(last_sent_rgb_step_seq)
            and latest_step_sync_index is not None
            and int(latest_step_sync_index) >= int(last_sent_rgb_step_seq)
        ):
            return "stop"
        return "wait"
    if last_sent_rgb_step_seq is None:
        return "send"
    if int(current_rgb_step_seq) < int(last_sent_rgb_step_seq):
        return "stop"
    if int(current_rgb_step_seq) == int(last_sent_rgb_step_seq):
        return "wait"
    if (
        latest_step_sync_index is None
        or int(latest_step_sync_index) < int(last_sent_rgb_step_seq)
    ):
        return "wait"
    return "send"
