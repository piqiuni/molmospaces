"""Logical-time timeout rules for the simulator-synchronous startup scan."""

from __future__ import annotations


def startup_scan_elapsed_control_s(
    acknowledged_control_steps: int,
    control_dt_s: float,
) -> float:
    """Return simulator control time confirmed by bridge step acknowledgements."""

    return max(0, int(acknowledged_control_steps)) * max(0.0, float(control_dt_s))


def startup_scan_timeout_reason(
    *,
    acknowledged_control_steps: int,
    control_dt_s: float,
    timeout_s: float,
    awaiting_ack_step: int | None,
    last_command_sent_monotonic_s: float | None,
    now_monotonic_s: float,
    step_sync_stall_timeout_s: float,
) -> str:
    """Return an explicit scan failure without conflating wall and sim time.

    ``scan_timeout`` is measured only in acknowledged evaluator control steps.
    Wall time is used solely after a command was emitted and its matching
    ``step_sync`` never arrives, which identifies a bridge/ROS stall rather
    than slow but healthy simulation throughput.
    """

    if (
        awaiting_ack_step is not None
        and last_command_sent_monotonic_s is not None
        and float(step_sync_stall_timeout_s) > 0.0
        and float(now_monotonic_s) - float(last_command_sent_monotonic_s)
        >= float(step_sync_stall_timeout_s)
    ):
        return "scan_step_sync_stall"
    if (
        float(timeout_s) > 0.0
        and startup_scan_elapsed_control_s(
            acknowledged_control_steps, control_dt_s
        ) >= float(timeout_s)
    ):
        return "scan_timeout"
    return ""
