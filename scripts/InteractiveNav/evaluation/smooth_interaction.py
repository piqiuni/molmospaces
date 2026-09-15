"""Run the native interaction state machine with evaluator-owned observations."""

from typing import Any, Callable

from scripts.InteractiveNav.force_interaction_bridge import AtomicForceInteractionController


def run_smooth_interaction(
    task: Any,
    command: dict,
    *,
    object_name: str,
    step: Callable,
    max_steps: int = 600,
) -> dict:
    # Do not subscribe to ROS here: the restricted adapter owns command routing
    # and publishes the sanitized result after private scoring.
    controller = AtomicForceInteractionController(
        interaction_execution_mode="smooth",
        interaction_transition_steps=5,
        drawer_observation_steps=10,
        bypass_unsafe_open_sweep=True,
        close_all_doors_on_prepare=False,
        object_id_resolver=lambda _: object_name,
    )
    controller.enqueue_command(command)
    result = None
    consumed = 0
    for index in range(max_steps):
        result = controller.before_step(task, index)
        if result is not None:
            break
        terminal = step(controller, index)
        consumed += 1
        if terminal is True:
            return {"result": {"success": True, "state": "open",
                               "interrupted_by_goal_status": True},
                    "task_steps_consumed": consumed, "events": controller._events,
                    "execution_mode": "ordinary_native_smooth"}
        result = controller.after_step(task, index)
        controller.after_task_step()
        if result is not None:
            break
    if result is None:
        raise RuntimeError("native smooth interaction exceeded its bounded task-step budget")
    if controller._restore_view_pending:
        controller.before_step(task, consumed)
        step(controller, consumed)
        consumed += 1
        controller.after_task_step()
    return {"result": result, "task_steps_consumed": consumed,
            "events": controller._events, "execution_mode": "ordinary_native_smooth"}
