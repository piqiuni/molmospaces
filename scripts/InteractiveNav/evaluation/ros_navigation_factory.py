"""Factory adapter for the current ROS interactive-navigation system.

The user's navigation algorithm is a ROS graph, rather than an in-process
Python policy.  Its active entrypoints are:

* ``nav_pkg/molmospaces_nav_system.launch`` with ``start_sim:=false``;
* ``explore_py_pkg/scripts/explore_py_node.py`` for frontier selection;
* ``move_base`` plus ``nav_pkg/relay_node`` for ``/cmd_vel`` to
  ``/cmd_vel_stamped`` conversion.

This factory attaches the standalone V3 evaluator's private MuJoCo replay to
that already-running graph through ``RosBridgePolicy``.  It deliberately does
not launch ROS, does not receive a task object, and does not publish realtime
GT.  Thus the graph sees exactly the same RGB/depth/point-cloud/odometry topics
as its ordinary live simulator run, but no ``interactive_nav`` record, object
names, controlling joints, oracle plan, or segmentation mask.

It is usable through the evaluator's generic factory protocol:

.. code-block:: bash

   --policy factory \\
   --policy-factory scripts.InteractiveNav.evaluation.ros_navigation_factory:create_current_ros_navigation_policy

For the default stack, ``--policy ros_bridge`` is an equivalent shorter
spelling.  Keeping this factory makes the concrete external-algorithm contract
explicit and provides a stable integration point if the ROS stack later gains
its own adapter parameters.
"""

from __future__ import annotations

from typing import Any, Mapping

from .benchmark_policies import RosBridgePolicyAdapter, build_ros_bridge_policy


CURRENT_ROS_NAVIGATION_FACTORY = (
    "scripts.InteractiveNav.evaluation.ros_navigation_factory:"
    "create_current_ros_navigation_policy"
)


def create_current_ros_navigation_policy(
    *,
    public_episode: Mapping[str, Any] | None = None,
    policy_dt_ms: float = 200.0,
    observation_topic: str = "/molmo_spaces/head_camera/image",
    action_topic: str = "/molmo_spaces/action",
    action_timeout_s: float = 5.0,
    cmd_vel_linear_gain: float = 3.0,
    require_move_base_active: bool = True,
    map_warmup_skip_frames: int = 0,
) -> RosBridgePolicyAdapter:
    """Return a GT-free adapter to an already-running ROS navigation graph.

    ``public_episode`` is accepted only to conform to the generic factory
    contract and is intentionally discarded.  It contains no V3 annotations,
    but the ROS algorithm does not need it for online navigation either.
    """

    del public_episode
    return build_ros_bridge_policy(
        policy_dt_ms=float(policy_dt_ms),
        observation_topic=str(observation_topic),
        action_topic=str(action_topic),
        action_timeout_s=float(action_timeout_s),
        cmd_vel_linear_gain=float(cmd_vel_linear_gain),
        require_move_base_active=bool(require_move_base_active),
        map_warmup_skip_frames=int(map_warmup_skip_frames),
    )


# A concise alias is convenient for factory specifications in experiment YAML.
build_policy = create_current_ros_navigation_policy
