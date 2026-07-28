from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any


@dataclass
class CompletionMonitorConfig:
    mode: str = "disabled"
    frontier_confirmations: int = 3
    post_completion_hold_steps: int = 0
    semantic_target_requires_distance_and_visibility: bool = False


class CompletionState:
    def __init__(self, config: CompletionMonitorConfig | None = None) -> None:
        self.config = config or CompletionMonitorConfig()
        # Native benchmark episodes publish a latched semantic status.  Keep
        # the expected lifecycle fields outside ``reset`` so a monitor can be
        # reset between rollouts without briefly accepting a terminal status
        # retained from the preceding episode.
        self.semantic_episode_id = ""
        self.semantic_episode_generation: int | None = None
        self.reset()

    def reset(self) -> None:
        self.frontier_confirmations = 0
        self.requested = False
        self.reason = ""
        self.detail: dict[str, Any] = {}
        self.last_semantic_status = ""
        self.target_goal_succeeded = False
        self.target_detail: dict[str, Any] = {}
        self.requested_at_wall_time = 0.0
        self.requested_at_step: int | None = None

    def configure_semantic_episode(
        self,
        *,
        episode_id: str = "",
        episode_generation: int | None = None,
    ) -> None:
        """Restrict semantic completion to one target lifecycle, when set."""

        self.semantic_episode_id = str(episode_id or "")
        if episode_generation is None:
            self.semantic_episode_generation = None
            return
        try:
            self.semantic_episode_generation = int(episode_generation)
        except (TypeError, ValueError):
            self.semantic_episode_generation = None

    def _matches_semantic_episode(self, payload: dict[str, Any]) -> bool:
        if not self.semantic_episode_id and self.semantic_episode_generation is None:
            return True
        target_context = payload.get("target_context")
        if not isinstance(target_context, dict):
            return False
        if (
            self.semantic_episode_id
            and str(target_context.get("episode_id") or "")
            != self.semantic_episode_id
        ):
            return False
        if self.semantic_episode_generation is not None:
            try:
                generation = int(target_context.get("episode_generation"))
            except (TypeError, ValueError):
                return False
            if generation != self.semantic_episode_generation:
                return False
        return True

    def update_frontier(self, payload: dict[str, Any]) -> bool:
        if self.requested or str(self.config.mode) != "frontier":
            return self.requested
        exhausted = bool(payload.get("frontier_exhausted", False))
        active_proposal_id = str(payload.get("active_proposal_id") or "")
        proposal_count = int(
            payload.get("proposal_count", len(payload.get("proposals") or [])) or 0
        )
        ready = bool(payload.get("ready", True))
        stable_empty = ready and exhausted and not active_proposal_id and proposal_count == 0
        self.frontier_confirmations = self.frontier_confirmations + 1 if stable_empty else 0
        if self.frontier_confirmations >= max(
            1, int(self.config.frontier_confirmations)
        ):
            self.request(
                "frontier_exhausted",
                {
                    "frontier_confirmations": self.frontier_confirmations,
                    "proposal_count": proposal_count,
                },
            )
        return self.requested

    def update_semantic(self, payload: dict[str, Any]) -> bool:
        if self.requested or str(self.config.mode) != "semantic":
            return self.requested
        if not self._matches_semantic_episode(payload):
            return self.requested
        status = str(payload.get("status") or "").upper()
        detail = dict(payload.get("detail") or {})
        reason = str(detail.get("reason") or "")
        self.last_semantic_status = status
        mission_mode = str(payload.get("mission_mode") or "").casefold()
        if status == "SUCCEEDED" and reason == "target_goal_succeeded":
            if self.config.semantic_target_requires_distance_and_visibility and not (
                bool(detail.get("target_visibility_passed", detail.get("target_visible", False)))
                and bool(detail.get("target_distance_passed", False))
            ):
                return self.requested
            self.target_goal_succeeded = True
            self.target_detail = detail
            if mission_mode in {"object_goal", "semantic_interaction_object_goal"}:
                self.request("target_goal_succeeded", detail)
                return self.requested
        if status == "EXPLORATION_EXHAUSTED" or (
            status == "SUCCEEDED"
            and reason
            in {
                "exploration_exhausted",
                "navigation_and_interaction_frontiers_exhausted",
            }
        ):
            detail.setdefault("target_goal_succeeded", self.target_goal_succeeded)
            if self.target_detail:
                detail.setdefault("target_detail", dict(self.target_detail))
            self.request(
                reason or "navigation_and_interaction_frontiers_exhausted",
                detail,
            )
        return self.requested

    def request(self, reason: str, detail: dict[str, Any] | None = None) -> None:
        if self.requested:
            return
        self.requested = True
        self.reason = str(reason or "completed")
        self.detail = dict(detail or {})
        self.requested_at_wall_time = time.time()

    def should_stop(self, completed_steps: int) -> bool:
        if not self.requested:
            return False
        if self.requested_at_step is None:
            self.requested_at_step = int(completed_steps)
        return int(completed_steps) >= self.requested_at_step + max(
            0, int(self.config.post_completion_hold_steps)
        )

    def is_holding(self, completed_steps: int) -> bool:
        return self.requested and not self.should_stop(completed_steps)

    def to_dict(self, completed_steps: int | None = None) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "requested": self.requested,
            "reason": self.reason,
            "detail": dict(self.detail),
            "last_semantic_status": self.last_semantic_status,
            "target_goal_succeeded": self.target_goal_succeeded,
            "target_detail": dict(self.target_detail),
            "semantic_episode_filter": {
                "episode_id": self.semantic_episode_id,
                "episode_generation": self.semantic_episode_generation,
            },
            "frontier_confirmations": self.frontier_confirmations,
            "requested_at_wall_time": self.requested_at_wall_time,
            "requested_at_step": self.requested_at_step,
            "completed_steps": completed_steps,
        }


class RosCompletionMonitor:
    def __init__(
        self,
        config: CompletionMonitorConfig,
        frontier_topic: str = "/explore_py/proposals",
        semantic_topic: str = "/semantic_decision/goal_status",
        output_path: str | Path | None = None,
    ) -> None:
        self.state = CompletionState(config)
        self.frontier_topic = str(frontier_topic)
        self.semantic_topic = str(semantic_topic)
        self.output_path = Path(output_path).expanduser().resolve() if output_path else None
        self.lock = threading.Lock()
        self.subscriber = None
        self._ensure_ros()

    def _ensure_ros(self) -> None:
        import rospy
        from std_msgs.msg import String

        mode = str(self.state.config.mode)
        if mode == "frontier":
            self.subscriber = rospy.Subscriber(
                self.frontier_topic, String, self._frontier_callback, queue_size=4
            )
        elif mode == "semantic":
            self.subscriber = rospy.Subscriber(
                self.semantic_topic, String, self._semantic_callback, queue_size=4
            )

    def prepare(self) -> None:
        with self.lock:
            self.state.reset()
            self._write_snapshot(None)

    def configure_semantic_episode(
        self,
        *,
        episode_id: str = "",
        episode_generation: int | None = None,
    ) -> None:
        with self.lock:
            self.state.configure_semantic_episode(
                episode_id=episode_id,
                episode_generation=episode_generation,
            )
            self._write_snapshot(None)

    def should_stop(self, completed_steps: int) -> bool:
        with self.lock:
            should_stop = self.state.should_stop(completed_steps)
            self._write_snapshot(completed_steps)
            return should_stop

    def is_holding(self, completed_steps: int) -> bool:
        with self.lock:
            return self.state.is_holding(completed_steps)

    def finalize(self, completed_steps: int) -> None:
        with self.lock:
            self._write_snapshot(completed_steps)

    def close(self) -> None:
        if self.subscriber is not None:
            self.subscriber.unregister()
            self.subscriber = None

    def _frontier_callback(self, message) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self.lock:
            self.state.update_frontier(payload)
            self._write_snapshot(None)

    def _semantic_callback(self, message) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self.lock:
            self.state.update_semantic(payload)
            self._write_snapshot(None)

    def _write_snapshot(self, completed_steps: int | None) -> None:
        if self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(
                self.state.to_dict(completed_steps),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
