from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .behavior_candidates import BehaviorCandidate


@dataclass
class MissionCompletionConfig:
    empty_candidate_confirmations: int = 3
    empty_candidate_min_steps: int = 50
    stagnation_failure_limit: int = 0


class MissionCompletionTracker:
    def __init__(self, config: MissionCompletionConfig | None = None) -> None:
        self.config = config or MissionCompletionConfig()
        self.reset()

    def reset(self) -> None:
        self.confirmations = 0
        self.last_sequence = -1
        self.complete = False
        self.reason = ""
        self.failure_streak = 0
        self.stalled = False
        self.empty_since_step: int | None = None

    def note_feedback(self, feedback: dict[str, Any]) -> None:
        status = str(feedback.get("status") or "")
        behavior_type = str(feedback.get("behavior_type") or "")
        detail = feedback.get("detail") or {}
        event = str(detail.get("event") or "") if isinstance(detail, dict) else ""
        no_progress_success = event == "frontier_unreachable_after_viewpoint_reached"
        if behavior_type not in {"EXPLORE", "NAVIGATE"}:
            return
        if status in {"FAILED", "REJECTED", "CANCELED"} or no_progress_success:
            self.failure_streak += 1
        elif status == "SUCCEEDED":
            self.failure_streak = 0

    def update(
        self,
        candidates_payload: dict[str, Any],
        *,
        has_active_behavior: bool,
        target_enabled: bool,
    ) -> bool:
        if self.complete:
            return True
        sequence = int(candidates_payload.get("sequence", 0) or 0)
        if sequence == self.last_sequence:
            return False
        self.last_sequence = sequence
        self.stalled = False
        exploration = candidates_payload.get("exploration_context") or {}
        initial_scan_complete = bool(exploration.get("initial_scan_complete", True))
        candidates = list(candidates_payload.get("candidates") or [])
        fallback_navigation_count = sum(
            str(candidate.get("behavior_type") or "") == "EXPLORE"
            for candidate in candidates
        )
        fallback_interaction_count = sum(
            str(candidate.get("behavior_type") or "") == "INTERACT"
            and not bool(
                (candidate.get("metadata") or {}).get(
                    "interaction_group_already_explored"
                )
            )
            for candidate in candidates
        )
        fallback_candidate_count = int(
            candidates_payload.get("candidate_count", len(candidates)) or 0
        )
        navigation_frontier_count = int(
            exploration.get(
                "navigation_frontier_count",
                fallback_navigation_count
                if candidates
                else fallback_candidate_count,
            )
            or 0
        )
        interaction_frontier_count = int(
            exploration.get(
                "interaction_frontier_count", fallback_interaction_count
            )
            or 0
        )
        combined_frontier_count = int(
            exploration.get(
                "combined_frontier_count",
                navigation_frontier_count + interaction_frontier_count,
            )
            or 0
        )
        navigation_frontier_exhausted = bool(
            exploration.get(
                "navigation_frontier_exhausted",
                exploration.get("frontier_exhausted", False),
            )
        )
        interaction_frontier_exhausted = bool(
            exploration.get(
                "interaction_frontier_exhausted",
                interaction_frontier_count == 0,
            )
        )
        exhausted = bool(
            exploration.get(
                "frontier_exhausted",
                navigation_frontier_exhausted and interaction_frontier_exhausted,
            )
        )
        ready_to_complete = (
            not has_active_behavior
            and initial_scan_complete
            and exhausted
            and navigation_frontier_exhausted
            and interaction_frontier_exhausted
            and combined_frontier_count == 0
        )
        observation_step = exploration.get("observation_step")
        try:
            observation_step = int(observation_step)
        except (TypeError, ValueError):
            observation_step = None
        stagnated = (
            not has_active_behavior
            and int(self.config.stagnation_failure_limit) > 0
            and self.failure_streak >= int(self.config.stagnation_failure_limit)
            and combined_frontier_count > 0
        )
        if stagnated:
            self.reason = "exploration_stalled_recovery"
            self.stalled = True
            self.failure_streak = 0
            self.confirmations = 0
            return False
        if not ready_to_complete:
            self.confirmations = 0
            self.empty_since_step = None
            return False
        self.confirmations += 1
        if observation_step is not None:
            if self.empty_since_step is None:
                self.empty_since_step = observation_step
            elapsed_empty_steps = observation_step - self.empty_since_step
            self.complete = elapsed_empty_steps >= max(
                0, int(self.config.empty_candidate_min_steps)
            )
        else:
            self.complete = self.confirmations >= max(
                1, int(self.config.empty_candidate_confirmations)
            )
        if self.complete:
            self.reason = "navigation_and_interaction_frontiers_exhausted"
        return self.complete


class TargetMissionTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.pending_interaction: dict[str, str] | None = None

    @staticmethod
    def _normalized(value: Any) -> str:
        return " ".join(str(value or "").casefold().replace("_", " ").split())

    def matches_target_interaction(
        self,
        *,
        target_context: dict[str, Any],
        feedback: dict[str, Any],
        candidates: list[dict[str, Any]] | None = None,
    ) -> bool:
        if not bool(target_context.get("enabled", True)):
            return False
        if self.matches_target_container_interaction(
            target_context=target_context,
            feedback=feedback,
        ):
            return True
        target_id = self._normalized(
            feedback.get("target_id") or feedback.get("node_id")
        )
        target_name = self._normalized(
            feedback.get("target_name")
            or feedback.get("object_id")
            or feedback.get("source_object_name")
            or feedback.get("object_name")
        )
        for candidate in candidates or []:
            metadata = candidate.get("metadata") or {}
            if not bool(metadata.get("target_goal")):
                continue
            candidate_id = self._normalized(candidate.get("target_id"))
            candidate_name = self._normalized(candidate.get("target_name"))
            if (target_id and candidate_id == target_id) or (
                target_name and candidate_name == target_name
            ):
                return True
        labels = [
            target_context.get("target_name"),
            *(target_context.get("object_labels") or []),
        ]
        target_text = f"{target_id} {target_name}".strip()
        return bool(target_text) and any(
            self._normalized(label) and self._normalized(label) in target_text
            for label in labels
        )

    def matches_target_container_interaction(
        self,
        *,
        target_context: dict[str, Any],
        feedback: dict[str, Any],
    ) -> bool:
        requested_source = self._normalized(
            target_context.get("target_container_source_object_name")
        )
        requested_instance = self._normalized(
            target_context.get("target_container_instance_id")
        )
        feedback_source = self._normalized(
            feedback.get("object_id")
            or feedback.get("source_object_name")
            or feedback.get("target_name")
            or (feedback.get("interaction_result") or {}).get("object_id")
            or (feedback.get("interaction_result") or {}).get("source_object_name")
        )
        feedback_instance = self._normalized(
            feedback.get("instance_id")
            or (feedback.get("interaction_result") or {}).get("instance_id")
        )
        if requested_source and requested_source == feedback_source:
            return True
        if requested_instance and requested_instance == feedback_instance:
            return True
        return False

    def on_behavior_succeeded(
        self,
        *,
        behavior_type: str,
        active_target_goal: bool,
        target_context: dict[str, Any],
        feedback: dict[str, Any],
        next_candidate_sequence: int,
    ) -> dict[str, Any]:
        if not active_target_goal:
            return {"phase": "none"}
        require_interaction = bool(target_context.get("require_interaction", False))
        if str(behavior_type) == "INTERACT" and self.matches_target_container_interaction(
            target_context=target_context,
            feedback=feedback,
        ):
            return {
                "phase": "container_opened",
                "detail": {
                    **(feedback.get("detail") or {}),
                    "target_container_interaction_complete": True,
                    "next_phase": "NAVIGATE_TARGET_OBJECT",
                },
            }
        if (
            str(behavior_type) == "NAVIGATE"
            and require_interaction
            and not target_context.get("target_container_source_object_name")
        ):
            self.pending_interaction = {
                "target_id": str(feedback.get("target_id") or ""),
                "target_name": str(feedback.get("target_name") or ""),
            }
            return {
                "phase": "target_reached",
                "minimum_candidate_sequence": int(next_candidate_sequence),
                "detail": {
                    **(feedback.get("detail") or {}),
                    "next_phase": "INTERACT_TARGET",
                },
            }
        detail = dict(feedback.get("detail") or {})
        if (
            str(behavior_type) == "NAVIGATE"
            and target_context.get("target_container_source_object_name")
        ):
            detail["target_object_navigation_complete"] = True
        if str(behavior_type) == "INTERACT":
            detail["target_interaction_complete"] = True
            self.pending_interaction = None
        return {"phase": "complete", "detail": detail}

    def filter_candidates(
        self, candidates: list[BehaviorCandidate]
    ) -> list[BehaviorCandidate]:
        if self.pending_interaction is None:
            return candidates
        pending = self.pending_interaction
        return [
            candidate
            for candidate in candidates
            if candidate.behavior_type == "INTERACT"
            and (
                str(candidate.target_id or "") == str(pending.get("target_id") or "")
                or str(candidate.target_name or "")
                == str(pending.get("target_name") or "")
            )
        ]
