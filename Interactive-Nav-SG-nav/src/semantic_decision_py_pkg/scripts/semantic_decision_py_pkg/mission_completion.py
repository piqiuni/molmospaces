from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .behavior_candidates import BehaviorCandidate


@dataclass
class MissionCompletionConfig:
    empty_candidate_confirmations: int = 3
    empty_candidate_min_steps: int = 50
    stagnation_failure_limit: int = 0
    retryable_frontier_stall_min_steps: int = 50
    retryable_frontier_stall_confirmations: int = 3


@dataclass
class InteractionApproachFailureLimitConfig:
    """Bound repeated navigation failures for one concrete interaction approach."""

    failure_limit: int = 3


class InteractionApproachFailureLimitTracker:
    """Mark an interaction candidate terminal after a bounded number of failures.

    The key is the concrete candidate ID rather than a room/visibility filter:
    remote containers remain valid global-planning candidates until the executor
    has actually failed them repeatedly.  Once terminal, a candidate stays
    excluded until the next episode; late or duplicate feedback must not make
    the same unreachable subgoal selectable again.
    """

    REASON = "interaction_approach_terminal_unreachable"
    VISUAL_PRECONDITION_REASON = "interaction_visual_precondition_terminal"
    EXECUTION_REASON = "interaction_execution_terminal_failed"
    APPROACH_EXHAUSTED_STAGE = "interaction_approach_exhausted"
    _IMMEDIATE_TERMINAL_REASONS = {
        "interaction_visual_precondition": VISUAL_PRECONDITION_REASON,
        "interaction_execution": EXECUTION_REASON,
    }

    def __init__(
        self, config: InteractionApproachFailureLimitConfig | None = None
    ) -> None:
        self.config = config or InteractionApproachFailureLimitConfig()
        self.reset()

    def reset(self) -> None:
        self.failure_counts: dict[str, int] = {}
        self.terminal_candidate_ids: set[str] = set()

    @property
    def failure_limit(self) -> int:
        return max(0, int(self.config.failure_limit))

    def note_feedback(
        self,
        *,
        candidate_id: str,
        behavior_type: str,
        status: str,
        failure_stage: str = "",
    ) -> dict[str, Any] | None:
        """Record one terminal feedback and return detail when the limit is met."""

        candidate_id = str(candidate_id or "")
        if not candidate_id or str(behavior_type or "").upper() != "INTERACT":
            return None
        normalized_status = str(status or "").upper()
        if candidate_id in self.terminal_candidate_ids:
            return None
        if normalized_status == "SUCCEEDED":
            self.failure_counts.pop(candidate_id, None)
            return None
        if normalized_status not in {"FAILED", "REJECTED"}:
            return None
        normalized_stage = str(failure_stage or "").strip().casefold()
        if normalized_stage == self.APPROACH_EXHAUSTED_STAGE:
            # The executor has already tried every preserved approach pose for
            # this concrete candidate. Requiring several complete rings used
            # to leave an unexecutable raw interaction alive indefinitely.
            # This is only episode-local reachability memory; it says nothing
            # about the physical state of the object.
            count = self.failure_counts.get(candidate_id, 0) + 1
            self.failure_counts[candidate_id] = count
            self.terminal_candidate_ids.add(candidate_id)
            return {
                "reason": self.REASON,
                "interaction_failure_count": count,
                "interaction_failure_limit": count,
                "interaction_approach_options_exhausted": True,
                "terminal_candidate_exclusion": True,
            }
        immediate_reason = self._IMMEDIATE_TERMINAL_REASONS.get(normalized_stage)
        if immediate_reason:
            # The executor has already consumed the bounded visual gate or a
            # concrete drawer execution attempt.  Do not send the same
            # impossible candidate back through generic cooldown/reselection.
            self.terminal_candidate_ids.add(candidate_id)
            self.failure_counts[candidate_id] = 1
            return {
                "reason": immediate_reason,
                "interaction_failure_count": 1,
                "interaction_failure_limit": 1,
                "terminal_candidate_exclusion": True,
            }
        if (
            normalized_stage != "interaction_approach_navigation"
            or self.failure_limit <= 0
        ):
            return None
        count = self.failure_counts.get(candidate_id, 0) + 1
        self.failure_counts[candidate_id] = count
        if count < self.failure_limit:
            return None
        self.terminal_candidate_ids.add(candidate_id)
        return {
            "reason": self.REASON,
            "interaction_failure_count": count,
            "interaction_failure_limit": self.failure_limit,
            "terminal_candidate_exclusion": True,
        }


@dataclass
class TerminalInteractionNoPlanExitConfig:
    """Bounded exit after repeated interaction approaches become terminal.

    This is deliberately narrower than generic action-timeout handling.  A
    missing ``cmd_vel`` and a single failed preflight can be transient
    ROS/costmap scheduling issues.  The decision layer first requires a
    concrete interaction candidate to reach its bounded approach-failure
    limit, then requires several later *observation steps* with no executable
    replacement before ending the episode.  Ordinary cooldown/recovery and the
    mandatory startup scan are therefore not converted into terminal outcomes.
    """

    enabled: bool = False
    no_executable_candidate_min_steps: int = 20
    no_executable_candidate_confirmations: int = 3


class TerminalInteractionNoPlanExitTracker:
    """Track a confirmed no-local-subgoal streak after terminal interaction planning.

    The decision node can receive multiple candidate messages for one simulator
    observation.  Therefore confirmations are keyed by
    ``exploration_context.observation_step``, never by ROS message count.
    """

    REASON = "no_executable_candidates_after_terminal_interaction_no_plan"
    TERMINAL_INTERACTION_REASONS = {
        InteractionApproachFailureLimitTracker.REASON,
        InteractionApproachFailureLimitTracker.VISUAL_PRECONDITION_REASON,
        InteractionApproachFailureLimitTracker.EXECUTION_REASON,
    }

    def __init__(
        self, config: TerminalInteractionNoPlanExitConfig | None = None
    ) -> None:
        self.config = config or TerminalInteractionNoPlanExitConfig()
        self.reset()

    def reset(self) -> None:
        self.complete = False
        self.reason = ""
        self.terminal_failure: dict[str, Any] = {}
        self.terminal_failure_observation_step: int | None = None
        self.no_executable_since_observation_step: int | None = None
        self.last_counted_observation_step: int | None = None
        self.no_executable_observation_confirmations = 0
        self.last_detail: dict[str, Any] = {}

    @staticmethod
    def is_terminal_interaction_no_plan(feedback: dict[str, Any]) -> bool:
        """Return true only for an executor-confirmed failed interaction plan."""

        status = str(feedback.get("status") or "").upper()
        behavior_type = str(feedback.get("behavior_type") or "").upper()
        detail = feedback.get("detail") or {}
        reason = str(detail.get("reason") or "").casefold()
        return (
            status in {"FAILED", "REJECTED"}
            and behavior_type == "INTERACT"
            and reason in TerminalInteractionNoPlanExitTracker.TERMINAL_INTERACTION_REASONS
        )

    def note_feedback(
        self, feedback: dict[str, Any], *, observation_step: int | None
    ) -> bool:
        """Arm the streak only after a confirmed terminal interaction failure."""

        if not self.config.enabled or not self.is_terminal_interaction_no_plan(feedback):
            return False
        detail = dict(feedback.get("detail") or {})
        self.complete = False
        self.reason = ""
        self.terminal_failure = {
            "candidate_id": str(feedback.get("candidate_id") or ""),
            "decision_id": str(feedback.get("decision_id") or ""),
            "behavior_type": str(feedback.get("behavior_type") or ""),
            "status": str(feedback.get("status") or ""),
            "detail": detail,
        }
        self.terminal_failure_observation_step = observation_step
        self.no_executable_since_observation_step = observation_step
        self.last_counted_observation_step = None
        self.no_executable_observation_confirmations = 0
        self.last_detail = {}
        return True

    def clear_for_recovery(self) -> None:
        """Forget an old terminal failure once an executable replacement exists."""

        self.reset()

    @staticmethod
    def _observation_step(candidate_snapshot: dict[str, Any]) -> int | None:
        exploration = candidate_snapshot.get("exploration_context") or {}
        try:
            return int(exploration.get("observation_step"))
        except (TypeError, ValueError):
            return None

    def update(
        self,
        candidate_snapshot: dict[str, Any],
        *,
        has_active_behavior: bool,
        has_executable_candidate: bool,
        startup_scan_pending: bool,
        eligible_candidate_count: int,
    ) -> bool:
        """Advance only on distinct post-failure simulator observations."""

        if self.complete:
            return True
        if not self.config.enabled or not self.terminal_failure:
            return False
        if has_active_behavior or has_executable_candidate or startup_scan_pending:
            self.clear_for_recovery()
            return False

        observation_step = self._observation_step(candidate_snapshot)
        # A missing observation-step token cannot establish a simulator-step
        # streak safely.  This avoids treating bursts of ROS candidate messages
        # as elapsed simulation time.
        if observation_step is None:
            return False
        if (
            self.terminal_failure_observation_step is not None
            and observation_step <= self.terminal_failure_observation_step
        ):
            return False
        if (
            self.last_counted_observation_step is not None
            and observation_step <= self.last_counted_observation_step
        ):
            return False

        self.last_counted_observation_step = observation_step
        if self.no_executable_since_observation_step is None:
            self.no_executable_since_observation_step = observation_step
        self.no_executable_observation_confirmations += 1
        elapsed_steps = max(
            0,
            observation_step - int(self.no_executable_since_observation_step),
        )
        raw_candidate_count = int(
            candidate_snapshot.get(
                "candidate_count", len(candidate_snapshot.get("candidates") or [])
            )
            or 0
        )
        self.last_detail = {
            "reason": self.REASON,
            "candidate_sequence": int(candidate_snapshot.get("sequence", 0) or 0),
            "observation_step": observation_step,
            "no_executable_since_observation_step": (
                self.no_executable_since_observation_step
            ),
            "no_executable_elapsed_steps": elapsed_steps,
            "no_executable_observation_confirmations": (
                self.no_executable_observation_confirmations
            ),
            "no_executable_candidate_min_steps": max(
                0, int(self.config.no_executable_candidate_min_steps)
            ),
            "no_executable_candidate_confirmations_required": max(
                1, int(self.config.no_executable_candidate_confirmations)
            ),
            "raw_candidate_count": raw_candidate_count,
            "eligible_candidate_count": max(0, int(eligible_candidate_count)),
            "terminal_interaction_failure": dict(self.terminal_failure),
        }
        self.complete = (
            elapsed_steps
            >= max(0, int(self.config.no_executable_candidate_min_steps))
            and self.no_executable_observation_confirmations
            >= max(1, int(self.config.no_executable_candidate_confirmations))
        )
        if self.complete:
            self.reason = self.REASON
        return self.complete


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
        self.terminal_stalled = False
        self.empty_since_step: int | None = None
        self.retryable_frontier_since_step: int | None = None
        self.retryable_frontier_confirmations = 0
        self.last_retryable_frontier_detail: dict[str, Any] = {}

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
        if self.terminal_stalled:
            return False
        sequence = int(candidates_payload.get("sequence", 0) or 0)
        if sequence == self.last_sequence:
            return False
        self.last_sequence = sequence
        self.stalled = False
        exploration = candidates_payload.get("exploration_context") or {}
        initial_scan_complete = bool(exploration.get("initial_scan_complete", True))
        retryable_filtered_frontier = bool(
            exploration.get("filtered_frontier_retryable", False)
        )
        candidates = list(candidates_payload.get("candidates") or [])
        # A producer counter can lag or reset while a candidate snapshot is
        # still live. Treat the raw snapshot as a conservative lower bound so
        # completion cannot report exhaustion while a visible frontier remains.
        raw_navigation_count = sum(
            str(candidate.get("behavior_type") or "") == "EXPLORE"
            for candidate in candidates
        )
        raw_interaction_count = sum(
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
                raw_navigation_count
                if candidates
                else fallback_candidate_count,
            )
            or 0
        )
        navigation_frontier_count = max(
            navigation_frontier_count, raw_navigation_count
        )
        interaction_frontier_count = int(
            exploration.get(
                "interaction_frontier_count", raw_interaction_count
            )
            or 0
        )
        interaction_frontier_count = max(
            interaction_frontier_count, raw_interaction_count
        )
        combined_frontier_count = int(
            exploration.get(
                "combined_frontier_count",
                navigation_frontier_count + interaction_frontier_count,
            )
            or 0
        )
        combined_frontier_count = max(
            combined_frontier_count,
            navigation_frontier_count + interaction_frontier_count,
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
        if raw_navigation_count or raw_interaction_count:
            navigation_frontier_exhausted = (
                navigation_frontier_exhausted and raw_navigation_count == 0
            )
            interaction_frontier_exhausted = (
                interaction_frontier_exhausted and raw_interaction_count == 0
            )
            exhausted = False
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
        if retryable_filtered_frontier:
            if self.retryable_frontier_since_step is None:
                self.retryable_frontier_since_step = observation_step
            self.retryable_frontier_confirmations += 1
            elapsed_steps = (
                max(
                    0,
                    observation_step - self.retryable_frontier_since_step,
                )
                if observation_step is not None
                and self.retryable_frontier_since_step is not None
                else 0
            )
            self.last_retryable_frontier_detail = {
                "reason": str(
                    exploration.get("filtered_frontier_reason")
                    or "material_frontier_without_safe_viewpoint"
                ),
                "raw_frontier_cluster_count": int(
                    exploration.get("raw_frontier_cluster_count", 0) or 0
                ),
                "raw_frontier_material_cluster_count": int(
                    exploration.get("raw_frontier_material_cluster_count", 0) or 0
                ),
                "filtered_no_viewpoint_frontier_cluster_count": int(
                    exploration.get(
                        "filtered_no_viewpoint_frontier_cluster_count", 0
                    )
                    or 0
                ),
                "retryable_frontier_since_observation_step": (
                    self.retryable_frontier_since_step
                ),
                "retryable_frontier_elapsed_steps": elapsed_steps,
                "retryable_frontier_has_observation_step": observation_step is not None,
                "retryable_frontier_confirmations": (
                    self.retryable_frontier_confirmations
                ),
                "retryable_frontier_stall_min_steps": max(
                    0, int(self.config.retryable_frontier_stall_min_steps)
                ),
                "retryable_frontier_stall_confirmations_required": max(
                    1, int(self.config.retryable_frontier_stall_confirmations)
                ),
            }
            self.confirmations = 0
            self.empty_since_step = None
            self.stalled = False
            enough_confirmations = self.retryable_frontier_confirmations >= max(
                1, int(self.config.retryable_frontier_stall_confirmations)
            )
            enough_elapsed_steps = (
                observation_step is None
                or elapsed_steps
                >= max(0, int(self.config.retryable_frontier_stall_min_steps))
            )
            if enough_confirmations and enough_elapsed_steps:
                self.stalled = True
                self.terminal_stalled = True
                self.reason = str(self.last_retryable_frontier_detail["reason"])
            return False
        self.retryable_frontier_since_step = None
        self.retryable_frontier_confirmations = 0
        self.last_retryable_frontier_detail = {}
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
    def priority_target_candidate(
        candidates: list[dict[str, Any]] | None,
        *,
        currently_visible_only: bool = False,
    ) -> dict[str, Any] | None:
        """Return a reliable concrete target-navigation candidate, preferring current visibility."""
        matches = []
        for candidate in candidates or []:
            metadata = candidate.get("metadata") or {}
            if str(candidate.get("behavior_type") or "").upper() != "NAVIGATE":
                continue
            if not bool(metadata.get("target_goal")) or not bool(
                metadata.get("target_reliably_observed")
            ):
                continue
            if currently_visible_only and not bool(metadata.get("target_visible_now")):
                continue
            matches.append(candidate)
        if not matches:
            return None
        return min(
            matches,
            key=lambda candidate: (
                0
                if bool((candidate.get("metadata") or {}).get("target_visible_now"))
                else 1,
                float(
                    (candidate.get("features") or {}).get("distance_m", float("inf"))
                    or 0.0
                ),
                str(candidate.get("candidate_id") or ""),
            ),
        )

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
