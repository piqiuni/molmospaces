"""Fail-closed visual STOP verifier for navigation-only ObjectGoal tasks.

This is a Module-3 decision seam, not an interaction executor.  Its complete
output vocabulary is STOP, CONTINUE, and RESCAN; it never constructs an
interaction command or a motion action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObjectGoalStopResult:
    decision: str
    confidence: float
    reason: str
    error: str = ""


class ObjectGoalStopVerifier:
    ALLOWED_DECISIONS = frozenset({"STOP", "CONTINUE", "RESCAN"})

    def __init__(self, client: Any, *, min_confidence: float = 0.80) -> None:
        self._client = client
        self._min_confidence = float(min_confidence)

    def verify(
        self,
        *,
        target_name: str,
        image_data_url: str,
        detector_confidence: float,
        public_candidate_distance_m: float,
        bbox_center_distance_m: float,
        max_bbox_center_distance_m: float,
        metrics_context: dict[str, Any] | None = None,
    ) -> ObjectGoalStopResult:
        schema = {
            "name": "objectgoal_stop_verification",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "decision": {"type": "string", "enum": sorted(self.ALLOWED_DECISIONS)},
                    "target_matches": {"type": "boolean"},
                    "target_visible": {"type": "boolean"},
                    "target_mostly_visible": {"type": "boolean"},
                    "target_distance_close_enough": {"type": "boolean"},
                    "stopping_view_is_clear": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reason": {"type": "string"},
                },
                "required": [
                    "decision",
                    "target_matches",
                    "target_visible",
                    "target_mostly_visible",
                    "target_distance_close_enough",
                    "stopping_view_is_clear",
                    "confidence",
                    "reason",
                ],
            },
        }
        response = self._client.request_json(
            role="objectgoal_stop_verification",
            instruction=(
                "You are the final navigation-only ObjectGoal STOP verifier. "
                f"The requested category is {target_name!r}. Inspect only the current RGB image. "
                "Return STOP only when a clearly identifiable requested object is presently visible, "
                "most of the target object's characteristic body and outline is inside the image, the target "
                "is not merely a small fragment, corner, edge crop, reflection, or heavily occluded instance, "
                "the robot appears to be at a sensible unobstructed viewing/standoff pose, and another forward "
                "move is unnecessary. Set target_mostly_visible=false whenever a substantial part of the object "
                "is outside the frame or hidden. Return RESCAN for ambiguity, severe cropping, or occlusion; "
                "otherwise return CONTINUE. The supplied bbox_center_distance_m is the public RGB-D planar "
                "distance from the robot to the detected target box centre. Set target_distance_close_enough=true "
                "only when it is no greater than max_bbox_center_distance_m. Return CONTINUE when the target is "
                "still too far, but do not treat a short distance alone as proof that STOP is safe. "
                "Never propose manipulation, interaction, or another motion command."
            ),
            context={
                "target_name": str(target_name),
                "detector_confidence": float(detector_confidence),
                "public_candidate_distance_m": float(public_candidate_distance_m),
                "bbox_center_distance_m": float(bbox_center_distance_m),
                "max_bbox_center_distance_m": float(max_bbox_center_distance_m),
                "allowed_outputs": sorted(self.ALLOWED_DECISIONS),
            },
            images=[image_data_url],
            response_schema=schema,
            metrics_context=dict(metrics_context or {}),
        )
        if response.error:
            return ObjectGoalStopResult("CONTINUE", 0.0, "model request failed", str(response.error))
        payload = response.payload or {}
        decision = str(payload.get("decision") or "CONTINUE").upper()
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        valid_stop = bool(
            decision == "STOP"
            and payload.get("target_matches") is True
            and payload.get("target_visible") is True
            and payload.get("target_mostly_visible") is True
            and payload.get("target_distance_close_enough") is True
            and payload.get("stopping_view_is_clear") is True
            and confidence >= self._min_confidence
            and float(bbox_center_distance_m) <= float(max_bbox_center_distance_m)
        )
        if decision not in self.ALLOWED_DECISIONS or (decision == "STOP" and not valid_stop):
            decision = "CONTINUE"
        return ObjectGoalStopResult(decision, confidence, str(payload.get("reason") or ""))
