"""Keep scheduling bounded and isolate passive M1 from backend state labels."""

import json
from copy import deepcopy
import threading


def physical_completion(payload):
    return (isinstance(payload, dict)
            and payload.get("action_executed") is not False
            and payload.get("observation_outcome") != "finish_without_action")


def result_step(payload, fallback=-1):
    for key in ("result_published_step", "step", "capture_step"):
        try:
            return int(payload[key])
        except (KeyError, TypeError, ValueError):
            pass
    return fallback


class PostActionVisualGuard:
    """An executed action invalidates old evidence, never supplies a state label."""

    def __init__(self):
        self.lock = threading.RLock()
        self.episode = ""
        self.boundaries = {}

    def observe_episode(self, episode):
        with self.lock:
            if episode and self.episode and episode != self.episode:
                self.boundaries.clear()
            if episode:
                self.episode = episode

    def record(self, payload, fallback_step=-1):
        if not physical_completion(payload):
            return
        with self.lock:
            self.observe_episode(str(payload.get("episode_id") or ""))
            for key in ("object_id", "instance_id", "node_id", "source_object_name"):
                identifier = str(payload.get(key) or "")
                if identifier:
                    self.boundaries[identifier] = max(
                        self.boundaries.get(identifier, -1), result_step(payload, fallback_step))

    def apply(self, snapshot):
        result = deepcopy(snapshot)
        with self.lock:
            self.observe_episode(str(snapshot.get("episode_id") or ""))
            boundaries = dict(self.boundaries)
        for candidate in result.get("candidates", []):
            if candidate.get("behavior_type") != "INTERACT":
                continue
            command = candidate.get("interaction_command") or {}
            keys = [candidate.get("target_id"), command.get("node_id"), command.get("object_id")]
            steps = [boundaries[key] for key in keys if key in boundaries]
            if not steps:
                continue
            boundary = max(steps)
            metadata = candidate.setdefault("metadata", {})
            if (metadata.get("perception_visual_ready")
                    and int(metadata.get("perception_visual_step", -1)) > boundary):
                continue
            metadata.update(observation_required=True, reobserve=True,
                            observation_only_reobserve=True,
                            observation_reason="post_action_visual_refresh",
                            interaction_observation_source="mllm_attribute_inference",
                            interaction_observation_min_capture_step=boundary + 1,
                            perception_post_action_boundary=boundary)
        return result


REFRESH_PROFILES = {
    "baseline": {},
    "continuous": {
        "success_refresh_interval_s": 30.0,
        "attribute_inference/portal_state_cooldown_steps": 60,
    },
}


def inference_node_class(base):
    class PerceptionOnlyInferenceNode(base):
        def _interaction_result_callback(self, message):
            try:
                payload = json.loads(message.data)
            except (ValueError, TypeError):
                return
            # Native observation-only completions share this topic with physical
            # actions. Clearing their votes destroys the very M1 result we need.
            if not physical_completion(payload):
                return
            # Completion invalidates a pre-action image; neither success nor
            # post_state may become visual evidence through record_authoritative.
            identifiers = {str(payload.get(key) or "") for key in
                           ("object_id", "instance_id", "node_id", "source_object_name")}
            identifiers.discard("")
            with self.lock:
                event = str(payload.get("event_id") or payload.get("command_id") or "")
                events = getattr(self, "_perception_completed_events", set())
                event_key = (str(payload.get("episode_id") or ""), event)
                if event and event_key in events:
                    return
                if event:
                    events.add(event_key)
                self._perception_completed_events = events
                object_ids = {self.aliases.get(key, key) for key in identifiers}
                for object_id in object_ids:
                    self.generations[object_id] = self.generations.get(object_id, 0) + 1
                    self.completed.pop(object_id, None)
                    self.last_request.pop(object_id, None)
                    self.pending.pop(object_id, None)
                    self.target_visual_history.pop(object_id, None)
                    self.portal_observation_streaks.pop(object_id, None)
            for object_id in object_ids:
                self.request_queue.discard(object_id)
                self.portal_state_consensus.clear(object_id)

    return PerceptionOnlyInferenceNode
