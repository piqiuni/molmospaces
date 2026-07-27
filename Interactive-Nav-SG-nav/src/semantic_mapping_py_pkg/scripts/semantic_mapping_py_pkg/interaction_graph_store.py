from __future__ import annotations

import time
import math
from collections import defaultdict

from .geometry_utils import grid_index, normalize_label, world_to_grid
from .graph_rules import (
    default_interaction_payload,
    distance_xy,
    infer_node_type,
    normalize_observation,
    room_node_label,
    sanitize_token,
)
from .graph_schema import NavigationHint, SceneGraphBundle, SceneGraphEdge, SceneGraphNode
from .room_inference_backends import WeightedRoomAttributeInferencer


class InteractionGraphStore:
    def __init__(
        self,
        scene_id="scene",
        match_distance=0.5,
        room_id_to_name=None,
        room_box_height=0.2,
        portal_room_max_radius_m=1.0,
        object_room_search_margin_m=0.75,
        object_room_priors=None,
        room_attribute_min_confidence=0.2,
        interaction_geometry_overrides=None,
    ):
        self.scene_id = str(scene_id or "scene")
        self.match_distance = float(match_distance)
        self.room_id_to_name = dict(room_id_to_name or {})
        self.room_box_height = float(room_box_height)
        self.portal_room_max_radius_m = max(0.1, float(portal_room_max_radius_m))
        self.object_room_search_margin_m = max(
            0.1, float(object_room_search_margin_m)
        )
        self.room_attribute_inferencer = WeightedRoomAttributeInferencer(
            object_room_priors or {},
            min_confidence=room_attribute_min_confidence,
        )
        self.interaction_geometry_overrides = {
            str(key): dict(value or {})
            for key, value in (interaction_geometry_overrides or {}).items()
        }
        self.room_geometries = {}
        self.room_geometry_candidates = {}
        self.room_geometry_stability_frames = 5
        self.room_redirects = {}
        self.nodes = {}
        self.edges = {}
        self.next_node_index = 1
        self.edge_counter = 1
        self.room_grid = None
        self.source_mode = "detector_online"
        self.episode_id = ""
        self.graph_revision = 0
        self.capture_step = None
        self.interaction_event_counter = 1
        self._ensure_scene_node()

    def reset(self, episode_id="", source_mode=None):
        self.episode_id = str(episode_id or "")
        if source_mode:
            self.source_mode = str(source_mode)
        self.room_geometries = {}
        self.room_geometry_candidates = {}
        self.room_geometry_stability_frames = 5
        self.room_redirects = {}
        self.nodes = {}
        self.edges = {}
        self.next_node_index = 1
        self.edge_counter = 1
        self.room_grid = None
        self.graph_revision = 0
        self.capture_step = None
        self.interaction_event_counter = 1
        self._ensure_scene_node()

    def update_room_grid(
        self,
        grid_info,
        scene_data,
        confidence_data=None,
        room_id_to_name=None,
        room_merges=None,
        geometry_stability_frames=5,
    ):
        self.room_geometry_stability_frames = max(
            1, int(geometry_stability_frames)
        )
        if room_id_to_name:
            self.room_id_to_name.update({int(k): str(v) for k, v in room_id_to_name.items()})
        self.room_grid = {
            "info": grid_info,
            "scene_data": list(scene_data or []),
            "confidence_data": list(confidence_data or []),
        }
        self._apply_room_merges(room_merges or {})
        self._refresh_room_nodes_from_grid()
        self._rebuild_relations()
        self._bump_revision()

    def set_room_geometries(self, rooms):
        self.room_geometries = {}
        for room in rooms or []:
            room_id = room.get("room_id")
            if room_id is None:
                continue
            room_id = int(room_id)
            self.room_geometries[room_id] = {
                "center": self._grounded_center(room.get("center") or room.get("aabb_center") or [float(room_id), 0.0, 0.0]),
                "aabb_center": self._grounded_center(room.get("aabb_center") or room.get("center") or [float(room_id), 0.0, 0.0]),
                "aabb_size": self._room_box_size(room.get("aabb_size") or [0.5, 0.5, self.room_box_height]),
                "name": str(room.get("name") or room_node_label(self.room_id_to_name, room_id)),
                "cell_count": int(room.get("cell_count", 0)),
            }
            self.room_id_to_name.setdefault(room_id, self.room_geometries[room_id]["name"])
            node = self._ensure_room_node(room_id)
            geom = self.room_geometries[room_id]
            node.name = geom["name"]
            node.centroid = list(geom["center"])
            node.aabb_center = list(geom["aabb_center"])
            node.aabb_size = list(geom["aabb_size"])
            node.attributes["cell_count"] = geom["cell_count"]

    def update_observations(
        self, observations, stamp=None, source_mode=None, capture_step=None
    ):
        now = float(stamp if stamp is not None else time.time())
        if source_mode:
            self.source_mode = str(source_mode)
        if capture_step is not None:
            self.capture_step = int(capture_step)
        if self.source_mode == "realtime_gt_observation":
            for node in self.nodes.values():
                if node.type not in {"scene", "room"}:
                    node.attributes["_was_visible_previous_update"] = bool(
                        node.is_currently_visible
                    )
                    node.is_currently_visible = False
        for raw_observation in observations:
            observation = normalize_observation(raw_observation)
            node = self._find_or_create_node(observation)
            self._apply_observation(node, observation, now)
        for node in self.nodes.values():
            node.attributes.pop("_was_visible_previous_update", None)
        self._refresh_room_nodes_from_grid()
        self._rebuild_relations(now=now)
        self._refresh_missing_room_nodes_from_observations()
        self._bump_revision()

    def update_interaction_result(self, result, stamp=None):
        node_id = str(result.get("node_id") or "")
        node = self.nodes.get(node_id)
        if node is None:
            instance_id = str(
                result.get("object_id") or result.get("instance_id") or ""
            )
            node = next(
                (candidate for candidate in self.nodes.values() if candidate.attributes.get("instance_id") == instance_id),
                None,
            )
        if node is None:
            source_object_name = str(result.get("source_object_name") or "")
            if source_object_name:
                node = next(
                    (
                        candidate
                        for candidate in self.nodes.values()
                        if candidate.attributes.get("source_object_name") == source_object_name
                        or candidate.name == source_object_name
                    ),
                    None,
                )
        if node is None:
            return False
        now = float(stamp if stamp is not None else time.time())
        pre_state = str(node.interaction.get("state", "unknown"))
        explicit_state = result.get("state") or result.get("post_state")
        if explicit_state is not None:
            node.interaction["state"] = str(explicit_state)
            node.interaction["state_source"] = str(
                result.get("source")
                or result.get("verification_source")
                or "interaction_result"
            )
            node.interaction["state_confidence"] = float(
                result.get("confidence", 1.0)
            )
        static_capability = bool(
            str(result.get("interaction_capability") or "").casefold() == "static"
            or str(result.get("reason") or "").casefold() == "non_articulated"
            or result.get("interactable") is False
        )
        if static_capability:
            node.interaction.update(
                {
                    "is_interactable": False,
                    "interaction_mode": "none",
                    "state": str(explicit_state or "static"),
                    "state_source": str(
                        result.get("source")
                        or result.get("verification_source")
                        or "interaction_capability_check"
                    ),
                    "state_confidence": float(result.get("confidence", 1.0)),
                    "capability": "static",
                    "failure_reason": str(
                        result.get("reason") or "non_articulated"
                    ),
                }
            )
        self._refresh_planner_state_fields(node)
        state = node.interaction.get("state", "unknown")

        history = list(node.interaction.get("operation_history") or [])
        event_id = str(result.get("event_id") or f"interaction_{self.interaction_event_counter:06d}")
        if not any(entry.get("event_id") == event_id for entry in history):
            self.interaction_event_counter += 1
            history_entry = {
                    "event_id": event_id,
                    "action": str(result.get("action") or result.get("interaction_mode") or "unknown"),
                    "timestamp": now,
                    "pre_state": pre_state,
                    "post_state": str(state),
                    "success": bool(result.get("success", True)),
                    "execution_cost": float(result.get("execution_cost", result.get("cost", 1.0))),
                    "verification_source": str(result.get("verification_source") or result.get("source") or "interaction_result"),
                }
            if result.get("interaction_group_id"):
                history_entry["interaction_group_id"] = str(
                    result["interaction_group_id"]
                )
            approach_goal = list(result.get("approach_goal_xyyaw") or [])
            if len(approach_goal) >= 2:
                history_entry["approach_goal_xyyaw"] = [
                    float(value) for value in approach_goal[:3]
                ]
            history.append(history_entry)
        node.interaction["operation_history"] = history
        self._update_interaction_group_memory(node, result)
        if explicit_state is not None or static_capability:
            node.attributes["interaction_state_override"] = {
                key: node.interaction.get(key)
                for key in (
                    "state",
                    "state_source",
                    "state_confidence",
                    "traversable",
                    "requires_interaction",
                    "is_interactable",
                    "interaction_mode",
                    "capability",
                    "failure_reason",
                )
                if key in node.interaction
            }
            node.attributes["interaction_state_override"]["event_id"] = event_id
            node.attributes["interaction_state_override"]["timestamp"] = now
        node.last_seen = now
        self._rebuild_relations(now=now)
        self._bump_revision()
        return True

    def _update_interaction_group_memory(self, node, result):
        sequence_results = list(result.get("interaction_group_results") or [])
        if sequence_results:
            for group_result in sequence_results:
                self._update_interaction_group_memory(node, dict(group_result or {}))
            return
        interaction = node.interaction
        group_id = str(
            result.get("interaction_group_id") or result.get("region_id") or ""
        )
        if not group_id:
            return
        success = bool(result.get("success", True))
        completed = {
            str(value)
            for value in interaction.get("completed_interaction_groups") or []
        }
        failed = {
            str(value) for value in interaction.get("failed_interaction_groups") or []
        }
        if success:
            completed.add(group_id)
            failed.discard(group_id)
        else:
            failed.add(group_id)
        interaction["completed_interaction_groups"] = sorted(completed)
        interaction["failed_interaction_groups"] = sorted(failed)

    @staticmethod
    def _refresh_planner_state_fields(node):
        interaction = node.interaction
        state = str(interaction.get("state") or "unknown")
        if node.type == "portal":
            interaction["traversable"] = state in {"open", "static_open"}
            interaction["requires_interaction"] = bool(
                interaction.get("is_interactable")
                and state not in {"open", "static_open"}
            )

            return
        interaction["traversable"] = (
            True
            if state in {"open", "ajar", "static_open"}
            else False
            if state == "closed"
            else None
        )
        interaction["requires_interaction"] = bool(
            interaction.get("is_interactable") and state in {"closed", "unknown"}
        )

    @staticmethod
    def _expanded_room_geometry(accepted_center, accepted_size, center, size):
        if (
            float(size[0]) <= float(accepted_size[0]) + 1e-6
            and float(size[1]) <= float(accepted_size[1]) + 1e-6
        ):
            return list(accepted_center), list(accepted_size)
        old_min_x = float(accepted_center[0]) - 0.5 * float(accepted_size[0])
        old_max_x = float(accepted_center[0]) + 0.5 * float(accepted_size[0])
        old_min_y = float(accepted_center[1]) - 0.5 * float(accepted_size[1])
        old_max_y = float(accepted_center[1]) + 0.5 * float(accepted_size[1])
        new_min_x = float(center[0]) - 0.5 * float(size[0])
        new_max_x = float(center[0]) + 0.5 * float(size[0])
        new_min_y = float(center[1]) - 0.5 * float(size[1])
        new_max_y = float(center[1]) + 0.5 * float(size[1])
        min_x, max_x = min(old_min_x, new_min_x), max(old_max_x, new_max_x)
        min_y, max_y = min(old_min_y, new_min_y), max(old_max_y, new_max_y)
        return (
            [0.5 * (min_x + max_x), 0.5 * (min_y + max_y), float(center[2])],
            [max(0.1, max_x - min_x), max(0.1, max_y - min_y), float(size[2])],
        )

    def apply_attribute_patch(self, patch, stamp=None):
        object_id = str(patch.get("object_id") or "")
        node = self.nodes.get(object_id)
        if node is None:
            node = next(
                (
                    candidate
                    for candidate in self.nodes.values()
                    if str(candidate.attributes.get("instance_id") or "") == object_id
                ),
                None,
            )
        if node is None:
            return False
        attribute_status = str(patch.get("attribute_status") or "ready")
        patch_stamp = float(stamp if stamp is not None else time.time())
        request_sequence = int(patch.get("request_sequence", 0) or 0)
        latest_request_sequence = int(
            node.attributes.get("attribute_request_sequence", 0) or 0
        )
        if request_sequence and latest_request_sequence and request_sequence < latest_request_sequence:
            return False
        patch_frame_index = patch.get("observation_frame_index")
        current_frame_index = node.attributes.get("last_observation_frame_index")
        try:
            patch_frame_index = int(patch_frame_index)
            current_frame_index = int(current_frame_index)
        except (TypeError, ValueError):
            patch_frame_index = None
            current_frame_index = None
        if (
            patch_frame_index is not None
            and current_frame_index is not None
            and patch_frame_index < current_frame_index
        ):
            node.attributes.update(
                {
                    "attribute_status": "stale",
                    "attribute_request_sequence": max(
                        latest_request_sequence, request_sequence
                    ),
                    "attribute_error": "observation_version_superseded",
                }
            )
            self._bump_revision()
            return True
        node.attributes.update(
            {
                "attribute_status": attribute_status,
                "attribute_request_sequence": max(latest_request_sequence, request_sequence),
                "attribute_response_lag_sec": float(
                    patch.get("response_lag_sec", 0.0) or 0.0
                ),
                "attribute_queue_lag_sec": float(patch.get("queue_lag_sec", 0.0) or 0.0),
                "attribute_total_lag_sec": float(patch.get("total_lag_sec", 0.0) or 0.0),
                "attribute_error": str(patch.get("error") or ""),
                "attribute_observation_stamp_sec": float(
                    patch.get("observation_stamp_sec", patch_stamp) or patch_stamp
                ),
                "attribute_observation_frame_index": patch_frame_index,
                "attribute_observation_signature": str(
                    patch.get("observation_signature") or ""
                ),
            }
        )
        if attribute_status in {"pending", "failed", "stale"}:
            self._bump_revision()
            return True
        verified_state_override = dict(
            node.attributes.get("interaction_state_override") or {}
        )
        has_verified_interaction_state = bool(
            verified_state_override.get("event_id")
        )
        confidence = float(patch.get("confidence", 0.0) or 0.0)
        interaction_class = normalize_label(patch.get("interaction_class"))
        if (
            not has_verified_interaction_state
            and confidence >= 0.5
            and interaction_class in {"portal", "container", "support", "object"}
        ):
            node.type = interaction_class
        parts = list(patch.get("interaction_parts") or [])
        for deprecated_key in (
            "interaction_groups",
            "interaction_group_source",
            "joint_infos",
            "observation_evidence",
        ):
            node.attributes.pop(deprecated_key, None)
        node.attributes.update(
            {
                "attribute_source": str(patch.get("source") or "mllm"),
                "attribute_model": str(patch.get("model_name") or ""),
                "attribute_confidence": confidence,
                "evidence_frame_ids": list(patch.get("evidence_frame_ids") or []),
                "affordances": list(patch.get("affordances") or []),
                "interaction_parts": parts,
                "mllm_interaction_parts": parts,
            }
        )
        previous_history = list(node.interaction.get("operation_history") or [])
        latest_operation_stamp = max(
            [
                float(item.get("timestamp", 0.0) or 0.0)
                for item in previous_history
                if isinstance(item, dict)
            ],
            default=0.0,
        )
        state_was_updated = (
            not has_verified_interaction_state
            and latest_operation_stamp <= patch_stamp
        )
        if state_was_updated:
            node.interaction.update(
                {
                    "is_interactable": bool(patch.get("interactable", False)),
                    "state": str(
                        patch.get("coarse_state")
                        or node.interaction.get("state")
                        or "unknown"
                    ),
                    "state_source": str(patch.get("source") or "mllm_attribute_inference"),
                    "state_confidence": confidence,
                    "expected_effect": str(
                        patch.get("expected_effect")
                        or (
                            "unlock_connectivity"
                            if node.type == "portal"
                            else "reveal_contents"
                        )
                    ),
                    "operation_history": previous_history,
                }
            )
        if state_was_updated:
            self._refresh_planner_state_fields(node)
            node.attributes["interaction_state_override"] = {
                key: node.interaction.get(key)
                for key in (
                    "state",
                    "state_source",
                    "state_confidence",
                    "traversable",
                    "requires_interaction",
                )
                if key in node.interaction
            }
            node.attributes["interaction_state_override"]["timestamp"] = patch_stamp

        node.attributes["attribute_updated_at"] = patch_stamp
        self._rebuild_relations(now=patch_stamp)
        self._bump_revision()
        return True

    def as_graph_bundle(self, stamp=None):
        now = float(stamp if stamp is not None else time.time())
        for node in self.nodes.values():
            node.state_age_sec = max(0.0, now - float(node.last_seen)) if node.last_seen is not None else 0.0
            node.graph_revision = self.graph_revision
        nodes = sorted(self.nodes.values(), key=lambda item: item.id)
        edges = sorted(self.edges.values(), key=lambda item: item.id)
        semantic_node_ids = [node.id for node in nodes]
        semantic_edge_ids = [edge.id for edge in edges]

        interaction_core = {node.id for node in nodes if node.type in {"portal", "support", "container"}}
        interaction_edge_ids = []
        for edge in edges:
            if edge.src_id in interaction_core or edge.dst_id in interaction_core:
                interaction_core.add(edge.src_id)
                interaction_core.add(edge.dst_id)
                interaction_edge_ids.append(edge.id)

        navigation_hints = self._build_navigation_hints()
        navigation_node_ids = sorted({hint.node_id for hint in navigation_hints})
        navigation_edge_ids = [
            edge.id
            for edge in edges
            if edge.src_id in navigation_node_ids or edge.dst_id in navigation_node_ids
        ]

        return SceneGraphBundle(
            scene_id=self.scene_id,
            episode_id=self.episode_id,
            source_mode=self.source_mode,
            graph_revision=self.graph_revision,
            timestamp=now,
            capture_step=self.capture_step,
            nodes=nodes,
            edges=edges,
            semantic_node_ids=semantic_node_ids,
            semantic_edge_ids=semantic_edge_ids,
            interaction_node_ids=sorted(interaction_core),
            interaction_edge_ids=sorted(set(interaction_edge_ids)),
            navigation_node_ids=navigation_node_ids,
            navigation_edge_ids=navigation_edge_ids,
            navigation_hints=navigation_hints,
        )

    def as_graph_dict(self, stamp=None):
        return self.as_graph_bundle(stamp=stamp).to_dict()

    def as_navigation_hints(self):
        return [hint.to_dict() for hint in self._build_navigation_hints()]

    def prune_stale_nodes(self, stale_after_sec, now=None):
        stale_after_sec = float(stale_after_sec)
        if stale_after_sec <= 0.0:
            return
        now = float(now if now is not None else time.time())
        stale_ids = [
            node_id
            for node_id, node in self.nodes.items()
            if node.type != "room" and node.last_seen is not None and now - float(node.last_seen) > stale_after_sec
        ]
        if not stale_ids:
            return
        stale_id_set = set(stale_ids)
        for node_id in stale_ids:
            self.nodes.pop(node_id, None)
        self.edges = {
            edge_id: edge
            for edge_id, edge in self.edges.items()
            if edge.src_id not in stale_id_set and edge.dst_id not in stale_id_set
        }
        self._rebuild_relations(now=now)

    def _find_or_create_node(self, observation):
        instance_id = observation.get("instance_id")
        if instance_id:
            for node in self.nodes.values():
                if node.attributes.get("instance_id") == instance_id:
                    return node

        node_type = infer_node_type(observation)
        label = normalize_label(observation.get("semantic_name"))
        best = None
        best_dist = None
        for node in self.nodes.values():
            if node.type == "room":
                continue
            if node.type != node_type or node.label != label:
                continue
            dist = distance_xy(node.centroid, observation["position"])
            if dist <= self.match_distance and (best is None or dist < best_dist):
                best = node
                best_dist = dist
        if best is not None:
            return best

        node_id = self._make_node_id(node_type, observation)
        node = SceneGraphNode(
            id=node_id,
            type=node_type,
            label=label or node_type,
            name=str(observation.get("name") or label or node_type),
        )
        self.nodes[node_id] = node
        return node

    def _ensure_scene_node(self):
        node_id = f"scene_{sanitize_token(self.episode_id or self.scene_id)}"
        existing = next((node for node in self.nodes.values() if node.type == "scene"), None)
        if existing is not None:
            if existing.id != node_id:
                self.nodes.pop(existing.id, None)
                existing.id = node_id
                self.nodes[node_id] = existing
            existing.name = self.episode_id or self.scene_id
            existing.label = normalize_label(self.scene_id) or "scene"
            existing.attributes.update(
                {
                    "scene_id": self.scene_id,
                    "episode_id": self.episode_id,
                    "source_mode": self.source_mode,
                }
            )
            return existing
        node = SceneGraphNode(
            id=node_id,
            type="scene",
            label=normalize_label(self.scene_id) or "scene",
            name=self.episode_id or self.scene_id,
            confidence=1.0,
            is_currently_visible=True,
            attributes={
                "scene_id": self.scene_id,
                "episode_id": self.episode_id,
                "source_mode": self.source_mode,
            },
        )
        self.nodes[node_id] = node
        return node

    def _make_node_id(self, node_type, observation):
        instance_id = sanitize_token(observation.get("instance_id") or "")
        if instance_id:
            node_id = f"{node_type}_{instance_id}"
            if node_id not in self.nodes:
                return node_id
        label = sanitize_token(observation.get("semantic_name") or node_type)
        while True:
            node_id = f"{node_type}_{label}_{self.next_node_index}"
            self.next_node_index += 1
            if node_id not in self.nodes:
                return node_id

    def _apply_observation(self, node, observation, now):
        minimal_gt = bool(observation.get("minimal_gt_observation"))
        interaction_state_override = dict(
            node.attributes.get("interaction_state_override") or {}
        )
        node.type = infer_node_type(observation)
        node.label = normalize_label(observation.get("semantic_name")) or node.type
        node.name = str(observation.get("name") or node.label or node.type)
        node.centroid = self._ground_non_room_centroid(observation["position"], observation["aabb_size"])
        node.aabb_center = self._ground_non_room_centroid(observation["aabb_center"], observation["aabb_size"])
        node.aabb_size = list(observation["aabb_size"])
        node.room_id = observation.get("room_id") if observation.get("room_id") is not None else node.room_id
        node.confidence = max(float(node.confidence), float(observation.get("confidence", 0.0)))
        node.observation_count += 1
        if node.first_seen is None:
            node.first_seen = now
        node.last_seen = now
        node.is_currently_visible = True
        max_visible_pixels = max(
            int(node.attributes.get("max_visible_pixels", 0) or 0),
            int(observation.get("visible_pixels", 0) or 0),
        )
        max_visible_fraction = max(
            float(node.attributes.get("max_visible_fraction", 0.0) or 0.0),
            float(observation.get("visible_fraction", 0.0) or 0.0),
        )
        consecutive_observations = int(
            observation.get("consecutive_observations", 0) or 0
        )
        if minimal_gt:
            consecutive_observations = (
                int(node.attributes.get("consecutive_observations", 0) or 0) + 1
                if bool(node.attributes.get("_was_visible_previous_update"))
                else 1
            )
        max_consecutive_observations = max(
            int(node.attributes.get("max_consecutive_observations", 0) or 0),
            consecutive_observations,
        )

        observation_attributes = {
                "instance_id": observation.get("instance_id") or node.attributes.get("instance_id") or "",
                "category": observation.get("category"),
                "candidate_labels": list(observation.get("candidate_labels") or []),
                "label_votes": dict(observation.get("label_votes") or {}),
                "connected_room_ids": list(observation.get("connected_room_ids") or []),
                "source": observation.get("source"),
                "source_object_name": observation.get("source_object_name"),
                "visible_pixels": int(observation.get("visible_pixels", 0)),
                "max_visible_pixels": max_visible_pixels,
                "visible_fraction": float(
                    observation.get("visible_fraction", 0.0) or 0.0
                ),
                "max_visible_fraction": max_visible_fraction,
                "bbox_2d": list(observation.get("bbox_2d") or []),
                "consecutive_observations": consecutive_observations,
                "max_consecutive_observations": max_consecutive_observations,
                "camera_name": observation.get("camera_name"),
                "frame_index": int(observation.get("frame_index", 0)),
                "last_observation_frame_index": int(
                    observation.get("frame_index", 0) or 0
                ),
                "episode_id": observation.get("episode_id"),
                "box_3d_frame_id": observation.get("box_3d_frame_id"),
                "viz_aabb_center": list(observation.get("viz_aabb_center") or observation["aabb_center"]),
                "viz_aabb_size": list(observation.get("viz_aabb_size") or observation["aabb_size"]),
            }
        if not minimal_gt:
            observation_attributes.update(
                {
                    "asset_id": observation.get("asset_id"),
                    "object_id": observation.get("object_id"),
                    "orientation": list(observation.get("orientation") or [0.0, 0.0, 0.0, 1.0]),
                    "interaction_approach_axis_xy": list(
                        observation.get("interaction_approach_axis_xy") or []
                    ),
                    "projected_bbox_2d": list(
                        observation.get("projected_bbox_2d") or []
                    ),
                }
            )
        node.attributes.update(observation_attributes)
        source_object_name = str(
            observation.get("source_object_name")
            or observation.get("instance_id")
            or node.name
            or ""
        )
        geometry_override = self.interaction_geometry_overrides.get(
            source_object_name
        )
        if geometry_override:
            for key in (
                "interaction_approach_axis_xy",
                "interaction_approach_pose_xyyaw",
                "interaction_reference_aabb_center",
                "interaction_reference_aabb_size",
            ):
                values = list(geometry_override.get(key) or [])
                if values:
                    node.attributes[key] = values
            node.attributes["interaction_geometry_source"] = str(
                geometry_override.get("source") or "configured_override"
            )
        for deprecated_key in (
            "parent",
            "children",
            "is_receptacle",
            "is_pickup_candidate",
            "is_articulable",
            "is_door",
            "is_movable_door",
            "joint_infos",
            "interaction_groups",
            "interaction_group_source",
            "observation_evidence",
        ):
            node.attributes.pop(deprecated_key, None)

        previous_interaction_memory = {
            key: node.interaction.get(key)
            for key in (
                "operation_history",
                "completed_interaction_groups",
                "failed_interaction_groups",
            )
            if key in node.interaction
        }
        node.interaction = default_interaction_payload(node.type, observation)
        if node.type == "portal":
            if "interaction_reference_aabb_center" not in node.attributes:
                node.attributes["interaction_reference_aabb_center"] = list(
                    observation["aabb_center"]
                )
                node.attributes["interaction_reference_aabb_size"] = list(
                    observation["aabb_size"]
                )
        if interaction_state_override:
            for key in (
                "state",
                "state_source",
                "state_confidence",
                "traversable",
                "requires_interaction",
                "is_interactable",
                "interaction_mode",
                "capability",
                "failure_reason",
            ):
                if key in interaction_state_override:
                    node.interaction[key] = interaction_state_override[key]

        node.interaction.update(previous_interaction_memory)

    def _refresh_room_nodes_from_grid(self, geometry_stability_frames=None):
        if not self.room_grid:
            return
        grid_info = self.room_grid["info"]
        scene_data = self.room_grid["scene_data"]
        confidence_data = self.room_grid["confidence_data"]
        if grid_info is None or not scene_data:
            return
        if geometry_stability_frames is None:
            geometry_stability_frames = self.room_geometry_stability_frames
        room_points = defaultdict(list)
        room_conf = defaultdict(list)
        width = int(grid_info.width)
        for idx, scene_id in enumerate(scene_data):
            scene_id = int(scene_id)
            if scene_id < 0:
                continue
            mx = idx % width
            my = idx // width
            wx = float(grid_info.origin.position.x) + (mx + 0.5) * float(grid_info.resolution)
            wy = float(grid_info.origin.position.y) + (my + 0.5) * float(grid_info.resolution)
            room_points[scene_id].append((wx, wy))
            if idx < len(confidence_data):
                room_conf[scene_id].append(float(confidence_data[idx]))
        for room_id, points in room_points.items():
            room_id = self._resolve_room_id(room_id)
            node = self._ensure_room_node(room_id)
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            center = [sum(xs) / len(xs), sum(ys) / len(ys), 0.5 * self.room_box_height]
            size = [
                max(float(grid_info.resolution), max(xs) - min(xs)),
                max(float(grid_info.resolution), max(ys) - min(ys)),
                self.room_box_height,
            ]
            stable_geometry = self._accept_room_geometry(
                room_id, center, size, geometry_stability_frames
            )
            if stable_geometry is not None:
                stable_center, stable_size = stable_geometry
                node.centroid = stable_center
                node.aabb_center = stable_center
                node.aabb_size = stable_size
            node.confidence = max(node.confidence, sum(room_conf[room_id]) / max(len(room_conf[room_id]), 1) / 100.0)
            node.attributes["cell_count"] = len(points)
            node.attributes["active"] = True

    def _accept_room_geometry(self, room_id, center, size, stability_frames):
        candidate = self.room_geometry_candidates.get(room_id)
        if candidate is None:
            candidate = {
                "center": list(center),
                "size": list(size),
                "count": 1,
                "accepted_center": list(center),
                "accepted_size": list(size),
            }
            self.room_geometry_candidates[room_id] = candidate
            return list(center), list(size)
        if self._room_geometry_close(
            candidate["center"], candidate["size"], center, size
        ):
            candidate["count"] += 1
        else:
            candidate["count"] = 1
        candidate["center"] = list(center)
        candidate["size"] = list(size)
        if candidate["count"] >= max(1, int(stability_frames)):
            candidate["accepted_center"], candidate["accepted_size"] = (
                self._expanded_room_geometry(
                    candidate["accepted_center"],
                    candidate["accepted_size"],
                    center,
                    size,
                )
            )
        self.room_geometry_candidates[room_id] = candidate
        return list(candidate["accepted_center"]), list(candidate["accepted_size"])

    @staticmethod
    def _room_geometry_close(old_center, old_size, center, size):
        center_delta = math.hypot(
            float(old_center[0]) - float(center[0]),
            float(old_center[1]) - float(center[1]),
        )
        size_delta = max(
            abs(float(old_size[0]) - float(size[0])),
            abs(float(old_size[1]) - float(size[1])),
        )
        return center_delta <= 0.40 and size_delta <= 0.60

    def _resolve_room_id(self, room_id):
        room_id = int(room_id)
        seen = set()
        while room_id in self.room_redirects and room_id not in seen:
            seen.add(room_id)
            room_id = int(self.room_redirects[room_id])
        return room_id

    def _apply_room_merges(self, merges):
        for secondary, primary in merges.items():
            secondary = self._resolve_room_id(secondary)
            primary = self._resolve_room_id(primary)
            if secondary == primary:
                continue
            self.room_redirects[secondary] = primary
            old_node = self.nodes.get(f"room_{secondary}")
            if old_node is not None:
                old_node.attributes["active"] = False
                old_node.attributes["merged_into"] = f"room_{primary}"
            for node in self.nodes.values():
                if node.type != "room" and node.room_id == secondary:
                    node.room_id = primary
                if node.type == "portal":
                    room_ids = node.attributes.get("connected_room_ids") or []
                    node.attributes["connected_room_ids"] = sorted(
                        {self._resolve_room_id(room_id) for room_id in room_ids}
                    )

    def _refresh_missing_room_nodes_from_observations(self):
        room_to_nodes = defaultdict(list)
        for node in self.nodes.values():
            if node.type == "room" or node.room_id is None:
                continue
            if int(node.room_id) in self.room_geometries:
                continue
            room_to_nodes[int(node.room_id)].append(node)
        for room_id, child_nodes in room_to_nodes.items():
            if not child_nodes:
                continue
            room_node = self._ensure_room_node(room_id)
            mins = []
            maxs = []
            for child in child_nodes:
                center = [float(v) for v in child.aabb_center]
                size = [float(v) for v in child.aabb_size]
                mins.append([center[i] - size[i] * 0.5 for i in range(3)])
                maxs.append([center[i] + size[i] * 0.5 for i in range(3)])
            min_corner = [min(point[i] for point in mins) for i in range(3)]
            max_corner = [max(point[i] for point in maxs) for i in range(3)]
            center = [(min_corner[i] + max_corner[i]) * 0.5 for i in range(3)]
            center[2] = 0.5 * self.room_box_height
            size = [max(max_corner[i] - min_corner[i], 0.1) for i in range(3)]
            size[2] = self.room_box_height
            room_node.centroid = center
            room_node.aabb_center = center
            room_node.aabb_size = size
            room_node.attributes["estimated_from_observations"] = True
            room_node.attributes["cell_count"] = len(child_nodes)

    def _ensure_room_node(self, room_id):
        room_id = int(room_id)
        node_id = f"room_{room_id}"
        node = self.nodes.get(node_id)
        if node is None:
            label = room_node_label(self.room_id_to_name, room_id)
            geometry = self.room_geometries.get(room_id, {})
            center = self._grounded_center(geometry.get("center") or self._default_room_center(room_id))
            aabb_center = self._grounded_center(geometry.get("aabb_center") or center)
            size = self._room_box_size(geometry.get("aabb_size") or self._default_room_size(room_id))
            node = SceneGraphNode(
                id=node_id,
                type="room",
                label=label,
                name=str(geometry.get("name") or f"{label}_{room_id}"),
                centroid=center,
                aabb_center=aabb_center,
                aabb_size=size,
                parent_id=self._ensure_scene_node().id,
                room_id=room_id,
            )
            if "cell_count" in geometry:
                node.attributes["cell_count"] = int(geometry["cell_count"])
            self.nodes[node_id] = node
        return node

    def _default_room_center(self, room_id):
        if not self.room_grid:
            return [float(room_id), 0.0, 0.0]
        grid_info = self.room_grid.get("info")
        scene_data = list(self.room_grid.get("scene_data") or [])
        if grid_info is None or not scene_data:
            return [float(room_id), 0.0, 0.0]
        width = int(grid_info.width)
        xs = []
        ys = []
        for idx, scene_id in enumerate(scene_data):
            if int(scene_id) != int(room_id):
                continue
            mx = idx % width
            my = idx // width
            xs.append(float(grid_info.origin.position.x) + (mx + 0.5) * float(grid_info.resolution))
            ys.append(float(grid_info.origin.position.y) + (my + 0.5) * float(grid_info.resolution))
        if xs and ys:
            return [sum(xs) / len(xs), sum(ys) / len(ys), 0.5 * self.room_box_height]
        return [float(room_id), 0.0, 0.0]

    def _default_room_size(self, room_id):
        if not self.room_grid:
            return [0.5, 0.5, 0.1]
        grid_info = self.room_grid.get("info")
        scene_data = list(self.room_grid.get("scene_data") or [])
        if grid_info is None or not scene_data:
            return [0.5, 0.5, 0.1]
        width = int(grid_info.width)
        xs = []
        ys = []
        for idx, scene_id in enumerate(scene_data):
            if int(scene_id) != int(room_id):
                continue
            mx = idx % width
            my = idx // width
            xs.append(float(grid_info.origin.position.x) + (mx + 0.5) * float(grid_info.resolution))
            ys.append(float(grid_info.origin.position.y) + (my + 0.5) * float(grid_info.resolution))
        if len(xs) > 1 and len(ys) > 1:
            return [max(max(xs) - min(xs), float(grid_info.resolution)), max(max(ys) - min(ys), float(grid_info.resolution)), self.room_box_height]
        resolution = float(grid_info.resolution) if grid_info is not None else 0.5
        return [resolution, resolution, self.room_box_height]

    def _ground_non_room_centroid(self, center, size):
        grounded = [float(center[0]), float(center[1]), float(center[2])]
        half_height = max(float(size[2]) * 0.5, 0.01)
        grounded[2] = max(grounded[2], half_height)
        return grounded

    def _grounded_center(self, center):
        grounded = list(center)
        if len(grounded) < 3:
            grounded.extend([0.0] * (3 - len(grounded)))
        grounded = [float(v) for v in grounded[:3]]
        grounded[2] = 0.5 * self.room_box_height
        return grounded

    def _room_box_size(self, size):
        size = list(size or [])
        if len(size) < 3:
            size.extend([0.0] * (3 - len(size)))
        room_size = [max(float(size[0]), 0.1), max(float(size[1]), 0.1), self.room_box_height]
        return room_size

    def _rebuild_relations(self, now=None):
        now = float(now if now is not None else time.time())
        self.edges = {}
        scene_node = self._ensure_scene_node()
        scene_node.attributes["source_mode"] = self.source_mode
        scene_node.attributes["episode_id"] = self.episode_id
        rooms = {
            node.id: node
            for node in self.nodes.values()
            if node.type == "room" and node.attributes.get("active", True)
        }
        for room_node in (
            node
            for node in self.nodes.values()
            if node.type == "room" and node.attributes.get("active", True)
        ):
            room_node.parent_id = scene_node.id
            self._upsert_edge(scene_node.id, "has_room", room_node.id, now=now)
        non_rooms = [node for node in self.nodes.values() if node.type not in {"scene", "room"}]

        for node in non_rooms:
            if node.type == "portal":
                node.parent_id = scene_node.id
            room_id = node.room_id
            if room_id is None:
                room_id = self._infer_room_id_from_node(node)
                node.room_id = room_id
            if room_id is not None:
                room_id = self._resolve_room_id(room_id)
                node.room_id = room_id
                room_node = self._ensure_room_node(room_id)
                if node.type != "portal":
                    node.parent_id = room_node.id
                self._upsert_edge(node.id, "in_room", room_node.id, now=now)
                self._upsert_edge(room_node.id, "has_child", node.id, now=now)

        for node in non_rooms:
            if node.type == "portal":
                connected_room_ids = list(node.attributes.get("connected_room_ids") or [])
                if self.source_mode != "gt_replay" or not connected_room_ids:
                    connected_room_ids = self._infer_portal_room_ids(node)
                connected_room_ids = [
                    self._resolve_room_id(room_id)
                    for room_id in connected_room_ids
                    if room_id is not None
                ]
                node.attributes["connected_room_ids"] = connected_room_ids
                node.attributes["connectivity_source"] = (
                    "observation"
                    if self.source_mode == "gt_replay" and connected_room_ids
                    else "room_segment_ring"
                )
                node.attributes["connectivity_status"] = (
                    "connected" if len(connected_room_ids) >= 2 else "partial" if len(connected_room_ids) == 1 else "unknown"
                )
                traversable = node.interaction.get("traversable")
                edge_attributes = {
                    "portal_node_id": node.id,
                    "state": node.interaction.get("state", "unknown"),
                    "traversable": traversable,
                    "requires_interaction": bool(node.interaction.get("requires_interaction")),
                    "interaction_mode": node.interaction.get("interaction_mode", "none"),
                    "interaction_cost": float(node.interaction.get("interaction_cost", 1.0)),
                    "expected_effect": "unlock_connectivity",
                    "connectivity_status": node.attributes["connectivity_status"],
                }
                for room_id in sorted(set(int(room) for room in connected_room_ids if room is not None)):
                    room_node = self._ensure_room_node(room_id)
                    self._upsert_edge(node.id, "connects", room_node.id, attributes=edge_attributes, now=now)
                    self._upsert_edge(room_node.id, "adjacent_via", node.id, attributes=edge_attributes, now=now)

        support_nodes = [node for node in non_rooms if node.type == "support"]
        container_nodes = [node for node in non_rooms if node.type == "container"]
        object_nodes = [node for node in non_rooms if node.type == "object"]
        id_lookup = {node.id: node for node in non_rooms}
        for container in container_nodes:
            container.attributes["inferred_child_ids"] = []
            container.attributes["inferred_child_count"] = 0

        for obj in object_nodes:
            previous_parent_id = obj.parent_id
            obj.parent_id = None
            parent = self._find_parent_node(
                obj,
                support_nodes,
                container_nodes,
                id_lookup,
                previous_parent_id=previous_parent_id,
            )
            if parent is None:
                if obj.room_id is not None:
                    obj.parent_id = self._ensure_room_node(obj.room_id).id
                continue
            obj.parent_id = parent.id
            if parent.type == "support":
                self._upsert_edge(parent.id, "supports", obj.id, now=now)
            elif parent.type == "container":
                self._upsert_edge(parent.id, "contains", obj.id, now=now)
                parent.attributes["inferred_child_ids"].append(obj.id)

        for container in container_nodes:
            child_ids = sorted(set(container.attributes["inferred_child_ids"]))
            container.attributes["inferred_child_ids"] = child_ids
            container.attributes["inferred_child_count"] = len(child_ids)

        for room_node in (
            node
            for node in self.nodes.values()
            if node.type == "room" and node.attributes.get("active", True)
        ):
            room_node.parent_id = scene_node.id
            self._upsert_edge(scene_node.id, "has_room", room_node.id, now=now)
        self._refresh_room_attributes()

    def _find_parent_node(
        self,
        obj,
        support_nodes,
        container_nodes,
        id_lookup,
        previous_parent_id=None,
    ):
        containing = [
            node
            for node in container_nodes
            if _is_plausible_container_content(obj, node)
            and self._is_inside_volume(obj, node)
        ]
        if containing:
            return sorted(containing, key=lambda node: volume(node.aabb_size))[0]

        if not obj.is_currently_visible and previous_parent_id:
            previous_parent = id_lookup.get(str(previous_parent_id))
            if previous_parent is not None and previous_parent.type == "container":
                return previous_parent

        supporting = [node for node in support_nodes if self._is_on_support(obj, node)]
        if supporting:
            return sorted(supporting, key=lambda node: abs(top_surface_z(node) - obj.centroid[2]))[0]
        return None

    def _upsert_edge(self, src_id, relation, dst_id, attributes=None, confidence=1.0, now=None):
        edge_id = f"edge_{relation}_{sanitize_token(src_id)}_{sanitize_token(dst_id)}"
        self.edges[edge_id] = SceneGraphEdge(
            id=edge_id,
            src_id=src_id,
            relation=relation,
            dst_id=dst_id,
            attributes=dict(attributes or {}),
            confidence=float(confidence),
            last_seen=now,
        )

    def _infer_room_id_from_node(self, node):
        if not self.room_grid:
            return None
        grid_info = self.room_grid["info"]
        scene_data = self.room_grid["scene_data"]
        if grid_info is None or not scene_data:
            return None
        candidates = defaultdict(int)
        center = node.aabb_center
        size = node.aabb_size
        half_x = max(float(size[0]) * 0.45, 0.02)
        half_y = max(float(size[1]) * 0.45, 0.02)
        sample_points = [
            (float(center[0]), float(center[1])),
            (float(center[0]) - half_x, float(center[1]) - half_y),
            (float(center[0]) - half_x, float(center[1]) + half_y),
            (float(center[0]) + half_x, float(center[1]) - half_y),
            (float(center[0]) + half_x, float(center[1]) + half_y),
        ]
        for px, py in sample_points:
            coords = world_to_grid(px, py, grid_info)
            if coords is None:
                continue
            idx = grid_index(coords[0], coords[1], grid_info.width)
            if idx < 0 or idx >= len(scene_data):
                continue
            room_id = int(scene_data[idx])
            if room_id >= 0:
                candidates[room_id] += 1
        if candidates:
            return max(sorted(candidates.keys()), key=lambda room_id: candidates[room_id])
        resolution = max(float(getattr(grid_info, "resolution", 0.05)), 0.01)
        margins = []
        margin = max(0.10, resolution * 2.0)
        while margin < self.object_room_search_margin_m:
            margins.append(margin)
            margin *= 2.0
        margins.append(self.object_room_search_margin_m)
        for margin in sorted(set(round(value, 6) for value in margins)):
            ring_candidates = defaultdict(int)
            expanded_half_x = max(float(size[0]) * 0.5 + margin, margin)
            expanded_half_y = max(float(size[1]) * 0.5 + margin, margin)
            for index in range(8):
                ratio = float(index) / 7.0
                x = float(center[0]) - expanded_half_x + 2.0 * expanded_half_x * ratio
                y = float(center[1]) - expanded_half_y + 2.0 * expanded_half_y * ratio
                perimeter_points = (
                    (x, float(center[1]) - expanded_half_y),
                    (x, float(center[1]) + expanded_half_y),
                    (float(center[0]) - expanded_half_x, y),
                    (float(center[0]) + expanded_half_x, y),
                )
                for px, py in perimeter_points:
                    coords = world_to_grid(px, py, grid_info)
                    if coords is None:
                        continue
                    idx = grid_index(coords[0], coords[1], grid_info.width)
                    if 0 <= idx < len(scene_data):
                        room_id = int(scene_data[idx])
                        if room_id >= 0:
                            ring_candidates[room_id] += 1
            if ring_candidates:
                return max(
                    sorted(ring_candidates.keys()),
                    key=lambda room_id: ring_candidates[room_id],
                )
        return None

    def _infer_portal_room_ids(self, node):
        if not self.room_grid:
            return [node.room_id] if node.room_id is not None else []
        grid_info = self.room_grid["info"]
        scene_data = self.room_grid["scene_data"]
        if grid_info is None or not scene_data:
            return [node.room_id] if node.room_id is not None else []
        counts = defaultdict(int)
        center_x, center_y = float(node.aabb_center[0]), float(node.aabb_center[1])
        radii = sorted(
            {
                min(self.portal_room_max_radius_m, radius)
                for radius in (0.25, 0.50, 0.75, self.portal_room_max_radius_m)
                if radius > 0.0
            }
        )
        for radius_index, radius in enumerate(radii):
            weight = len(radii) - radius_index
            for index in range(48):
                angle = 2.0 * math.pi * float(index) / 48.0
                coords = world_to_grid(
                    center_x + radius * math.cos(angle),
                    center_y + radius * math.sin(angle),
                    grid_info,
                )
                if coords is None:
                    continue
                data_index = grid_index(coords[0], coords[1], grid_info.width)
                if 0 <= data_index < len(scene_data):
                    room_id = int(scene_data[data_index])
                    if room_id >= 0:
                        counts[room_id] += weight
        ranked = sorted(counts, key=lambda room_id: (-counts[room_id], room_id))
        return ranked[:2]

    def _refresh_room_attributes(self):
        room_nodes = {
            int(node.room_id): node
            for node in self.nodes.values()
            if node.type == "room"
            and node.room_id is not None
            and node.attributes.get("active", True)
        }
        evidence_by_room = defaultdict(list)
        for node in self.nodes.values():
            if node.type in {"scene", "room", "portal"} or node.room_id is None:
                continue
            room_id = self._resolve_room_id(node.room_id)
            evidence_by_room[room_id].append(
                {
                    "node_id": node.id,
                    "semantic_name": node.label,
                    "category": node.attributes.get("category"),
                    "confidence": node.confidence,
                }
            )
        for room_id, room_node in room_nodes.items():
            result = self.room_attribute_inferencer.infer(
                evidence_by_room.get(room_id, [])
            )
            room_node.attributes.update(
                {
                    "room_attribute": result["room_attribute"],
                    "room_attribute_confidence": float(result["confidence"]),
                    "room_attribute_scores": dict(result["scores"]),
                    "room_attribute_evidence": list(result["evidence"]),
                    "room_attribute_source": "weighted_object_types",
                }
            )

    def _bump_revision(self):
        self.graph_revision += 1
        for node in self.nodes.values():
            node.graph_revision = self.graph_revision

    def _build_navigation_hints(self):
        hints = []
        counter = 1
        for node in sorted(self.nodes.values(), key=lambda item: item.id):
            if node.type == "room" and node.attributes.get("active", True):
                hints.append(
                    self._make_hint(counter, "room_center", node, False, node.id, "none", "known_room_center")
                )
                counter += 1
            elif node.type == "object":
                hints.append(
                    self._make_hint(counter, "target_object", node, False, None, "none", "detected_object")
                )
                counter += 1
            elif node.type == "portal":
                hints.append(
                    self._make_hint(
                        counter,
                        "interactive_portal",
                        node,
                        bool(node.interaction.get("requires_interaction")),
                        node.id,
                        node.interaction.get("interaction_mode", "none"),
                        "door_may_unlock_room",
                    )
                )
                counter += 1
            elif node.type == "container":
                hints.append(
                    self._make_hint(
                        counter,
                        "interactive_container",
                        node,
                        bool(node.interaction.get("is_interactable")),
                        node.id,
                        node.interaction.get("interaction_mode", "none"),
                        "container_may_reveal_object",
                    )
                )
                counter += 1
            elif node.type == "support":
                hints.append(
                    self._make_hint(
                        counter,
                        "support_surface",
                        node,
                        bool(node.interaction.get("is_interactable")),
                        node.id,
                        node.interaction.get("interaction_mode", "none"),
                        "support_surface_context",
                    )
                )
                counter += 1
        return hints

    def _make_hint(self, index, hint_type, node, requires_interaction, interaction_node_id, interaction_mode, reason):
        return NavigationHint(
            hint_id=f"hint_{index:04d}",
            type=hint_type,
            node_id=node.id,
            position=list(node.centroid),
            room_id=node.room_id,
            priority=1.0,
            confidence=float(node.confidence),
            requires_interaction=bool(requires_interaction),
            interaction_node_id=interaction_node_id,
            interaction_mode=interaction_mode,
            state=node.interaction.get("state", "unknown"),
            reason=reason,
        )


def volume(size):
    return max(float(size[0]), 0.0) * max(float(size[1]), 0.0) * max(float(size[2]), 0.0)


def top_surface_z(node):
    return float(node.aabb_center[2]) + float(node.aabb_size[2]) * 0.5


def point_inside_2d(point, center, size, margin=0.05):
    half_x = float(size[0]) * 0.5 + margin
    half_y = float(size[1]) * 0.5 + margin
    return (
        abs(float(point[0]) - float(center[0])) <= half_x
        and abs(float(point[1]) - float(center[1])) <= half_y
    )


def point_inside_3d(point, center, size, margin=0.05):
    return (
        abs(float(point[0]) - float(center[0])) <= float(size[0]) * 0.5 + margin
        and abs(float(point[1]) - float(center[1])) <= float(size[1]) * 0.5 + margin
        and abs(float(point[2]) - float(center[2])) <= float(size[2]) * 0.5 + margin
    )


def _vertical_gap(obj, support):
    return abs(float(obj.centroid[2]) - top_surface_z(support))


def _support_height_limit(support):
    return max(0.25, float(support.aabb_size[2]) + 0.6)


def _same_room_or_unknown(obj, candidate):
    return candidate.room_id is None or obj.room_id is None or candidate.room_id == obj.room_id


def _support_xy_match(obj, support):
    return point_inside_2d(obj.centroid, support.aabb_center, support.aabb_size, margin=0.08)


def _object_above_support(obj, support):
    return float(obj.centroid[2]) >= top_surface_z(support) - 0.12


def _object_not_too_high(obj, support):
    return _vertical_gap(obj, support) <= _support_height_limit(support)


def _container_contains(obj, container, min_axis_fraction=0.90):
    for axis in range(3):
        object_center = float(obj.aabb_center[axis])
        object_size = max(0.0, float(obj.aabb_size[axis]))
        container_center = float(container.aabb_center[axis])
        container_size = max(0.0, float(container.aabb_size[axis]))
        container_min = container_center - 0.5 * container_size
        container_max = container_center + 0.5 * container_size
        if not container_min <= object_center <= container_max:
            return False
        if object_size <= 1e-8:
            continue
        object_min = object_center - 0.5 * object_size
        object_max = object_center + 0.5 * object_size
        overlap = max(0.0, min(object_max, container_max) - max(object_min, container_min))
        if overlap / object_size < float(min_axis_fraction):
            return False
    return True


def _is_plausible_container_content(obj, container):
    label = normalize_label(obj.label or obj.name)
    if any(token in label for token in ("plant", "flower", "tree")):
        return False
    container_volume = volume(container.aabb_size)
    object_volume = volume(obj.aabb_size)
    return container_volume > 1e-6 and object_volume <= min(0.10, 0.10 * container_volume)


InteractionGraphStore._is_inside_volume = staticmethod(_container_contains)
InteractionGraphStore._is_on_support = staticmethod(
    lambda obj, support: _same_room_or_unknown(obj, support)
    and _support_xy_match(obj, support)
    and _object_above_support(obj, support)
    and _object_not_too_high(obj, support)
)
