from collections import defaultdict

from .geometry_utils import normalize_label


class RoomInferenceBackend:
    def infer(self, detections):
        raise NotImplementedError


class WeightedRoomAttributeInferencer(RoomInferenceBackend):
    def __init__(self, object_room_priors, min_confidence=0.2):
        self.min_confidence = float(min_confidence)
        self.object_to_room_scores = defaultdict(dict)
        for room, priors in (object_room_priors or {}).items():
            for obj_name, score in (priors or {}).items():
                self.object_to_room_scores[normalize_label(obj_name)][
                    normalize_label(room)
                ] = float(score)

    def infer(self, detections):
        strongest_evidence = {}
        for detection in detections or []:
            label = normalize_label(
                detection.get("semantic_name")
                or detection.get("semantic_class")
                or detection.get("class")
                or detection.get("category")
            )
            confidence = float(
                detection.get("confidence", detection.get("conf", 1.0)) or 0.0
            )
            for room, prior in self.object_to_room_scores.get(label, {}).items():
                vote = confidence * prior
                evidence_key = (room, label)
                evidence_record = {
                    "node_id": str(detection.get("node_id") or ""),
                    "object_label": label,
                    "object_confidence": confidence,
                    "prior_weight": prior,
                    "vote": vote,
                }
                previous = strongest_evidence.get(evidence_key)
                if previous is None or vote > float(previous["vote"]):
                    strongest_evidence[evidence_key] = evidence_record

        room_scores = defaultdict(float)
        evidence = defaultdict(list)
        for (room, _label), evidence_record in strongest_evidence.items():
            room_scores[room] += float(evidence_record["vote"])
            evidence[room].append(evidence_record)

        if not room_scores:
            return {
                "room_attribute": "unknown",
                "confidence": 0.0,
                "scores": {},
                "evidence": [],
            }

        room, score = max(room_scores.items(), key=lambda item: (item[1], item[0]))
        total_score = sum(max(value, 0.0) for value in room_scores.values())
        confidence = min(1.0, max(score, 0.0)) * (
            max(score, 0.0) / total_score if total_score > 1e-8 else 0.0
        )
        inferred_room = room if confidence >= self.min_confidence else "unknown"
        ranked_evidence = sorted(
            evidence[room],
            key=lambda item: (-float(item["vote"]), item["object_label"], item["node_id"]),
        )
        return {
            "room_attribute": inferred_room,
            "confidence": confidence,
            "scores": dict(sorted(room_scores.items())),
            "evidence": ranked_evidence,
        }


class ObjectRulesRoomInference(RoomInferenceBackend):
    def __init__(self, object_room_priors, min_confidence=0.2):
        self.object_room_priors = object_room_priors or {}
        self.min_confidence = float(min_confidence)
        self.object_to_room_scores = defaultdict(dict)
        for room, priors in self.object_room_priors.items():
            for obj_name, score in (priors or {}).items():
                self.object_to_room_scores[normalize_label(obj_name)][normalize_label(room)] = float(score)

    def infer(self, detections):
        room_scores = defaultdict(float)
        evidence = defaultdict(list)
        for det in detections:
            label = normalize_label(det.get("semantic_class") or det.get("class") or det.get("semantic_name"))
            confidence = float(det.get("confidence", det.get("conf", 1.0)) or 0.0)
            for room, prior in self.object_to_room_scores.get(label, {}).items():
                vote = confidence * prior
                room_scores[room] += vote
                evidence[room].append(label)

        if not room_scores:
            return {"scene_attribute": "unknown", "confidence": 0.0, "evidence": []}

        room, score = max(room_scores.items(), key=lambda item: item[1])
        confidence = min(1.0, score)
        if confidence < self.min_confidence:
            return {"scene_attribute": "unknown", "confidence": confidence, "evidence": evidence[room]}
        return {"scene_attribute": room, "confidence": confidence, "evidence": sorted(set(evidence[room]))}


def make_room_backend(kind, config, object_room_priors):
    kind = str(kind or "object_rules")
    if kind == "object_rules":
        return ObjectRulesRoomInference(
            object_room_priors=object_room_priors,
            min_confidence=config.get("min_confidence", 0.2),
        )
    return ObjectRulesRoomInference(object_room_priors=object_room_priors)
