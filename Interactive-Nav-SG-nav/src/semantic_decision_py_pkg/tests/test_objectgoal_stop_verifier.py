from types import SimpleNamespace

from semantic_decision_py_pkg.objectgoal_stop_verifier import ObjectGoalStopVerifier


class _Client:
    def __init__(self, payload):
        self.payload = payload
        self.kwargs = None

    def request_json(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(payload=dict(self.payload), error="")


def _payload(*, mostly_visible: bool):
    return {
        "decision": "STOP",
        "target_matches": True,
        "target_visible": True,
        "target_mostly_visible": mostly_visible,
        "target_distance_close_enough": True,
        "stopping_view_is_clear": True,
        "confidence": 0.95,
        "reason": "test",
    }


def test_stop_requires_most_of_target_to_be_visible() -> None:
    client = _Client(_payload(mostly_visible=False))
    result = ObjectGoalStopVerifier(client).verify(
        target_name="television",
        image_data_url="data:image/jpeg;base64,AA==",
        detector_confidence=0.9,
        public_candidate_distance_m=0.2,
        bbox_center_distance_m=0.7,
        max_bbox_center_distance_m=0.85,
    )

    assert result.decision == "CONTINUE"
    assert "target_mostly_visible" in client.kwargs["response_schema"]["schema"]["required"]
    assert "small fragment" in client.kwargs["instruction"]


def test_stop_accepts_clear_mostly_visible_target() -> None:
    client = _Client(_payload(mostly_visible=True))
    result = ObjectGoalStopVerifier(client).verify(
        target_name="television",
        image_data_url="data:image/jpeg;base64,AA==",
        detector_confidence=0.9,
        public_candidate_distance_m=0.2,
        bbox_center_distance_m=0.7,
        max_bbox_center_distance_m=0.85,
    )

    assert result.decision == "STOP"


def test_stop_is_rejected_when_public_bbox_center_is_too_far() -> None:
    client = _Client(_payload(mostly_visible=True))
    result = ObjectGoalStopVerifier(client).verify(
        target_name="television",
        image_data_url="data:image/jpeg;base64,AA==",
        detector_confidence=0.9,
        public_candidate_distance_m=0.2,
        bbox_center_distance_m=1.2,
        max_bbox_center_distance_m=0.85,
    )

    assert result.decision == "CONTINUE"
    assert client.kwargs["context"]["bbox_center_distance_m"] == 1.2
