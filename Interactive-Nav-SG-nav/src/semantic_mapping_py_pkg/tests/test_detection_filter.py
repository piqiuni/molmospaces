from semantic_mapping_py_pkg.detection_filter import DetectionFilter
from semantic_mapping_py_pkg.detector_backends import FilteredRawDetectionProvider, YoloeLocalProvider


FILTER_CONFIG = {
    "enabled": True,
    "excluded_raw_labels": ["mirror", "bathroom mirror", "lamp", "ceiling_light", "extrude"],
    "excluded_semantic_labels": [],
    "aliases": {"exit": "door", "exit_door": "door"},
}


class _StaticProvider:
    def __init__(self, detections):
        self.detections = detections

    def detect_2d(self, *_args):
        return self.detections


def test_filter_removes_non_target_raw_labels_and_preserves_objects():
    detections = [
        {"semantic_class_raw": "bathroom mirror", "semantic_class": "unknown_open_set"},
        {"semantic_class_raw": "lamp", "semantic_class": "lamp"},
        {"semantic_class_raw": "extrude", "semantic_class": "unknown_open_set"},
        {"semantic_class_raw": "cabinet", "semantic_class": "cabinet"},
    ]
    assert DetectionFilter(FILTER_CONFIG).apply(detections) == [
        {"semantic_class_raw": "cabinet", "semantic_class": "cabinet"}
    ]


def test_filter_maps_exit_to_door_without_losing_raw_provenance():
    result = DetectionFilter(FILTER_CONFIG).apply(
        [{"semantic_class_raw": "exit", "semantic_class": "unknown_open_set", "confidence": 0.8}]
    )
    assert result == [{"semantic_class_raw": "exit", "semantic_class": "door", "confidence": 0.8}]


def test_provider_filter_runs_before_projection_consumers():
    provider = FilteredRawDetectionProvider(
        _StaticProvider(
            [
                {"semantic_class_raw": "ceiling light", "semantic_class": "lamp"},
                {"semantic_class_raw": "drawer", "semantic_class": "drawer"},
            ]
        ),
        FILTER_CONFIG,
    )
    assert provider.detect_2d(None, None, None, None) == [
        {"semantic_class_raw": "drawer", "semantic_class": "drawer"}
    ]


def test_disabled_filter_is_a_noop_for_model_payloads():
    source = [{"semantic_class_raw": "mirror", "semantic_class": "mirror"}]
    assert DetectionFilter({"enabled": False}).apply(source) == source


def test_yoloe_mapping_accepts_filter_aliases_before_unknown_drop():
    provider = YoloeLocalProvider(
        model_path="unused.pt",
        class_mapping={},
        class_aliases={"exit": "door"},
    )
    assert provider.class_mapping["exit"] == "door"
