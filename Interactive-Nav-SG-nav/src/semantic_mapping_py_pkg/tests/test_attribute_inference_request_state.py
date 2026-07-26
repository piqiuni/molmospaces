import threading

import pytest

pytest.importorskip("rospy")

from interaction_attribute_inference_node import InteractionAttributeInferenceNode


def test_initial_generation_zero_request_is_current() -> None:
    node = object.__new__(InteractionAttributeInferenceNode)
    node.lock = threading.Lock()
    node.current_episode_id = "episode_1"
    node.pending = {
        "gt_000001": {
            "request_sequence": 1,
            "generation": 0,
            "episode_id": "episode_1",
        }
    }
    node.generations = {"gt_000001": 0}

    assert node._is_current_request("gt_000001", "episode_1", 0, 1)
