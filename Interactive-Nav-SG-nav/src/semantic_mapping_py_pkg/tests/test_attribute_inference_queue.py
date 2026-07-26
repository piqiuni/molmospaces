from semantic_mapping_py_pkg.attribute_inference_queue import LatestPriorityRequestQueue


def request(object_id: str, priority: float, sequence: int) -> dict:
    return {
        "object_id": object_id,
        "priority": priority,
        "request_sequence": sequence,
    }


def test_queue_retains_higher_priority_request_when_full() -> None:
    queue = LatestPriorityRequestQueue(max_size=2)
    assert queue.put(request("door", 10.0, 1)) == (True, None)
    assert queue.put(request("cabinet", 5.0, 2)) == (True, None)

    accepted, dropped = queue.put(request("drawer", 1.0, 3))
    assert accepted is False
    assert dropped["object_id"] == "drawer"
    assert queue.get(0.0)["object_id"] == "door"
    assert queue.get(0.0)["object_id"] == "cabinet"


def test_queue_evicts_lower_priority_and_replaces_same_object() -> None:
    queue = LatestPriorityRequestQueue(max_size=2)
    queue.put(request("door", 10.0, 1))
    queue.put(request("cabinet", 2.0, 2))
    accepted, dropped = queue.put(request("fridge", 6.0, 3))
    assert accepted is True
    assert dropped["object_id"] == "cabinet"

    accepted, dropped = queue.put(request("door", 11.0, 4))
    assert accepted is True
    assert dropped["request_sequence"] == 1
    assert queue.get(0.0)["request_sequence"] == 4
    assert queue.get(0.0)["object_id"] == "fridge"


def test_queue_close_discards_pending_requests_and_rejects_new_work() -> None:
    queue = LatestPriorityRequestQueue(max_size=2)
    queue.put(request("door", 10.0, 1))

    queue.close()

    assert queue.get(0.0) is None
    accepted, dropped = queue.put(request("drawer", 5.0, 2))
    assert accepted is False
    assert dropped["object_id"] == "drawer"
