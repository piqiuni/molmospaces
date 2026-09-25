from semantic_decision_py_pkg.frontier_failure_memory import FrontierFailureMemory


def test_failed_region_cooldown_then_single_retry():
    memory = FrontierFailureMemory()
    memory.record_failure([1, 2], "map", 38)
    assert memory.rejection([1.2, 2.1], "/map", 133) == "frontier_region_failure_cooldown"
    assert memory.rejection([1.2, 2.1], "map", 158) == ""
    memory.record_failure([1.2, 2.1], "map", 160)
    assert memory.rejection([1, 2], "map", 2000) == "frontier_region_failed_requires_topology_change"
    memory.clear()
    assert memory.rejection([1, 2], "map", 2001) == ""


def test_other_regions_frames_and_invalid_geometry_not_blocked():
    memory = FrontierFailureMemory()
    memory.record_failure([1, 2], "map", 10)
    memory.record_failure([], "map", 10)
    memory.record_failure([float("nan"), 0], "map", 10)
    assert len(memory.entries) == 1
    assert memory.rejection([4, 2], "map", 11) == ""
    assert memory.rejection([1, 2], "odom", 11) == ""
    assert memory.rejection([], "map", 11) == ""
