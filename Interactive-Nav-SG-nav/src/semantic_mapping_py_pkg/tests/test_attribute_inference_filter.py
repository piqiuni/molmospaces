from semantic_mapping_py_pkg.attribute_filter import is_interaction_attribute_candidate


def test_filter_keeps_portals_and_named_containers() -> None:
    include = ("door", "fridge", "cabinet", "drawer")
    exclude = ("toilet", "sofa", "safe")
    assert is_interaction_attribute_candidate(
        {"semantic_name": "Door", "is_door": True}, include, exclude
    )
    assert is_interaction_attribute_candidate(
        {"semantic_name": "Fridge"}, include, exclude
    )
    assert is_interaction_attribute_candidate(
        {"semantic_name": "unknown", "is_receptacle": True, "is_articulable": True},
        include,
        exclude,
    )


def test_filter_rejects_noninteractive_and_excluded_objects() -> None:
    include = ("door", "fridge", "cabinet", "drawer")
    exclude = ("toilet", "sofa", "safe")
    assert not is_interaction_attribute_candidate({"semantic_name": "Sofa"}, include, exclude)
    assert not is_interaction_attribute_candidate({"semantic_name": "Safe"}, include, exclude)
    assert not is_interaction_attribute_candidate(
        {"semantic_name": "CounterTop", "is_receptacle": True, "is_articulable": False},
        include,
        exclude,
    )
