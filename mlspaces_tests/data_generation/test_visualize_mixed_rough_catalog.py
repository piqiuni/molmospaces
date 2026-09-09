from __future__ import annotations

from collections import Counter

from scripts.InteractiveNav import visualize_mixed_rough_catalog as visualize


def candidate(
    index: int,
    *,
    house: int,
    container: str,
    length: float,
    door_count: int,
    target: str,
) -> dict:
    return {
        "_candidate_index": index,
        "case_id": f"case_{index}",
        "house_index": house,
        "container_category": container,
        "target_category": target,
        "all_open_path_length_m": length,
        "crossed_door_roots": [f"door_{value}" for value in range(door_count)],
    }


def test_path_and_door_count_bins_cover_boundaries():
    assert visualize.path_length_bin(4.99) == "lt_5m"
    assert visualize.path_length_bin(5.0) == "5_to_8m"
    assert visualize.path_length_bin(8.0) == "8_to_12m"
    assert visualize.path_length_bin(12.0) == "12_to_20m"
    assert visualize.path_length_bin(20.0) == "ge_20m"
    assert visualize.crossed_door_count_bin(1) == "1_door"
    assert visualize.crossed_door_count_bin(2) == "2_doors"
    assert visualize.crossed_door_count_bin(4) == "3plus_doors"


def test_balanced_selection_keeps_houses_unique_and_container_types_even():
    rows = []
    for index in range(12):
        rows.append(
            candidate(
                index,
                house=index,
                container="Dresser" if index % 2 == 0 else "Fridge",
                length=[3.0, 6.0, 10.0, 15.0, 22.0][index % 5],
                door_count=3 if index == 0 else 2 if index in {1, 2} else 1,
                target=f"target_{index}",
            )
        )

    selected = visualize.balanced_select_candidates(
        rows, max_samples=6, seed=4, unique_houses=True
    )

    assert len(selected) == 6
    assert len({row["house_index"] for row in selected}) == 6
    assert Counter(row["container_category"] for row in selected) == {
        "Dresser": 3,
        "Fridge": 3,
    }
    assert any(len(row["crossed_door_roots"]) >= 3 for row in selected)


def test_select_candidate_rows_filters_candidate_type_and_explicit_house():
    catalog = {
        "candidates": [
            {
                **candidate(
                    0,
                    house=4,
                    container="Fridge",
                    length=7.0,
                    door_count=1,
                    target="apple",
                ),
                "rough_candidate_type": "door_crossing_only",
            },
            {
                **candidate(
                    1,
                    house=5,
                    container="Dresser",
                    length=9.0,
                    door_count=1,
                    target="book",
                ),
                "rough_candidate_type": "mixed_required_verified",
            },
        ]
    }
    for row in catalog["candidates"]:
        row.pop("_candidate_index")

    selected = visualize.select_candidate_rows(
        catalog,
        candidate_type="door_crossing_only",
        case_ids=None,
        house_indices={4},
        max_samples=None,
        seed=0,
        unique_houses=True,
    )

    assert len(selected) == 1
    assert selected[0]["_candidate_index"] == 0
    assert selected[0]["case_id"] == "case_0"
