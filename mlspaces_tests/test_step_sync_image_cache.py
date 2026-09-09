from __future__ import annotations

import sys
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDER_SCRIPT_DIR = (
    REPO_ROOT / "Interactive-Nav-SG-nav" / "src" / "explore_py_pkg" / "scripts"
)
if str(RECORDER_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(RECORDER_SCRIPT_DIR))

from step_sync_image_cache import CachedStepImage, ExactStepImageCache


def _frame(seq: int, stamp_ns: int, value: int = 7) -> CachedStepImage:
    return CachedStepImage(
        source_seq=seq,
        stamp_ns=stamp_ns,
        stamp=stamp_ns / 1_000_000_000.0,
        width=2,
        height=1,
        rgb=bytes([value] * 6),
    )


def test_step_sync_waits_for_exact_rgb_arriving_on_another_callback() -> None:
    cache = ExactStepImageCache(max_size=4)

    def publish_image() -> None:
        time.sleep(0.02)
        cache.put(_frame(12, 345, value=19))

    publisher = threading.Thread(target=publish_image)
    publisher.start()
    matched = cache.wait_pop(12, 345, timeout_sec=0.5)
    publisher.join()

    assert matched is not None
    assert matched.rgb == bytes([19] * 6)
    assert cache.wait_pop(12, 345, timeout_sec=0.0) is None


def test_step_sync_never_substitutes_a_different_step_image() -> None:
    cache = ExactStepImageCache(max_size=2)
    cache.put(_frame(4, 100))
    cache.put(_frame(5, 200))

    assert cache.wait_pop(4, 999, timeout_sec=0.0) is None
    assert cache.wait_pop(4, 100, timeout_sec=0.0) is not None
    assert cache.wait_pop(5, 200, timeout_sec=0.0) is not None


def test_step_sync_allows_float_timestamp_round_trip_for_same_sequence() -> None:
    cache = ExactStepImageCache(max_size=2)
    cache.put(_frame(8, 1_000_000_123))

    matched = cache.wait_pop(
        8,
        1_000_000_000,
        timeout_sec=0.0,
        max_stamp_delta_ns=200,
    )
    assert matched is not None
    assert matched.source_seq == 8


def test_ros_bridge_contract_can_pair_by_common_stamp_when_seq_callback_lags() -> None:
    cache = ExactStepImageCache(max_size=4)
    # RosBridge publishes RGB before step_sync with a shared common_stamp, but
    # recorder callback completion and sequence visibility are asynchronous.
    cache.put(_frame(40, 2_000_000_100, value=31))

    selected = cache.wait_select(
        source_seq=41,
        stamp_ns=2_000_000_000,
        timeout_sec=0.0,
        max_stamp_delta_ns=500,
        fallback_max_age_ns=2_000_000_000,
    )
    assert selected is not None
    assert selected.source == "timestamp_image"
    assert selected.reused is False
    assert selected.frame.rgb == bytes([31] * 6)


def test_dropped_image_reuses_nearest_real_frame_instead_of_black_placeholder() -> None:
    cache = ExactStepImageCache(max_size=4)
    cache.put(_frame(9, 1_000_000_000, value=43))

    first = cache.wait_select(
        source_seq=10,
        stamp_ns=2_000_000_000,
        timeout_sec=0.0,
        max_stamp_delta_ns=5_000_000,
        fallback_max_age_ns=2_500_000_000,
    )
    second = cache.wait_select(
        source_seq=11,
        stamp_ns=3_000_000_000,
        timeout_sec=0.0,
        max_stamp_delta_ns=5_000_000,
        fallback_max_age_ns=2_500_000_000,
    )

    assert first is not None and first.source == "nearest_image"
    assert first.reused is False
    assert second is not None and second.source == "latest_image"
    assert second.reused is True
    assert second.frame.rgb == bytes([43] * 6)


def test_fallback_is_causal_and_never_uses_a_future_observation() -> None:
    cache = ExactStepImageCache(max_size=4)
    cache.put(_frame(4, 1_000_000_000, value=11))
    cache.put(_frame(6, 3_000_000_000, value=99))

    selected = cache.wait_select(
        source_seq=5,
        stamp_ns=2_000_000_000,
        timeout_sec=0.0,
        max_stamp_delta_ns=5_000_000,
        fallback_max_age_ns=2_000_000_000,
    )
    assert selected is not None
    assert selected.frame.stamp_ns == 1_000_000_000
    assert selected.frame.rgb == bytes([11] * 6)

    future_only = ExactStepImageCache(max_size=2)
    future_only.put(_frame(6, 2_100_000_000, value=99))
    assert (
        future_only.wait_select(
            source_seq=5,
            stamp_ns=2_000_000_000,
            timeout_sec=0.0,
            max_stamp_delta_ns=5_000_000,
            fallback_max_age_ns=2_000_000_000,
        )
        is None
    )


def test_newer_image_proves_missing_step_without_waiting_full_timeout() -> None:
    cache = ExactStepImageCache(max_size=4)
    cache.put(_frame(4, 1_000_000_000, value=11))
    cache.put(_frame(6, 3_000_000_000, value=99))

    started = time.monotonic()
    selected = cache.wait_select(
        source_seq=5,
        stamp_ns=2_000_000_000,
        timeout_sec=0.25,
        max_stamp_delta_ns=5_000_000,
        fallback_max_age_ns=2_000_000_000,
    )
    elapsed = time.monotonic() - started

    assert selected is not None
    assert selected.source == "nearest_image"
    assert selected.frame.source_seq == 4
    assert elapsed < 0.05


def test_consecutive_missing_steps_pay_only_one_wait_per_image_generation() -> None:
    cache = ExactStepImageCache(max_size=4)
    cache.put(_frame(1, 1_000_000_000, value=27))

    started = time.monotonic()
    selected = [
        cache.wait_select(
            source_seq=seq,
            stamp_ns=seq * 1_000_000_000,
            timeout_sec=0.04,
            max_stamp_delta_ns=5_000_000,
            fallback_max_age_ns=10_000_000_000,
        )
        for seq in range(2, 8)
    ]
    elapsed = time.monotonic() - started

    assert all(item is not None for item in selected)
    assert selected[0] is not None and selected[0].source == "nearest_image"
    assert all(item is not None and item.source == "latest_image" for item in selected[1:])
    assert all(item is not None and item.reused for item in selected[1:])
    # Six independent 40 ms waits would take about 240 ms.  One generation is
    # allowed one bounded wait; later queued markers reuse the latest real RGB.
    assert 0.035 <= elapsed < 0.12


def test_step_sync_cache_is_bounded() -> None:
    cache = ExactStepImageCache(max_size=2)
    cache.put(_frame(1, 10))
    cache.put(_frame(2, 20))
    cache.put(_frame(3, 30))

    assert cache.wait_pop(1, 10, timeout_sec=0.0) is None
    assert cache.wait_pop(2, 20, timeout_sec=0.0) is not None
    assert cache.wait_pop(3, 30, timeout_sec=0.0) is not None
