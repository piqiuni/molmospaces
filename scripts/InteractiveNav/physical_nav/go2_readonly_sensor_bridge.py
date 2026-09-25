#!/usr/bin/env python3
"""Go2-side D435i and state publisher.

This process intentionally imports only Unitree *subscriber* types.  It does
not construct SportClient, ObstaclesAvoidClient, a publisher, or any motion
API.  The only outbound channel is a WebSocket carrying sensor frames and
read-only telemetry to the policy machine.

The script is designed to run on the Go2 Jetson.  ``--dry-run`` produces a
synthetic RGB-D stream for protocol and web UI tests on a development machine.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import socket
import threading
import time
from typing import Any

from physical_protocol import (
    clock_probe_packet,
    encode_wire_packet,
    hello_packet,
    image_packet,
    sample_clock_pair,
    telemetry_packet,
)

_IMU_GRAVITY_TAU_S = -0.01 / math.log(0.98)  # Existing alpha at configured 100 Hz.
_IMU_REFERENCE_DURATION_S = 2.5
_IMU_REFERENCE_MAX_GAP_S = .25
_IMU_REFERENCE_MAX_ACCEL_ERROR_M_S2 = .75
_IMU_REFERENCE_MAX_GYRO_RAD_S = .08
_IMU_REFERENCE_MAX_TILT_SPAN_RAD = .035
_STANDARD_GRAVITY_M_S2 = 9.80665
_IMU_CONTINUITY_RECOVERY_S = .5
_VIDEO_PAIR_MAX_DELTA_MS = 50.
_DEVICE_CLOCK_OFFSET_WINDOW_S = 30.
_DEVICE_CLOCK_WARMUP_SAMPLES = 8
_DEVICE_CLOCK_WARMUP_SPAN_S = .5
_DEVICE_CLOCK_MAX_DRIFT_PPM = 250.
_DEVICE_CLOCK_ANCHOR_POSITIVE_GATE_S = .005
_DEVICE_EPOCH_MAX_SKEW_S = 300.
_DEVICE_EPOCH_FUTURE_TOLERANCE_S = .05
_EMPTY_STAGE_ITEM = object()


def _canonical_timestamp_domain(value: Any) -> str:
    """Return only timestamp domains whose librealsense semantics are known."""

    normalized = str(value or "").strip().casefold()
    # ``sensor_hardware_clock`` was emitted by an earlier bridge revision for
    # SENSOR_TIMESTAMP metadata.  Accept it as a compatibility alias, but do
    # not create a fourth clock domain: D435i IMU and depth use the same depth
    # hardware clock.
    if normalized in {
        "hardware_clock",
        "sensor_hardware_clock",
        "timestamp_domain.hardware_clock",
        "rs2_timestamp_domain.hardware_clock",
    }:
        return "hardware_clock"
    if normalized in {
        "system_time",
        "timestamp_domain.system_time",
        "rs2_timestamp_domain.system_time",
    }:
        return "system_time"
    if normalized in {
        "global_time",
        "timestamp_domain.global_time",
        "rs2_timestamp_domain.global_time",
    }:
        return "global_time"
    return ""


class _VideoPair:
    """Minimal frameset facade for callback backends yielding single frames."""

    def __init__(self, color: Any, depth: Any) -> None:
        self._color = color
        self._depth = depth

    def get_color_frame(self) -> Any:
        return self._color

    def get_depth_frame(self) -> Any:
        return self._depth


class _LatestOnlyStage:
    """Run one processor while retaining at most one newer pending item."""

    def __init__(self, processor: Any, sink: Any, *, name: str,
                 on_error: Any = None) -> None:
        self._processor = processor
        self._sink = sink
        self._on_error = on_error
        self._condition = threading.Condition()
        self._pending: Any = _EMPTY_STAGE_ITEM
        self._closed = False
        self._submitted = 0
        self._replaced = 0
        self._processed = 0
        self._errors = 0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, item: Any) -> bool:
        with self._condition:
            if self._closed:
                return False
            self._submitted += 1
            if self._pending is not _EMPTY_STAGE_ITEM:
                self._replaced += 1
            self._pending = item
            self._condition.notify()
            return True

    def stats(self) -> dict[str, int | bool]:
        with self._condition:
            return {
                "submitted": self._submitted,
                "replaced": self._replaced,
                "processed": self._processed,
                "errors": self._errors,
                "alive": self._thread.is_alive(),
            }

    def close(self, *, drain: bool = False, timeout: float = 1.0) -> bool:
        with self._condition:
            self._closed = True
            if not drain:
                self._pending = _EMPTY_STAGE_ITEM
            self._condition.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=max(float(timeout), 0.0))
        return not self._thread.is_alive()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or self._pending is not _EMPTY_STAGE_ITEM
                )
                if self._closed and self._pending is _EMPTY_STAGE_ITEM:
                    return
                item = self._pending
                self._pending = _EMPTY_STAGE_ITEM
            try:
                result = self._processor(item)
                self._sink(result)
                with self._condition:
                    self._processed += 1
            except Exception as exc:
                with self._condition:
                    self._errors += 1
                if self._on_error is not None:
                    try:
                        self._on_error(exc)
                    except Exception:
                        pass


class _FrameContinuityTracker:
    """Count per-stream SDK identities without assuming delivery is complete."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._streams: dict[str, dict[str, Any]] = {}
        self._epoch = 0
        self._reset_count = 0
        self._last_reset_reason = "initial"
        self._reset_candidates: dict[
            str, tuple[float, str, int, float, int | None]
        ] = {}
        self._domain_candidates: dict[
            str, tuple[float, str, int, float | None, int | None]
        ] = {}
        self._reset_candidate_window_s = .5

    @staticmethod
    def _new_stream_state() -> dict[str, Any]:
        return {
            "dequeued": 0, "unique": 0, "epoch_unique": 0,
            "duplicates": 0, "late": 0, "missing": 0, "gaps": 0,
            "timestamp_regressions": 0, "epoch_resets": 0,
            "last_frame_number": None, "first_timestamp_ms": None,
            "last_timestamp_ms": None, "timestamp_domain": "",
            "continuous_since_timestamp_ms": None, "epoch": 0,
        }

    def _reset_epoch_locked(
        self,
        reason: str,
        *,
        stream_domains: dict[str, str] | None = None,
    ) -> None:
        self._epoch += 1
        self._reset_count += 1
        self._last_reset_reason = str(reason)
        self._reset_candidates.clear()
        self._domain_candidates.clear()
        for state in self._streams.values():
            state["gaps"] += 1
            state["epoch_resets"] += 1
            state["epoch_unique"] = 0
            state["last_frame_number"] = None
            state["first_timestamp_ms"] = None
            state["last_timestamp_ms"] = None
            state["timestamp_domain"] = ""
            state["continuous_since_timestamp_ms"] = None
            state["epoch"] = self._epoch
        # A confirmed SDK domain transition applies to every corroborating
        # stream, including a stream whose last candidate was consumed before
        # the frame that completed confirmation.  Seed only the new domain,
        # not an identity watermark, so its next real sample is accepted while
        # a delayed frame from the retired domain remains quarantined.
        for stream, domain in (stream_domains or {}).items():
            state = self._streams.setdefault(stream, self._new_stream_state())
            state["timestamp_domain"] = domain
            state["epoch"] = self._epoch

    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "epoch": self._epoch,
                "reset_count": self._reset_count,
                "last_reset_reason": self._last_reset_reason,
                "reset_candidate_streams": sorted(self._reset_candidates),
                "domain_candidate_streams": sorted(self._domain_candidates),
            }

    def observe(
        self,
        stream: str,
        *,
        frame_number: int | None,
        timestamp_ms: float | None,
        timestamp_domain: str = "",
    ) -> bool:
        with self._lock:
            state = self._streams.setdefault(stream, self._new_stream_state())
            state["dequeued"] += 1
            previous_domain = state["timestamp_domain"]
            previous_number = state["last_frame_number"]
            previous_timestamp = state["last_timestamp_ms"]
            number_duplicate = bool(
                frame_number is not None
                and previous_number is not None
                and frame_number == previous_number
            )
            number_late = bool(
                frame_number is not None
                and previous_number is not None
                and frame_number < previous_number
            )
            # Identity and timestamp diagnostics are independent: a late
            # frame whose timestamp also regresses must be visible in both
            # counters even though neither watermark is advanced.
            if number_duplicate:
                state["duplicates"] += 1
            elif number_late:
                state["late"] += 1
            if (
                timestamp_domain
                and previous_domain
                and previous_domain != timestamp_domain
            ):
                if number_duplicate or number_late:
                    # A delayed frame from an old SDK domain must not reset a
                    # healthy current epoch merely because its domain differs.
                    return False
                # A single stream can briefly report a foreign SDK domain
                # under load.  Quarantine it instead of clearing the complete
                # RGB-D/IMU epoch.  A real librealsense domain transition is
                # accepted only after two progressing observations in at
                # least two streams agree on the new domain.
                now = time.monotonic()
                previous_candidate = self._domain_candidates.get(stream)
                candidate_count = 1
                if previous_candidate is not None:
                    (
                        _candidate_arrival,
                        candidate_domain,
                        prior_count,
                        candidate_timestamp,
                        candidate_number,
                    ) = previous_candidate
                    number_progressed = bool(
                        frame_number is None
                        or candidate_number is None
                        or frame_number > candidate_number
                    )
                    timestamp_progressed = bool(
                        timestamp_ms is None
                        or candidate_timestamp is None
                        or timestamp_ms > candidate_timestamp
                    )
                    if (
                        candidate_domain == timestamp_domain
                        and number_progressed
                        and timestamp_progressed
                    ):
                        candidate_count = prior_count + 1
                self._domain_candidates[stream] = (
                    now,
                    timestamp_domain,
                    candidate_count,
                    timestamp_ms,
                    frame_number,
                )
                cutoff = now - self._reset_candidate_window_s
                self._domain_candidates = {
                    name: candidate
                    for name, candidate in self._domain_candidates.items()
                    if candidate[0] >= cutoff
                }
                candidate_domains = {
                    candidate[1]
                    for candidate in self._domain_candidates.values()
                    if candidate[1]
                }
                confirmed_streams = sum(
                    candidate[2] >= 2
                    for candidate in self._domain_candidates.values()
                )
                if confirmed_streams >= 2 and len(candidate_domains) == 1:
                    switched_domains = {
                        name: candidate[1]
                        for name, candidate in self._domain_candidates.items()
                        if candidate[2] >= 2
                    }
                    self._reset_epoch_locked(
                        "timestamp_domain_change",
                        stream_domains=switched_domains,
                    )
                    state = self._streams[stream]
                    previous_number = None
                    previous_timestamp = None
                    number_duplicate = False
                    number_late = False
                else:
                    return False
            else:
                # Returning to the established domain disproves a quarantined
                # per-stream transition candidate.
                self._domain_candidates.pop(stream, None)
            timestamp_regressed = bool(
                timestamp_ms is not None
                and math.isfinite(timestamp_ms)
                and previous_timestamp is not None
                and timestamp_ms + 1e-6 < previous_timestamp
            )
            coherent_reset_candidate = bool(
                timestamp_regressed
                and previous_timestamp - float(timestamp_ms) >= 50.
            )
            if coherent_reset_candidate:
                state["timestamp_regressions"] += 1
                now = time.monotonic()
                previous_candidate = self._reset_candidates.get(stream)
                candidate_count = 1
                if previous_candidate is not None:
                    (
                        _candidate_arrival,
                        candidate_domain,
                        prior_count,
                        candidate_timestamp,
                        candidate_number,
                    ) = previous_candidate
                    number_progressed = bool(
                        frame_number is None
                        or candidate_number is None
                        or frame_number > candidate_number
                    )
                    if (
                        candidate_domain == timestamp_domain
                        and float(timestamp_ms) > candidate_timestamp
                        and number_progressed
                    ):
                        candidate_count = prior_count + 1
                self._reset_candidates[stream] = (
                    now,
                    timestamp_domain,
                    candidate_count,
                    float(timestamp_ms),
                    frame_number,
                )
                cutoff = now - self._reset_candidate_window_s
                self._reset_candidates = {
                    name: candidate
                    for name, candidate in self._reset_candidates.items()
                    if candidate[0] >= cutoff
                }
                candidate_domains = {
                    candidate[1]
                    for candidate in self._reset_candidates.values()
                    if candidate[1]
                }
                confirmed_streams = sum(
                    candidate[2] >= 2
                    for candidate in self._reset_candidates.values()
                )
                if (
                    confirmed_streams >= 2
                    and (len(candidate_domains) <= 1)
                ):
                    self._reset_epoch_locked("coherent_device_clock_reset")
                    state = self._streams[stream]
                    previous_number = None
                    previous_timestamp = None
                    timestamp_regressed = False
                else:
                    return False
            elif timestamp_regressed:
                # A lone small regression is an out-of-order SDK delivery,
                # not a new epoch.  Do not advance either identity watermark.
                state["timestamp_regressions"] += 1
                return False
            else:
                self._reset_candidates.pop(stream, None)

            if timestamp_domain:
                state["timestamp_domain"] = timestamp_domain

            advances = frame_number is None or previous_number is None or frame_number > previous_number
            if frame_number is not None and previous_number is not None:
                if frame_number == previous_number:
                    advances = False
                elif frame_number < previous_number:
                    advances = False
                elif frame_number > previous_number + 1:
                    state["missing"] += frame_number - previous_number - 1
                    state["gaps"] += 1
                    state["continuous_since_timestamp_ms"] = timestamp_ms
            if advances:
                state["unique"] += 1
                state["epoch_unique"] += 1
                if frame_number is not None:
                    state["last_frame_number"] = frame_number

            if timestamp_ms is not None and math.isfinite(timestamp_ms):
                if state["first_timestamp_ms"] is None:
                    state["first_timestamp_ms"] = timestamp_ms
                    state["continuous_since_timestamp_ms"] = timestamp_ms
                if advances:
                    state["last_timestamp_ms"] = timestamp_ms
            return advances

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            result: dict[str, dict[str, Any]] = {}
            for name, state in self._streams.items():
                item = dict(state)
                first = item.get("first_timestamp_ms")
                last = item.get("last_timestamp_ms")
                unique = int(item.get("epoch_unique", 0))
                span_s = (last - first) * .001 if first is not None and last is not None else 0.
                item["effective_hz"] = (unique - 1) / span_s if unique > 1 and span_s > 0. else 0.
                continuous_since = item.get("continuous_since_timestamp_ms")
                continuous_s = (
                    (last - continuous_since) * .001
                    if last is not None and continuous_since is not None
                    else 0.
                )
                item["continuous_s"] = max(0., continuous_s)
                item["healthy"] = continuous_s >= _IMU_CONTINUITY_RECOVERY_S
                result[name] = item
            return result


class _DeviceClockMapper:
    """Map a known RealSense timestamp domain into the Go2 host clocks.

    Hardware-clock timestamps have no epoch.  Their lower-envelope
    host-minus-device offset is therefore only an estimate: it exposes
    *variable* SDK/USB backlog, while the unknown minimum transport latency
    remains in the offset.  A connection-lifetime minimum is retained as the
    baseline, while the selected recent envelope can move only at a bounded
    oscillator-scale ppm rate so ordinary clock drift does not eventually
    make every capture stale.  The estimate is not selected as a packet stamp
    until a small warm-up window has been observed.  GLOBAL_TIME and
    SYSTEM_TIME already use the host epoch and are handled directly; the
    caller still records whether their semantic is exposure, readout, or
    arrival.
    """

    def __init__(
        self,
        window_s: float = _DEVICE_CLOCK_OFFSET_WINDOW_S,
        *,
        warmup_samples: int = _DEVICE_CLOCK_WARMUP_SAMPLES,
        warmup_span_s: float = _DEVICE_CLOCK_WARMUP_SPAN_S,
        max_drift_ppm: float = _DEVICE_CLOCK_MAX_DRIFT_PPM,
        anchor_positive_gate_s: float = _DEVICE_CLOCK_ANCHOR_POSITIVE_GATE_S,
    ) -> None:
        self._window_s = max(1., float(window_s))
        self._warmup_samples = max(2, int(warmup_samples))
        self._warmup_span_s = max(0., float(warmup_span_s))
        self._max_adjust_rate_s_per_s = max(0., float(max_drift_ppm)) * 1e-6
        self._anchor_positive_gate_s = max(
            0., float(anchor_positive_gate_s)
        )
        self._lock = threading.Lock()
        self._offset_samples: dict[str, deque[tuple[float, float]]] = {}
        self._minimum_offsets: dict[str, float] = {}
        self._minimum_offset_monotonic: dict[str, float] = {}
        self._selected_offsets: dict[str, float] = {}
        self._last_adjust_monotonic: dict[str, float] = {}
        self._ready_domains: set[str] = set()
        self._last_anchor_monotonic: dict[str, float] = {}
        self._holdover_since_monotonic: dict[str, float] = {}
        self._rejected_anchor_counts: dict[str, int] = {}
        self._lower_reanchor_counts: dict[str, int] = {}
        self._clock_anchors: dict[
            str, deque[tuple[int, float, float]]
        ] = {}
        self._estimated_rates_s_per_s: dict[str, float] = {}
        self._anchor_bucket_s = 1.
        self._rate_min_span_s = 60.
        self._last_stream_timestamp_ms: dict[tuple[str, str], float] = {}
        self._reset_count = 0
        self._last_reset_reason = "initial"
        self._latest: dict[str, Any] = {}

    def reset(self, reason: str = "external_epoch_reset") -> None:
        """Forget every device-domain anchor after an ingress epoch change."""

        with self._lock:
            self._offset_samples.clear()
            self._minimum_offsets.clear()
            self._minimum_offset_monotonic.clear()
            self._selected_offsets.clear()
            self._last_adjust_monotonic.clear()
            self._ready_domains.clear()
            self._last_anchor_monotonic.clear()
            self._holdover_since_monotonic.clear()
            self._rejected_anchor_counts.clear()
            self._lower_reanchor_counts.clear()
            self._clock_anchors.clear()
            self._estimated_rates_s_per_s.clear()
            self._last_stream_timestamp_ms.clear()
            self._reset_count += 1
            self._last_reset_reason = str(reason)
            self._latest = {
                "valid": False,
                "source_epoch_valid": False,
                "failure_reason": str(reason),
                "clock_mapping_method": "host_read_fallback",
                "clock_mapping_quality": "fallback",
            }

    def _record_clock_anchor_locked(
        self, domain: str, monotonic: float, offset: float,
    ) -> None:
        """Update a sparse low-delay envelope and its robust clock rate."""

        anchors = self._clock_anchors.setdefault(domain, deque(maxlen=128))
        bucket = int(monotonic // self._anchor_bucket_s)
        if anchors and anchors[-1][0] == bucket:
            previous = anchors[-1]
            if offset < previous[2]:
                anchors[-1] = (bucket, monotonic, offset)
            return
        else:
            anchors.append((bucket, monotonic, offset))
        if len(anchors) < 2:
            return
        _latest_bucket, latest_time, latest_offset = anchors[-1]
        slopes = [
            (latest_offset - earlier_offset) / (latest_time - earlier_time)
            for _bucket, earlier_time, earlier_offset in anchors
            if latest_time - earlier_time >= self._rate_min_span_s
        ]
        if not slopes:
            return
        slopes.sort()
        midpoint = len(slopes) // 2
        median_slope = (
            slopes[midpoint]
            if len(slopes) % 2
            else .5 * (slopes[midpoint - 1] + slopes[midpoint])
        )
        limit = self._max_adjust_rate_s_per_s
        self._estimated_rates_s_per_s[domain] = max(
            -limit, min(limit, median_slope)
        )

    def map(
        self,
        timestamp_ms: Any,
        timestamp_domain: Any,
        *,
        stream: str,
        observed_monotonic: float | None = None,
        observed_wall: float | None = None,
    ) -> dict[str, Any]:
        try:
            if observed_monotonic is None or observed_wall is None:
                sampled_wall, sampled_monotonic, _pair_span_s = sample_clock_pair()
                if observed_monotonic is None:
                    observed_monotonic = sampled_monotonic
                if observed_wall is None:
                    observed_wall = sampled_wall
            observed_monotonic = float(observed_monotonic)
            observed_wall = float(observed_wall)
        except (TypeError, ValueError, OverflowError):
            observed_wall, observed_monotonic, _pair_span_s = sample_clock_pair()
        raw_domain = str(timestamp_domain or "").strip()
        domain = _canonical_timestamp_domain(raw_domain)
        fallback = {
            "valid": False,
            "source_epoch_valid": False,
            "stream": str(stream),
            "timestamp_domain": domain,
            "timestamp_domain_raw": raw_domain,
            "device_timestamp_ms": None,
            "capture_monotonic": observed_monotonic,
            "capture_wall": observed_wall,
            "dequeue_age_s": 0.,
            "dequeue_monotonic": observed_monotonic,
            "dequeue_wall": observed_wall,
            "clock_mapping_method": "host_read_fallback",
            "clock_mapping_quality": "fallback",
            "warmup_complete": False,
            "warmup_samples": 0,
            "warmup_span_s": 0.,
            "minimum_transport_delay_known": False,
        }
        try:
            stamp_ms = float(timestamp_ms)
        except (TypeError, ValueError, OverflowError):
            fallback["failure_reason"] = "invalid_timestamp"
            with self._lock:
                self._latest = dict(fallback)
            return fallback
        if (
            not domain
            or not math.isfinite(stamp_ms)
            or stamp_ms < 0.
            or not math.isfinite(observed_monotonic)
            or not math.isfinite(observed_wall)
        ):
            fallback["failure_reason"] = (
                "unsupported_timestamp_domain" if not domain else "invalid_timestamp"
            )
            with self._lock:
                self._latest = dict(fallback)
            return fallback

        device_s = stamp_ms * .001
        stream_key = (domain, str(stream))
        with self._lock:
            previous = self._last_stream_timestamp_ms.get(stream_key)
            if previous is not None and stamp_ms + 1e-6 < previous:
                # A device/stream restart invalidates the old clock epoch.
                self._offset_samples.pop(domain, None)
                self._minimum_offsets.pop(domain, None)
                self._minimum_offset_monotonic.pop(domain, None)
                self._selected_offsets.pop(domain, None)
                self._last_adjust_monotonic.pop(domain, None)
                self._ready_domains.discard(domain)
                self._last_anchor_monotonic.pop(domain, None)
                self._holdover_since_monotonic.pop(domain, None)
                self._rejected_anchor_counts.pop(domain, None)
                self._clock_anchors.pop(domain, None)
                self._estimated_rates_s_per_s.pop(domain, None)
                for old_key in tuple(self._last_stream_timestamp_ms):
                    if old_key[0] == domain:
                        self._last_stream_timestamp_ms.pop(old_key, None)
                self._reset_count += 1
                self._last_reset_reason = "stream_timestamp_regression"
            self._last_stream_timestamp_ms[stream_key] = stamp_ms

            if domain in {"global_time", "system_time"}:
                epoch_age_s = observed_wall - device_s
                if (
                    epoch_age_s < -_DEVICE_EPOCH_FUTURE_TOLERANCE_S
                    or epoch_age_s > _DEVICE_EPOCH_MAX_SKEW_S
                ):
                    fallback.update(
                        device_timestamp_ms=stamp_ms,
                        failure_reason="epoch_timestamp_out_of_range",
                        epoch_age_s=epoch_age_s,
                    )
                    self._latest = dict(fallback)
                    return fallback
                # A tiny future value can occur from clock interpolation.  It
                # cannot be a causal capture, so clamp only within the explicit
                # tolerance and expose the amount for field diagnostics.
                dequeue_age_s = max(0., epoch_age_s)
                capture_wall = min(observed_wall, device_s)
                result = {
                    "valid": True,
                    "source_epoch_valid": True,
                    "stream": str(stream),
                    "timestamp_domain": domain,
                    "timestamp_domain_raw": raw_domain,
                    "device_timestamp_ms": stamp_ms,
                    "capture_monotonic": observed_monotonic - dequeue_age_s,
                    "capture_wall": capture_wall,
                    "dequeue_age_s": dequeue_age_s,
                    "dequeue_monotonic": observed_monotonic,
                    "dequeue_wall": observed_wall,
                    "epoch_future_s": max(0., -epoch_age_s),
                    "clock_mapping_method": f"sdk_{domain}",
                    "clock_mapping_quality": "sdk_host_epoch",
                    "warmup_complete": True,
                    "warmup_samples": 1,
                    "warmup_span_s": 0.,
                    "minimum_transport_delay_known": domain == "global_time",
                }
                self._latest = dict(result)
                return result

            samples = self._offset_samples.setdefault(domain, deque())
            samples.append((observed_monotonic, observed_monotonic - device_s))
            cutoff = observed_monotonic - self._window_s
            while samples and samples[0][0] < cutoff:
                samples.popleft()
            observed_offset_s = observed_monotonic - device_s
            if (
                domain not in self._minimum_offsets
                or observed_offset_s < self._minimum_offsets[domain]
            ):
                self._minimum_offsets[domain] = observed_offset_s
                self._minimum_offset_monotonic[domain] = observed_monotonic
            minimum_offset_s = self._minimum_offsets[domain]
            recent_offset_s = min(offset for _, offset in samples)
            warmup_span_s = max(0., samples[-1][0] - samples[0][0])
            warmup_complete = (
                len(samples) >= self._warmup_samples
                and warmup_span_s >= self._warmup_span_s
            )
            if domain not in self._ready_domains:
                # Before the first valid mapping, freely select the best
                # warm-up anchor.  No packet timestamp consumes it yet.
                self._selected_offsets[domain] = minimum_offset_s
                if warmup_complete:
                    self._ready_domains.add(domain)
                    self._last_anchor_monotonic[domain] = observed_monotonic
                    self._record_clock_anchor_locked(
                        domain, observed_monotonic, recent_offset_s,
                    )
            else:
                previous_selected = self._selected_offsets[domain]
                last_adjust = self._last_adjust_monotonic.get(
                    domain, observed_monotonic,
                )
                allowance = max(
                    0., observed_monotonic - last_adjust,
                ) * self._max_adjust_rate_s_per_s
                difference = recent_offset_s - previous_selected
                if difference < 0.:
                    # A new lower dequeue envelope is evidence that the
                    # startup baseline contained avoidable SDK/USB backlog.
                    # Applying a ppm rate here can preserve a 200 ms startup
                    # error for many minutes.  Re-anchor downward immediately;
                    # upward changes remain rate-limited because they are
                    # indistinguishable from additional queueing.
                    self._selected_offsets[domain] = recent_offset_s
                    self._lower_reanchor_counts[domain] = (
                        self._lower_reanchor_counts.get(domain, 0) + 1
                    )
                    self._last_anchor_monotonic[domain] = observed_monotonic
                    self._holdover_since_monotonic.pop(domain, None)
                    self._record_clock_anchor_locked(
                        domain, observed_monotonic, recent_offset_s,
                    )
                elif difference <= self._anchor_positive_gate_s:
                    self._selected_offsets[domain] = previous_selected + max(
                        -allowance, min(allowance, difference)
                    )
                    self._last_anchor_monotonic[domain] = observed_monotonic
                    self._holdover_since_monotonic.pop(domain, None)
                    self._record_clock_anchor_locked(
                        domain, observed_monotonic, recent_offset_s,
                    )
                else:
                    self._rejected_anchor_counts[domain] = (
                        self._rejected_anchor_counts.get(domain, 0) + 1
                    )
                    self._holdover_since_monotonic.setdefault(
                        domain, observed_monotonic,
                    )
                    estimated_rate = self._estimated_rates_s_per_s.get(
                        domain, 0.,
                    )
                    self._selected_offsets[domain] = previous_selected + max(
                        -allowance,
                        min(allowance, estimated_rate * max(
                            0., observed_monotonic - last_adjust,
                        )),
                    )
            self._last_adjust_monotonic[domain] = observed_monotonic
            offset_s = self._selected_offsets[domain]
            baseline_age_s = max(
                0.,
                observed_monotonic
                - self._minimum_offset_monotonic.get(domain, observed_monotonic),
            )
            selected_from_baseline_s = offset_s - minimum_offset_s
            last_anchor_age_s = (
                max(
                    0., observed_monotonic
                    - self._last_anchor_monotonic[domain],
                )
                if domain in self._last_anchor_monotonic else None
            )
            holdover_age_s = (
                max(
                    0., observed_monotonic
                    - self._holdover_since_monotonic[domain],
                )
                if domain in self._holdover_since_monotonic else 0.
            )
            anchors = self._clock_anchors.get(domain, ())
            anchor_span_s = (
                max(0., anchors[-1][1] - anchors[0][1])
                if len(anchors) >= 2 else 0.
            )
            capture_monotonic = min(observed_monotonic, device_s + offset_s)
            dequeue_age_s = max(0., observed_monotonic - capture_monotonic)
            result = {
                # ``valid`` is retained for protocol-v1 consumers and means
                # only "safe to use as a source epoch", not "absolute exposure
                # time is known".  The latter is attached by the metadata
                # extractor below.
                "valid": warmup_complete,
                "source_epoch_valid": warmup_complete,
                "stream": str(stream),
                "timestamp_domain": domain,
                "timestamp_domain_raw": raw_domain,
                "device_timestamp_ms": stamp_ms,
                "capture_monotonic": capture_monotonic,
                "capture_wall": observed_wall - dequeue_age_s,
                "dequeue_age_s": dequeue_age_s,
                "dequeue_monotonic": observed_monotonic,
                "dequeue_wall": observed_wall,
                "offset_s": offset_s,
                "minimum_offset_s": minimum_offset_s,
                "recent_offset_s": recent_offset_s,
                "minimum_offset_baseline_age_s": baseline_age_s,
                "selected_minus_minimum_s": selected_from_baseline_s,
                "selected_baseline_drift_ppm": (
                    selected_from_baseline_s / baseline_age_s * 1e6
                    if baseline_age_s > 0. else None
                ),
                "max_adjust_rate_ppm": self._max_adjust_rate_s_per_s * 1e6,
                "anchor_positive_gate_s": self._anchor_positive_gate_s,
                "last_anchor_age_s": last_anchor_age_s,
                "holdover_age_s": holdover_age_s,
                "rejected_anchor_count": self._rejected_anchor_counts.get(
                    domain, 0,
                ),
                "lower_reanchor_count": self._lower_reanchor_counts.get(
                    domain, 0,
                ),
                "observed_residual_s": observed_offset_s - offset_s,
                "recent_envelope_residual_s": recent_offset_s - offset_s,
                "estimated_queue_excess_s": max(
                    0., observed_offset_s - offset_s,
                ),
                "estimated_rate_ppm": self._estimated_rates_s_per_s.get(
                    domain, 0.,
                ) * 1e6,
                "trusted_anchor_count": len(anchors),
                "anchor_span_s": anchor_span_s,
                "clock_mapping_method": "recent_min_offset",
                "clock_mapping_quality": "estimated",
                "warmup_complete": warmup_complete,
                "warmup_samples": len(samples),
                "warmup_span_s": warmup_span_s,
                "minimum_transport_delay_known": False,
            }
            self._latest = dict(result)
            return result

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "window_s": self._window_s,
                "warmup_samples_required": self._warmup_samples,
                "warmup_span_s_required": self._warmup_span_s,
                "reset_count": self._reset_count,
                "last_reset_reason": self._last_reset_reason,
                "sample_counts": {
                    domain: len(samples)
                    for domain, samples in self._offset_samples.items()
                },
                "minimum_offsets_s": dict(self._minimum_offsets),
                "selected_offsets_s": dict(self._selected_offsets),
                "minimum_offset_baseline_monotonic": dict(
                    self._minimum_offset_monotonic
                ),
                "max_adjust_rate_ppm": self._max_adjust_rate_s_per_s * 1e6,
                "anchor_positive_gate_s": self._anchor_positive_gate_s,
                "rejected_anchor_counts": dict(self._rejected_anchor_counts),
                "lower_reanchor_counts": dict(self._lower_reanchor_counts),
                "estimated_rates_ppm": {
                    domain: rate * 1e6
                    for domain, rate in self._estimated_rates_s_per_s.items()
                },
                "latest": dict(self._latest),
            }


class ReadOnlyState:
    def __init__(self, history_size: int = 512) -> None:
        self.lock = threading.Lock()
        self.telemetry: dict[str, Any] = {}
        self.seq = 0
        self._history: deque[tuple[float, dict[str, Any]]] = deque(
            maxlen=max(2, int(history_size))
        )

    def update(self, value: dict[str, Any]) -> None:
        with self.lock:
            self.telemetry = dict(value)
            try:
                stamp = float(value.get("received_at"))
            except (AttributeError, TypeError, ValueError, OverflowError):
                return
            if not math.isfinite(stamp) or stamp <= 0.:
                return
            snapshot = dict(value)
            if self._history and abs(self._history[-1][0] - stamp) <= 1e-9:
                self._history[-1] = (stamp, snapshot)
            elif not self._history or stamp > self._history[-1][0]:
                self._history.append((stamp, snapshot))
            else:
                # A wall-clock step starts a new causal history generation.
                self._history.clear()
                self._history.append((stamp, snapshot))

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.telemetry)

    def snapshot_at(
        self, stamp: float, *, max_delta_s: float = .15,
    ) -> tuple[dict[str, Any], float | None]:
        try:
            target = float(stamp)
            limit = max(0., float(max_delta_s))
        except (TypeError, ValueError, OverflowError):
            return {}, None
        if not math.isfinite(target) or target <= 0.:
            return {}, None
        with self.lock:
            if not self._history:
                return {}, None
            selected_stamp, selected = min(
                self._history,
                # On an exact tie, an older sample cannot contain motion known
                # to occur after the image exposure.
                key=lambda item: (abs(item[0] - target), item[0] > target),
            )
            delta = selected_stamp - target
            return (dict(selected), delta) if abs(delta) <= limit else ({}, delta)


def _field_sequence(message: Any, name: str, *, cast: Any = float) -> list[Any]:
    """Read optional SDK sequences without dropping the whole pose packet."""
    try:
        values = getattr(message, name, None)
        return [] if values is None else [cast(value) for value in values]
    except (TypeError, ValueError):
        return []


def _unitree_state_reader(state: ReadOnlyState, interface: str) -> None:
    """Subscribe to state topics without loading any command client."""
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_
    except ImportError as exc:
        print(f"warning: Unitree SDK unavailable; telemetry disabled: {exc}", flush=True)
        return

    try:
        ChannelFactoryInitialize(0, interface)
        pose_lock = threading.Lock()
        latest: dict[str, Any] = {}

        def on_sport(msg: Any) -> None:
            try:
                imu = getattr(msg, "imu_state")
                q = _field_sequence(imu, "quaternion")
                if len(q) != 4:
                    raise ValueError("IMU quaternion is missing or does not contain four values")
                w, x, y, z = q
                yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                position = (_field_sequence(msg, "position") + [0.0, 0.0, 0.0])[:3]
                velocity = (_field_sequence(msg, "velocity") + [0.0, 0.0, 0.0])[:3]
                value = {
                    "received_at": time.time(),
                    "error_code": int(getattr(msg, "error_code", 0)),
                    "position": position,
                    "velocity": velocity,
                    "yaw_speed": float(getattr(msg, "yaw_speed", 0.0)),
                    "mode": int(getattr(msg, "mode", 0)),
                    "progress": float(getattr(msg, "progress", 0.0)),
                    "gait_type": int(getattr(msg, "gait_type", 0)),
                    "foot_raise_height": float(getattr(msg, "foot_raise_height", 0.0)),
                    "body_height": float(getattr(msg, "body_height", 0.0)),
                    "range_obstacle": _field_sequence(msg, "range_obstacle"),
                    "foot_force": _field_sequence(msg, "foot_force", cast=int),
                    "imu": {
                        "quaternion": q,
                        "gyroscope": _field_sequence(imu, "gyroscope"),
                        "accelerometer": _field_sequence(imu, "accelerometer"),
                        "rpy": _field_sequence(imu, "rpy"),
                        "temperature": int(getattr(imu, "temperature", 0)),
                    },
                    "yaw": yaw,
                }
                with pose_lock:
                    latest.update(value)
                    state.update(dict(latest))
            except Exception as exc:  # SDK message layouts differ by release.
                print(f"warning: could not decode rt/sportmodestate: {exc}", flush=True)

        def on_low(msg: Any) -> None:
            try:
                bms = msg.bms_state
                value = {
                    "received_at": time.time(),
                    "soc": int(getattr(bms, "soc", 0)),
                    "bms_current": int(getattr(bms, "current", 0)),
                    "cycle": int(getattr(bms, "cycle", 0)),
                    "cell_vol": [int(v) for v in getattr(bms, "cell_vol", [])],
                    "bq_ntc": [int(v) for v in getattr(bms, "bq_ntc", [])],
                    "mcu_ntc": [int(v) for v in getattr(bms, "mcu_ntc", [])],
                    "voltage": float(getattr(msg, "power_v", 0.0)),
                    "current": float(getattr(msg, "power_a", 0.0)),
                    "power": float(getattr(msg, "power_v", 0.0)) * float(getattr(msg, "power_a", 0.0)),
                    "temperature_ntc1": int(getattr(msg, "temperature_ntc1", 0)),
                    "temperature_ntc2": int(getattr(msg, "temperature_ntc2", 0)),
                    "fan_frequency": [int(v) for v in getattr(msg, "fan_frequency", [])],
                }
                with pose_lock:
                    latest["battery"] = value
                    state.update(dict(latest))
            except Exception as exc:
                print(f"warning: could not decode rt/lowstate: {exc}", flush=True)

        # Subscribers are the only Unitree objects constructed in this file.
        subscribers = [ChannelSubscriber("rt/sportmodestate", SportModeState_), ChannelSubscriber("rt/lowstate", LowState_)]
        subscribers[0].Init(on_sport, 10); subscribers[1].Init(on_low, 10)
        print("read-only Unitree state subscribers started", flush=True)
        while True:
            time.sleep(1.0)
    except Exception as exc:
        print(f"warning: state subscribers stopped: {exc}", flush=True)


def _finite_motion_vector(value: Any) -> tuple[float, float, float] | None:
    try:
        if value is None or len(value) < 3:
            return None
        vector = tuple(float(item) for item in value[:3])
        return vector if all(math.isfinite(item) for item in vector) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _new_motion_timestamp(value: Any, previous: float | None) -> float | None:
    try:
        stamp = float(value)
        if math.isfinite(stamp) and stamp >= 0. and (previous is None or stamp > previous):
            return stamp
    except (TypeError, ValueError, OverflowError):
        pass
    return None


class D435iSource:
    def __init__(self, width: int, height: int, fps: int, enable_motion: bool = False,
                 color_width: int | None = None, color_height: int | None = None,
                 color_fps: int | None = None, align_to: str = "depth",
                 imu_gravity_tau_s: float = _IMU_GRAVITY_TAU_S,
                 frame_queue_capacity: int = 128) -> None:
        self._imu_gravity_tau_s = float(imu_gravity_tau_s)
        if not math.isfinite(self._imu_gravity_tau_s) or self._imu_gravity_tau_s <= 0.:
            raise ValueError("imu_gravity_tau_s must be finite and positive")
        if align_to not in ("none", "color", "depth"):
            raise ValueError("align_to must be none, color, or depth")
        self._frame_queue_capacity = int(frame_queue_capacity)
        if self._frame_queue_capacity < 8:
            raise ValueError("frame_queue_capacity must be at least 8")
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self._closed_event = threading.Event()
        self._close_lock = threading.Lock()
        self._pipeline_stopped = False
        depth_width, depth_height, depth_fps = width, height, fps
        color_width, color_height, color_fps = color_width or width, color_height or height, color_fps or fps
        self._video_pair_max_delta_ms = min(
            _VIDEO_PAIR_MAX_DELTA_MS,
            500.0 / max(min(int(depth_fps), int(color_fps)), 1),
        )
        self.config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, depth_fps)
        self.config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, color_fps)
        self.motion_enabled = False
        try:
            # D435i exposes gyro/accelerometer streams, but not a standalone
            # visual-odometry pose stream.  Keep these measurements alongside
            # the Go2 body quaternion so the policy host can use dynamic tilt.
            if enable_motion:
                self.config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
                self.config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
                self.motion_enabled = True
        except Exception as exc:
            print(f"D435i motion streams unavailable: {exc}", flush=True)
        if enable_motion:
            # After a previous bridge restart the D435i motion endpoint can
            # remain in a stale state on the Jetson USB2 controller.  Reset
            # once before resolving the combined RGB-D+IMU requests.
            try:
                devices = self.rs.context().query_devices()
                if len(devices):
                    devices[0].hardware_reset()
                    time.sleep(3.0)
            except Exception as exc:
                print(f"D435i hardware reset skipped: {exc}", flush=True)
        pipeline_started = False
        try:
            # The C++ queue buffers motion samples while Python briefly copies
            # a video frame. No Python runs on librealsense's callback thread.
            # Pipeline synchronizes video into native composite framesets while
            # motion frames bypass that synchronizer. Sending the callback into
            # a C++ queue preserves every dequeued IMU sample without running
            # Python on librealsense's callback thread, for every align mode.
            self._frame_queue = rs.frame_queue(self._frame_queue_capacity, False)
            self.profile = self.pipeline.start(self.config, self._frame_queue)
            pipeline_started = True
            self.align_to = align_to
            # ``align(depth)`` leaves the native depth frame unchanged and only
            # warps the colour frame onto the depth grid.  This bridge deliberately
            # transmits the native 1280x720 RGB (to avoid the aligned canvas'
            # black border) and performs RGB-mask -> depth projection locally.
            # Calling RealSense's align processor and then discarding its warped
            # colour image was nevertheless costing a full extra per-frame image
            # registration pass. Keep the object for API/back-end compatibility,
            # but execute it only for ``align(color)``, where the depth image
            # itself must be resampled.
            self.align = None if align_to == "none" else rs.align(getattr(rs.stream, align_to))
            depth_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
            color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.depth_scale = float(self.profile.get_device().first_depth_sensor().get_depth_scale())
            c, d = color_profile.get_intrinsics(), depth_profile.get_intrinsics()
            self.intrinsics = {
                "fx": c.fx,
                "fy": c.fy,
                "cx": c.ppx,
                "cy": c.ppy,
                "width": c.width,
                "height": c.height,
                "distortion_model": str(getattr(c, "model", "")),
                "distortion": [float(v) for v in getattr(c, "coeffs", [])],
            }
            self.depth_intrinsics = {
                "fx": d.fx, "fy": d.fy, "cx": d.ppx, "cy": d.ppy,
                "width": d.width, "height": d.height,
                "distortion_model": str(getattr(d, "model", "")),
                "distortion": [float(v) for v in getattr(d, "coeffs", [])],
            }
            if align_to == "color":
                self.depth_intrinsics = dict(self.intrinsics)
            ext = depth_profile.get_extrinsics_to(color_profile)
            self.depth_to_color_extrinsics = {
                "rotation": [float(v) for v in ext.rotation],
                "translation": [float(v) for v in ext.translation],
            }
            target_stream = rs.stream.depth if align_to == "depth" else rs.stream.color
            self.camera_frame = f"{self.profile.get_stream(target_stream).stream_name()}_frame"
            self.latest_motion: dict[str, Any] = {}
            self._imu_rpy = [0.0, 0.0, 0.0]
            self._imu_reference_rpy = None
            self._imu_calibration_samples = 0
            self._imu_calibration_sum = [0.0, 0.0, 0.0]
            self._imu_calibration_window_start_ts = None
            self._imu_calibration_window_last_ts = None
            self._imu_calibration_anchor = None
            self._imu_calibration_gravity_sum = [0.0, 0.0, 0.0]
            self._imu_calibration_window_samples = 0
            self._imu_calibration_reset_reason = "waiting_for_stable_gravity"
            self._imu_last_gyro_ts = None
            self._imu_last_accel_ts = None
            self._imu_accel_updates = 0
            self._imu_gyro_updates = 0
            self._imu_component_received_at = {"accel": None, "gyro": None}
            self._motion_lock = threading.Lock()
            self._imu_history: deque[tuple[float, str, dict[str, Any]]] = deque(maxlen=512)
            self._capture_motion: dict[str, Any] = {}
            self._continuity = _FrameContinuityTracker()
            self._continuity_epoch_seen = self._continuity.epoch()
            self._device_clock_mapper = _DeviceClockMapper()
            self._capture_time: dict[str, Any] = {}
            self._imu_component_sample_at = {"accel": None, "gyro": None}
            self._imu_component_sample_valid = {"accel": False, "gyro": False}
            self._video_frames_replaced = 0
            self._queue_dequeued = 0
            self._queue_high_water = 0
            self._invalid_ingress_frames = 0
            self._pending_video_frames: dict[str, tuple[Any, float | None, str, int] | None] = {
                "color": None,
                "depth": None,
            }
            self._video_arrival_order = 0
            self._unpaired_video_frames = 0
            self._video_pair_domain_mismatches = 0
            self._video_pair_skew_exceeded = 0
            self._unsupported_align_color_video_pairs = 0
            print(
                f"D435i started color {c.width}x{c.height}@{color_fps} + "
                f"depth {d.width}x{d.height}@{depth_fps}, align={align_to}, "
                f"ingress={'frame_queue' if self._frame_queue is not None else 'synchronized'}, "
                f"depth_scale={self.depth_scale}",
                flush=True,
            )
        except BaseException:
            if pipeline_started:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
            raise

    def _ensure_ingress_state(self) -> None:
        if not hasattr(self, "_closed_event"):
            self._closed_event = threading.Event()
        if not hasattr(self, "_close_lock"):
            self._close_lock = threading.Lock()
        if not hasattr(self, "_pipeline_stopped"):
            self._pipeline_stopped = False
        if not hasattr(self, "_motion_lock"):
            self._motion_lock = threading.Lock()
        if not hasattr(self, "_imu_history"):
            self._imu_history = deque(maxlen=512)
        if not hasattr(self, "_capture_motion"):
            self._capture_motion = {}
        if not hasattr(self, "_continuity"):
            self._continuity = _FrameContinuityTracker()
        if not hasattr(self, "_continuity_epoch_seen"):
            self._continuity_epoch_seen = self._continuity.epoch()
        if not hasattr(self, "_device_clock_mapper"):
            self._device_clock_mapper = _DeviceClockMapper()
        if not hasattr(self, "_capture_time"):
            self._capture_time = {}
        if not hasattr(self, "_imu_component_sample_at"):
            self._imu_component_sample_at = {"accel": None, "gyro": None}
        if not hasattr(self, "_imu_component_sample_valid"):
            self._imu_component_sample_valid = {"accel": False, "gyro": False}
        if not hasattr(self, "_video_frames_replaced"):
            self._video_frames_replaced = 0
        if not hasattr(self, "_queue_dequeued"):
            self._queue_dequeued = 0
        if not hasattr(self, "_queue_high_water"):
            self._queue_high_water = 0
        if not hasattr(self, "_invalid_ingress_frames"):
            self._invalid_ingress_frames = 0
        if not hasattr(self, "_pending_video_frames"):
            self._pending_video_frames = {"color": None, "depth": None}
        if not hasattr(self, "_video_arrival_order"):
            self._video_arrival_order = 0
        if not hasattr(self, "_unpaired_video_frames"):
            self._unpaired_video_frames = 0
        if not hasattr(self, "_video_pair_domain_mismatches"):
            self._video_pair_domain_mismatches = 0
        if not hasattr(self, "_video_pair_skew_exceeded"):
            self._video_pair_skew_exceeded = 0
        if not hasattr(self, "_video_pair_max_delta_ms"):
            self._video_pair_max_delta_ms = _VIDEO_PAIR_MAX_DELTA_MS
        if not hasattr(self, "_unsupported_align_color_video_pairs"):
            self._unsupported_align_color_video_pairs = 0

    def _consume_continuity_epoch(self) -> None:
        """Reset timestamp-dependent state once per confirmed SDK epoch."""

        epoch = self._continuity.epoch()
        if epoch == self._continuity_epoch_seen:
            return
        self._continuity_epoch_seen = epoch
        self._device_clock_mapper.reset("ingress_continuity_epoch")
        with self._motion_lock:
            self.latest_motion = {}
            self._imu_history.clear()
            self._capture_motion = {}
            self._capture_time = {}
        self._imu_last_gyro_ts = None
        self._imu_last_accel_ts = None
        self._imu_component_received_at = {"accel": None, "gyro": None}
        self._imu_component_sample_at = {"accel": None, "gyro": None}
        self._imu_component_sample_valid = {"accel": False, "gyro": False}
        self._reset_imu_reference_candidate("device_clock_epoch_reset")
        self._pending_video_frames = {"color": None, "depth": None}

    @staticmethod
    def _timestamp_domain_name(value: Any) -> str:
        return _canonical_timestamp_domain(value)

    @staticmethod
    def _frame_identity(frame: Any) -> tuple[int | None, float | None, str]:
        try:
            number = int(frame.get_frame_number())
        except (AttributeError, TypeError, ValueError, OverflowError):
            number = None
        try:
            timestamp_ms = float(frame.get_timestamp())
            if not math.isfinite(timestamp_ms):
                timestamp_ms = None
        except (AttributeError, TypeError, ValueError, OverflowError):
            timestamp_ms = None
        try:
            timestamp_domain = D435iSource._timestamp_domain_name(
                frame.get_frame_timestamp_domain()
            )
        except Exception:
            timestamp_domain = ""
        return number, timestamp_ms, timestamp_domain

    def _capture_timestamp_identity(
        self, frame: Any,
    ) -> tuple[float | None, str, str, dict[str, Any]]:
        """Return the best available video acquisition timestamp in ms.

        SENSOR_TIMESTAMP is the device-reported exposure midpoint and is
        specified in microseconds.  When the SDK timestamp is GLOBAL_TIME,
        FRAME_TIMESTAMP from the same raw depth frame supplies the device-clock
        delta needed to move that global readout stamp to the exposure midpoint.
        Kernels/firmware without metadata support fall back to get_timestamp(),
        whose exact meaning is preserved rather than overstated as exposure.
        """
        _, sdk_timestamp_ms, sdk_domain = self._frame_identity(frame)
        metadata_values: dict[str, Any] = {
            "sdk_frame_timestamp_ms": sdk_timestamp_ms,
            "sdk_frame_timestamp_domain": sdk_domain,
        }
        try:
            exposure_metadata = self.rs.frame_metadata_value.actual_exposure
            if frame.supports_frame_metadata(exposure_metadata):
                actual_exposure_us = float(frame.get_frame_metadata(exposure_metadata))
                if math.isfinite(actual_exposure_us) and actual_exposure_us >= 0.:
                    metadata_values["actual_exposure_us"] = actual_exposure_us
        except Exception:
            pass
        frame_timestamp_us: float | None = None
        try:
            frame_metadata = self.rs.frame_metadata_value.frame_timestamp
            if frame.supports_frame_metadata(frame_metadata):
                value = float(frame.get_frame_metadata(frame_metadata))
                if math.isfinite(value) and value >= 0.:
                    frame_timestamp_us = value
                    metadata_values["frame_readout_timestamp_us"] = value
        except Exception:
            pass
        try:
            metadata = self.rs.frame_metadata_value.sensor_timestamp
            if frame.supports_frame_metadata(metadata):
                sensor_timestamp_us = float(frame.get_frame_metadata(metadata))
                timestamp_ms = sensor_timestamp_us * .001
                if math.isfinite(timestamp_ms) and timestamp_ms >= 0.:
                    metadata_values["sensor_exposure_timestamp_us"] = sensor_timestamp_us
                    # D435i motion packets and depth frames share the depth
                    # hardware clock.  Preserve this raw identity separately
                    # even when get_timestamp() has been converted to GLOBAL_TIME.
                    metadata_values["imu_reference_timestamp_ms"] = timestamp_ms
                    metadata_values["imu_reference_timestamp_domain"] = "hardware_clock"
                    if (
                        sdk_domain == "global_time"
                        and sdk_timestamp_ms is not None
                        and frame_timestamp_us is not None
                    ):
                        metadata_values["sensor_to_frame_delta_us"] = (
                            sensor_timestamp_us - frame_timestamp_us
                        )
                        return (
                            sdk_timestamp_ms
                            + (sensor_timestamp_us - frame_timestamp_us) * .001,
                            "global_time",
                            "sensor_exposure_midpoint_global",
                            metadata_values,
                        )
                    return (
                        timestamp_ms,
                        "hardware_clock",
                        "sensor_exposure_midpoint",
                        metadata_values,
                    )
        except Exception:
            pass
        metadata_values["imu_reference_timestamp_ms"] = sdk_timestamp_ms
        metadata_values["imu_reference_timestamp_domain"] = sdk_domain
        if sdk_domain == "system_time":
            timestamp_kind = "host_time_of_arrival"
        elif sdk_domain in {"hardware_clock", "global_time"}:
            timestamp_kind = "frame_readout_timestamp"
        else:
            timestamp_kind = "unknown_frame_timestamp"
        return sdk_timestamp_ms, sdk_domain, timestamp_kind, metadata_values

    def _motion_frames(self, candidate: Any) -> list[tuple[str, Any]]:
        frames: list[tuple[str, Any]] = []
        for stream_name, key in (
            (self.rs.stream.gyro, "gyro"),
            (self.rs.stream.accel, "accel"),
        ):
            try:
                frame = candidate.first_or_default(stream_name)
                if frame:
                    frames.append((key, frame))
            except Exception:
                pass
        if frames:
            return frames
        try:
            stream_type = candidate.get_profile().stream_type()
        except Exception:
            return []
        if stream_type == self.rs.stream.gyro:
            return [("gyro", candidate)]
        if stream_type == self.rs.stream.accel:
            return [("accel", candidate)]
        return []

    def _process_motion_frame(self, key: str, frame: Any) -> None:
        try:
            value = frame.as_motion_frame().get_motion_data()
            vector = [float(value.x), float(value.y), float(value.z)]
        except Exception:
            return
        number, timestamp_ms, timestamp_domain = self._frame_identity(frame)
        self._ensure_ingress_state()
        advances = self._continuity.observe(
            key,
            frame_number=number,
            timestamp_ms=timestamp_ms,
            timestamp_domain=timestamp_domain,
        )
        self._consume_continuity_epoch()
        if not advances:
            return
        observed_wall, observed_monotonic, _pair_span_s = sample_clock_pair()
        sample_time = self._device_clock_mapper.map(
            timestamp_ms,
            timestamp_domain,
            stream=key,
            observed_monotonic=observed_monotonic,
            observed_wall=observed_wall,
        )
        motion: dict[str, Any] = {
            "source": "d435i_imu", "received_at": observed_wall,
            "sample_at": sample_time["capture_wall"],
            "sample_time_valid": sample_time["valid"],
            "dequeue_age_s": sample_time["dequeue_age_s"],
            "timestamp_domain": timestamp_domain,
        }
        if number is not None:
            motion["frame_number"] = number
        if key == "gyro":
            motion["gyroscope"] = vector
            motion["gyro_timestamp_ms"] = timestamp_ms
        else:
            motion["accelerometer"] = vector
            motion["accel_timestamp_ms"] = timestamp_ms
        if not self._update_imu_orientation(motion):
            return
        if timestamp_ms is None:
            return
        with self._motion_lock:
            self.latest_motion = motion
            self._imu_history.append(
                (timestamp_ms, timestamp_domain, dict(motion))
            )

    @staticmethod
    def _as_video_frameset(candidate: Any) -> Any:
        try:
            if candidate.is_frameset():
                candidate = candidate.as_frameset()
        except Exception:
            pass
        try:
            return candidate if candidate.get_color_frame() and candidate.get_depth_frame() else None
        except Exception:
            return None

    def _single_video_frame(self, candidate: Any) -> tuple[str, Any] | None:
        try:
            stream_type = candidate.get_profile().stream_type()
        except Exception:
            return None
        if stream_type == self.rs.stream.color:
            return "color", candidate
        if stream_type == self.rs.stream.depth:
            return "depth", candidate
        return None

    def _video_pair_compatible(
        self,
        first: tuple[Any, float | None, str, int],
        second: tuple[Any, float | None, str, int],
    ) -> bool:
        _, first_stamp, first_domain, _ = first
        _, second_stamp, second_domain, _ = second
        if first_stamp is None or second_stamp is None:
            return False
        # SDK timestamp values are comparable only within the same known
        # domain. A missing domain is diagnostic data, not proof of alignment.
        if not first_domain or not second_domain or first_domain != second_domain:
            return False
        return abs(first_stamp - second_stamp) < self._video_pair_max_delta_ms

    @staticmethod
    def _older_video_key(
        color: tuple[Any, float | None, str, int],
        depth: tuple[Any, float | None, str, int],
    ) -> str:
        """Choose the frame that cannot belong to a future ordered pair."""
        color_stamp, depth_stamp = color[1], depth[1]
        same_domain = bool(color[2] and depth[2] and color[2] == depth[2])
        if (
            same_domain
            and color_stamp is not None
            and depth_stamp is not None
            and color_stamp != depth_stamp
        ):
            return "color" if color_stamp < depth_stamp else "depth"
        # Timestamps from different domains are not comparable. Arrival order
        # is the only safe deterministic eviction rule in that case.
        return "color" if color[3] < depth[3] else "depth"

    def _process_single_video(self, key: str, frame: Any) -> Any:
        self._ensure_ingress_state()
        number, timestamp_ms, timestamp_domain = self._frame_identity(frame)
        advances = self._continuity.observe(
            key,
            frame_number=number,
            timestamp_ms=timestamp_ms,
            timestamp_domain=timestamp_domain,
        )
        self._consume_continuity_epoch()
        if not advances:
            return None

        self._video_arrival_order += 1
        if self._pending_video_frames[key] is not None:
            # Only one pending frame per stream is useful.  A newer frame makes
            # the older orphan impossible to return without increasing latency.
            self._unpaired_video_frames += 1
        self._pending_video_frames[key] = (
            frame, timestamp_ms, timestamp_domain, self._video_arrival_order,
        )
        color = self._pending_video_frames["color"]
        depth = self._pending_video_frames["depth"]
        if color is None or depth is None:
            return None
        if not self._video_pair_compatible(color, depth):
            if not color[2] or not depth[2] or color[2] != depth[2]:
                self._video_pair_domain_mismatches += 1
            elif color[1] is not None and depth[1] is not None:
                self._video_pair_skew_exceeded += 1
            older = self._older_video_key(color, depth)
            self._pending_video_frames[older] = None
            self._unpaired_video_frames += 1
            return None

        self._pending_video_frames["color"] = None
        self._pending_video_frames["depth"] = None
        return _VideoPair(color[0], depth[0])

    def _process_candidate(self, candidate: Any) -> Any:
        self._ensure_ingress_state()
        motion_frames: list[tuple[str, Any]] = []
        if self.motion_enabled:
            motion_frames = self._motion_frames(candidate)
            for key, frame in motion_frames:
                self._process_motion_frame(key, frame)
        frameset = self._as_video_frameset(candidate)
        if frameset is None:
            single_video = self._single_video_frame(candidate)
            if single_video is not None:
                pair = self._process_single_video(*single_video)
                if pair is not None and self.align_to == "color":
                    # rs.align accepts only a native composite_frame.  Standard
                    # pipeline delivery reaches the frameset path above; if a
                    # backend emits individual video frames, fail closed rather
                    # than passing a duck-typed pair into native code.
                    self._unsupported_align_color_video_pairs += 1
                    return None
                return pair
            if not motion_frames:
                self._invalid_ingress_frames += 1
            return None
        pending_count = sum(
            item is not None for item in self._pending_video_frames.values()
        )
        if pending_count:
            # Do not let stale individual frames survive a native synchronized
            # frameset and pair with a later frame from a different delivery
            # mode. Per-stream continuity below will reject exact duplicates.
            self._pending_video_frames = {"color": None, "depth": None}
            self._unpaired_video_frames += pending_count
        color = frameset.get_color_frame()
        depth = frameset.get_depth_frame()
        color_number, color_stamp, color_domain = self._frame_identity(color)
        depth_number, depth_stamp, depth_domain = self._frame_identity(depth)
        self._ensure_ingress_state()
        # A pipeline frameset has already passed librealsense synchronization;
        # preserve it for compatibility, but expose suspect matching so a
        # device/profile regression is not silent in field diagnostics.
        if color_domain and depth_domain and color_domain != depth_domain:
            self._video_pair_domain_mismatches += 1
        elif (
            color_stamp is not None
            and depth_stamp is not None
            and abs(color_stamp - depth_stamp) >= self._video_pair_max_delta_ms
        ):
            self._video_pair_skew_exceeded += 1
        color_ok = self._continuity.observe(
            "color", frame_number=color_number, timestamp_ms=color_stamp,
            timestamp_domain=color_domain,
        )
        self._consume_continuity_epoch()
        depth_ok = self._continuity.observe(
            "depth", frame_number=depth_number, timestamp_ms=depth_stamp,
            timestamp_domain=depth_domain,
        )
        self._consume_continuity_epoch()
        return frameset if color_ok and depth_ok else None

    def _motion_at(
        self,
        timestamp_ms: float,
        timestamp_domain: str,
        *,
        capture_wall: float | None = None,
        capture_wall_valid: bool = False,
    ) -> dict[str, Any]:
        self._ensure_ingress_state()
        try:
            mapped_capture_wall = float(capture_wall)
        except (TypeError, ValueError, OverflowError):
            mapped_capture_wall = math.nan
        mapped_time_usable = bool(
            capture_wall_valid
            and math.isfinite(mapped_capture_wall)
            and mapped_capture_wall > 0.
        )
        try:
            raw_reference_ms = float(timestamp_ms)
        except (TypeError, ValueError, OverflowError):
            raw_reference_ms = math.nan
        reference_domain = _canonical_timestamp_domain(timestamp_domain)
        with self._motion_lock:
            candidates: list[tuple[float, int, dict[str, Any]]] = []
            if mapped_time_usable:
                for order, (_stamp, _domain, motion) in enumerate(self._imu_history):
                    if motion.get("sample_time_valid") is not True:
                        continue
                    try:
                        sample_at = float(motion.get("sample_at"))
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if (
                        math.isfinite(sample_at)
                        and sample_at > 0.
                        and sample_at <= mapped_capture_wall
                    ):
                        candidates.append((sample_at, order, motion))

            # During device-clock warm-up there is no mapped epoch yet.  The
            # fallback is safe only when both raw SDK domains are known and
            # identical; an empty/unknown domain is never evidence that two
            # independent clocks are comparable.
            if not candidates and reference_domain and math.isfinite(raw_reference_ms):
                for order, (stamp, domain, motion) in enumerate(self._imu_history):
                    motion_domain = _canonical_timestamp_domain(domain)
                    try:
                        motion_stamp_ms = float(stamp)
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if (
                        motion_domain == reference_domain
                        and math.isfinite(motion_stamp_ms)
                        and motion_stamp_ms <= raw_reference_ms
                    ):
                        candidates.append((motion_stamp_ms, order, motion))
            result = (
                dict(max(candidates, key=lambda item: (item[0], item[1]))[2])
                if candidates else {}
            )
        if result:
            result["imu_ingress"] = self.ingress_stats()
        return result

    def latest_motion_snapshot(self) -> dict[str, Any]:
        self._ensure_ingress_state()
        with self._motion_lock:
            return dict(self.latest_motion)

    def capture_motion_snapshot(self) -> dict[str, Any]:
        self._ensure_ingress_state()
        with self._motion_lock:
            return dict(self._capture_motion)

    def capture_time_snapshot(self) -> dict[str, Any]:
        self._ensure_ingress_state()
        with self._motion_lock:
            return dict(self._capture_time)

    def ingress_stats(self) -> dict[str, Any]:
        self._ensure_ingress_state()
        return {
            "streams": self._continuity.snapshot(),
            "continuity_epoch": self._continuity.diagnostics(),
            "video_replaced": self._video_frames_replaced,
            "queue_dequeued": self._queue_dequeued,
            "queue_high_water": self._queue_high_water,
            "invalid_frames": self._invalid_ingress_frames,
            "unpaired_video_frames": self._unpaired_video_frames,
            "video_pair_domain_mismatches": self._video_pair_domain_mismatches,
            "video_pair_skew_exceeded": self._video_pair_skew_exceeded,
            "unsupported_align_color_video_pairs": self._unsupported_align_color_video_pairs,
            "queue_enabled": getattr(self, "_frame_queue", None) is not None,
            "queue_capacity": (
                getattr(self, "_frame_queue_capacity", 0)
                if getattr(self, "_frame_queue", None) is not None else 0
            ),
            "device_clock": self._device_clock_mapper.snapshot(),
        }

    def _record_queue_dequeue(self, frame_queue: Any) -> None:
        self._ensure_ingress_state()
        self._queue_dequeued += 1
        try:
            self._queue_high_water = max(
                self._queue_high_water, int(frame_queue.size()) + 1,
            )
        except Exception:
            pass

    def read(self) -> tuple[Any, Any, float]:
        self._ensure_ingress_state()
        if self._closed_event.is_set():
            raise RuntimeError("D435i source is closed")
        deadline = time.monotonic() + 0.5
        raw_frames = None
        raw_dequeue_monotonic: float | None = None
        raw_dequeue_wall: float | None = None
        frame_queue = getattr(self, "_frame_queue", None)
        while not self._closed_event.is_set() and time.monotonic() < deadline:
            try:
                candidate = (
                    frame_queue.wait_for_frame(100)
                    if frame_queue is not None
                    else self.pipeline.wait_for_frames(100)
                )
            except Exception:
                if self._closed_event.is_set():
                    break
                time.sleep(0.001)
                continue
            (
                candidate_dequeue_wall,
                candidate_dequeue_monotonic,
                _candidate_clock_pair_span_s,
            ) = sample_clock_pair()
            if frame_queue is not None:
                self._record_queue_dequeue(frame_queue)
            processed_frames = self._process_candidate(candidate)
            if processed_frames is not None:
                raw_frames = processed_frames
                raw_dequeue_monotonic = candidate_dequeue_monotonic
                raw_dequeue_wall = candidate_dequeue_wall
                break
        if raw_frames is None:
            if self._closed_event.is_set():
                raise RuntimeError("D435i source closed during frame poll")
            raise RuntimeError("D435i RGB-D frame poll timeout")

        # A backed-up queue may contain older video composites. Process every
        # already-buffered motion event, but keep only the newest complete
        # RGB-D frameset so latency does not grow with queue depth.
        if frame_queue is not None and hasattr(frame_queue, "poll_for_frame"):
            for _ in range(getattr(self, "_frame_queue_capacity", 128)):
                if self._closed_event.is_set():
                    break
                try:
                    candidate = frame_queue.poll_for_frame()
                except Exception:
                    break
                if not candidate:
                    break
                (
                    candidate_dequeue_wall,
                    candidate_dequeue_monotonic,
                    _candidate_clock_pair_span_s,
                ) = sample_clock_pair()
                self._record_queue_dequeue(frame_queue)
                newer_frames = self._process_candidate(candidate)
                if newer_frames is not None:
                    raw_frames = newer_frames
                    raw_dequeue_monotonic = candidate_dequeue_monotonic
                    raw_dequeue_wall = candidate_dequeue_wall
                    self._video_frames_replaced += 1

        # Keep the complete native color image.  On D435i, retrieving color
        # from an ``align(depth)`` frameset can return a depth-sized canvas
        # with black padding/cropping.  Only depth should come from the
        # aligned frameset; RGB remains the original 1280x720 stream.
        color = raw_frames.get_color_frame()
        raw_depth = raw_frames.get_depth_frame()
        if not color or not raw_depth:
            raise RuntimeError("D435i returned an incomplete native RGB-D frame")
        _, color_timestamp_ms, color_timestamp_domain = self._frame_identity(color)
        depth_frame_number, depth_timestamp_ms, depth_timestamp_domain = (
            self._frame_identity(raw_depth)
        )
        sync_valid = bool(
            color_timestamp_ms is not None
            and depth_timestamp_ms is not None
            and color_timestamp_domain
            and color_timestamp_domain == depth_timestamp_domain
        )
        sync_ms = (
            abs(color_timestamp_ms - depth_timestamp_ms)
            if sync_valid else -1.
        )
        (
            capture_timestamp_ms,
            capture_timestamp_domain,
            capture_timestamp_kind,
            capture_metadata,
        ) = self._capture_timestamp_identity(raw_depth)
        # Bind the host observation at the instant this particular raw video
        # candidate left wait/poll.  Queue draining and align.process() happen
        # later and must not be mistaken for USB/SDK backlog.
        if raw_dequeue_monotonic is None or raw_dequeue_wall is None:
            raw_dequeue_monotonic = time.monotonic()
            raw_dequeue_wall = time.time()
        capture_time = self._device_clock_mapper.map(
            capture_timestamp_ms,
            capture_timestamp_domain,
            stream="depth",
            observed_monotonic=raw_dequeue_monotonic,
            observed_wall=raw_dequeue_wall,
        )
        imu_reference_timestamp_ms = capture_metadata.get(
            "imu_reference_timestamp_ms", capture_timestamp_ms,
        )
        imu_reference_domain = str(capture_metadata.get(
            "imu_reference_timestamp_domain", capture_timestamp_domain,
        ) or "")
        capture_motion = (
            self._motion_at(
                float(imu_reference_timestamp_ms),
                imu_reference_domain,
                capture_wall=capture_time.get("capture_wall"),
                capture_wall_valid=capture_time.get("valid") is True,
            )
            if imu_reference_timestamp_ms is not None else {}
        )
        capture_time.update(
            version=1,
            reference_stream="depth",
            depth_frame_number=depth_frame_number,
            timestamp_kind=capture_timestamp_kind,
            frame_timestamp_ms=depth_timestamp_ms,
            frame_timestamp_domain=depth_timestamp_domain,
            color_frame_timestamp_ms=color_timestamp_ms,
            color_frame_timestamp_domain=color_timestamp_domain,
            color_depth_sync_valid=sync_valid,
            exposure_identity_valid=capture_timestamp_kind.startswith(
                "sensor_exposure_midpoint"
            ),
            absolute_exposure_time_valid=(
                capture_timestamp_kind == "sensor_exposure_midpoint_global"
                and capture_time.get("source_epoch_valid") is True
            ),
            **capture_metadata,
        )
        # Motion samples were already extracted from this frameset in the
        # polling loop above.  Re-reading gyro/accel here performs the same
        # SDK calls twice per RGB-D frame and adds avoidable Python/USB jitter.
        frames = (
            self.align.process(raw_frames)
            if self.align is not None and self.align_to == "color"
            else raw_frames
        )
        depth = frames.get_depth_frame()
        if not depth:
            raise RuntimeError("D435i returned an incomplete RGB-D frame")
        self._ensure_ingress_state()
        with self._motion_lock:
            self._capture_motion = capture_motion
            self._capture_time = capture_time
        import numpy as np
        return np.asanyarray(color.get_data()), np.asanyarray(depth.get_data()), sync_ms

    def _reset_imu_reference_candidate(self, reason: str) -> None:
        self._imu_calibration_window_start_ts = None
        self._imu_calibration_window_last_ts = None
        self._imu_calibration_anchor = None
        self._imu_calibration_gravity_sum = [0.0, 0.0, 0.0]
        self._imu_calibration_window_samples = 0
        self._imu_calibration_reset_reason = reason

    def _update_imu_orientation(self, motion: dict[str, Any]) -> bool:
        """Estimate camera roll/pitch from gravity and yaw by gyro integration.

        D435i motion frames do not provide an absolute quaternion.  The
        roll/pitch are time-smoothed gravity estimates; only yaw integrates
        gyro here. Yaw is relative and reset on process start.
        """
        accel = _finite_motion_vector(motion.get("accelerometer"))
        gyro = _finite_motion_vector(motion.get("gyroscope"))
        accel_norm = None
        if accel is not None:
            accel_norm = norm = math.hypot(*accel)
            if not math.isfinite(norm) or norm <= 1e-6:
                accel = None
        accel_ts = _new_motion_timestamp(
            motion.get("accel_timestamp_ms"), getattr(self, "_imu_last_accel_ts", None),
        ) if accel is not None else None
        if accel_ts is None:
            accel = None
        gyro_ts = _new_motion_timestamp(
            motion.get("gyro_timestamp_ms"), self._imu_last_gyro_ts,
        ) if gyro is not None else None
        if accel_ts is None and gyro_ts is None:
            return False
        try:
            received_at = float(motion.get("received_at"))
            if not math.isfinite(received_at) or received_at <= 0.:
                received_at = None
        except (TypeError, ValueError, OverflowError):
            received_at = None
        component_times = getattr(self, "_imu_component_received_at", None)
        if component_times is None:
            component_times = self._imu_component_received_at = {"accel": None, "gyro": None}
        component_sample_times = getattr(self, "_imu_component_sample_at", None)
        if component_sample_times is None:
            component_sample_times = self._imu_component_sample_at = {
                "accel": None, "gyro": None,
            }
        component_sample_valid = getattr(self, "_imu_component_sample_valid", None)
        if component_sample_valid is None:
            component_sample_valid = self._imu_component_sample_valid = {
                "accel": False, "gyro": False,
            }
        sample_time_valid = bool(motion.get("sample_time_valid", False))
        try:
            sample_at = float(motion.get("sample_at", received_at))
            if not math.isfinite(sample_at) or sample_at <= 0.:
                sample_at = received_at
        except (TypeError, ValueError, OverflowError):
            sample_at = received_at
        if accel is not None:
            ax, ay, az = accel
            ax, ay, az = ax / norm, ay / norm, az / norm
            gravity_roll = math.atan2(ay, az)
            gravity_pitch = math.atan2(-ax, max((ay * ay + az * az) ** 0.5, 1e-6))
            previous_accel_ts = getattr(self, "_imu_last_accel_ts", None)
            sample_dt = (accel_ts - previous_accel_ts) * .001 if previous_accel_ts is not None else .01
            # A long outage contains no intermediate gravity observations.
            # Resume with one nominal sample instead of snapping to the first
            # possibly acceleration-contaminated measurement after the gap.
            gravity_gap = sample_dt > _IMU_REFERENCE_MAX_GAP_S
            gravity_dt = .01 if gravity_gap else sample_dt
            tau = getattr(self, "_imu_gravity_tau_s", _IMU_GRAVITY_TAU_S)
            alpha = math.exp(-gravity_dt / tau)
            self._imu_rpy[0] = alpha * self._imu_rpy[0] + (1.0 - alpha) * gravity_roll
            self._imu_rpy[1] = alpha * self._imu_rpy[1] + (1.0 - alpha) * gravity_pitch
            motion["imu_gravity_dt_s"] = gravity_dt
            motion["imu_gravity_gap"] = gravity_gap
            self._imu_last_accel_ts = accel_ts
            self._imu_accel_updates = getattr(self, "_imu_accel_updates", 0) + 1
            component_times["accel"] = received_at
            component_sample_times["accel"] = sample_at
            component_sample_valid["accel"] = sample_time_valid
        if gyro_ts is not None:
            if self._imu_last_gyro_ts is not None:
                dt = min(max((gyro_ts - self._imu_last_gyro_ts) * 1e-3, 0.0), 0.05)
                self._imu_rpy[2] += gyro[2] * dt
            self._imu_last_gyro_ts = gyro_ts
            self._imu_gyro_updates = getattr(self, "_imu_gyro_updates", 0) + 1
            component_times["gyro"] = received_at
            component_sample_times["gyro"] = sample_at
            component_sample_valid["gyro"] = sample_time_valid
        # Reference capture is time based. Counting 250 polls made startup
        # take 25 s at 10 Hz but only 1.25 s at 200 Hz, and could lock a
        # moving mast or a filter that had not converged. Accumulate raw unit
        # gravity only during one continuous, quiet interval.
        if self._imu_reference_rpy is None:
            if not hasattr(self, "_imu_calibration_window_start_ts"):
                self._reset_imu_reference_candidate("waiting_for_stable_gravity")

            gyro_moving = gyro_ts is not None and math.hypot(*gyro) > _IMU_REFERENCE_MAX_GYRO_RAD_S
            if gyro_moving:
                self._reset_imu_reference_candidate("angular_motion")
            if accel is not None:
                self._imu_calibration_samples += 1  # Diagnostic only; never the readiness gate.
                gravity = tuple(value / accel_norm for value in accel)
                accel_unstable = abs(accel_norm - _STANDARD_GRAVITY_M_S2) > _IMU_REFERENCE_MAX_ACCEL_ERROR_M_S2
                if accel_unstable:
                    self._reset_imu_reference_candidate("non_gravity_acceleration")
                elif not gyro_moving:
                    start_new = gravity_gap or self._imu_calibration_window_start_ts is None
                    if not start_new:
                        dot = max(-1.0, min(1.0, sum(
                            current * anchor for current, anchor in zip(gravity, self._imu_calibration_anchor)
                        )))
                        start_new = math.acos(dot) > _IMU_REFERENCE_MAX_TILT_SPAN_RAD
                        if start_new:
                            self._reset_imu_reference_candidate("tilt_motion")
                    if start_new:
                        self._imu_calibration_window_start_ts = accel_ts
                        self._imu_calibration_anchor = gravity
                        self._imu_calibration_gravity_sum = list(gravity)
                        self._imu_calibration_window_samples = 1
                    else:
                        for index, value in enumerate(gravity):
                            self._imu_calibration_gravity_sum[index] += value
                        self._imu_calibration_window_samples += 1
                    self._imu_calibration_window_last_ts = accel_ts
                    elapsed = max(0.0, (accel_ts - self._imu_calibration_window_start_ts) * 1e-3)
                    self._imu_calibration_reset_reason = "stable"
                    if elapsed + 1e-9 >= _IMU_REFERENCE_DURATION_S:
                        mean = self._imu_calibration_gravity_sum
                        mean_norm = max(math.hypot(*mean), 1e-12)
                        gx, gy, gz = (value / mean_norm for value in mean)
                        reference_roll = math.atan2(gy, gz)
                        reference_pitch = math.atan2(-gx, max(math.hypot(gy, gz), 1e-12))
                        reference_yaw = self._imu_rpy[2]
                        self._imu_reference_rpy = [reference_roll, reference_pitch, reference_yaw]
                        # Enable correction continuously at zero instead of
                        # exposing low-pass startup residue as a TF jump.
                        self._imu_rpy[:] = self._imu_reference_rpy
                        self._imu_calibration_reset_reason = "calibrated"

        r, p, y = self._imu_rpy
        cr, sr = math.cos(r / 2), math.sin(r / 2)
        cp, sp = math.cos(p / 2), math.sin(p / 2)
        cy, sy = math.cos(y / 2), math.sin(y / 2)
        motion["rpy"] = [r, p, y]
        motion["imu_valid_updates"] = {
            "accel": getattr(self, "_imu_accel_updates", 0),
            "gyro": getattr(self, "_imu_gyro_updates", 0),
        }
        motion["component_received_at"] = dict(component_times)
        motion["component_sample_at"] = dict(component_sample_times)
        motion["component_sample_time_valid"] = dict(component_sample_valid)
        reference = self._imu_reference_rpy or [0.0, 0.0, 0.0]
        motion["imu_calibrating"] = self._imu_reference_rpy is None
        calibration_start = getattr(self, "_imu_calibration_window_start_ts", None)
        calibration_last = getattr(self, "_imu_calibration_window_last_ts", None)
        motion["imu_calibration_elapsed_s"] = (
            max(0.0, (calibration_last - calibration_start) * 1e-3)
            if calibration_start is not None and calibration_last is not None else 0.0
        )
        motion["imu_calibration_status"] = getattr(
            self, "_imu_calibration_reset_reason", "waiting_for_stable_gravity",
        )
        motion["correction_rpy"] = [0.0, 0.0, 0.0] if self._imu_reference_rpy is None else [
            math.atan2(math.sin(value - baseline), math.cos(value - baseline))
            for value, baseline in zip((r, p, y), reference)
        ]
        motion["quaternion"] = [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
        # At least one hardware sample advanced; old/invalid polls return
        # before filtering, quaternion conversion and metadata allocation.
        return True

    def close(self) -> None:
        self._ensure_ingress_state()
        with self._close_lock:
            if self._pipeline_stopped:
                return
            # Set before pipeline.stop so an SDK wait error makes read() exit
            # immediately.  Only mark stop complete after success, allowing a
            # caller to retry cleanup if the SDK raises transiently.
            self._closed_event.set()
            self.pipeline.stop()
            self._pipeline_stopped = True


def _encode(
    rgb: Any,
    depth: Any,
    *,
    depth_png_compression: int = 4,
) -> tuple[bytes, bytes]:
    import cv2
    ok_rgb, rgb_buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 82])
    # OpenCV's implicit PNG settings produce unusually large 16-bit D435
    # depth frames.  An explicit moderate compression level is still exactly
    # lossless, but nearly halves the bytes sent through the Go2 Wi-Fi link.
    ok_depth, depth_buf = cv2.imencode(
        ".png",
        depth,
        [cv2.IMWRITE_PNG_COMPRESSION, int(depth_png_compression)],
    )
    if not ok_rgb or not ok_depth:
        raise RuntimeError("failed to encode D435i frame")
    return bytes(rgb_buf), bytes(depth_buf)


def _encode_parallel(
    rgb: Any,
    depth: Any,
    *,
    depth_png_compression: int = 4,
    executor: ThreadPoolExecutor,
) -> tuple[bytes, bytes]:
    """Encode RGB JPEG and uint16 depth PNG concurrently.

    OpenCV releases the GIL while JPEG/PNG codecs run.  The previous capture
    loop encoded the two independent images serially, so a normal 1280x720 +
    848x480 pair could spend most of a 100 ms period in codecs before the
    network sender even saw the frame.  A persistent two-worker executor
    overlaps those codecs without changing the wire format or sample values.
    The synchronous ``_encode`` helper remains available for small replay
    tests and callers that do not own an executor.
    """
    import cv2

    rgb_future = executor.submit(
        cv2.imencode, ".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 82]
    )
    depth_future = executor.submit(
        cv2.imencode,
        ".png",
        depth,
        [cv2.IMWRITE_PNG_COMPRESSION, int(depth_png_compression)],
    )
    ok_rgb, rgb_buf = rgb_future.result()
    ok_depth, depth_buf = depth_future.result()
    if not ok_rgb or not ok_depth:
        raise RuntimeError("failed to encode D435i frame")
    return bytes(rgb_buf), bytes(depth_buf)


def _synthetic_frame(width: int, height: int) -> tuple[Any, Any, float, dict[str, float]]:
    import numpy as np
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, 1] = 32
    rgb[height // 4 : height * 3 // 4, width // 3 : width * 2 // 3] = (60, 150, 230)
    depth = np.full((height, width), 1500, dtype=np.uint16)
    depth[height // 4 : height * 3 // 4, width // 3 : width * 2 // 3] = 1000
    intr = {"fx": width * 0.9, "fy": width * 0.9, "cx": width / 2, "cy": height / 2, "width": width, "height": height}
    return rgb, depth, 0.0, intr


async def _publish_impl(args: argparse.Namespace, cleanup_callbacks: list[Any]) -> None:
    import websocket

    state = ReadOnlyState()
    threading.Thread(target=_unitree_state_reader, args=(state, args.interface), daemon=True).start()
    source = None
    try:
        source = None if args.dry_run else D435iSource(
            args.depth_width, args.depth_height, args.depth_fps,
            enable_motion=args.enable_camera_imu,
            color_width=args.color_width, color_height=args.color_height,
            color_fps=args.color_fps, align_to=args.align_to,
            imu_gravity_tau_s=args.imu_gravity_tau_s,
            frame_queue_capacity=getattr(args, "sensor_queue_capacity", 128),
        )
    except BaseException:
        # D435iSource stops a partially-started pipeline in its constructor.
        # Keep this boundary explicit so later resource additions cannot hide
        # an acquisition startup failure.
        if source is not None:
            try:
                source.close()
            except Exception:
                pass
        raise
    # Capture, codecs, and TCP are three independent stages. Both hand-offs
    # are latest-only: slow codecs or a network write may drop obsolete
    # images, but must not stop draining the RealSense pipeline (including its
    # higher-rate IMU frames).
    frame_lock = threading.Lock()
    frame_ready = threading.Event()
    latest_frame: dict[str, Any] = {}
    shutdown_event = threading.Event()
    # JPEG and PNG codecs are independent and release the GIL. Keep the
    # executor alive for the whole bridge rather than constructing two worker
    # threads for every frame.
    try:
        encode_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="go2-rgbd-encode"
        )
    except BaseException:
        if source is not None:
            try:
                source.close()
            except Exception:
                pass
        raise

    codec_stage: _LatestOnlyStage | None = None

    def capture_loop() -> None:
        seq = 0
        diagnostic_started = time.monotonic()
        diagnostic_frames = 0
        diagnostic_attempts = 0
        diagnostic_errors = 0
        consecutive_errors = 0
        diagnostic_read_s = 0.0
        diagnostic_copy_s = 0.0
        diagnostic_last_success = diagnostic_started
        diagnostic_last_warning = 0.0
        while not shutdown_event.is_set():
            started = time.monotonic()
            diagnostic_attempts += 1
            try:
                read_started = time.monotonic()
                try:
                    if source is None:
                        rgb, depth, sync_ms, intr = _synthetic_frame(args.width, args.height)
                        capture_motion: dict[str, Any] = {}
                        capture_time: dict[str, Any] = {}
                    else:
                        rgb, depth, sync_ms = source.read()
                        intr = source.intrinsics
                        # One immutable, depth-exposure-causal IMU snapshot is
                        # shared by both packet locations. Reading
                        # latest_motion twice could mix two ingress updates.
                        capture_motion = source.capture_motion_snapshot()
                        capture_time_fn = getattr(source, "capture_time_snapshot", None)
                        capture_time = (
                            capture_time_fn() if callable(capture_time_fn) else {}
                        )
                finally:
                    # Include failed/timed-out SDK reads in acquisition cost.
                    diagnostic_read_s += time.monotonic() - read_started
                observed_wall, observed_monotonic, _pair_span_s = sample_clock_pair()
                try:
                    if capture_time.get("valid") is not True:
                        raise ValueError("capture clock mapping is unavailable")
                    capture_stamp = float(capture_time.get("capture_wall"))
                    capture_monotonic = float(capture_time.get("capture_monotonic"))
                    if (
                        not math.isfinite(capture_stamp)
                        or capture_stamp <= 0.
                        or not math.isfinite(capture_monotonic)
                        or capture_monotonic <= 0.
                        or capture_monotonic > observed_monotonic + 1e-3
                    ):
                        raise ValueError("invalid mapped capture time")
                except (TypeError, ValueError, OverflowError):
                    capture_stamp = observed_wall
                    capture_monotonic = observed_monotonic
                    stamp_source = "host_read_fallback"
                else:
                    stamp_source = {
                        "recent_min_offset": "device_clock_estimated",
                        "sdk_global_time": "sdk_global_time",
                        "sdk_system_time": "sdk_system_time",
                    }.get(
                        str(capture_time.get("clock_mapping_method") or ""),
                        "device_clock_estimated",
                    )
                capture_time = dict(capture_time)
                capture_time["stamp_source"] = stamp_source
                # These are the values actually selected for the protocol and
                # stale-frame clock.  During hardware-clock warm-up they differ
                # from the shadow estimate retained in capture_wall/monotonic.
                capture_time["selected_capture_wall"] = capture_stamp
                capture_time["selected_capture_monotonic"] = capture_monotonic
                # SDK-backed arrays can reference reusable frame-pool memory.
                # The encoder owns these copies while capture immediately
                # returns to wait_for_frames.
                copy_started = time.monotonic()
                rgb, depth = rgb.copy(), depth.copy()
                diagnostic_copy_s += time.monotonic() - copy_started
                state_snapshot, state_delta = state.snapshot_at(
                    capture_stamp,
                    max_delta_s=float(
                        getattr(args, "capture_telemetry_max_delta_s", .15)
                    ),
                )
                # Only the pose fields needed for capture-time TF/YOLO are
                # copied into every RGB-D packet. Battery/range diagnostics
                # remain on the lower-rate telemetry channel, avoiding a
                # sizeable bandwidth increase at 10 Hz.
                capture_telemetry = {
                    key: state_snapshot[key]
                    for key in ("received_at", "position", "velocity", "yaw", "imu")
                    if key in state_snapshot
                }
                if capture_motion:
                    capture_telemetry["camera_imu"] = capture_motion
                diagnostic_frames += 1
                consecutive_errors = 0
                diagnostic_last_success = time.monotonic()
                seq += 1
                raw_frame = {
                    "seq": seq,
                    "stamp": capture_stamp,
                    "captured_monotonic": capture_monotonic,
                    "capture_timing": capture_time,
                    "capture_telemetry_delta_s": state_delta,
                    "rgb": rgb,
                    "depth": depth,
                    "intrinsics": intr,
                    "rgb_intrinsics": source.intrinsics if source is not None else intr,
                    "depth_intrinsics": source.depth_intrinsics if source is not None else intr,
                    "depth_to_color_extrinsics": source.depth_to_color_extrinsics if source is not None else {},
                    "sync_ms": sync_ms,
                    "depth_scale": args.depth_scale if source is None else source.depth_scale,
                    "camera_imu": capture_motion,
                    "telemetry": capture_telemetry,
                }
                if codec_stage is not None:
                    codec_stage.submit(raw_frame)
            except Exception as exc:
                diagnostic_errors += 1
                consecutive_errors += 1
                now = time.monotonic()
                # A disconnected camera previously printed at 10 Hz and hid
                # the useful rate diagnostics. Keep the first error and one
                # sample per second; the heartbeat below carries the counts.
                if diagnostic_errors == 1 or now - diagnostic_last_warning >= 1.0:
                    print(
                        "camera capture warning: "
                        f"{exc} consecutive_errors={consecutive_errors}",
                        flush=True,
                    )
                    diagnostic_last_warning = now
                shutdown_event.wait(0.1)
            now = time.monotonic()
            if now - diagnostic_started >= 10.0:
                wall = max(now - diagnostic_started, 1e-6)
                stage_stats = codec_stage.stats() if codec_stage is not None else {
                    "processed": 0, "errors": 0, "replaced": 0, "alive": False,
                }
                latest_motion = source.latest_motion_snapshot() if source is not None else {}
                ingress = source.ingress_stats() if source is not None else {}
                print(
                    "sensor capture "
                    f"attempt_hz={diagnostic_attempts / wall:.2f} "
                    f"read_hz={diagnostic_frames / wall:.2f} "
                    f"capture_errors={diagnostic_errors} "
                    f"consecutive_errors={consecutive_errors} "
                    f"last_success_age_ms={1000.0 * max(0.0, now - diagnostic_last_success):.1f} "
                    f"avg_read_ms={1000.0 * diagnostic_read_s / max(diagnostic_attempts, 1):.1f} "
                    f"avg_copy_ms={1000.0 * diagnostic_copy_s / max(diagnostic_frames, 1):.1f} "
                    f"capture_seq={seq} "
                    f"codec_processed={stage_stats['processed']} "
                    f"codec_errors={stage_stats['errors']} "
                    f"codec_replaced={stage_stats['replaced']} "
                    f"codec_alive={stage_stats['alive']} "
                    f"imu_updates={latest_motion.get('imu_valid_updates', {})} "
                    f"imu_calibration={latest_motion.get('imu_calibration_status', 'off')}/"
                    f"{float(latest_motion.get('imu_calibration_elapsed_s', 0.0)):.2f}s "
                    f"ingress={ingress}",
                    flush=True,
                )
                diagnostic_started = now
                diagnostic_frames = 0
                diagnostic_attempts = 0
                diagnostic_errors = 0
                diagnostic_read_s = 0.0
                diagnostic_copy_s = 0.0
            if source is None:
                shutdown_event.wait(max(0.0, 1.0 / args.fps - (time.monotonic() - started)))

    codec_diagnostics = {
        "started": time.monotonic(), "frames": 0, "encode_s": 0.0,
        "max_encode_s": 0.0, "last_replaced": 0, "last_warning": 0.0,
    }

    def report_encode_error(exc: Exception) -> None:
        now = time.monotonic()
        if now - codec_diagnostics["last_warning"] >= 1.0:
            print(f"camera encode warning: {exc}", flush=True)
            codec_diagnostics["last_warning"] = now

    def encode_frame(captured: dict[str, Any]) -> dict[str, Any]:
        captured = dict(captured)
        rgb = captured.pop("rgb")
        depth = captured.pop("depth")
        encode_started = time.monotonic()
        rgb_jpeg, depth_png = _encode_parallel(
            rgb,
            depth,
            depth_png_compression=args.depth_png_compression,
            executor=encode_executor,
        )
        elapsed = time.monotonic() - encode_started
        codec_diagnostics["frames"] += 1
        codec_diagnostics["encode_s"] += elapsed
        codec_diagnostics["max_encode_s"] = max(codec_diagnostics["max_encode_s"], elapsed)
        captured.update(
            rgb_jpeg=rgb_jpeg,
            depth_png=depth_png,
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
        )
        now = time.monotonic()
        if now - codec_diagnostics["started"] >= 10.0:
            wall = max(now - codec_diagnostics["started"], 1e-6)
            stats = codec_stage.stats()
            replaced = stats["replaced"] - codec_diagnostics["last_replaced"]
            print(
                "sensor codec "
                f"encode_hz={codec_diagnostics['frames'] / wall:.2f} "
                f"avg_codec_ms={1000.0 * codec_diagnostics['encode_s'] / max(codec_diagnostics['frames'], 1):.1f} "
                f"max_codec_ms={1000.0 * codec_diagnostics['max_encode_s']:.1f} "
                f"latest_capture_seq={captured['seq']} replaced_raw={replaced}",
                flush=True,
            )
            codec_diagnostics.update(
                started=now, frames=0, encode_s=0.0, max_encode_s=0.0,
                last_replaced=stats["replaced"],
            )
        return captured

    def publish_encoded(frame: dict[str, Any]) -> None:
        with frame_lock:
            latest_frame.clear()
            latest_frame.update(frame)
            frame_ready.set()

    capture_thread: threading.Thread | None = None
    cleanup_lock = threading.Lock()
    cleanup_done = False

    def cleanup(_future: Any = None) -> None:
        nonlocal cleanup_done
        with cleanup_lock:
            if cleanup_done:
                return
            cleanup_done = True
        shutdown_event.set()
        if source is not None:
            try:
                source.close()
            except Exception:
                pass
        if capture_thread is not None:
            capture_thread.join(timeout=1.0)
            if capture_thread.is_alive():
                print("camera capture shutdown warning: worker still alive", flush=True)
        codec_closed = True
        if codec_stage is not None:
            codec_closed = codec_stage.close(timeout=1.0)
            if not codec_closed:
                print("camera codec shutdown warning: worker still alive", flush=True)
        # Once the codec stage joined, all of its executor futures completed;
        # wait=True then releases both native worker threads without extending
        # shutdown. If a native codec is stuck, preserve bounded cleanup.
        encode_executor.shutdown(wait=codec_closed, cancel_futures=True)

    cleanup_callbacks.append(cleanup)
    try:
        codec_stage = _LatestOnlyStage(
            encode_frame,
            publish_encoded,
            name="d435i-codec",
            on_error=report_encode_error,
        )
        capture_thread = threading.Thread(
            target=capture_loop, name="d435i-capture", daemon=True,
        )
        capture_thread.start()
    except BaseException:
        cleanup()
        raise

    telemetry_seq = 0
    last_sent_seq = -1
    while True:
        ws = None
        try:
            ws = websocket.create_connection(args.url, timeout=args.connect_timeout, enable_multithread=True)
            # A short connect timeout is useful, but applying the same timeout
            # to a large RGB-D send creates a reconnect storm on brief Wi-Fi
            # congestion.  Capture remains independent and latest-only while
            # this send waits, so allowing the TCP channel to drain is safe.
            ws.settimeout(args.send_timeout)
            raw_socket = getattr(ws.sock, "sock", ws.sock)
            try:
                raw_socket.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_SNDBUF,
                    int(args.send_buffer_kb) * 1024,
                )
                # Wake senders as soon as a small amount can leave the kernel
                # instead of accumulating several stale RGB-D frames.
                if hasattr(socket, "TCP_NOTSENT_LOWAT"):
                    raw_socket.setsockopt(
                        socket.IPPROTO_TCP,
                        socket.TCP_NOTSENT_LOWAT,
                        min(32 * 1024, int(args.send_buffer_kb) * 1024),
                    )
            except OSError as exc:
                print(f"sensor socket tuning warning: {exc}", flush=True)
            ws.send(json.dumps(hello_packet(host=socket.gethostname(), streams={"camera": "d435i", "fps": args.fps})))
            # Two additional tiny, ordered send-time samples let the local
            # bridge choose a three-sample lower envelope (hello + probes)
            # before the first large RGB-D packet. This is startup-only and
            # does not consume camera/IMU bandwidth or add per-frame latency.
            for clock_probe_seq in range(2):
                # Temporal diversity prevents three packets sharing one
                # transient scheduler/network delay from immediately defining
                # the cross-host epoch.  This is a one-time startup cost only.
                await asyncio.sleep(.025)
                ws.send(json.dumps(clock_probe_packet(seq=clock_probe_seq)))
            clock_probe_seq = 2
            last_clock_probe = time.monotonic()
            last_telemetry = 0.0
            next_publish = time.monotonic()
            diagnostic_started = time.monotonic()
            diagnostic_sends = 0
            diagnostic_send_s = 0.0
            diagnostic_max_send_s = 0.0
            diagnostic_packet_s = 0.0
            diagnostic_delivery_age_s = 0.0
            diagnostic_max_delivery_age_s = 0.0
            diagnostic_bytes = 0
            diagnostic_stale_frames = 0
            diagnostic_last_stale_warning = 0.0
            print(f"connected to policy WebSocket {args.url}", flush=True)
            while True:
                now = time.monotonic()
                if now < next_publish:
                    await asyncio.sleep(min(0.01, next_publish - now))
                    continue
                if not frame_ready.wait(timeout=0.02):
                    await asyncio.sleep(0)
                    continue
                with frame_lock:
                    frame = dict(latest_frame)
                    frame_seq = int(frame.get("seq", -1))
                    if not frame or frame_seq == last_sent_seq:
                        frame_ready.clear()
                if not frame or frame_seq == last_sent_seq:
                    await asyncio.sleep(0.002)
                    continue
                try:
                    frame_age = time.monotonic() - float(frame["captured_monotonic"])
                except (KeyError, TypeError, ValueError, OverflowError):
                    frame_age = float("inf")
                if not math.isfinite(frame_age) or frame_age < 0.0 or frame_age > args.max_frame_age_s:
                    last_sent_seq = frame_seq
                    with frame_lock:
                        if int(latest_frame.get("seq", -1)) == frame_seq:
                            frame_ready.clear()
                    diagnostic_stale_frames += 1
                    now = time.monotonic()
                    if (
                        diagnostic_stale_frames == 1
                        or now - diagnostic_last_stale_warning >= 1.0
                    ):
                        print(
                            "dropping stale encoded frame "
                            f"seq={frame_seq} age_s={frame_age:.3f} "
                            f"dropped_since_connect={diagnostic_stale_frames}",
                            flush=True,
                        )
                        diagnostic_last_stale_warning = now
                    # A stale result used to leave next_publish in the past,
                    # creating a tight loop which could starve cancellation
                    # and other asyncio work. There is no value in bursting
                    # stale frames, so resume at the next regular period.
                    next_publish = now + 1.0 / args.publish_fps
                    await asyncio.sleep(0)
                    continue
                packet_started = time.monotonic()
                packet = image_packet(
                    seq=frame_seq, stamp=float(frame["stamp"]),
                    rgb_jpeg=frame["rgb_jpeg"], depth_png=frame["depth_png"],
                    width=int(frame["width"]), height=int(frame["height"]),
                    camera_frame=args.camera_frame,
                    # ``align(depth)`` leaves native depth on its own optical
                    # frame; ``align(none)`` does the same.  Only
                    # ``align(color)`` resamples depth onto the RGB frame.
                    # Keeping this distinction is required for the local
                    # bridge to compose the RealSense depth->colour baseline
                    # and prevents an unaligned mode from silently publishing
                    # depth points in the RGB TF frame.
                    depth_frame=("d435i_depth_optical_frame" if source is not None and args.align_to in ("depth", "none") else args.camera_frame),
                    depth_scale=float(frame["depth_scale"]),
                    intrinsics=frame["intrinsics"], color_depth_sync_ms=frame["sync_ms"],
                    rgb_intrinsics=frame.get("rgb_intrinsics"), depth_intrinsics=frame.get("depth_intrinsics"),
                    depth_to_color_extrinsics=frame.get("depth_to_color_extrinsics"),
                    camera_imu=frame.get("camera_imu"),
                    telemetry=frame.get("telemetry"),
                    capture_timing=frame.get("capture_timing"),
                )
                # Send a tiny clock probe before (never after) the large RGB-D
                # envelope.  The periodic clean lane lets the receiver follow
                # oscillator drift even when image sends or decode are slow;
                # it does not touch camera capture or inference cadence.
                probe_now = time.monotonic()
                if probe_now - last_clock_probe >= float(
                    getattr(args, "clock_probe_period", 1.0)
                ):
                    ws.send(json.dumps(
                        clock_probe_packet(seq=clock_probe_seq),
                        separators=(",", ":"),
                    ))
                    clock_probe_seq += 1
                    last_clock_probe = probe_now
                payload = encode_wire_packet(packet, compression_level=1)
                diagnostic_packet_s += time.monotonic() - packet_started
                send_started = time.monotonic()
                ws.send_binary(payload)
                send_s = time.monotonic() - send_started
                delivery_age_s = time.monotonic() - float(frame["captured_monotonic"])
                diagnostic_sends += 1
                diagnostic_send_s += send_s
                diagnostic_max_send_s = max(diagnostic_max_send_s, send_s)
                diagnostic_delivery_age_s += delivery_age_s
                diagnostic_max_delivery_age_s = max(
                    diagnostic_max_delivery_age_s, delivery_age_s,
                )
                diagnostic_bytes += len(payload)
                last_sent_seq = frame_seq
                with frame_lock:
                    if int(latest_frame.get("seq", -1)) == frame_seq:
                        frame_ready.clear()
                now = time.monotonic()
                if now - last_telemetry >= args.telemetry_period:
                    telemetry_seq += 1
                    telemetry = state.snapshot()
                    if source is not None:
                        latest_motion = source.latest_motion_snapshot()
                        if latest_motion:
                            telemetry["camera_imu"] = latest_motion
                    ws.send(json.dumps(telemetry_packet(seq=telemetry_seq, telemetry=telemetry), separators=(",", ":")))
                    last_telemetry = now
                if now - diagnostic_started >= 10.0:
                    wall = max(now - diagnostic_started, 1e-6)
                    print(
                        "sensor transport "
                        f"send_hz={diagnostic_sends / wall:.2f} "
                        f"avg_send_ms={1000.0 * diagnostic_send_s / max(diagnostic_sends, 1):.1f} "
                        f"avg_packet_ms={1000.0 * diagnostic_packet_s / max(diagnostic_sends, 1):.1f} "
                        f"max_send_ms={1000.0 * diagnostic_max_send_s:.1f} "
                        f"avg_delivery_age_ms={1000.0 * diagnostic_delivery_age_s / max(diagnostic_sends, 1):.1f} "
                        f"max_delivery_age_ms={1000.0 * diagnostic_max_delivery_age_s:.1f} "
                        f"stale_frames={diagnostic_stale_frames} "
                        f"wire_mbps={diagnostic_bytes * 8.0 / wall / 1e6:.2f} "
                        f"capture_seq={frame['seq']}",
                        flush=True,
                    )
                    diagnostic_started = now
                    diagnostic_sends = 0
                    diagnostic_send_s = 0.0
                    diagnostic_packet_s = 0.0
                    diagnostic_max_send_s = 0.0
                    diagnostic_delivery_age_s = 0.0
                    diagnostic_max_delivery_age_s = 0.0
                    diagnostic_bytes = 0
                # Schedule against the original cadence.  Adding the encode
                # time to ``now`` made a 10 Hz target degrade to ~6 Hz even
                # when capture and socket send were fast.
                next_publish += 1.0 / args.publish_fps
                if next_publish < now:
                    next_publish = now
        except Exception as exc:
            print(f"sensor link disconnected: {exc}; retrying in {args.reconnect_s}s", flush=True)
            await asyncio.sleep(args.reconnect_s)
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass


async def publish(args: argparse.Namespace) -> None:
    """Publish until cancelled and synchronously release every owned worker."""
    cleanup_callbacks: list[Any] = []
    try:
        await _publish_impl(args, cleanup_callbacks)
    finally:
        while cleanup_callbacks:
            cleanup_callbacks.pop()()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:12334")
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--color-width", type=int, default=1280)
    parser.add_argument("--color-height", type=int, default=720)
    parser.add_argument("--color-fps", type=int, default=10)
    parser.add_argument("--depth-width", type=int, default=848)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--depth-fps", type=int, default=10)
    parser.add_argument("--align-to", choices=("none", "color", "depth"), default="none")
    parser.add_argument("--publish-fps", type=float, default=10.0)
    parser.add_argument("--depth-png-compression", type=int, default=4)
    parser.add_argument(
        "--sensor-queue-capacity", type=int, default=128,
        help="bounded librealsense callback queue for mixed RGB-D and IMU frames",
    )
    parser.add_argument("--telemetry-period", type=float, default=0.2)
    parser.add_argument(
        "--clock-probe-period", type=float, default=1.0,
        help="periodic small pre-RGB-D clock probe interval in seconds",
    )
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--send-timeout", type=float, default=3.0)
    parser.add_argument("--max-frame-age-s", type=float, default=0.5,
                        help="drop an encoded RGB-D frame if it waited this long before send")
    parser.add_argument("--send-buffer-kb", type=int, default=128)
    parser.add_argument("--reconnect-s", type=float, default=2.0)
    parser.add_argument("--camera-frame", default="d435i_color_optical_frame")
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--imu-gravity-tau-s", type=float, default=_IMU_GRAVITY_TAU_S,
                        help="Gravity smoothing time constant; default matches alpha=0.98 at 100 Hz")
    parser.add_argument(
        "--enable-camera-imu",
        action="store_true",
        help="also stream the D435i gyro/accelerometer (disabled by default for RGB-D stability)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.imu_gravity_tau_s) or args.imu_gravity_tau_s <= 0.:
        parser.error("--imu-gravity-tau-s must be finite and positive")
    if args.publish_fps <= 0:
        parser.error("--publish-fps must be positive")
    if not 0 <= args.depth_png_compression <= 9:
        parser.error("--depth-png-compression must be in [0, 9]")
    if args.send_timeout <= 0:
        parser.error("--send-timeout must be positive")
    if not math.isfinite(args.max_frame_age_s) or args.max_frame_age_s <= 0.:
        parser.error("--max-frame-age-s must be finite and positive")
    if args.send_buffer_kb < 32:
        parser.error("--send-buffer-kb must be at least 32")
    if args.sensor_queue_capacity < 8:
        parser.error("--sensor-queue-capacity must be at least 8")
    try:
        asyncio.run(publish(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
