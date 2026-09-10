#!/usr/bin/env python3
"""Direct Go2 RGB-D WebSocket to ROS bridge.

The sensor socket and ROS publishers are independent from the dashboard. The
dashboard may receive a latest-only mirror, but it is never on the sensor path.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
import math
import os
import secrets
import sys
import threading
import time
import urllib.request
from typing import Any

import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header, String
import tf2_ros
from PIL import Image as PILImage

try:
    import cv2
except Exception:  # pragma: no cover - minimal ROS/replay installs may omit OpenCV
    cv2 = None

from physical_protocol import decode_wire_packet, sample_clock_pair


def _attach_transport_metadata(
    telemetry: dict[str, Any],
    session: Any = None,
    connection: Any = None,
) -> dict[str, Any]:
    """Copy telemetry while retaining the private source-generation markers.

    RGB-D packets already carry these markers.  Telemetry-only receipts must
    carry them too, otherwise a persistent web gateway can reject a fresh
    post-restart pose because its source timestamp/sequence reset.  The
    markers are additive and are removed by ``RuntimeState`` before public
    snapshots are returned.
    """

    payload = dict(telemetry) if isinstance(telemetry, dict) else {}
    if session not in (None, ""):
        payload["_transport_session"] = str(session)
        try:
            payload["_transport_connection"] = int(connection)
        except (TypeError, ValueError):
            # A session token alone is still useful for generation resets;
            # retain compatibility with callers that do not know the socket
            # connection counter.
            pass
    return payload


class _SourceEpochMapper:
    """Estimate the Go2-wall to local-ROS-wall offset from small packets.

    Camera receipts are deliberately *not* observations: their age includes
    exposure, codecs, buffering and network transfer and must remain visible to
    the mapper/GMapping stale checks.  The hello and telemetry envelopes are
    stamped immediately before their small WebSocket send, so their one-way
    arrival offset is a bounded estimate of the two hosts' wall-clock offset.
    Three ordered probes establish the lower envelope before it is usable.
    Later telemetry refines the recent lower envelope at an oscillator-scale
    bounded rate while retaining the connection-lifetime minimum as a stable
    diagnostic baseline.  A delayed first hello
    therefore cannot alone define the epoch, while one transient timing sample
    cannot make ROS timestamps jump by hundreds of milliseconds between
    adjacent 10 Hz captures.  ``project`` is a pure read, avoiding a race
    between the ROS publisher thread and the asyncio telemetry handler.
    """

    def __init__(
        self,
        window_s: float = 30.,
        *,
        min_observations: int = 3,
        max_adjust_rate_s_per_s: float = 500e-6,
        anchor_positive_gate_s: float = .02,
        host_clock_step_tolerance_s: float = .05,
        source_clock_step_tolerance_s: float = .05,
        legacy_interval_tolerance_s: float = 1.,
        min_observation_span_s: float = .02,
        clock_pair_max_span_s: float = .005,
    ) -> None:
        self._window_s = max(1., float(window_s))
        self._min_observations = max(2, int(min_observations))
        self._max_adjust_rate_s_per_s = max(
            0., float(max_adjust_rate_s_per_s)
        )
        self._anchor_positive_gate_s = max(0., float(anchor_positive_gate_s))
        self._host_clock_step_tolerance_s = max(
            .05, float(host_clock_step_tolerance_s)
        )
        self._source_clock_step_tolerance_s = max(
            .05, float(source_clock_step_tolerance_s)
        )
        self._legacy_interval_tolerance_s = max(
            .1, float(legacy_interval_tolerance_s)
        )
        self._min_observation_span_s = max(
            0., float(min_observation_span_s)
        )
        self._clock_pair_max_span_s = max(
            1e-6, float(clock_pair_max_span_s)
        )
        self._lock = threading.Lock()
        self._generation: tuple[str | None, int] | None = None
        self._retired_generations: set[tuple[str | None, int]] = set()
        self._samples: deque[tuple[float, float]] = deque()
        self._minimum_offset_s: float | None = None
        self._minimum_offset_monotonic: float | None = None
        self._selected_offset_s: float | None = None
        self._ready = False
        self._last_adjust_monotonic: float | None = None
        self._last_observation: tuple[float, float, float, float | None] | None = None
        self._observation_count = 0
        self._observation_started_monotonic: float | None = None
        self._total_observation_count = 0
        self._invalid_count = 0
        self._stale_generation_count = 0
        self._out_of_order_count = 0
        self._mapping_epoch = 0
        self._clock_step_count = 0
        self._legacy_interval_suspect = False
        self._source_phase_suspect = False
        self._source_phase_candidate_s: float | None = None
        self._source_phase_candidate_count = 0
        self._source_phase_consistency_s = .005
        self._source_step_deferred_count = 0
        self._host_phase_suspect = False
        self._host_phase_candidate_s: float | None = None
        self._host_phase_candidate_count = 0
        self._host_phase_consistency_s = .005
        self._host_step_deferred_count = 0
        self._clock_pair_rejected_count = 0
        self._last_clock_pair_rejection = ""
        self._clock_pair_suspect = False
        self._last_anchor_monotonic: float | None = None
        self._holdover_since_monotonic: float | None = None
        self._rejected_anchor_count = 0
        self._lower_reanchor_count = 0
        self._observed_residual_s: float | None = None
        self._clock_anchors: deque[tuple[int, float, float]] = deque(maxlen=128)
        self._estimated_rate_s_per_s = 0.
        self._anchor_bucket_s = 2.
        self._rate_min_span_s = 60.
        self._last_reset_reason = "uninitialized"

    def _record_clock_anchor_locked(self, monotonic: float, offset: float) -> None:
        """Retain a sparse low-delay envelope and estimate bounded drift."""

        bucket = int(monotonic // self._anchor_bucket_s)
        if self._clock_anchors and self._clock_anchors[-1][0] == bucket:
            previous = self._clock_anchors[-1]
            if offset < previous[2]:
                self._clock_anchors[-1] = (bucket, monotonic, offset)
            return
        else:
            self._clock_anchors.append((bucket, monotonic, offset))
        if len(self._clock_anchors) < 2:
            return
        _latest_bucket, latest_time, latest_offset = self._clock_anchors[-1]
        slopes = [
            (latest_offset - earlier_offset) / (latest_time - earlier_time)
            for _bucket, earlier_time, earlier_offset in self._clock_anchors
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
        self._estimated_rate_s_per_s = max(
            -limit, min(limit, median_slope)
        )

    @staticmethod
    def _valid_number(value: Any, *, positive: bool = False) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(number) or (positive and number <= 0.):
            return None
        return number

    def observe(
        self,
        source_send_wall: Any,
        received_wall: Any,
        received_monotonic: Any,
        generation: tuple[str | None, int],
        source_send_monotonic: Any = None,
        source_clock_pair_span_s: Any = None,
        received_clock_pair_span_s: Any = None,
    ) -> dict[str, Any]:
        source = self._valid_number(source_send_wall, positive=True)
        received = self._valid_number(received_wall, positive=True)
        monotonic = self._valid_number(received_monotonic, positive=True)
        source_monotonic = self._valid_number(
            source_send_monotonic, positive=True,
        )
        source_pair_span = self._valid_number(source_clock_pair_span_s)
        received_pair_span = self._valid_number(received_clock_pair_span_s)
        with self._lock:
            if source is None or received is None or monotonic is None:
                self._invalid_count += 1
                return self._snapshot_locked()
            for label, raw_value, pair_span in (
                ("source", source_clock_pair_span_s, source_pair_span),
                ("receiver", received_clock_pair_span_s, received_pair_span),
            ):
                # Missing fields are an explicit protocol-v1 compatibility
                # path. A peer that advertises pair quality must not have a
                # preempted/invalid sample admitted to the clock envelope.
                if raw_value is not None and (
                    pair_span is None
                    or pair_span < 0.
                    or pair_span > self._clock_pair_max_span_s
                ):
                    self._clock_pair_rejected_count += 1
                    self._last_clock_pair_rejection = label
                    self._clock_pair_suspect = True
                    return self._snapshot_locked()
            self._clock_pair_suspect = False
            if generation in self._retired_generations:
                self._stale_generation_count += 1
                return self._snapshot_locked()
            if generation != self._generation:
                if (
                    self._generation is not None
                    and generation[0] == self._generation[0]
                    and generation[1] < self._generation[1]
                ):
                    self._stale_generation_count += 1
                    return self._snapshot_locked()
                if self._generation is not None:
                    self._retired_generations.add(self._generation)
                self._generation = generation
                self._mapping_epoch += 1
                self._last_reset_reason = "transport_generation"
                self._samples.clear()
                self._minimum_offset_s = None
                self._minimum_offset_monotonic = None
                self._selected_offset_s = None
                self._ready = False
                self._last_adjust_monotonic = monotonic
                self._last_observation = None
                self._observation_count = 0
                self._observation_started_monotonic = None
                self._legacy_interval_suspect = False
                self._source_phase_suspect = False
                self._source_phase_candidate_s = None
                self._source_phase_candidate_count = 0
                self._host_phase_suspect = False
                self._host_phase_candidate_s = None
                self._host_phase_candidate_count = 0
                self._clock_pair_suspect = False
                self._last_anchor_monotonic = None
                self._holdover_since_monotonic = None
                self._observed_residual_s = None
                self._clock_anchors.clear()
                self._estimated_rate_s_per_s = 0.
            elif self._last_observation is not None:
                (
                    last_source,
                    last_received,
                    last_monotonic,
                    last_source_monotonic,
                ) = self._last_observation
                monotonic_elapsed = monotonic - last_monotonic
                if monotonic_elapsed < 0.:
                    self._out_of_order_count += 1
                    return self._snapshot_locked()
                source_elapsed = source - last_source
                received_elapsed = received - last_received
                reset_reason = None
                host_phase = received - monotonic
                last_host_phase = last_received - last_monotonic
                if abs(host_phase - last_host_phase) > self._host_clock_step_tolerance_s:
                    # A local wall step is confirmed twice just like a source
                    # step. One corrupt clock read must gate one observation,
                    # not discard a healthy mapping epoch.
                    if (
                        self._host_phase_candidate_s is not None
                        and abs(
                            host_phase - self._host_phase_candidate_s
                        ) <= self._host_phase_consistency_s
                    ):
                        self._host_phase_candidate_count += 1
                    else:
                        self._host_phase_candidate_s = host_phase
                        self._host_phase_candidate_count = 1
                    self._host_phase_suspect = True
                    self._host_step_deferred_count += 1
                    if self._host_phase_candidate_count >= 2:
                        reset_reason = "host_wall_step"
                    else:
                        return self._snapshot_locked()
                else:
                    self._host_phase_suspect = False
                    self._host_phase_candidate_s = None
                    self._host_phase_candidate_count = 0
                if reset_reason is None and (
                    source_monotonic is not None
                    and last_source_monotonic is not None
                ):
                    source_phase = source - source_monotonic
                    last_source_phase = last_source - last_source_monotonic
                    phase_discontinuous = bool(
                        source_monotonic < last_source_monotonic
                        or abs(source_phase - last_source_phase)
                        > self._source_clock_step_tolerance_s
                    )
                    if phase_discontinuous:
                        # Wall and monotonic are sampled on the Go2, so HOL
                        # delay cancels.  Still require two consistent packets
                        # before resetting: one corrupt stamp must only pause
                        # world geometry for a single observation.
                        if (
                            self._source_phase_candidate_s is not None
                            and abs(
                                source_phase - self._source_phase_candidate_s
                            ) <= self._source_phase_consistency_s
                        ):
                            self._source_phase_candidate_count += 1
                        else:
                            self._source_phase_candidate_s = source_phase
                            self._source_phase_candidate_count = 1
                        self._source_phase_suspect = True
                        self._source_step_deferred_count += 1
                        if self._source_phase_candidate_count >= 2:
                            reset_reason = "source_wall_step"
                        else:
                            return self._snapshot_locked()
                    else:
                        self._source_phase_suspect = False
                        self._source_phase_candidate_s = None
                        self._source_phase_candidate_count = 0
                        self._legacy_interval_suspect = False
                elif reset_reason is None and source_elapsed < -0.05:
                    # A legacy sender has no independent clock with which to
                    # validate the rollback.  Preserve the historical direct
                    # recovery for an explicit backwards wall timestamp.
                    reset_reason = "source_wall_step"
                elif reset_reason is None and (
                    (source_monotonic is None or last_source_monotonic is None)
                    and
                    abs(source_elapsed - monotonic_elapsed)
                    > self._legacy_interval_tolerance_s
                ):
                    # Legacy peers do not expose their monotonic clock, so a
                    # source/local interval mismatch cannot be distinguished
                    # from TCP head-of-line delay.  Fail closed for this probe
                    # and retain the established epoch; do not guess a clock
                    # step from one-way arrival timing.  A normal later probe
                    # clears the suspect state, while a real forward wall step
                    # remains gated until reconnect/upgrade.
                    self._legacy_interval_suspect = True
                    self._source_step_deferred_count += 1
                    return self._snapshot_locked()
                else:
                    self._legacy_interval_suspect = False
                    self._source_phase_suspect = False
                    self._source_phase_candidate_s = None
                    self._source_phase_candidate_count = 0
                if reset_reason is not None:
                    # A wall-clock step invalidates the selected offset even
                    # though the TCP connection itself is unchanged.  Start a
                    # fresh three-probe epoch; sensor receipts remain excluded.
                    self._mapping_epoch += 1
                    self._clock_step_count += 1
                    self._last_reset_reason = reset_reason
                    self._samples.clear()
                    self._minimum_offset_s = None
                    self._minimum_offset_monotonic = None
                    self._selected_offset_s = None
                    self._ready = False
                    self._last_adjust_monotonic = monotonic
                    self._observation_count = 0
                    self._observation_started_monotonic = None
                    self._legacy_interval_suspect = False
                    self._source_phase_suspect = False
                    self._source_phase_candidate_s = None
                    self._source_phase_candidate_count = 0
                    self._host_phase_suspect = False
                    self._host_phase_candidate_s = None
                    self._host_phase_candidate_count = 0
                    self._clock_pair_suspect = False
                    self._last_anchor_monotonic = None
                    self._holdover_since_monotonic = None
                    self._observed_residual_s = None
                    self._clock_anchors.clear()
                    self._estimated_rate_s_per_s = 0.
            offset = received - source
            if self._observation_started_monotonic is None:
                self._observation_started_monotonic = monotonic
            self._samples.append((monotonic, offset))
            cutoff = monotonic - self._window_s
            while self._samples and self._samples[0][0] < cutoff:
                self._samples.popleft()
            self._observation_count += 1
            self._total_observation_count += 1
            if self._minimum_offset_s is None or offset < self._minimum_offset_s:
                self._minimum_offset_s = offset
                self._minimum_offset_monotonic = monotonic
            minimum = self._minimum_offset_s
            recent_minimum = min(sample[1] for sample in self._samples)
            if self._selected_offset_s is None or not self._ready:
                self._selected_offset_s = minimum
                observation_span_s = max(
                    0., monotonic - self._observation_started_monotonic,
                )
                self._ready = (
                    self._observation_count >= self._min_observations
                    and observation_span_s + 1e-9
                    >= self._min_observation_span_s
                )
                if self._ready:
                    self._last_anchor_monotonic = monotonic
                    self._record_clock_anchor_locked(monotonic, recent_minimum)
            else:
                elapsed = max(
                    0., monotonic - float(self._last_adjust_monotonic or monotonic)
                )
                allowance = elapsed * self._max_adjust_rate_s_per_s
                # The lifetime minimum alone cannot follow relative oscillator
                # drift in the positive direction.  Follow the recent lower
                # envelope in both directions, but only at a ppm-scale rate so
                # minutes of network backlog remain visible rather than being
                # mistaken for clock motion.
                difference = recent_minimum - self._selected_offset_s
                if difference < 0.:
                    # A smaller one-way lower envelope removes avoidable
                    # startup/network delay. Correct it immediately; a ppm
                    # limiter here can retain a 200 ms bias for minutes.
                    self._selected_offset_s = recent_minimum
                    self._lower_reanchor_count += 1
                    self._last_anchor_monotonic = monotonic
                    self._holdover_since_monotonic = None
                    self._record_clock_anchor_locked(monotonic, recent_minimum)
                elif difference <= self._anchor_positive_gate_s:
                    self._selected_offset_s += max(
                        -allowance, min(allowance, difference)
                    )
                    self._last_anchor_monotonic = monotonic
                    self._holdover_since_monotonic = None
                    self._record_clock_anchor_locked(monotonic, recent_minimum)
                else:
                    # A sudden positive residual is indistinguishable from
                    # extra one-way queueing, and must not become clock drift.
                    # Hold the already established clock model until a future
                    # low-delay probe returns near its envelope.
                    self._rejected_anchor_count += 1
                    if self._holdover_since_monotonic is None:
                        self._holdover_since_monotonic = monotonic
                    # Continue the previously learned oscillator drift during
                    # backlog instead of freezing the clock.  No queued sample
                    # is admitted as a new anchor in this holdover path.
                    self._selected_offset_s += max(
                        -allowance,
                        min(
                            allowance,
                            self._estimated_rate_s_per_s * elapsed,
                        ),
                    )
            self._observed_residual_s = offset - self._selected_offset_s
            self._last_adjust_monotonic = monotonic
            self._last_observation = (
                source, received, monotonic, source_monotonic,
            )
            return self._snapshot_locked()

    def project(
        self, source_wall: Any, generation: tuple[str | None, int],
    ) -> tuple[float, dict[str, Any]]:
        source = self._valid_number(source_wall, positive=True)
        with self._lock:
            valid = (
                source is not None
                and generation == self._generation
                and self._selected_offset_s is not None
                and self._ready
                and not self._legacy_interval_suspect
                and not self._source_phase_suspect
                and not self._host_phase_suspect
                and not self._clock_pair_suspect
            )
            if not valid:
                info = self._snapshot_locked()
                info.update(valid=False, projected_wall=source)
                return float(source) if source is not None else float("nan"), info
            projected = source + float(self._selected_offset_s)
            info = self._snapshot_locked()
            info.update(valid=True, projected_wall=projected)
            return projected, info

    def _snapshot_locked(self) -> dict[str, Any]:
        recent_minimum = (
            min(sample[1] for sample in self._samples)
            if self._samples else None
        )
        minimum = self._minimum_offset_s
        selected = self._selected_offset_s
        latest_monotonic = (
            self._last_observation[2] if self._last_observation is not None else None
        )
        baseline_age_s = (
            max(0., latest_monotonic - self._minimum_offset_monotonic)
            if latest_monotonic is not None
            and self._minimum_offset_monotonic is not None
            else None
        )
        selected_from_baseline_s = (
            selected - minimum
            if selected is not None and minimum is not None else None
        )
        last_anchor_age_s = (
            max(0., latest_monotonic - self._last_anchor_monotonic)
            if latest_monotonic is not None
            and self._last_anchor_monotonic is not None
            else None
        )
        holdover_age_s = (
            max(0., latest_monotonic - self._holdover_since_monotonic)
            if latest_monotonic is not None
            and self._holdover_since_monotonic is not None
            else 0.
        )
        anchor_span_s = (
            max(0., self._clock_anchors[-1][1] - self._clock_anchors[0][1])
            if len(self._clock_anchors) >= 2 else 0.
        )
        observation_span_s = (
            max(0., latest_monotonic - self._observation_started_monotonic)
            if latest_monotonic is not None
            and self._observation_started_monotonic is not None
            else 0.
        )
        return {
            "version": 1,
            "generation": self._generation,
            "valid": (
                self._ready
                and selected is not None
                and not self._legacy_interval_suspect
                and not self._source_phase_suspect
                and not self._host_phase_suspect
                and not self._clock_pair_suspect
            ),
            "selected_offset_s": selected,
            "minimum_observed_offset_s": minimum,
            "recent_minimum_observed_offset_s": recent_minimum,
            "minimum_offset_baseline_age_s": baseline_age_s,
            "selected_minus_minimum_s": (
                selected_from_baseline_s
            ),
            "selected_baseline_drift_ppm": (
                selected_from_baseline_s / baseline_age_s * 1e6
                if selected_from_baseline_s is not None
                and baseline_age_s is not None and baseline_age_s > 0.
                else None
            ),
            "anchor_positive_gate_s": self._anchor_positive_gate_s,
            "last_anchor_age_s": last_anchor_age_s,
            "holdover_age_s": holdover_age_s,
            "rejected_anchor_count": self._rejected_anchor_count,
            "lower_reanchor_count": self._lower_reanchor_count,
            "observed_residual_s": self._observed_residual_s,
            "recent_envelope_residual_s": (
                recent_minimum - selected
                if recent_minimum is not None and selected is not None
                else None
            ),
            "estimated_queue_excess_s": max(
                0., float(self._observed_residual_s or 0.)
            ),
            "estimated_rate_ppm": self._estimated_rate_s_per_s * 1e6,
            "trusted_anchor_count": len(self._clock_anchors),
            "anchor_span_s": anchor_span_s,
            "observation_count": self._observation_count,
            "total_observation_count": self._total_observation_count,
            "min_observations": self._min_observations,
            "observation_span_s": observation_span_s,
            "min_observation_span_s": self._min_observation_span_s,
            "invalid_count": self._invalid_count,
            "stale_generation_count": self._stale_generation_count,
            "out_of_order_count": self._out_of_order_count,
            "mapping_epoch": self._mapping_epoch,
            "clock_step_count": self._clock_step_count,
            "legacy_interval_suspect": self._legacy_interval_suspect,
            "source_phase_suspect": self._source_phase_suspect,
            "source_phase_candidate_count": self._source_phase_candidate_count,
            "host_phase_suspect": self._host_phase_suspect,
            "host_phase_candidate_count": self._host_phase_candidate_count,
            "host_clock_step_tolerance_s": self._host_clock_step_tolerance_s,
            "source_clock_step_tolerance_s": self._source_clock_step_tolerance_s,
            "legacy_interval_tolerance_s": self._legacy_interval_tolerance_s,
            "source_step_deferred_count": self._source_step_deferred_count,
            "host_step_deferred_count": self._host_step_deferred_count,
            "clock_pair_max_span_s": self._clock_pair_max_span_s,
            "clock_pair_rejected_count": self._clock_pair_rejected_count,
            "last_clock_pair_rejection": self._last_clock_pair_rejection,
            "clock_pair_suspect": self._clock_pair_suspect,
            "last_reset_reason": self._last_reset_reason,
            "ready": self._ready,
            "phase": (
                "clock_pair_rejected"
                if self._clock_pair_suspect
                else (
                "legacy_interval_suspect"
                if self._legacy_interval_suspect
                else (
                    "source_phase_suspect"
                    if self._source_phase_suspect
                    else (
                        "host_phase_suspect"
                        if self._host_phase_suspect
                        else ("ready" if self._ready else "warming")
                    )
                )
                )
            ),
            "max_adjust_rate_s_per_s": self._max_adjust_rate_s_per_s,
            "max_adjust_rate_ppm": self._max_adjust_rate_s_per_s * 1e6,
            "method": "small_packet_one_way_offset",
            "one_way_delay_included": True,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()


def _patch_roslogging_findcaller_for_py311() -> None:
    """Avoid rosgraph's Python 3.11 ``findCaller`` recursion.

    Under the Conda 3.11 interpreter ``rospy.init_node`` otherwise busy-loops
    inside ``rosgraph.roslogging.RospyLogger.findCaller`` before registration,
    consuming a core and never publishing. Mirrors the gateway workaround so
    this bridge can run with the Conda interpreter (required for ``websockets``).
    """
    if sys.version_info < (3, 11):
        return
    try:
        import rosgraph.roslogging as roslogging
    except Exception:
        return
    if getattr(roslogging.RospyLogger.findCaller, "_physical_sensor_safe", False):
        return

    def _safe_find_caller(self, *args, **kwargs):
        result = logging.Logger.findCaller(self, *args, **kwargs)
        if len(result) == 4:
            result = result[:3]
        return result

    _safe_find_caller._physical_sensor_safe = True
    roslogging.RospyLogger.findCaller = _safe_find_caller


def _decode(value: Any, prefer_bgr: bool = False) -> np.ndarray:
    encoding = ""
    if isinstance(value, dict):
        encoding = str(value.get("encoding", "") or "").casefold()
        value = value.get("data", "")
    payload = base64.b64decode(value)
    if cv2 is not None and encoding:
        # OpenCV's JPEG decoder is materially faster than PIL for the
        # 1280x720 stream. The public helper keeps its historical RGB default;
        # the live ROS path requests BGR directly because Image.encoding is
        # ``bgr8`` and YOLO/ROS would otherwise pay a second full-frame swap.
        flags = cv2.IMREAD_UNCHANGED if "png" in encoding else cv2.IMREAD_COLOR
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), flags)
        if decoded is not None:
            if (
                flags == cv2.IMREAD_COLOR
                and decoded.ndim == 3
                and not prefer_bgr
            ):
                decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            return np.asarray(decoded)
    decoded = np.asarray(PILImage.open(io.BytesIO(payload)))
    if prefer_bgr and decoded.ndim == 3 and decoded.shape[2] >= 3:
        return np.ascontiguousarray(decoded[:, :, :3][:, :, ::-1])
    return decoded


def _sample_depth_points(
    depth: np.ndarray,
    intrinsics: dict[str, Any],
    stride: int,
    scale: float,
    max_depth_m: float,
    no_return_depth_m: float,
    *,
    projection_cache: tuple[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, tuple[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] | None]:
    """Project one native-depth image into a compact XYZ array.

    The camera intrinsics and sampling stride are constant for a stream, but
    the old implementation rebuilt row/column arrays and multiplied both
    coordinates for every frame.  The returned cache contains the two static
    ray coefficients; each frame only converts the valid depth samples and
    applies the range sentinel.  Keeping this helper independent of ROS also
    makes its latest-frame/geometry contract easy to test offline.
    """
    depth_array = np.asarray(depth)
    if depth_array.ndim != 2:
        raise ValueError(f"depth must be a 2-D array, got {depth_array.shape}")
    stride = max(1, int(stride))
    height, width = (int(value) for value in depth_array.shape)
    key = (
        height,
        width,
        stride,
        round(float(intrinsics.get("fx", 0.0)), 9),
        round(float(intrinsics.get("fy", 0.0)), 9),
        round(float(intrinsics.get("cx", 0.0)), 9),
        round(float(intrinsics.get("cy", 0.0)), 9),
    )
    if projection_cache is not None and projection_cache[0] == key:
        cached_key, (x_coefficients, y_coefficients) = projection_cache
    else:
        fx = float(intrinsics.get("fx", 0.0))
        fy = float(intrinsics.get("fy", 0.0))
        if fx <= 0.0 or fy <= 0.0:
            return np.empty((0, 3), dtype=np.float32), None
        rows = np.arange(0, height, stride, dtype=np.float32)
        cols = np.arange(0, width, stride, dtype=np.float32)
        # Broadcast once, then retain flat contiguous coefficients.  A
        # 848x480 stream at stride 6 has only ~11k rays, so this cache is
        # small while eliminating two per-frame meshgrid allocations.
        x_row = ((cols - float(intrinsics.get("cx", 0.0))) / fx)[None, :]
        y_col = ((rows - float(intrinsics.get("cy", 0.0))) / fy)[:, None]
        x_coefficients = np.broadcast_to(
            x_row, (rows.size, cols.size)
        ).reshape(-1).copy()
        y_coefficients = np.broadcast_to(
            y_col, (rows.size, cols.size)
        ).reshape(-1).copy()
        cached_key = key
        projection_cache = (cached_key, (x_coefficients, y_coefficients))

    sampled = np.asarray(depth_array[::stride, ::stride], dtype=np.float32)
    flat = sampled.reshape(-1)
    depth_scale = float(scale)
    if not math.isfinite(depth_scale) or depth_scale <= 0.0:
        return np.empty((0, 3), dtype=np.float32), projection_cache
    valid_indices = np.flatnonzero(flat > 0.0)
    if valid_indices.size == 0:
        return np.empty((0, 3), dtype=np.float32), projection_cache
    # Convert the native integer units before applying the metre-based range
    # gate.  D435 depth is normally millimetres (scale=0.001), but replay
    # fixtures may provide 32FC1 metres.
    z = flat[valid_indices].copy() * depth_scale
    max_depth = max(0.0, float(max_depth_m))
    no_return = max(max_depth + 1e-3, float(no_return_depth_m))
    if max_depth > 0.0:
        z[z > max_depth] = no_return
    points = np.empty((valid_indices.size, 3), dtype=np.float32)
    points[:, 0] = x_coefficients[valid_indices] * z
    points[:, 1] = y_coefficients[valid_indices] * z
    points[:, 2] = z
    return points, projection_cache


def _insert_capture_telemetry(
    history,
    stamp: float,
    snapshot: dict[str, Any],
) -> None:
    """Insert one pose sample in timestamp order while retaining its bound.

    Capture packets are decoded on a different thread from the small telemetry
    envelopes, so a valid older capture sample can arrive after a newer live
    pose. A plain ``deque.append`` would either lose causal ordering or evict
    the wrong end of a full history. Equal timestamps describe the same pose
    instant; retain the latest receipt without consuming another history slot.
    """

    sample_stamp = float(stamp)
    entries = [
        (float(existing_stamp), existing_snapshot)
        for existing_stamp, existing_snapshot in history
        if float(existing_stamp) != sample_stamp
    ]
    entries.append((sample_stamp, dict(snapshot)))
    entries.sort(key=lambda item: item[0])
    maxlen = getattr(history, "maxlen", None)
    if maxlen is not None:
        maxlen = int(maxlen)
        entries = entries[-maxlen:] if maxlen > 0 else []
    history.clear()
    history.extend(entries)


def _nearest_capture_telemetry(history, stamp, max_delta_sec=0.15):
    """Select the latest causal pose; never wait or use a future sample."""
    try:
        capture = float(stamp)
        limit = float(max_delta_sec)
    except (TypeError, ValueError, OverflowError):
        return {}, None
    if not math.isfinite(capture) or capture <= 0. or not math.isfinite(limit) or limit < 0.:
        return {}, None
    best = None
    best_stamp = -float("inf")
    best_index = -1
    # Callback fixtures are not required to be pre-sorted. At an identical
    # timestamp the later receipt wins, matching duplicate replacement in the
    # production history.
    for index, item in enumerate(history):
        try:
            sample_stamp, snapshot = item
            sample_stamp = float(sample_stamp)
        except (TypeError, ValueError, OverflowError):
            continue
        if (
            not isinstance(snapshot, dict)
            or not math.isfinite(sample_stamp)
            or sample_stamp <= 0.
            or sample_stamp > capture
        ):
            continue
        if sample_stamp > best_stamp or (
            sample_stamp == best_stamp and index > best_index
        ):
            best_stamp = sample_stamp
            best_index = index
            best = snapshot
    if best is None:
        return {}, None
    delta = capture - best_stamp
    if delta > limit:
        return {}, delta
    return dict(best), delta


def _telemetry_body_position(telemetry: Any) -> tuple[float, float, float] | None:
    if not isinstance(telemetry, dict):
        return None
    try:
        position = np.asarray(telemetry.get("position"), dtype=float)
        if position.ndim != 1 or position.size < 3 or not np.isfinite(position[:3]).all():
            return None
        return tuple(float(value) for value in position[:3])
    except (TypeError, ValueError, OverflowError):
        return None


def _telemetry_body_quaternion(telemetry: Any) -> tuple[float, float, float, float] | None:
    """One normalized base-link orientation for sensor TF and embedded boxes.

    Unitree IMU lists are wxyz; explicit mappings and other body-pose lists
    are xyzw. Camera pose/IMU fields cannot substitute for chassis orientation.
    Missing/invalid orientation stays missing, rather than becoming identity.
    """
    if not isinstance(telemetry, dict):
        return None
    for name in ("imu", "base_pose", "odom", "pose"):
        source = telemetry.get(name)
        if not isinstance(source, dict):
            continue
        for field in ("quaternion", "orientation"):
            raw = source.get(field)
            try:
                if isinstance(raw, dict):
                    values = [float(raw[key]) for key in ("x", "y", "z", "w")]
                elif isinstance(raw, (list, tuple, np.ndarray)) and len(raw) >= 4:
                    values = [float(value) for value in raw[:4]]
                    if name == "imu":
                        values = [values[1], values[2], values[3], values[0]]
                else:
                    continue
                norm = math.hypot(*values)
                if math.isfinite(norm) and norm > 1e-6:
                    return tuple(value / norm for value in values)
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
    try:
        yaw = float(telemetry["yaw"] if "yaw" in telemetry else telemetry["imu"]["rpy"][2])
        if math.isfinite(yaw):
            return (0., 0., math.sin(yaw / 2.), math.cos(yaw / 2.))
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        pass
    return None


def _quat_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def _quat_multiply(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def _quat_normalize(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Return a finite unit quaternion (xyzw), or identity on bad input."""
    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        return (0.0, 0.0, 0.0, 1.0)
    norm = float(np.linalg.norm(values))
    if norm <= 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(float(value / norm) for value in values)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Interpret ROS/CLI bool parameters without treating ``"false"`` as true."""
    if value is None:
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "no", "off", "none"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


def _camera_imu_capture_match(telemetry, source_stamp, *, enabled=True, use_yaw=False, max_delta_sec=.15):
    """Check component receipts in the Go2 source clock, not host arrival time."""
    result = {"status": "unavailable", "accepted": True, "ages_sec": {},
              "continuity": {}, "max_delta_sec": max_delta_sec,
              "time_basis": "none"}
    if not enabled:
        result["status"] = "disabled"
        return result
    motion = telemetry.get("camera_imu") if isinstance(telemetry, dict) else None
    if not isinstance(motion, dict):
        return result
    if _coerce_bool(motion.get("imu_calibrating", False), False):
        result["status"] = "calibrating"
        return result
    if "component_received_at" not in motion:
        # Old recordings do not prove independent accel/gyro freshness.
        result["status"] = "legacy_unverified"
        return result
    required_components = ("accel", "gyro") if use_yaw else ("accel",)
    if "imu_ingress" in motion:
        try:
            streams = motion["imu_ingress"]["streams"]
            for component in required_components:
                health = streams[component]
                result["continuity"][component] = {
                    key: health.get(key)
                    for key in (
                        "healthy", "continuous_s", "effective_hz", "missing",
                        "duplicates", "late", "timestamp_regressions",
                    )
                }
                if health.get("healthy") is not True:
                    result.update(status="discontinuous", accepted=False)
                    return result
        except (TypeError, KeyError):
            result.update(status="invalid_continuity", accepted=False)
            return result
    time_field = "component_received_at"
    sample_times = motion.get("component_sample_at")
    sample_valid = motion.get("component_sample_time_valid")
    if (
        isinstance(sample_times, dict)
        and isinstance(sample_valid, dict)
        and all(sample_valid.get(component) is True for component in required_components)
    ):
        time_field = "component_sample_at"
        result["time_basis"] = "device_clock_mapped_sample"
    else:
        result["time_basis"] = "source_receipt"
    result.update(status="invalid_time", accepted=False)
    try:
        reference = float(source_stamp)
        limit = float(max_delta_sec)
        if not math.isfinite(reference) or reference <= 0. or not math.isfinite(limit) or limit < 0.:
            return result
        times = motion[time_field]
        for component in required_components:
            value = float(times[component])
            if not math.isfinite(value) or value <= 0.:
                return result
            result["ages_sec"][component] = reference - value
        if any(abs(age) > limit + 1e-9 for age in result["ages_sec"].values()):
            result["status"] = "stale"
        else:
            result.update(status="matched", accepted=True)
    except (TypeError, KeyError, ValueError, OverflowError):
        pass
    return result


def _camera_imu_correction_quaternion(
    telemetry: dict[str, Any] | None,
    *,
    enabled: bool = True,
    use_yaw: bool = False,
    max_roll_rad: float = 0.7,
    max_pitch_rad: float = 0.7,
    max_yaw_rad: float = 0.35,
) -> tuple[tuple[float, float, float, float], bool]:
    """Read a calibrated D435i correction and return ``(q, valid)``.

    ``go2_readonly_sensor_bridge`` publishes ``camera_imu.correction_rpy``
    after its gravity/gyro reference has converged.  The values are Euler
    angles in the D435/depth optical axes (x=right, y=down, z=forward), not
    Go2 ``base_link`` axes.  The caller converts this sensor-frame rotation to
    the parent frame before applying it to TF.  During calibration, with
    missing data, or for an implausible jump we deliberately return identity;
    this preserves the existing static extrinsic instead of injecting a bad
    TF into SLAM.

    Yaw is disabled by default because the D435i gyro yaw is relative and has
    no absolute heading reference.  Roll/pitch are gravity-observable and are
    therefore safe to use for pole sway correction once calibrated.
    """
    identity = (0.0, 0.0, 0.0, 1.0)
    if not enabled or not isinstance(telemetry, dict):
        return identity, False
    camera_imu = telemetry.get("camera_imu")
    if not isinstance(camera_imu, dict) or _coerce_bool(
        camera_imu.get("imu_calibrating", False), False
    ):
        return identity, False
    values = camera_imu.get("correction_rpy")
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        return identity, False
    try:
        roll, pitch, yaw = (float(value) for value in values[:3])
    except (TypeError, ValueError):
        return identity, False
    if not all(math.isfinite(value) for value in (roll, pitch, yaw)):
        return identity, False
    limits = (max(0.0, float(max_roll_rad)), max(0.0, float(max_pitch_rad)),
              max(0.0, float(max_yaw_rad)))
    # Ignore the relative gyro yaw entirely unless explicitly requested.  A
    # long run can accumulate yaw drift far beyond ``max_yaw_rad``; that must
    # not invalidate otherwise useful gravity roll/pitch correction.
    if abs(roll) > limits[0] or abs(pitch) > limits[1] or (
        use_yaw and abs(yaw) > limits[2]
    ):
        return identity, False
    if not use_yaw:
        yaw = 0.0
    return _quat_normalize(_quat_rpy(roll, pitch, yaw)), True


def _camera_imu_parent_correction_quaternion(
    telemetry: dict[str, Any] | None,
    mount_quaternion: tuple[float, float, float, float],
    **kwargs: Any,
) -> tuple[tuple[float, float, float, float], bool]:
    """Convert a D435-frame correction into the TF parent frame.

    ``mount_quaternion`` maps the nominal camera optical frame into the TF
    parent (``base_link``).  A rotation measured around D435 axes must be
    conjugated by that nominal mount before it can be pre-multiplied onto the
    parent->camera transform:

    ``R_parent = R_mount R_d435 R_mountᵀ``.

    This is the axis substitution that is easy to miss when the camera is
    mounted with the REP-103 optical frame (camera roll maps mostly to Go2
    base yaw/side axes, and camera pitch does not map one-to-one to base
    pitch).  Invalid/calibrating samples retain identity.
    """
    sensor_quaternion, valid = _camera_imu_correction_quaternion(
        telemetry, **kwargs
    )
    if not valid:
        return (0.0, 0.0, 0.0, 1.0), False
    mount = _quat_normalize(mount_quaternion)
    inverse_mount = (-mount[0], -mount[1], -mount[2], mount[3])
    parent_quaternion = _quat_multiply(
        _quat_multiply(mount, sensor_quaternion), inverse_mount
    )
    return _quat_normalize(parent_quaternion), True


def _apply_camera_imu_correction(
    mount_quaternion: tuple[float, float, float, float],
    mount_translation: tuple[float, float, float],
    correction_quaternion: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float, float], tuple[float, float, float]]:
    """Apply a parent-frame correction to a camera rigid transform.

    The correction acts on the complete parent<-camera transform (the ROS TF
    transform that maps camera coordinates into the parent), not only on its
    orientation: ``T' = T_correction * T_mount``.  In particular,
    ``t' = R_correction * t`` accounts for the horizontal displacement caused
    by the 0.98 m elevated camera when the two aluminium rails lean.
    """
    correction = _quat_normalize(correction_quaternion)
    mount = _quat_normalize(mount_quaternion)
    corrected_quaternion = _quat_normalize(_quat_multiply(correction, mount))
    corrected_translation = _quat_rotate(correction, mount_translation)
    return corrected_quaternion, corrected_translation


def _quat_rotate(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate a vector by an xyzw quaternion without ROS geometry helpers."""
    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _rotation_matrix_to_quaternion(rotation: Any) -> tuple[float, float, float, float]:
    """Convert a RealSense row-major 3x3 rotation to an xyzw quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(max(trace + 1.0, 1e-12))
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        scale = 2.0 * math.sqrt(
            max(
                1e-12,
                1.0
                + float(diagonal[index])
                - float(diagonal[next_index])
                - float(diagonal[last_index]),
            )
        )
        values = [0.0, 0.0, 0.0, 0.0]
        values[index] = 0.25 * scale
        values[3] = (matrix[last_index, next_index] - matrix[next_index, last_index]) / scale
        values[next_index] = (matrix[next_index, index] + matrix[index, next_index]) / scale
        values[last_index] = (matrix[last_index, index] + matrix[index, last_index]) / scale
        x, y, z, w = values
    values = np.asarray([x, y, z, w], dtype=np.float64)
    norm = max(float(np.linalg.norm(values)), 1e-12)
    return tuple(float(value / norm) for value in values)


def _depth_static_extrinsic(
    color_quaternion: tuple[float, float, float, float],
    color_translation: tuple[float, float, float],
    depth_frame: str,
    color_frame: str,
    extrinsics: dict[str, Any] | None,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float]]:
    """Compose base->depth from base->color and RealSense depth->color data.

    ``get_extrinsics_to(color_profile)`` returns
    ``p_color = R_dc p_depth + t_dc``.  Therefore
    ``T_base_depth = T_base_color * T_color_depth``.  Older/replay packets may
    omit the calibration; in that case retaining the color mount transform is
    the least surprising fallback.
    """
    if depth_frame == color_frame or not isinstance(extrinsics, dict):
        return color_quaternion, color_translation
    try:
        rotation = extrinsics.get("rotation")
        translation = extrinsics.get("translation")
        if rotation is None or translation is None or len(translation) < 3:
            return color_quaternion, color_translation
        q_dc = _rotation_matrix_to_quaternion(rotation)
        q_depth = _quat_multiply(color_quaternion, q_dc)
        offset = _quat_rotate(
            color_quaternion,
            (float(translation[0]), float(translation[1]), float(translation[2])),
        )
        t_depth = tuple(
            float(color_translation[index]) + float(offset[index])
            for index in range(3)
        )
        return q_depth, t_depth
    except (TypeError, ValueError, IndexError, OverflowError):
        return color_quaternion, color_translation


class SensorRosBridge:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.last_stamp = -float("inf")
        self.last_seq = -1
        # ``stamp`` is produced on the Go2 and can jump backwards when the
        # bridge/realsense process is restarted (or when NTP steps its clock).
        # Keep an internal transport generation and a host-local monotonically
        # increasing receipt sequence.  Downstream latest-only consumers use
        # these keys; the source wall clock remains available as the capture
        # timestamp for TF/message filters.
        self._transport_session = f"{os.getpid()}-{secrets.token_hex(8)}"
        self._transport_connection = 0
        self._active_transport_session: str | None = None
        self._active_transport_connection = -1
        self._retired_transport_sessions: set[str] = set()
        self._bridge_seq = 0
        self._last_published_bridge_seq = -1
        self._source_seq = -1
        # ROS consumers (TF/message_filters/GMapping) require timestamps in
        # the local ROS clock domain.  A Go2 sensor process can restart while
        # its capture clock/sequence starts over; accepting the new transport
        # generation is correct, but forwarding its lower wall stamp makes
        # every fresh cloud look stale and the mapper drops it.  Keep the
        # source stamp for ordering/diagnostics and apply an offset only after
        # a genuine rollback.  Normal traffic therefore remains byte-for-byte
        # capture stamped.
        self._stamp_generation: tuple[str | None, int] | None = None
        self._stamp_source_epoch: int | None = None
        self._stamp_offset_sec = 0.0
        self._last_source_capture_stamp = -float("inf")
        self._last_ros_capture_stamp = -float("inf")
        self._stamp_rebase_count = 0
        self._stamp_regression_tolerance_sec = 0.05
        self._source_epoch_mapper = _SourceEpochMapper()
        self._last_stamp_mapping: dict[str, Any] = {
            "source_capture_stamp": None,
            "ros_stamp": None,
            "rebased": False,
            "transport_session": None,
            "transport_connection": -1,
        }
        self._telemetry_stamp_generation: tuple[str | None, int] | None = None
        self._telemetry_stamp_source_epoch: int | None = None
        self._telemetry_stamp_offset_sec = 0.0
        self._last_telemetry_capture_stamp = -float("inf")
        self._last_telemetry_ros_stamp = -float("inf")
        self._telemetry_stamp_rebase_count = 0
        self._last_telemetry_stamp_mapping: dict[str, Any] = {
            "source_capture_stamp": None,
            "ros_stamp": None,
            "rebased": False,
            "transport_session": None,
            "transport_connection": -1,
            "stream": "telemetry",
        }
        self.telemetry: dict[str, Any] = {}
        self._telemetry_history: deque[tuple[float, dict[str, Any]]] = deque(maxlen=64)
        self._telemetry_lock = threading.Lock()
        # Prevent the 50 ms refresh timer from re-dating an unchanged pose
        # and making stale TF look current.
        self._last_tf_source_stamp = float("-inf")
        self._last_tf_ros_stamp_ns = -1
        self._published_tf_samples = {}
        self._tf_binding_floor_ns = -1
        self._tf_stamp_conflict_count = 0
        self._pose_publish_lock = threading.RLock()
        self._telemetry_ros_stamp = None
        self._telemetry_source_stamp = float("-inf")
        # RGB-D packet sequence is local to this bridge process and remains
        # monotonic even when the Go2 wall clock is stepped backwards.  Use it
        # to keep capture-time TF/odom advancing instead of freezing on a
        # stale ``received_at`` comparison.
        self._telemetry_capture_seq = -1
        self.static_frames: set[str] = set()
        # StaticTransformBroadcaster latches the last TFMessage rather than
        # accumulating individual calls. Keep the complete set so a late
        # depth-frame discovery cannot replace the colour transform.
        self._static_transforms: dict[str, TransformStamped] = {}
        self._depth_to_color_extrinsics: dict[str, Any] | None = None
        self._last_depth_frame: str | None = None
        # D435i gravity/gyro correction is applied to the complete camera
        # mount transform below.  Keep it opt-out, but enabled by default;
        # invalid or still-calibrating samples resolve to the identity.
        self._camera_imu_enabled = _coerce_bool(
            getattr(args, "camera_imu_enabled", True), True
        )
        self._camera_imu_use_yaw = _coerce_bool(
            getattr(args, "camera_imu_use_yaw", False), False
        )
        self._camera_imu_max_roll_rad = float(
            getattr(args, "camera_imu_max_roll_rad", 0.7)
        )
        self._camera_imu_max_pitch_rad = float(
            getattr(args, "camera_imu_max_pitch_rad", 0.7)
        )
        self._camera_imu_max_yaw_rad = float(
            getattr(args, "camera_imu_max_yaw_rad", 0.35)
        )
        self.rgb_pub = rospy.Publisher("/physical_nav/rgb/image_raw", Image, queue_size=1, tcp_nodelay=True)
        self.depth_pub = rospy.Publisher("/physical_nav/depth/image_raw", Image, queue_size=1, tcp_nodelay=True)
        self.info_pub = rospy.Publisher("/physical_nav/camera_info", CameraInfo, queue_size=1, latch=True)
        self.depth_info_pub = rospy.Publisher("/physical_nav/depth_camera_info", CameraInfo, queue_size=1, latch=True)
        # CameraInfo has no field for the D435 factory depth scale or the
        # depth->colour baseline. Publish those calibration values once so
        # local YOLO can lift native-depth masks without guessing identity
        # extrinsics/1 mm units.
        self.calibration_pub = rospy.Publisher(
            "/physical_nav/depth_calibration", String, queue_size=1, latch=True
        )
        self.capture_context_pub = rospy.Publisher(
            "/physical_nav/capture_context", String, queue_size=1, tcp_nodelay=True
        )
        self._last_calibration_key = None
        self.odom_pub = rospy.Publisher("/physical_nav/odom", Odometry, queue_size=1, tcp_nodelay=True)
        # YOLO and other local consumers need capture-time pose without
        # depending on the dashboard HTTP mirror.  Keep this topic latest-only
        # and publish the same timestamped telemetry that drives odometry.
        self.telemetry_pub = rospy.Publisher("/physical_nav/telemetry", String, queue_size=1, tcp_nodelay=True)
        self.cloud_pub = rospy.Publisher("/physical_nav/points", PointCloud2, queue_size=1, tcp_nodelay=True)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self._packet_lock = threading.Lock()
        self._packet_event = threading.Event()
        self._latest_packet: dict[str, Any] | None = None
        # JPEG and depth-PNG decoding are independent.  Keep this small
        # executor persistent so the ROS sensor publisher does not spend two
        # serial codec passes (and block the next 10 Hz packet) before image
        # callbacks can be delivered.
        self._decode_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="sensor-rgbd-decode"
        )
        threading.Thread(target=self._publish_loop, name="sensor-ros-publisher", daemon=True).start()
        # PointCloud2 projection is independent of RGB/depth delivery. Keep a
        # second latest-only slot so a slow cloud subscriber or NumPy packing
        # cannot delay the timestamped image callbacks consumed by YOLO.
        self._cloud_lock = threading.Lock()
        self._cloud_event = threading.Event()
        self._latest_cloud: tuple[
            np.ndarray, dict[str, Any], rospy.Time, str, float, int,
            tuple[str | None, int]
        ] | None = None
        self._cloud_superseded_count = 0
        self._cloud_projection_cache: tuple[
            tuple[Any, ...], tuple[np.ndarray, np.ndarray]
        ] | None = None
        threading.Thread(target=self._cloud_loop, name="sensor-cloud-publisher", daemon=True).start()
        self._mirror_enabled = bool(args.web_mirror_enabled and args.web_url)
        self._mirror_lock = threading.Lock()
        self._mirror_event = threading.Event()
        self._latest_mirror_packet: dict[str, Any] | None = None
        # The dashboard used to receive telemetry over the legacy 12334
        # WebSocket.  The direct transport delivers it separately, so mirror a
        # latest-only copy for the same dashboard/record telemetry state.
        self._latest_mirror_telemetry: dict[str, Any] | None = None
        if self._mirror_enabled:
            threading.Thread(target=self._mirror_loop, name="sensor-web-mirror", daemon=True).start()
        rospy.Timer(rospy.Duration(0.05), self.refresh_tf)

    def new_transport_connection(self) -> tuple[str, int]:
        """Return a token for one Go2 WebSocket connection.

        A reconnect is a new source generation even when its first packet has
        a lower sequence/timestamp than a delayed packet from the old socket.
        The token is internal metadata and is ignored by older protocol peers.
        """
        with self._packet_lock:
            self._transport_connection += 1
            return self._transport_session, self._transport_connection

    @staticmethod
    def _packet_transport(packet: dict[str, Any]) -> tuple[str | None, int]:
        session = packet.get("_transport_session")
        session = str(session) if session not in (None, "") else None
        try:
            connection = int(packet.get("_transport_connection", 0) or 0)
        except (TypeError, ValueError):
            connection = 0
        return session, connection

    def observe_source_epoch(
        self,
        packet: dict[str, Any],
        *,
        received_wall: float,
        received_monotonic: float,
    ) -> dict[str, Any]:
        """Feed only hello/telemetry send stamps to the cross-host mapper."""

        mapper = getattr(self, "_source_epoch_mapper", None)
        if mapper is None:
            return {"valid": False, "method": "unavailable"}
        generation = self._packet_transport(packet)
        if (
            generation[0] == getattr(self, "_transport_session", None)
            and generation[1] < int(getattr(self, "_transport_connection", 0))
        ):
            result = mapper.snapshot()
            result.update(valid=False, observation_status="stale_transport")
            return result
        return mapper.observe(
            packet.get("stamp"),
            received_wall,
            received_monotonic,
            generation,
            packet.get("source_monotonic"),
            packet.get("source_clock_pair_span_s"),
            packet.get("_host_clock_pair_span_s"),
        )

    def _accept_transport_generation(self, packet: dict[str, Any]) -> bool:
        """Select/reject source generations before comparing frame stamps."""
        session, connection = self._packet_transport(packet)
        # Legacy callers/tests have no generation metadata.  Keep their old
        # stamp/sequence behaviour rather than treating every packet as a
        # reconnect.  Packets without metadata are rejected after a tagged
        # stream is active so a late legacy socket cannot overwrite it.
        if session is None:
            return self._active_transport_session is None
        if session in self._retired_transport_sessions:
            return False
        if self._active_transport_session is None:
            self._active_transport_session = session
            self._active_transport_connection = connection
            self._source_seq = -1
            return True
        if session != self._active_transport_session:
            self._retired_transport_sessions.add(self._active_transport_session)
            self._active_transport_session = session
            self._active_transport_connection = connection
            self._source_seq = -1
            self._latest_packet = None
            return True
        if connection < self._active_transport_connection:
            return False
        if connection > self._active_transport_connection:
            self._active_transport_connection = connection
            self._source_seq = -1
            self._latest_packet = None
        return True

    def _transport_generation_is_current(self, packet: dict[str, Any]) -> bool:
        session, connection = self._packet_transport(packet)
        if session is None:
            return self._active_transport_session is None
        with self._packet_lock:
            return (
                session == self._active_transport_session
                and connection == self._active_transport_connection
            )

    def _normalise_ros_capture_stamp(
        self,
        packet: dict[str, Any],
        source_stamp: float,
        *,
        stream: str = "sensor",
    ) -> tuple[float, bool]:
        """Map a source capture stamp into the local ROS clock domain.

        The direct bridge deliberately accepts a new transport generation even
        when the Go2 source timestamp goes backwards.  Without this boundary,
        ``header.stamp`` can be seconds/minutes behind ``rospy.Time.now()``;
        ``slam_gmapping`` then rejects the otherwise newest PointCloud2 as an
        old cloud and no OCC update is produced.  Keep normal frames exact and
        only rebase the generation containing the rollback.  The return value
        includes whether a rebase was applied for diagnostics/tests.

        ``_source_capture_stamp`` and ``_ros_capture_stamp`` are intentionally
        internal state rather than public wire fields: the source value is
        still present in packet telemetry and ROS headers use the corrected
        causal time.  A monotonic floor prevents two adjacent frames from
        becoming equal after a coarse/stepped source clock.

        Sensor frames and telemetry-only packets have independent capture
        clocks: telemetry packets are stamped when sent (usually between two
        camera captures), so sharing one rollback baseline would falsely
        classify every following RGB-D frame as stale.  Keep one state per
        stream while retaining the historical sensor attribute names used by
        replay tests.
        """

        try:
            source_stamp = float(source_stamp)
        except (TypeError, ValueError):
            source_stamp = float("nan")
        source_stamp_fallback = not math.isfinite(source_stamp) or source_stamp <= 0.0
        if source_stamp_fallback:
            try:
                source_stamp = float(packet.get("_host_received_wall"))
            except (TypeError, ValueError, OverflowError):
                source_stamp = time.time()
            if not math.isfinite(source_stamp) or source_stamp <= 0.:
                source_stamp = time.time()

        stream_name = "telemetry" if str(stream).casefold() == "telemetry" else "sensor"
        if stream_name == "sensor":
            generation_attr = "_stamp_generation"
            epoch_attr = "_stamp_source_epoch"
            offset_attr = "_stamp_offset_sec"
            source_attr = "_last_source_capture_stamp"
            ros_attr = "_last_ros_capture_stamp"
            count_attr = "_stamp_rebase_count"
            mapping_attr = "_last_stamp_mapping"
        else:
            generation_attr = "_telemetry_stamp_generation"
            epoch_attr = "_telemetry_stamp_source_epoch"
            offset_attr = "_telemetry_stamp_offset_sec"
            source_attr = "_last_telemetry_capture_stamp"
            ros_attr = "_last_telemetry_ros_stamp"
            count_attr = "_telemetry_stamp_rebase_count"
            mapping_attr = "_last_telemetry_stamp_mapping"

        generation = self._packet_transport(packet)
        epoch_mapper = getattr(self, "_source_epoch_mapper", None)
        if epoch_mapper is not None and not source_stamp_fallback:
            projected_source_stamp, epoch_mapping = epoch_mapper.project(
                source_stamp, generation,
            )
        else:
            projected_source_stamp = source_stamp
            epoch_mapping = {
                "valid": False,
                "selected_offset_s": None,
                "method": (
                    "host_receipt_fallback" if source_stamp_fallback
                    else "legacy_source_epoch"
                ),
            }
        previous_generation = getattr(self, generation_attr, None)
        previous_epoch = getattr(self, epoch_attr, None)
        previous_source = float(getattr(self, source_attr, -float("inf")))
        previous_ros = float(getattr(self, ros_attr, -float("inf")))
        tolerance = max(
            0.0,
            float(
                getattr(
                    self,
                    "_stamp_regression_tolerance_sec",
                    0.05,
                )
            ),
        )

        epoch_mapping_valid = epoch_mapping.get("valid") is True
        try:
            source_epoch = int(epoch_mapping.get("mapping_epoch"))
        except (TypeError, ValueError, OverflowError):
            source_epoch = None
        source_epoch_changed = bool(
            epoch_mapping_valid
            and previous_epoch is not None
            and source_epoch != previous_epoch
        )
        tagged_epoch_warming = bool(
            generation[0] is not None
            and epoch_mapper is not None
            and not epoch_mapping_valid
        )
        generation_changed = generation != previous_generation

        if tagged_epoch_warming:
            # Never run a tagged but uncalibrated source through the legacy
            # rollback rebase.  That path writes a host-now offset which would
            # later be added *again* after the epoch mapper becomes ready.
            # Images remain observable under their local receipt time, while
            # publish() gates telemetry/TF/world geometry.  Crucially, none of
            # the ready-stream baselines are mutated during this phase.
            try:
                warming_stamp = float(packet.get("_host_received_wall"))
            except (TypeError, ValueError, OverflowError):
                warming_stamp = source_stamp
            if not math.isfinite(warming_stamp) or warming_stamp <= 0.:
                warming_stamp = source_stamp
            setattr(self, mapping_attr, {
                "source_capture_stamp": float(source_stamp),
                "source_stamp_fallback": source_stamp_fallback,
                "source_epoch_projected_stamp": float(projected_source_stamp),
                "ros_stamp": float(warming_stamp),
                "rebased": False,
                "monotonic_floor_applied": False,
                "source_regression_detected": False,
                "generation_changed": generation_changed,
                "source_epoch_changed": False,
                "epoch_warming_receipt_fallback": True,
                "transport_session": generation[0],
                "transport_connection": generation[1],
                "stream": stream_name,
                "source_epoch_mapping": epoch_mapping,
            })
            return float(warming_stamp), False

        # A new tagged generation or mapper epoch may legitimately have a
        # lower raw Go2 wall value.  Compare raw stamps only within one epoch.
        # For legacy packets (no generation metadata), the same check protects
        # against NTP/source-clock steps without changing normal ordering.
        source_regressed = (
            not generation_changed
            and not source_epoch_changed
            and math.isfinite(previous_source)
            and source_stamp + tolerance < previous_source
        )
        # Offsets belong to one source generation.  Carrying a previous
        # generation's offset into a healthy reconnect would shift an already
        # synchronized clock by the old rollback amount.
        if generation_changed or source_epoch_changed:
            setattr(self, offset_attr, 0.0)
        candidate = projected_source_stamp + float(getattr(self, offset_attr, 0.0))
        collision = math.isfinite(previous_ros) and candidate <= previous_ros
        floor_applied = False
        # Once a tagged source has a measured cross-host epoch, never date a
        # delayed/non-increasing packet with ``time.time()``.  Doing so made an
        # old RGB-D frame look current and defeated both transport-age and
        # GMapping stale checks.  A 100-us ordering floor is sufficient for ROS
        # consumers while preserving essentially all of the real backlog.
        if epoch_mapping_valid:
            needs_rebase = False
            if collision:
                candidate = previous_ros + 1e-4
                floor_applied = True
        else:
            # Legacy/replay packets have no cross-host estimator.  Preserve the
            # historical rollback recovery for those callers only.
            needs_rebase = source_regressed or collision
            if needs_rebase:
                anchor = max(time.time(), previous_ros + 1e-4)
                setattr(self, offset_attr, anchor - projected_source_stamp)
                candidate = projected_source_stamp + float(getattr(self, offset_attr))
                setattr(
                    self,
                    count_attr,
                    int(getattr(self, count_attr, 0)) + 1,
                )

        setattr(self, generation_attr, generation)
        if epoch_mapping_valid:
            setattr(self, epoch_attr, source_epoch)
        setattr(self, source_attr, source_stamp)
        setattr(self, ros_attr, max(previous_ros, candidate))
        setattr(self, mapping_attr, {
            "source_capture_stamp": float(source_stamp),
            "source_stamp_fallback": source_stamp_fallback,
            "source_epoch_projected_stamp": float(projected_source_stamp),
            "ros_stamp": float(candidate),
            "rebased": bool(needs_rebase),
            "monotonic_floor_applied": floor_applied,
            "source_regression_detected": source_regressed,
            "generation_changed": generation_changed,
            "source_epoch_changed": source_epoch_changed,
            "epoch_warming_receipt_fallback": False,
            "transport_session": generation[0],
            "transport_connection": generation[1],
            "stream": stream_name,
            "source_epoch_mapping": epoch_mapping,
        })
        # The source value is retained in ``last_stamp`` for legacy queue
        # ordering; only ROS-facing headers use the normalized candidate.
        return float(candidate), bool(needs_rebase)

    @staticmethod
    def _image_msg(
        array: np.ndarray,
        encoding: str,
        stamp: rospy.Time,
        frame: str,
        seq: int = 0,
    ) -> Image:
        array = np.ascontiguousarray(array)
        msg = Image()
        # Useful before transport/replay only: rospy serialization overwrites
        # this field with a per-topic counter. The normalized capture stamp is
        # the shared identity that downstream RGB-D consumers must match.
        msg.header.seq = max(0, int(seq))
        msg.header.stamp, msg.header.frame_id = stamp, frame
        msg.height, msg.width = array.shape[:2]
        msg.encoding, msg.is_bigendian = encoding, 0
        msg.step, msg.data = int(array.strides[0]), array.tobytes(order="C")
        return msg

    def enqueue_sensor_packet(self, packet: dict[str, Any]) -> None:
        stamp = float(packet.get("stamp", 0.0) or 0.0)
        try:
            source_seq = int(packet.get("seq", -1))
        except (TypeError, ValueError):
            source_seq = -1
        with self._packet_lock:
            if not self._accept_transport_generation(packet):
                return
            pending_packet = self._latest_packet
            pending_seq = int(pending_packet.get("seq", -1)) if pending_packet else -1
            # A valid source sequence is the primary ordering key.  Comparing
            # its wall stamp here was the old freeze bug: a perfectly new
            # frame after a Go2 clock step was discarded before publish().
            # Stamp remains a fallback for legacy/replay packets with no seq.
            if source_seq >= 0:
                if source_seq <= max(self._source_seq, pending_seq):
                    return
            else:
                pending_stamp = float(
                    pending_packet.get("stamp", -float("inf"))
                    if pending_packet
                    else -float("inf")
                )
                if stamp <= max(self.last_stamp, pending_stamp):
                    return
            self._bridge_seq += 1
            accepted = dict(packet)
            accepted["_bridge_seq"] = self._bridge_seq
            self._source_seq = max(self._source_seq, source_seq)
            self._latest_packet = accepted
            self._packet_event.set()
        if self._mirror_enabled:
            with self._mirror_lock:
                self._latest_mirror_packet = accepted
                self._mirror_event.set()

    def _publish_loop(self) -> None:
        while not rospy.is_shutdown():
            self._packet_event.wait(0.5)
            with self._packet_lock:
                packet = self._latest_packet
                self._latest_packet = None
                self._packet_event.clear()
            if packet is None:
                continue
            try:
                self.publish(packet)
            except Exception as exc:
                rospy.logwarn_throttle(5.0, "sensor ROS publish: %s", exc)

    def _queue_cloud(
        self,
        depth: np.ndarray,
        intrinsics: dict[str, Any],
        stamp: rospy.Time,
        frame: str,
        scale: float,
        seq: int = 0,
        transport_generation: tuple[str | None, int] | None = None,
    ) -> None:
        if transport_generation is None:
            with self._packet_lock:
                transport_generation = (
                    self._active_transport_session, self._active_transport_connection
                )
        with self._cloud_lock:
            self._latest_cloud = (
                depth,
                dict(intrinsics),
                stamp,
                frame,
                float(scale),
                max(0, int(seq)),
                transport_generation,
            )
            self._cloud_event.set()

    def _cloud_loop(self) -> None:
        while not rospy.is_shutdown():
            self._cloud_event.wait(0.5)
            with self._cloud_lock:
                pending = self._latest_cloud
                self._latest_cloud = None
                self._cloud_event.clear()
            if pending is None:
                continue
            try:
                # Latest-only applies to waiting work, not cancellation of
                # every in-flight projection when input is faster than compute.
                if not self._cloud_generation_is_current(pending[-1]):
                    self._cloud_superseded_count += 1
                    continue
                self.publish_cloud(*pending)
            except Exception as exc:
                rospy.logwarn_throttle(5.0, "sensor cloud publish: %s", exc)

    def _cloud_generation_is_current(self, generation: tuple[str | None, int]) -> bool:
        """Reject retired connections without starving same-stream progress."""
        with self._packet_lock:
            if generation[0] is None:
                return self._active_transport_session is None
            return generation == (
                self._active_transport_session, self._active_transport_connection
            )

    def _mirror_loop(self) -> None:
        base_url = self.args.web_url.rstrip("/")
        while not rospy.is_shutdown():
            # The event is set for every newest RGB-D receipt.  The timeout is
            # only a shutdown poll; it must not be treated as a 2-Hz throttle.
            self._mirror_event.wait(1.0)
            with self._mirror_lock:
                packet = self._latest_mirror_packet
                self._latest_mirror_packet = None
                telemetry_snapshot = self._latest_mirror_telemetry
                self._latest_mirror_telemetry = None
                self._mirror_event.clear()
            # A capture packet already contains its capture-time telemetry.
            # Post it first; a separate mirror snapshot must never overwrite a
            # newer heading before the packet reaches RuntimeState.
            packet_telemetry = packet.get("telemetry") if isinstance(packet, dict) else None
            if packet is not None and isinstance(packet_telemetry, dict) and packet_telemetry:
                telemetry_snapshot = None
            if telemetry_snapshot is not None:
                try:
                    # ``update_telemetry`` normally stores the accepted
                    # generation in the snapshot.  Re-attach it here as a
                    # defensive measure for snapshots produced by older
                    # callers/tests that only populated ``self.telemetry``.
                    with self._telemetry_lock:
                        session = getattr(self, "_telemetry_transport_session", None)
                        connection = getattr(self, "_telemetry_transport_connection", -1)
                    mirror_payload = _attach_transport_metadata(
                        telemetry_snapshot, session, connection
                    )
                    body = json.dumps(mirror_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    request = urllib.request.Request(base_url + "/api/telemetry", data=body, headers={"Content-Type": "application/json"}, method="POST")
                    with urllib.request.urlopen(request, timeout=0.5):
                        pass
                except Exception as exc:
                    rospy.logwarn_throttle(10.0, "optional dashboard telemetry mirror: %s", exc)
            if packet is None:
                continue
            try:
                body = json.dumps(packet, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                request = urllib.request.Request(base_url + "/api/raw-frame", data=body, headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(request, timeout=0.5):
                    pass
            except Exception as exc:
                rospy.logwarn_throttle(10.0, "optional dashboard sensor mirror: %s", exc)

    def publish(self, packet: dict[str, Any]) -> None:
        decode_started = time.perf_counter()
        try:
            raw_source_stamp_value = float(packet.get("stamp"))
        except (TypeError, ValueError, OverflowError):
            raw_source_stamp_value = float("nan")
        source_stamp_valid = (
            math.isfinite(raw_source_stamp_value)
            and raw_source_stamp_value > 0.0
        )
        if source_stamp_valid:
            source_stamp_value = raw_source_stamp_value
        else:
            try:
                source_stamp_value = float(packet.get("_host_received_wall"))
            except (TypeError, ValueError, OverflowError):
                source_stamp_value = time.time()
            if not math.isfinite(source_stamp_value) or source_stamp_value <= 0.0:
                source_stamp_value = time.time()
        # Keep the source value for duplicate/order checks.  The ROS stamp is
        # normalized only after the transport-generation check below, so a
        # delayed packet from a retired socket cannot move the rebase anchor.
        stamp_value = source_stamp_value
        seq = int(packet.get("seq", -1))
        bridge_seq = int(packet.get("_bridge_seq", -1) or -1)
        if bridge_seq >= 0:
            if bridge_seq <= self._last_published_bridge_seq:
                return
            self._last_published_bridge_seq = bridge_seq
        elif seq >= 0:
            # Legacy/direct unit-test path: source seq is authoritative when
            # present, even if its capture stamp moved backwards.
            if seq <= self.last_seq:
                return
        elif stamp_value <= self.last_stamp:
            return
        self.last_stamp = max(self.last_stamp, source_stamp_value)
        self.last_seq = max(self.last_seq, seq)
        # Decode directly to BGR for ROS/YOLO; the default RGB helper remains
        # available for replay callers that rely on its historical contract.
        rgb_future = self._decode_executor.submit(_decode, packet["rgb"], True)
        depth_future = self._decode_executor.submit(_decode, packet["depth"])
        rgb = rgb_future.result()
        depth = depth_future.result().astype(np.uint16, copy=False)
        if not self._transport_generation_is_current(packet):
            return
        # ``publish()`` runs on the ROS publisher thread while telemetry-only
        # packets are handled by the asyncio thread. Serialize the tiny
        # generation/offset state so a reconnect cannot interleave two
        # rebases and produce a non-monotonic header stamp.
        with self._packet_lock:
            stamp_value, stamp_rebased = self._normalise_ros_capture_stamp(
                packet, raw_source_stamp_value
            )
        transport_generation = self._packet_transport(packet)
        source_epoch_mapping = dict(
            getattr(self, "_last_stamp_mapping", {}).get(
                "source_epoch_mapping", {}
            )
        )
        stamp_mapping = dict(getattr(self, "_last_stamp_mapping", {}))
        source_clock_boundary = bool(
            stamp_rebased or stamp_mapping.get("source_epoch_changed") is True
        )
        try:
            host_received_wall = float(packet.get("_host_received_wall"))
        except (TypeError, ValueError, OverflowError):
            host_received_wall = float("nan")
        host_transport_age_s = (
            host_received_wall - stamp_value
            if math.isfinite(host_received_wall) else None
        )
        # Tagged production traffic must establish its cross-host epoch before
        # publishing world geometry. New Go2 clients send hello + two tiny clock
        # probes before RGB-D, so this normally costs no camera frames. Legacy
        # untagged replay retains its historical exact-stamp behavior.
        epoch_gate_reason = "ready"
        if transport_generation[0] is None:
            source_epoch_ready = True
            epoch_gate_reason = "legacy_untagged"
        elif getattr(self, "_source_epoch_mapper", None) is None:
            source_epoch_ready = True
            epoch_gate_reason = "mapper_unavailable_compatibility"
        elif source_epoch_mapping.get("valid") is not True:
            source_epoch_ready = False
            epoch_gate_reason = "warming"
        elif (
            stamp_mapping.get("source_regression_detected") is True
            and stamp_mapping.get("generation_changed") is not True
        ):
            source_epoch_ready = False
            epoch_gate_reason = "source_clock_regression"
        elif stamp_mapping.get("monotonic_floor_applied") is True:
            # A clock correction can leave several subsequent source stamps
            # below the last published ROS watermark.  The first frame is
            # labelled as a regression, but later frames may be increasing in
            # source time while still advancing only through the synthetic
            # 100-us ordering floor.  Keep images observable, but do not date
            # TF/world geometry with that floor until the projected capture
            # clock has genuinely caught up.
            source_epoch_ready = False
            epoch_gate_reason = "capture_stamp_floor"
        elif (
            host_transport_age_s is not None
            and host_transport_age_s < -0.1
        ):
            # The one-way estimator may place captures a few milliseconds in
            # the future (its minimum network delay is unobservable), but a
            # 100-ms lead signals a source/host clock step. Keep images visible
            # while withholding incorrect TF/world geometry until probes warm.
            source_epoch_ready = False
            epoch_gate_reason = "projected_capture_in_future"
        else:
            source_epoch_ready = True
        source_epoch_mapping.update({
            "host_transport_age_s": host_transport_age_s,
            "gate_ready": source_epoch_ready,
            "gate_reason": epoch_gate_reason,
        })
        stamp = rospy.Time.from_sec(stamp_value)
        if stamp_rebased:
            rospy.logwarn_throttle(
                2.0,
                "sensor source stamp regressed; rebasing ROS capture stamp "
                "source=%.6f ros=%.6f generation=%s/%s count=%d",
                source_stamp_value,
                stamp_value,
                self._packet_transport(packet)[0] or "legacy",
                self._packet_transport(packet)[1],
                int(getattr(self, "_stamp_rebase_count", 0)),
            )
        rospy.loginfo_throttle(
            10.0,
            "sensor RGB-D decode: %.1f ms rgb=%dx%d depth=%dx%d",
            (time.perf_counter() - decode_started) * 1000.0,
            int(rgb.shape[1]),
            int(rgb.shape[0]),
            int(depth.shape[1]),
            int(depth.shape[0]),
        )
        rgb_frame = str(packet.get("camera_frame") or self.args.camera_frame)
        depth_frame = str(packet.get("depth_frame") or rgb_frame)
        # Keep the frame/calibration pair for the periodic TF refresh.  A
        # refresh that only republishes the colour child leaves a native-depth
        # cloud using the previous depth transform after a dropped packet.
        self._last_depth_frame = depth_frame
        self.rgb_pub.publish(self._image_msg(rgb, "bgr8", stamp, rgb_frame, bridge_seq))
        self.depth_pub.publish(self._image_msg(depth, "16UC1", stamp, depth_frame, bridge_seq))
        rgb_intrinsics = packet.get("rgb_intrinsics") or packet.get("intrinsics", {})
        depth_intrinsics = packet.get("depth_intrinsics") or rgb_intrinsics
        self.info_pub.publish(self.camera_info(rgb_intrinsics, stamp, rgb_frame))
        self.depth_info_pub.publish(self.camera_info(depth_intrinsics, stamp, depth_frame))
        calibration = {
            "depth_scale": float(packet.get("depth_scale", 0.001) or 0.001),
            "telemetry_max_delta_sec": float(getattr(self.args, "telemetry_max_delta_sec", 0.15)),
            "depth_to_color_extrinsics": packet.get("depth_to_color_extrinsics") or {},
            "rgb_frame": rgb_frame,
            "depth_frame": depth_frame,
            "depth_width": int(depth_intrinsics.get("width", depth.shape[1]) or depth.shape[1]),
            "depth_height": int(depth_intrinsics.get("height", depth.shape[0]) or depth.shape[0]),
        }
        try:
            calibration_key = json.dumps(
                calibration, sort_keys=True, separators=(",", ":")
            )
            if calibration_key != self._last_calibration_key:
                self.calibration_pub.publish(String(data=calibration_key))
                self._last_calibration_key = calibration_key
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "sensor calibration publish: %s", exc)
        if source_epoch_ready and isinstance(packet.get("telemetry"), dict):
            self.update_telemetry(
                packet["telemetry"],
                stamp_value,
                capture_seq=bridge_seq if bridge_seq >= 0 else seq,
                source_reference_stamp=source_stamp_value,
                source_clock_rebased=source_clock_boundary,
                transport_session=packet.get("_transport_session"),
                transport_connection=packet.get("_transport_connection"),
            )
        # Publish the capture-time odom/TF before handing the cloud to its
        # worker. MessageFilter/gmapping can then consume the cloud without
        # waiting for a transform that is published a few milliseconds later.
        matched_telemetry, telemetry_delta = self.telemetry_at(stamp_value)
        imu_match = _camera_imu_capture_match(
            matched_telemetry,
            raw_source_stamp_value if source_stamp_valid else source_stamp_value,
            enabled=self._camera_imu_enabled, use_yaw=self._camera_imu_use_yaw,
            max_delta_sec=calibration["telemetry_max_delta_sec"],
        )
        # One authoritative pose/calibration decision per capture. Consumers
        # join by the serialized ROS stamp, not independent topic counters.
        camera_transforms = self._camera_transforms(
            matched_telemetry, stamp, depth_frame, calibration["depth_to_color_extrinsics"],
        )
        tf_status = "invalid_capture_pose"
        if not source_epoch_ready:
            tf_status = "source_epoch_" + epoch_gate_reason
        elif not imu_match["accepted"]:
            tf_status = imu_match["status"] + "_camera_imu"
        elif matched_telemetry:
            # Date odometry with the camera capture stamp. This lets the
            # mapper request the historical pose matching this cloud instead
            # of pairing a delayed frame with the current robot pose.
            tf_status = self.publish_pose(
                matched_telemetry,
                stamp,
                depth_frame,
                calibration["depth_to_color_extrinsics"],
                camera_transforms=camera_transforms,
            )
        elif telemetry_delta is not None:
            rospy.logwarn_throttle(
                5.0,
                "no capture-time telemetry for RGB-D stamp (nearest delta %.3fs); "
                "skipping world geometry for this capture",
                telemetry_delta,
            )
        capture_timing = (
            dict(packet.get("capture_timing"))
            if isinstance(packet.get("capture_timing"), dict) else {}
        )
        capture_timing.update({
            "source_packet_stamp": (
                raw_source_stamp_value if source_stamp_valid else None
            ),
            "source_packet_stamp_valid": source_stamp_valid,
            "ros_capture_stamp": stamp_value,
            "source_to_ros_offset_s": (
                stamp_value - raw_source_stamp_value
                if source_stamp_valid else None
            ),
            "source_clock_rebased": source_clock_boundary,
            "host_received_wall": packet.get("_host_received_wall"),
            "host_received_monotonic": packet.get("_host_received_monotonic"),
            "host_transport_age_s": host_transport_age_s,
            "source_epoch_mapping": source_epoch_mapping,
            "transport_session": packet.get("_transport_session"),
            "transport_connection": packet.get("_transport_connection"),
        })
        self.publish_capture_context(
            stamp, bridge_seq, matched_telemetry, telemetry_delta,
            rgb_intrinsics, depth_intrinsics, calibration,
            camera_transforms=camera_transforms, tf_status=tf_status, imu_match=imu_match,
            capture_timing=capture_timing,
        )
        if tf_status not in {"published", "duplicate"}:
            rospy.logwarn_throttle(5.0, "capture TF rejected: %s; skipping cloud", tf_status)
            return
        self._queue_cloud(
            depth,
            depth_intrinsics,
            stamp,
            depth_frame,
            float(packet.get("depth_scale", 0.001)),
            bridge_seq,
            self._packet_transport(packet),
        )

    def publish_capture_context(
        self, stamp, capture_seq, telemetry, telemetry_delta,
        rgb_intrinsics, depth_intrinsics, calibration,
        *, camera_transforms=None, tf_status="unchecked", imu_match=None,
        capture_timing=None,
    ) -> list[TransformStamped]:
        if camera_transforms is None:
            camera_transforms = self._camera_transforms(
                telemetry, stamp, calibration["depth_frame"],
                calibration["depth_to_color_extrinsics"],
            )
        depth_transform = next(t for t in camera_transforms
                               if t.child_frame_id == calibration["depth_frame"])
        q = depth_transform.transform.rotation
        p = depth_transform.transform.translation
        payload = {
            "version": 1,
            "stamp": stamp.to_sec(),
            "seq": max(0, int(capture_seq)),
            "telemetry": telemetry,
            "capture_tf_status": tf_status,
            "capture_tf_conflict_count": getattr(self, "_tf_stamp_conflict_count", 0),
            "camera_imu_match": imu_match or {"status": "unchecked"},
            "capture_timing": (
                dict(capture_timing) if isinstance(capture_timing, dict) else {}
            ),
            "capture_pose_match": {
                "delta_sec": telemetry_delta,
                "max_delta_sec": calibration["telemetry_max_delta_sec"],
                "accepted": bool(telemetry),
                "source": "sensor_capture_context",
            },
            "rgb_intrinsics": rgb_intrinsics,
            "depth_intrinsics": depth_intrinsics,
            "depth_scale": calibration["depth_scale"],
            "depth_to_color_extrinsics": calibration["depth_to_color_extrinsics"],
            "camera_frame": calibration["rgb_frame"],
            "depth_frame": calibration["depth_frame"],
            "depth_to_base": {
                "parent_frame": depth_transform.header.frame_id,
                "child_frame": depth_transform.child_frame_id,
                "quaternion_xyzw": [q.x, q.y, q.z, q.w],
                "translation": [p.x, p.y, p.z],
            },
        }
        self.capture_context_pub.publish(String(data=json.dumps(
            payload, separators=(",", ":"), ensure_ascii=False,
        )))
        return camera_transforms

    def update_telemetry(
        self,
        telemetry: dict[str, Any],
        stamp: float,
        *,
        capture_seq: int | None = None,
        source_reference_stamp: float | None = None,
        source_clock_rebased: bool = False,
        transport_session: Any = None,
        transport_connection: Any = None,
    ) -> None:
        snapshot = dict(telemetry)
        try:
            source_stamp = float(snapshot.get("received_at"))
        except (TypeError, ValueError):
            source_stamp = float("nan")
        if not math.isfinite(source_stamp) and source_reference_stamp is None:
            source_stamp = float(stamp)
        pose_ros_stamp = float(stamp)
        sample_time_valid = True
        if source_reference_stamp is not None:
            # received_at is the Go2 sport-state sample receipt, not the
            # enclosing camera capture / telemetry-send time. Apply the same
            # source->ROS offset without relabeling an old cached pose as new.
            try:
                reference = float(source_reference_stamp)
                pose_ros_stamp = source_stamp + (float(stamp) - reference)
                future_limit = float(getattr(self.args, "telemetry_max_delta_sec", .15))
                sample_time_valid = (
                    all(math.isfinite(v) and v > 0. for v in (source_stamp, reference, pose_ros_stamp))
                    and source_stamp - reference <= future_limit
                )
            except (TypeError, ValueError, OverflowError):
                sample_time_valid = False
        try:
            capture_seq_int = int(capture_seq) if capture_seq is not None else None
        except (TypeError, ValueError):
            capture_seq_int = None
        capture_embedded = (
            capture_seq_int is not None
            and capture_seq_int >= 0
            and source_reference_stamp is not None
        )
        with self._telemetry_lock:
            # A new transport resets both source-clock and capture-sequence
            # watermarks. Untagged packets cannot impersonate an active stream.
            session = (
                str(transport_session)
                if transport_session not in (None, "")
                else None
            )
            try:
                connection = int(transport_connection or 0)
            except (TypeError, ValueError):
                connection = 0
            current_session = getattr(self, "_telemetry_transport_session", None)
            current_connection = int(
                getattr(self, "_telemetry_transport_connection", -1)
            )
            if session is None and current_session is not None:
                return
            if session is not None:
                retired = getattr(self, "_telemetry_retired_sessions", set())
                if session in retired:
                    return
                if current_session is None:
                    self._telemetry_transport_session = session
                    self._telemetry_transport_connection = connection
                    self._telemetry_source_stamp = float("-inf")
                    self._telemetry_capture_seq = -1
                    self._telemetry_history.clear()
                elif session != current_session:
                    retired.add(current_session)
                    self._telemetry_retired_sessions = retired
                    self._telemetry_transport_session = session
                    self._telemetry_transport_connection = connection
                    self._telemetry_source_stamp = float("-inf")
                    self._telemetry_capture_seq = -1
                    self._telemetry_history.clear()
                elif connection < current_connection:
                    return
                elif connection > current_connection:
                    self._telemetry_transport_connection = connection
                    self._telemetry_source_stamp = float("-inf")
                    self._telemetry_capture_seq = -1
                    self._telemetry_history.clear()
            if source_clock_rebased:
                # The transport can stay connected while NTP or the Go2 wall
                # clock starts a new mapper epoch. Raw sport-state timestamps
                # from the old epoch must not keep every new capture classified
                # as stale, and old history must not win across the boundary.
                self._telemetry_source_stamp = float("-inf")
                self._telemetry_history.clear()
            pose_keys = {
                "received_at", "position", "velocity", "yaw", "yaw_speed",
                "imu", "camera_imu", "mode", "progress", "gait_type",
                "body_height",
                "base_pose", "odom", "pose", "camera_pose", "d435i_pose",
            }
            capture_is_fresh = (
                capture_seq_int is not None
                and capture_seq_int > self._telemetry_capture_seq
            )
            stale_pose = (
                not sample_time_valid
                or (capture_seq_int is not None and capture_seq_int < self._telemetry_capture_seq)
                or (source_stamp < self._telemetry_source_stamp - 1e-6
                    and not (capture_is_fresh and (
                        source_reference_stamp is None or source_clock_rebased
                    )))
            )
            history_only_capture = stale_pose and capture_embedded
            telemetry_wire = None
            if history_only_capture:
                # A small telemetry envelope can overtake RGB-D decode. The
                # capture still owns a valid historical pose for its own TF
                # lookup, but must not rewind or republish the latest state.
                self._telemetry_capture_seq = max(
                    self._telemetry_capture_seq, capture_seq_int
                )
                if sample_time_valid:
                    _insert_capture_telemetry(
                        self._telemetry_history, pose_ros_stamp, snapshot,
                    )
                telemetry_wire = None
            elif stale_pose:
                # Keep diagnostics from a late packet, but never let its old
                # pose replace the newest capture-time state used for TF.
                merged = dict(self.telemetry)
                for key, value in snapshot.items():
                    if key not in pose_keys:
                        merged[key] = value
                snapshot = merged
            else:
                # A live clock step must be confirmed by the envelope clock
                # normalizer, not merely by a newer frame carrying cached
                # sport state. Accepted steps must still reset the watermark.
                self._telemetry_source_stamp = source_stamp
                if capture_seq_int is not None:
                    self._telemetry_capture_seq = max(
                        self._telemetry_capture_seq, capture_seq_int
                    )
                _insert_capture_telemetry(
                    self._telemetry_history, pose_ros_stamp, snapshot,
                )
                self._telemetry_ros_stamp = pose_ros_stamp
            if not history_only_capture:
                self.telemetry = snapshot
            # Keep the accepted telemetry snapshot and its transport
            # generation together for both optional web lanes.  In
            # particular, the ROS telemetry topic must not lose the marker;
            # physical_ros_gateway forwards that topic when the direct HTTP
            # mirror is delayed or disabled.
            if not history_only_capture:
                telemetry_wire = _attach_transport_metadata(
                    snapshot,
                    getattr(self, "_telemetry_transport_session", None),
                    getattr(self, "_telemetry_transport_connection", -1),
                )
        # A late packet may update local diagnostics, but the merged current
        # pose must never be republished under the late packet's old stamp.
        if not stale_pose:
            try:
                self.telemetry_pub.publish(
                    String(
                        data=json.dumps(
                            {"stamp": pose_ros_stamp, "telemetry": telemetry_wire},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                )
            except Exception as exc:
                rospy.logwarn_throttle(5.0, "sensor telemetry publish: %s", exc)
        if self._mirror_enabled and telemetry_wire is not None:
            # Keep the optional dashboard mirror in the same transport
            # generation as the direct RGB-D receipt.  Without these private
            # ordering markers, a persistent web gateway can receive a
            # low-rate telemetry packet from the pre-restart source after the
            # new bridge has connected; its old timestamp/sequence may then
            # be rejected (or, after a clock step, incorrectly win) and leave
            # the browser arrow on the previous heading.  RuntimeState strips
            # the markers before exposing telemetry to clients.
            with self._mirror_lock:
                self._latest_mirror_telemetry = telemetry_wire
                self._mirror_event.set()

    def telemetry_at(self, stamp: float) -> tuple[dict[str, Any], float | None]:
        """Return the nearest capture pose and its timestamp delta.

        A stale pose is worse than a short TF wait: pairing an old IMU/odom
        sample with a new point cloud creates the map skew that previously
        appeared as a bent/rotated OCC. The maximum accepted delta is kept
        configurable and defaults to 0.15 s. No untimestamped latest-pose
        fallback is permitted when history is empty.
        """
        with self._telemetry_lock:
            return _nearest_capture_telemetry(
                self._telemetry_history, stamp,
                getattr(self.args, "telemetry_max_delta_sec", 0.15),
            )

    @staticmethod
    def camera_info(intrinsics: dict[str, Any], stamp: rospy.Time, frame: str) -> CameraInfo:
        msg = CameraInfo()
        msg.header.stamp, msg.header.frame_id = stamp, frame
        msg.width, msg.height = int(intrinsics.get("width", 0)), int(intrinsics.get("height", 0))
        msg.K = [float(intrinsics.get("fx", 0)), 0.0, float(intrinsics.get("cx", 0)), 0.0, float(intrinsics.get("fy", 0)), float(intrinsics.get("cy", 0)), 0.0, 0.0, 1.0]
        msg.D = [float(value) for value in (intrinsics.get("distortion") or [])[:5]]
        msg.distortion_model = str(intrinsics.get("distortion_model", "plumb_bob") or "plumb_bob")
        return msg

    def publish_cloud(
        self,
        depth: np.ndarray,
        intrinsics: dict[str, Any],
        stamp: rospy.Time,
        frame: str,
        scale: float,
        seq: int = 0,
        transport_generation: tuple[str | None, int] | None = None,
    ) -> None:
        if transport_generation is None:
            with self._packet_lock:
                transport_generation = (
                    self._active_transport_session, self._active_transport_connection
                )
        if not self._cloud_generation_is_current(transport_generation):
            self._cloud_superseded_count += 1
            return
        points, self._cloud_projection_cache = _sample_depth_points(
            depth,
            intrinsics,
            int(self.args.point_stride),
            float(scale),
            float(self.args.max_depth_m),
            float(self.args.no_return_depth_m),
            projection_cache=self._cloud_projection_cache,
        )
        if points.shape[0] == 0:
            return
        header = Header(stamp=stamp, frame_id=frame)
        # rospy overwrites this with the cloud topic's own publication count.
        # Capture stamp is the cross-topic identity (see capture_context).
        header.seq = max(0, int(seq))
        msg = PointCloud2(
            header=header, height=1, width=int(points.shape[0]),
            fields=[PointField("x", 0, PointField.FLOAT32, 1), PointField("y", 4, PointField.FLOAT32, 1), PointField("z", 8, PointField.FLOAT32, 1)],
            is_bigendian=False, point_step=12, row_step=int(points.shape[0] * 12),
            data=points.tobytes(order="C"), is_dense=False,
        )
        # Allow same-generation work to finish; otherwise sustained >100 ms
        # projection can drop every result under a 10 Hz input. Preserve the
        # capture stamp so downstream age/TF checks still see its real age.
        if not self._cloud_generation_is_current(transport_generation):
            self._cloud_superseded_count += 1
            return
        self.cloud_pub.publish(msg)

    def publish_pose(
        self,
        telemetry: dict[str, Any],
        stamp: rospy.Time,
        depth_frame: str | None = None,
        depth_to_color_extrinsics: dict[str, Any] | None = None,
        *, camera_transforms: list[TransformStamped] | None = None,
    ) -> str:
        # The capture worker and periodic timer may run concurrently. Serialize
        # their publications and watermark updates; no network wait is added.
        if not hasattr(self, "_pose_publish_lock"):
            self._pose_publish_lock = threading.RLock()  # Minimal offline fixtures.
        with self._pose_publish_lock:
            return self._publish_pose_locked(telemetry, stamp, depth_frame, depth_to_color_extrinsics,
                                             camera_transforms=camera_transforms)

    def _publish_pose_locked(
        self,
        telemetry: dict[str, Any],
        stamp: rospy.Time,
        depth_frame: str | None = None,
        depth_to_color_extrinsics: dict[str, Any] | None = None,
        *, camera_transforms: list[TransformStamped] | None = None,
    ) -> str:
        position = _telemetry_body_position(telemetry)
        quaternion = _telemetry_body_quaternion(telemetry)
        if position is None or quaternion is None:
            rospy.logwarn_throttle(5.0, "sensor pose missing/invalid; skipping odom and TF publication")
            return "invalid_capture_pose"
        if depth_to_color_extrinsics is None and self._depth_to_color_extrinsics is not None:
            depth_to_color_extrinsics = self._depth_to_color_extrinsics
        try:
            source_stamp = float(telemetry.get("received_at"))
        except (TypeError, ValueError, AttributeError):
            source_stamp = float("nan")
        if not math.isfinite(source_stamp):
            source_stamp = float(stamp.to_sec())
        stamp_ns = stamp.to_nsec()
        newest_odom = stamp_ns > getattr(self, "_last_tf_ros_stamp_ns", -1)
        velocity = list(telemetry.get("velocity") or [0.0, 0.0, 0.0]) + [0.0, 0.0, 0.0]
        x, y, z, w = quaternion
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id = stamp, "tf_frame_odom"
        odom.child_frame_id = "tf_frame_base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, position[:3])
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y = x, y
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = z, w
        odom.twist.twist.linear.x, odom.twist.twist.linear.y = map(float, velocity[:2])
        transform = TransformStamped(header=odom.header, child_frame_id=odom.child_frame_id)
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = map(float, position[:3])
        transform.transform.rotation = odom.pose.pose.orientation
        if camera_transforms is None:
            camera_transforms = self._camera_transforms(
                telemetry, stamp, depth_frame, depth_to_color_extrinsics,
            )
        samples = getattr(self, "_published_tf_samples", None)
        if samples is None:
            samples = self._published_tf_samples = {}
        if stamp_ns <= getattr(self, "_tf_binding_floor_ns", -1):
            return "expired_capture_tf_stamp"
        previous_sample = samples.get(stamp_ns, {})
        candidates = [transform, *camera_transforms]
        signatures = {}
        for candidate in candidates:
            p, q = candidate.transform.translation, candidate.transform.rotation
            quaternion = (q.x, q.y, q.z, q.w)
            numbers = (p.x, p.y, p.z, *quaternion)
            if not all(math.isfinite(value) for value in numbers):
                return "invalid_capture_transform"
            signature = (candidate.header.frame_id, *numbers)
            child = candidate.child_frame_id
            previous = previous_sample.get(child)
            if previous is not None and (
                signature[0] != previous[0]
                or any(abs(a - b) > 1e-9 for a, b in zip(signature[1:4], previous[1:4]))
                # q and -q are equivalent. Compare both signs without a
                # discontinuous sign convention at near-zero components.
                or (any(abs(a - b) > 1e-9 for a, b in zip(signature[4:], previous[4:]))
                    and any(abs(a + b) > 1e-9 for a, b in zip(signature[4:], previous[4:])))
            ):
                self._tf_stamp_conflict_count = getattr(self, "_tf_stamp_conflict_count", 0) + 1
                return "conflicting_capture_tf_stamp"
            signatures[child] = signature
        new_body = transform.child_frame_id not in previous_sample
        new_cameras = [t for t in camera_transforms if t.child_frame_id not in previous_sample]
        if newest_odom:
            self.odom_pub.publish(odom)
        if new_body:
            self.tf_broadcaster.sendTransform(transform)
            samples.setdefault(stamp_ns, {})[transform.child_frame_id] = signatures[transform.child_frame_id]
        if depth_to_color_extrinsics is not None:
            self._depth_to_color_extrinsics = dict(depth_to_color_extrinsics)
        self.static_frames.update(t.child_frame_id for t in camera_transforms)

        if self._camera_imu_enabled:
            # A dynamic child must not also be published on /tf_static.
            if new_cameras:
                self.tf_broadcaster.sendTransform(new_cameras)
        else:
            # The latched static message must retain every discovered child;
            # publish only when calibration actually changes.
            changed = False
            for transform in camera_transforms:
                child = transform.child_frame_id
                previous = self._static_transforms.get(child)
                if previous is None or any(
                    abs(float(current) - float(old)) > 1e-9
                    for current, old in (
                        (transform.transform.translation.x, previous.transform.translation.x),
                        (transform.transform.translation.y, previous.transform.translation.y),
                        (transform.transform.translation.z, previous.transform.translation.z),
                        (transform.transform.rotation.x, previous.transform.rotation.x),
                        (transform.transform.rotation.y, previous.transform.rotation.y),
                        (transform.transform.rotation.z, previous.transform.rotation.z),
                        (transform.transform.rotation.w, previous.transform.rotation.w),
                    )
                ):
                    self._static_transforms[child] = transform
                    changed = True
            if changed:
                self.static_broadcaster.sendTransform(list(self._static_transforms.values()))
        samples.setdefault(stamp_ns, {}).update(signatures)
        # Keep a bounded publication history. Never republish a forgotten old
        # timestamp: a consumer may still retain its first TF sample.
        while len(samples) > 512:
            oldest = min(samples)
            del samples[oldest]
            self._tf_binding_floor_ns = max(getattr(self, "_tf_binding_floor_ns", -1), oldest)
        if newest_odom:
            self._last_tf_ros_stamp_ns = stamp_ns
            self._last_tf_source_stamp = source_stamp
        return "published" if new_body or new_cameras else "duplicate"

    def _camera_transforms(self, telemetry, stamp, depth_frame, depth_to_color_extrinsics):
        """Compute the camera branch once for TF and the capture context."""
        color_frame = str(self.args.camera_frame)
        color_translation = (
            float(self.args.camera_x),
            float(self.args.camera_y),
            float(self.args.camera_z),
        )
        color_quaternion = _quat_multiply(
            _quat_rpy(self.args.camera_roll, self.args.camera_pitch, self.args.camera_yaw),
            (0.5, -0.5, 0.5, -0.5),
        )
        correction_quaternion, correction_valid = _camera_imu_parent_correction_quaternion(
            telemetry,
            color_quaternion,
            enabled=self._camera_imu_enabled,
            use_yaw=self._camera_imu_use_yaw,
            max_roll_rad=self._camera_imu_max_roll_rad,
            max_pitch_rad=self._camera_imu_max_pitch_rad,
            max_yaw_rad=self._camera_imu_max_yaw_rad,
        )
        frames = {color_frame, str(depth_frame or color_frame)}
        camera_transforms = []
        for frame in frames:
            nominal_quaternion, nominal_translation = (
                _depth_static_extrinsic(
                    color_quaternion,
                    color_translation,
                    frame,
                    color_frame,
                    depth_to_color_extrinsics,
                )
                if frame != color_frame
                else (color_quaternion, color_translation)
            )
            frame_quaternion, frame_translation = _apply_camera_imu_correction(
                nominal_quaternion,
                nominal_translation,
                correction_quaternion,
            )
            dynamic = TransformStamped()
            dynamic.header.stamp, dynamic.header.frame_id, dynamic.child_frame_id = (
                stamp,
                self.args.camera_parent,
                frame,
            )
            dynamic.transform.translation.x, dynamic.transform.translation.y, dynamic.transform.translation.z = frame_translation
            dynamic.transform.rotation.x, dynamic.transform.rotation.y, dynamic.transform.rotation.z, dynamic.transform.rotation.w = frame_quaternion
            camera_transforms.append(dynamic)
        return camera_transforms

    def refresh_tf(self, _event: Any) -> None:
        if not hasattr(self, "_pose_publish_lock"):
            self._pose_publish_lock = threading.RLock()
        with self._pose_publish_lock:
            with self._telemetry_lock:
                telemetry = dict(self.telemetry)
                ros_stamp = getattr(self, "_telemetry_ros_stamp", None)
            if not telemetry or ros_stamp is None:
                return
            try:
                ros_stamp = float(ros_stamp)
            except (TypeError, ValueError, OverflowError):
                return
            if not math.isfinite(ros_stamp) or ros_stamp <= 0.:
                return
            refresh_stamp = rospy.Time.from_sec(ros_stamp)
            # Compare in ROS nanoseconds, not the sensor clock or a rounded
            # float. A repeated timer must not re-date an unchanged pose.
            if refresh_stamp.to_nsec() <= getattr(self, "_last_tf_ros_stamp_ns", -1):
                return
            imu_match = _camera_imu_capture_match(
                telemetry, telemetry.get("received_at"),
                enabled=self._camera_imu_enabled,
                use_yaw=self._camera_imu_use_yaw,
                max_delta_sec=getattr(self.args, "telemetry_max_delta_sec", .15),
            )
            if not imu_match["accepted"]:
                rospy.logwarn_throttle(
                    5.0, "sensor periodic camera TF skipped: %s IMU; body odom retained",
                    imu_match["status"],
                )
            self._publish_pose_locked(
                telemetry, refresh_stamp, self._last_depth_frame,
                self._depth_to_color_extrinsics,
                # Do not restamp held tilt or fall back to a nominal mount.
                # The independent body pose remains valid for navigation.
                camera_transforms=None if imu_match["accepted"] else [],
            )


async def run(args: argparse.Namespace) -> None:
    bridge = SensorRosBridge(args)
    import websockets

    async def handler(websocket: Any, *_path: Any) -> None:
        transport_session, transport_connection = bridge.new_transport_connection()
        async for raw in websocket:
            try:
                # Bind the local receipt before JSON/zlib decode.  This is the
                # endpoint used by the small-packet clock estimator; codec work
                # must not be interpreted as network or source-clock offset.
                (
                    host_received_wall,
                    host_received_monotonic,
                    host_clock_pair_span_s,
                ) = sample_clock_pair()
                packet = decode_wire_packet(raw)
                # Internal metadata is deliberately additive so old Go2
                # bridge binaries remain wire-compatible.  The local ROS and
                # web latest-only queues use it to distinguish reconnects.
                packet = dict(packet)
                packet["_transport_session"] = transport_session
                packet["_transport_connection"] = transport_connection
                packet["_host_received_wall"] = host_received_wall
                packet["_host_received_monotonic"] = host_received_monotonic
                packet["_host_clock_pair_span_s"] = host_clock_pair_span_s
                if packet.get("type") == "sensor_frame":
                    bridge.enqueue_sensor_packet(packet)
                elif packet.get("type") in {"hello", "clock_probe"}:
                    bridge.observe_source_epoch(
                        packet,
                        received_wall=host_received_wall,
                        received_monotonic=host_received_monotonic,
                    )
                elif packet.get("type") == "telemetry" and isinstance(packet.get("telemetry"), dict):
                    # Telemetry-only receipts share the source clock with
                    # RGB-D packets but do not pass through ``publish()``.
                    # Normalize their ROS history at the same boundary so a
                    # reconnect/clock step cannot insert an old timestamp
                    # between two fresh capture poses.  The helper is
                    # generation-aware and is a no-op for healthy traffic.
                    epoch_status = bridge.observe_source_epoch(
                        packet,
                        received_wall=host_received_wall,
                        received_monotonic=host_received_monotonic,
                    )
                    # An old client may provide only one hello.  Its first two
                    # low-rate telemetry packets complete the same three-sample
                    # startup envelope, but must not seed TF/odom under an
                    # uncalibrated cross-host epoch in the meantime.
                    if epoch_status.get("valid") is not True:
                        rospy.logwarn_throttle(
                            5.0,
                            "source epoch warming; skipping telemetry pose "
                            "generation=%s/%s samples=%s/%s",
                            transport_session,
                            transport_connection,
                            epoch_status.get("observation_count", 0),
                            epoch_status.get("min_observations", 3),
                        )
                        continue
                    try:
                        source_stamp = float(packet.get("stamp", time.time()) or time.time())
                    except (TypeError, ValueError):
                        source_stamp = time.time()
                    with bridge._packet_lock:
                        ros_stamp, _stamp_rebased = bridge._normalise_ros_capture_stamp(
                            packet, source_stamp, stream="telemetry"
                        )
                        telemetry_stamp_mapping = dict(
                            getattr(bridge, "_last_telemetry_stamp_mapping", {})
                        )
                    bridge.update_telemetry(
                        packet["telemetry"],
                        ros_stamp,
                        source_reference_stamp=source_stamp,
                        source_clock_rebased=bool(
                            _stamp_rebased
                            or telemetry_stamp_mapping.get("source_epoch_changed") is True
                        ),
                        transport_session=transport_session,
                        transport_connection=transport_connection,
                    )
            except Exception as exc:
                rospy.logwarn_throttle(5.0, "sensor packet: %s", exc)

    async with websockets.serve(
        handler,
        args.host,
        args.port,
        max_size=args.max_message_mb * 1024 * 1024,
        # Application processing is already latest-only. Prevent the
        # websocket library from accumulating/decompressing a hidden backlog
        # of stale multi-megabyte RGB-D packets before that gate.
        max_queue=1,
        ping_interval=None,
    ):
        rospy.loginfo("direct sensor ROS WebSocket: ws://%s:%d", args.host, args.port)
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12335)
    parser.add_argument("--url", default="")
    parser.add_argument("--point-stride", type=int, default=6)
    parser.add_argument("--max-depth-m", type=float, default=8.0)
    parser.add_argument("--no-return-depth-m", type=float, default=8.05)
    parser.add_argument(
        "--telemetry-max-delta-sec",
        type=float,
        default=0.15,
        help="maximum capture-time RGB-D/telemetry timestamp mismatch",
    )
    parser.add_argument("--camera-frame", default="d435i_color_optical_frame")
    parser.add_argument("--camera-parent", default="tf_frame_base_link")
    parser.add_argument("--camera-x", type=float, default=0.03)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--camera-z", type=float, default=0.98)
    parser.add_argument("--camera-roll", type=float, default=0.0)
    parser.add_argument("--camera-pitch", type=float, default=0.1396263)
    parser.add_argument("--camera-yaw", type=float, default=0.0)
    # The D435i stream is optional on older Go2 firmware.  Keep correction
    # enabled by default here so a valid calibrated sample is used whenever
    # it is present, while identity is retained during calibration/missing
    # telemetry.  ``--disable-camera-imu`` is intentionally explicit because
    # argparse's BooleanOptionalAction is not available on ROS Noetic Python.
    parser.add_argument("--camera-imu-enabled", action="store_true", default=True)
    parser.add_argument(
        "--disable-camera-imu", action="store_false", dest="camera_imu_enabled"
    )
    parser.add_argument("--camera-imu-use-yaw", action="store_true", default=False)
    parser.add_argument("--camera-imu-max-roll-rad", type=float, default=0.7)
    parser.add_argument("--camera-imu-max-pitch-rad", type=float, default=0.7)
    parser.add_argument("--camera-imu-max-yaw-rad", type=float, default=0.35)
    parser.add_argument("--max-message-mb", type=int, default=16)
    parser.add_argument("--web-url", default="http://127.0.0.1:8765")
    parser.add_argument("--web-mirror-enabled", action="store_true")
    # roslaunch appends ROS remapping and log arguments after the executable.
    # They are consumed by rospy, not this argparse interface.
    args, _unknown_ros_args = parser.parse_known_args()
    _patch_roslogging_findcaller_for_py311()
    rospy.init_node("physical_sensor_ros_bridge")
    for name in ("host", "port", "url", "point_stride", "max_depth_m", "no_return_depth_m", "telemetry_max_delta_sec", "camera_frame", "camera_parent", "camera_x", "camera_y", "camera_z", "camera_roll", "camera_pitch", "camera_yaw", "camera_imu_enabled", "camera_imu_use_yaw", "camera_imu_max_roll_rad", "camera_imu_max_pitch_rad", "camera_imu_max_yaw_rad", "max_message_mb", "web_url", "web_mirror_enabled"):
        setattr(args, name, rospy.get_param("~" + name, getattr(args, name)))
    if args.url:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(str(args.url))
            if parsed.port:
                args.port = parsed.port
        except ValueError:
            rospy.logwarn("invalid sensor WebSocket URL: %s", args.url)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
