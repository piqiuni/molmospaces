"""Small causal RGB cache used by the debug recorder.

The simulator publishes the RGB image and the recorder's ``step_sync`` marker
on separate ROS connections.  ROS does not guarantee callback ordering across
those connections, so the recorder needs a bounded wait for the image carrying
the same ``(header.seq, header.stamp)`` before it renders a step.  Keeping this
piece independent of rospy makes the pairing behavior easy to test.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class CachedStepImage:
    """An immutable snapshot of one decoded RGB image."""

    source_seq: int
    stamp_ns: int
    stamp: float
    width: int
    height: int
    rgb: bytes


@dataclass(frozen=True)
class StepImageSelection:
    """Result of pairing one step marker with a decoded camera image."""

    frame: CachedStepImage
    source: str
    reused: bool
    stamp_delta_ns: int


class ExactStepImageCache:
    """Bounded cache supporting exact and nearest causal selection."""

    def __init__(self, max_size: int = 64) -> None:
        self.max_size = max(1, int(max_size))
        self._condition = threading.Condition()
        self._frames: OrderedDict[tuple[int, int], CachedStepImage] = OrderedDict()
        self._latest_received_frame: CachedStepImage | None = None
        self._latest_selected_frame: CachedStepImage | None = None
        self._put_generation = 0
        self._last_wait_exhausted_generation: int | None = None
        self._closed = False

    def put(self, frame: CachedStepImage) -> None:
        key = (int(frame.source_seq), int(frame.stamp_ns))
        with self._condition:
            if self._closed:
                return
            self._frames[key] = frame
            self._frames.move_to_end(key)
            self._latest_received_frame = frame
            self._put_generation += 1
            while len(self._frames) > self.max_size:
                self._frames.popitem(last=False)
            self._condition.notify_all()

    def wait_select(
        self,
        source_seq: int,
        stamp_ns: int,
        timeout_sec: float,
        max_stamp_delta_ns: int,
        fallback_max_age_ns: int,
    ) -> StepImageSelection | None:
        """Select the exact image, then the nearest real causal image.

        RosBridge publishes RGB before ``step_sync`` with a shared logical
        stamp, but subscriber callback completion is not ordered.  A slow or
        dropped image callback must therefore not turn the video black.  We
        wait for an exact sequence/stamp match first, accept a timestamp match
        when the sequence metadata differs, and finally reuse the nearest real
        image within a bounded age while reporting that reuse explicitly.
        """

        source_seq = int(source_seq)
        stamp_ns = int(stamp_ns)
        max_stamp_delta_ns = max(0, int(max_stamp_delta_ns))
        fallback_max_age_ns = max(max_stamp_delta_ns, int(fallback_max_age_ns))

        def key_for_immediate_match() -> tuple[tuple[int, int], str] | None:
            exact = [
                key
                for key in self._frames
                if key[0] == source_seq
                and abs(key[1] - stamp_ns) <= max_stamp_delta_ns
            ]
            if exact:
                return min(exact, key=lambda key: abs(key[1] - stamp_ns)), "exact_image"
            timestamp_matches = [
                key
                for key in self._frames
                if abs(key[1] - stamp_ns) <= max_stamp_delta_ns
            ]
            if timestamp_matches:
                return (
                    min(timestamp_matches, key=lambda key: abs(key[1] - stamp_ns)),
                    "timestamp_image",
                )
            return None

        def stream_has_passed_target() -> bool:
            """Return whether the ordered image stream has moved past this step.

            ``rospy`` invokes callbacks from one image connection in order, and
            the recorder also rejects non-increasing image sequence numbers.  A
            decoded image newer than this marker therefore proves that waiting
            longer cannot produce its exact image.  This is what prevents a
            delayed step-sync backlog from paying one full timeout per dropped
            frame.
            """

            latest = self._latest_received_frame
            if latest is None:
                return False
            return (
                int(latest.source_seq) > source_seq
                or int(latest.stamp_ns) > stamp_ns + max_stamp_delta_ns
            )

        def nearest_fallback() -> tuple[tuple[int, int], str] | None:
            candidates = [
                key
                for key in self._frames
                if key[1] <= stamp_ns + max_stamp_delta_ns
                and stamp_ns - key[1] <= fallback_max_age_ns
            ]
            if not candidates:
                return None
            return (
                min(candidates, key=lambda key: abs(key[1] - stamp_ns)),
                "nearest_image",
            )

        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        with self._condition:
            matched = key_for_immediate_match()
            should_wait = (
                matched is None
                and not stream_has_passed_target()
                and self._last_wait_exhausted_generation != self._put_generation
            )
            while not self._closed and matched is None and should_wait:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self._last_wait_exhausted_generation = self._put_generation
                    break
                self._condition.wait(timeout=remaining)
                matched = key_for_immediate_match()
                if matched is None and stream_has_passed_target():
                    break
            if matched is None:
                matched = nearest_fallback()
            if matched is None and self._latest_selected_frame is not None:
                latest_key = (
                    int(self._latest_selected_frame.source_seq),
                    int(self._latest_selected_frame.stamp_ns),
                )
                if (
                    latest_key[1] <= stamp_ns + max_stamp_delta_ns
                    and stamp_ns - latest_key[1] <= fallback_max_age_ns
                ):
                    matched = latest_key, "latest_image"
            if matched is None:
                return None
            key, source = matched
            frame = self._frames.pop(key, None)
            if frame is None and self._latest_selected_frame is not None:
                latest_key = (
                    int(self._latest_selected_frame.source_seq),
                    int(self._latest_selected_frame.stamp_ns),
                )
                frame = self._latest_selected_frame if latest_key == key else None
            if frame is None and self._latest_received_frame is not None:
                latest_key = (
                    int(self._latest_received_frame.source_seq),
                    int(self._latest_received_frame.stamp_ns),
                )
                frame = self._latest_received_frame if latest_key == key else None
            if frame is None:
                return None
            reused = source == "latest_image"
            self._latest_selected_frame = frame
            return StepImageSelection(
                frame=frame,
                source=source,
                reused=reused,
                stamp_delta_ns=abs(int(frame.stamp_ns) - stamp_ns),
            )

    def wait_pop(
        self,
        source_seq: int,
        stamp_ns: int,
        timeout_sec: float,
        max_stamp_delta_ns: int = 0,
    ) -> CachedStepImage | None:
        """Wait briefly for and consume the matching image, if it arrives.

        ``step_sync`` carries its timestamp through JSON as a floating-point
        number, while ``sensor_msgs/Image`` retains integer nanoseconds.  At
        epoch-sized timestamps that round-trip can differ by a few hundred
        nanoseconds.  The sequence must therefore match exactly, while callers
        may permit a small timestamp tolerance.
        """

        key = (int(source_seq), int(stamp_ns))
        max_stamp_delta_ns = max(0, int(max_stamp_delta_ns))

        def matching_key() -> tuple[int, int] | None:
            if key in self._frames:
                return key
            candidates = [
                candidate
                for candidate in self._frames
                if candidate[0] == key[0]
                and abs(candidate[1] - key[1]) <= max_stamp_delta_ns
            ]
            if not candidates:
                return None
            return min(candidates, key=lambda candidate: abs(candidate[1] - key[1]))

        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        with self._condition:
            matched_key = matching_key()
            while not self._closed and matched_key is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(timeout=remaining)
                matched_key = matching_key()
            return None if matched_key is None else self._frames.pop(matched_key, None)

    @property
    def active_size(self) -> int:
        with self._condition:
            return len(self._frames)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
