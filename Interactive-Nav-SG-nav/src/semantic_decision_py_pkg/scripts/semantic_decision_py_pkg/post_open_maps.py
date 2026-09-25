"""ROS-independent post-open map receipts and causal freshness contracts.

The executor owns counters, admission caches and synchronization. This module
only evaluates immutable receipt snapshots; it never subscribes or waits.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PostInteractionCostmapBaseline:
    """Map publications observed before a successful portal continuation.

    The post-open gate must follow the actual planner input, not merely a
    convenient costmap notification: raw SLAM occupancy -> semantic planning
    occupancy -> global costmap.  Receipt counters are local to the executor
    rather than ROS header sequences, which can reset with move_base.
    """

    portal_id: str
    source_event_id: str
    receipt_count: int
    header_seq: int | None = None
    update_receipt_count: int = 0
    update_header_seq: int | None = None
    raw_occupancy_receipt_count: int = 0
    raw_occupancy_header_seq: int | None = None
    raw_occupancy_header_stamp_sec: float | None = None
    planning_occupancy_receipt_count: int = 0
    planning_occupancy_header_seq: int | None = None
    planning_occupancy_header_stamp_sec: float | None = None
    interaction_result_stamp_sec: float | None = None


@dataclass(frozen=True)
class PostInteractionRawMapBarrier:
    """One raw map receipt admitted after a portal-open result.

    ``planning_occupancy_receipt_count`` is captured in the raw-map callback,
    so a planning map already received before this raw map can never satisfy
    the next stage merely because the executor wakes late.
    """

    receipt_count: int
    header_seq: int | None
    header_stamp_sec: float | None
    planning_occupancy_receipt_count: int
    planning_occupancy_header_seq: int | None = None
    planning_occupancy_header_stamp_sec: float | None = None
    # Snapshots taken in the raw-map callback let a physical deployment use a
    # fast raw-OCC -> costmap path without mistaking a costmap receipt that
    # arrived before the raw map for a causal update.
    global_costmap_receipt_count: int = 0
    global_costmap_update_receipt_count: int = 0
    global_costmap_header_stamp_sec: float | None = None
    global_costmap_update_header_stamp_sec: float | None = None


@dataclass(frozen=True)
class PostInteractionPlanningMapBarrier:
    """One planning-map receipt after a qualifying raw map.

    The costmap counters are sampled at this receipt.  A later global full or
    incremental update is therefore causally downstream of the planner input
    accepted by this barrier.
    """

    raw_map: PostInteractionRawMapBarrier
    raw_fresh_source: str
    receipt_count: int
    header_seq: int | None
    header_stamp_sec: float | None
    planning_fresh_source: str
    costmap_receipt_count: int
    costmap_header_seq: int | None
    costmap_update_receipt_count: int
    costmap_update_header_seq: int | None


def _positive_finite_stamp(value: object) -> float | None:
    try:
        stamp = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(stamp) or stamp <= 0.0:
        return None
    return stamp


def post_interaction_raw_occupancy_fresh_source(
    baseline: PostInteractionCostmapBaseline | None,
    current_receipt_count: int,
    current_header_stamp_sec: float | None,
    *,
    allow_callback_reorder: bool = False,
) -> str:
    """Return how a raw OCC receipt proves it followed the open result.

    When both stamps are available, require the source map's stamp to be
    strictly newer than the evaluator result.  A zero/missing stamp falls back
    to the executor's post-result receipt boundary and is labelled explicitly
    for benchmark diagnostics.
    """

    if baseline is None:
        return ""
    try:
        receipt_advanced = int(current_receipt_count) > int(
            baseline.raw_occupancy_receipt_count
        )
    except (TypeError, ValueError):
        return ""
    result_stamp = _positive_finite_stamp(baseline.interaction_result_stamp_sec)
    raw_stamp = _positive_finite_stamp(current_header_stamp_sec)
    # The raw-map callback and interaction-result callback are independent ROS
    # connections.  A map generated after the successful action can therefore
    # be delivered just before the result callback snapshots its counters.  A
    # strictly newer source stamp proves that ordering even when the local
    # receipt count is already in the baseline; without that timestamp proof
    # retain the normal post-result receipt boundary.
    if not receipt_advanced:
        if (
            allow_callback_reorder
            and result_stamp is not None
            and raw_stamp is not None
            and raw_stamp > result_stamp
        ):
            return "header_stamp_reordered"
        return ""
    if result_stamp is not None and raw_stamp is not None:
        return "header_stamp" if raw_stamp > result_stamp else ""
    if result_stamp is not None:
        return "receipt_after_result_no_raw_stamp"
    return "receipt_after_result"


def post_interaction_planning_occupancy_fresh_source(
    raw_barrier: PostInteractionRawMapBarrier | None,
    current_receipt_count: int,
    current_header_stamp_sec: float | None,
) -> str:
    """Return how a planning OCC receipt proves it follows the raw map.

    Semantic mapping preserves the raw occupancy stamp on
    ``planning_occ_map``.  If both stamps are present, require that source
    relationship.  Otherwise the raw callback's receipt snapshot still
    guarantees a strictly later local receipt.
    """

    if raw_barrier is None or int(current_receipt_count) <= int(
        raw_barrier.planning_occupancy_receipt_count
    ):
        return ""
    raw_stamp = _positive_finite_stamp(raw_barrier.header_stamp_sec)
    planning_stamp = _positive_finite_stamp(current_header_stamp_sec)
    if raw_stamp is not None and planning_stamp is not None:
        # Matching is normal because semantic_mapping copies the raw header;
        # a later stamp is also valid if a downstream map rebuild occurs.
        return "source_header_stamp" if planning_stamp >= raw_stamp else ""
    if raw_stamp is not None:
        return "receipt_after_raw_no_planning_stamp"
    return "receipt_after_raw"


def post_interaction_costmap_is_fresh(
    baseline: PostInteractionCostmapBaseline | None,
    current_receipt_count: int,
    current_update_receipt_count: int = 0,
) -> bool:
    """Return whether either global-map stream advanced after the open result."""

    return bool(
        post_interaction_costmap_fresh_source(
            baseline,
            current_receipt_count,
            current_update_receipt_count,
        )
    )


def post_interaction_costmap_fresh_source(
    baseline: PostInteractionCostmapBaseline | None,
    current_receipt_count: int,
    current_update_receipt_count: int = 0,
) -> str:
    """Identify the stream that made a post-open planner-map gate fresh.

    Prefer the costmap delta topic whenever both streams are new: it is the
    low-latency actual update path and avoids tying correctness to expensive
    full-grid publications.
    """

    if baseline is None:
        return ""
    return post_interaction_costmap_receipts_fresh_source(
        baseline.receipt_count,
        baseline.update_receipt_count,
        current_receipt_count,
        current_update_receipt_count,
    )


def post_interaction_costmap_receipts_fresh_source(
    baseline_receipt_count: int,
    baseline_update_receipt_count: int,
    current_receipt_count: int,
    current_update_receipt_count: int = 0,
) -> str:
    """Return a later costmap source from explicit receipt-counter bounds."""

    if int(current_update_receipt_count) > int(baseline_update_receipt_count):
        return "costmap_update"
    if int(current_receipt_count) > int(baseline_receipt_count):
        return "full"
    return ""


def post_interaction_global_costmap_fresh_source(
    raw_barrier: PostInteractionRawMapBarrier | None,
    current_receipt_count: int,
    current_update_receipt_count: int = 0,
    current_header_stamp_sec: float | None = None,
    current_update_header_stamp_sec: float | None = None,
    *,
    allow_callback_reorder: bool = False,
) -> str:
    """Return a global-costmap stream causally downstream of a raw OCC.

    Normally the executor receives the raw OCC before the global-costmap
    callback, so a local receipt-counter advance is sufficient.  ROS does not
    guarantee callback ordering across topics, however.  A caller may opt in
    to accepting a callback delivered first when the costmap publisher's
    header stamp is a trusted source-map stamp.  This is deliberately disabled
    by default: the physical costmap publisher stamps messages with
    ``ros::Time::now()``, which is not evidence that the map was built from the
    new raw OCC.  In that lane the receipt boundary is the only causal proof.
    """

    if raw_barrier is None:
        return ""
    raw_stamp = _positive_finite_stamp(raw_barrier.header_stamp_sec)

    def stream_source(
        current_count: int,
        barrier_count: int,
        current_stamp: float | None,
        stream_name: str,
    ) -> str:
        try:
            current_count = int(current_count)
            barrier_count = int(barrier_count)
        except (TypeError, ValueError):
            return ""
        source_stamp = _positive_finite_stamp(current_stamp)
        if current_count > barrier_count:
            if (
                raw_stamp is not None
                and source_stamp is not None
                and source_stamp < raw_stamp
            ):
                # A newer local callback carrying an older source map is a
                # residual/stale costmap and must not release make_plan.
                return ""
            return stream_name
        if (
            allow_callback_reorder
            and current_count > 0
            and raw_stamp is not None
            and source_stamp is not None
            and source_stamp >= raw_stamp
        ):
            return f"{stream_name}_header_stamp_reordered"
        return ""

    # Incremental updates are the low-latency planner signal.  Prefer them
    # whenever both streams can prove freshness.
    update_source = stream_source(
        current_update_receipt_count,
        raw_barrier.global_costmap_update_receipt_count,
        current_update_header_stamp_sec,
        "raw_occupancy_to_global_costmap_update",
    )
    if update_source:
        return update_source
    return stream_source(
        current_receipt_count,
        raw_barrier.global_costmap_receipt_count,
        current_header_stamp_sec,
        "raw_occupancy_to_global_costmap_full",
    )


def post_interaction_costmap_baseline_keys(
    source_event_id: object,
    portal_id: object,
) -> tuple[str, ...]:
    """Return exact-event then portal fallback keys for a traversal barrier."""

    event_id = str(source_event_id or "").strip()
    normalized_portal_id = str(portal_id or "").strip()
    keys: list[str] = []
    if event_id:
        keys.append(f"event:{event_id}")
    if normalized_portal_id:
        keys.append(f"portal:{normalized_portal_id}")
    return tuple(keys)


@dataclass(frozen=True)
class MapReceipt:
    count: int = 0
    seq: int | None = None
    stamp_sec: float | None = None


@dataclass(frozen=True)
class MapReceiptSnapshot:
    """Counters only, never a copy of the occupancy arrays."""

    raw: MapReceipt
    planning: MapReceipt
    full: MapReceipt
    update: MapReceipt


@dataclass(frozen=True)
class MapFreshness:
    stage: str
    source: str = ""
    direct_raw_costmap: bool = False

    @property
    def fresh(self):
        return bool(self.source)

    @property
    def timeout_reason(self):
        return {
            "waiting_raw_occupancy": "post_open_raw_occ_refresh_timeout",
            "waiting_planning_occupancy": "post_open_planning_occ_refresh_timeout",
        }.get(self.stage, "post_open_costmap_refresh_timeout")


@dataclass(frozen=True)
class PostOpenMapPolicy:
    """Select the actual StaticLayer input, not a simulator/robot class name."""

    direct_raw_costmap: bool = False

    @property
    def requires_planning_occupancy(self):
        return not self.direct_raw_costmap

    def evaluate(
        self,
        baseline: PostInteractionCostmapBaseline | None,
        raw_barrier: PostInteractionRawMapBarrier | None,
        planning_barrier: PostInteractionPlanningMapBarrier | None,
        snapshot: MapReceiptSnapshot,
    ) -> MapFreshness:
        if baseline is None or raw_barrier is None:
            return MapFreshness("waiting_raw_occupancy")
        if self.direct_raw_costmap:
            # Receipt order is required: a publisher's local-now header alone
            # cannot prove that it consumed the just-admitted raw occupancy.
            source = post_interaction_global_costmap_fresh_source(
                raw_barrier, snapshot.full.count, snapshot.update.count,
                snapshot.full.stamp_sec, snapshot.update.stamp_sec,
                allow_callback_reorder=False,
            )
        elif planning_barrier is None:
            return MapFreshness("waiting_planning_occupancy")
        else:
            source = post_interaction_costmap_receipts_fresh_source(
                planning_barrier.costmap_receipt_count,
                planning_barrier.costmap_update_receipt_count,
                snapshot.full.count, snapshot.update.count,
            )
        return MapFreshness("ready" if source else "waiting_global_costmap",
                            source, self.direct_raw_costmap)


def post_open_map_detail(
    candidate, baseline_key, baseline, raw_barrier, raw_fresh_source,
    planning_barrier, snapshot, freshness, *, elapsed_s, timeout_s,
):
    """Preserve public trace keys for both source-map wiring profiles."""
    current_full_count = snapshot.full.count
    current_full_header_seq = snapshot.full.seq
    current_full_header_stamp_sec = snapshot.full.stamp_sec
    current_update_count = snapshot.update.count
    current_update_header_seq = snapshot.update.seq
    current_update_header_stamp_sec = snapshot.update.stamp_sec
    current_raw_count = snapshot.raw.count
    current_raw_header_seq = snapshot.raw.seq
    current_raw_header_stamp_sec = snapshot.raw.stamp_sec
    current_planning_count = snapshot.planning.count
    current_planning_header_seq = snapshot.planning.seq
    current_planning_header_stamp_sec = snapshot.planning.stamp_sec
    causal_stage = freshness.stage
    costmap_baseline_full_count = (
        None
        if baseline is None
        else (
            planning_barrier.costmap_receipt_count
            if planning_barrier is not None
            else baseline.receipt_count
        )
    )
    costmap_baseline_full_header_seq = (
        None
        if baseline is None
        else (
            planning_barrier.costmap_header_seq
            if planning_barrier is not None
            else baseline.header_seq
        )
    )
    costmap_baseline_update_count = (
        None
        if baseline is None
        else (
            planning_barrier.costmap_update_receipt_count
            if planning_barrier is not None
            else baseline.update_receipt_count
        )
    )
    costmap_baseline_update_header_seq = (
        None
        if baseline is None
        else (
            planning_barrier.costmap_update_header_seq
            if planning_barrier is not None
            else baseline.update_header_seq
        )
    )
    detail = {
        "opened_portal_id": str(
            (candidate.get("metadata") or {}).get("opened_portal_id")
            or candidate.get("target_id")
            or ""
        ),
        "post_open_costmap_baseline_key": baseline_key,
        "post_open_costmap_baseline_receipt_count": (
            costmap_baseline_full_count
        ),
        "post_open_costmap_baseline_header_seq": (
            costmap_baseline_full_header_seq
        ),
        "post_open_costmap_latest_receipt_count": current_full_count,
        "post_open_costmap_latest_header_seq": current_full_header_seq,
        "post_open_costmap_latest_header_stamp_sec": (
            current_full_header_stamp_sec
        ),
        "post_open_costmap_baseline_update_receipt_count": (
            costmap_baseline_update_count
        ),
        "post_open_costmap_baseline_update_header_seq": (
            costmap_baseline_update_header_seq
        ),
        "post_open_costmap_latest_update_receipt_count": (
            current_update_count
        ),
        "post_open_costmap_latest_update_header_seq": (
            current_update_header_seq
        ),
        "post_open_costmap_latest_update_header_stamp_sec": (
            current_update_header_stamp_sec
        ),
        "post_open_costmap_wait_elapsed_s": elapsed_s,
        "post_open_costmap_wait_timeout_s": (
            timeout_s
        ),
        "post_open_causal_map_stage": causal_stage,
        "post_open_result_stamp_sec": (
            None
            if baseline is None
            else baseline.interaction_result_stamp_sec
        ),
        "post_open_raw_occ_baseline_receipt_count": (
            None
            if baseline is None
            else baseline.raw_occupancy_receipt_count
        ),
        "post_open_raw_occ_baseline_header_seq": (
            None
            if baseline is None
            else baseline.raw_occupancy_header_seq
        ),
        "post_open_raw_occ_baseline_header_stamp_sec": (
            None
            if baseline is None
            else baseline.raw_occupancy_header_stamp_sec
        ),
        "post_open_raw_occ_latest_receipt_count": current_raw_count,
        "post_open_raw_occ_latest_header_seq": current_raw_header_seq,
        "post_open_raw_occ_latest_header_stamp_sec": (
            current_raw_header_stamp_sec
        ),
        "post_open_raw_occ_admitted_receipt_count": (
            None if raw_barrier is None else raw_barrier.receipt_count
        ),
        "post_open_raw_occ_admitted_header_seq": (
            None if raw_barrier is None else raw_barrier.header_seq
        ),
        "post_open_raw_occ_admitted_header_stamp_sec": (
            None
            if raw_barrier is None
            else raw_barrier.header_stamp_sec
        ),
        "post_open_raw_occ_fresh_source": raw_fresh_source,
        "post_open_planning_occ_baseline_receipt_count": (
            None
            if baseline is None
            else baseline.planning_occupancy_receipt_count
        ),
        "post_open_planning_occ_baseline_header_seq": (
            None
            if baseline is None
            else baseline.planning_occupancy_header_seq
        ),
        "post_open_planning_occ_baseline_header_stamp_sec": (
            None
            if baseline is None
            else baseline.planning_occupancy_header_stamp_sec
        ),
        "post_open_planning_occ_latest_receipt_count": (
            current_planning_count
        ),
        "post_open_planning_occ_latest_header_seq": (
            current_planning_header_seq
        ),
        "post_open_planning_occ_latest_header_stamp_sec": (
            current_planning_header_stamp_sec
        ),
        "post_open_planning_occ_admitted_receipt_count": (
            None
            if planning_barrier is None
            else planning_barrier.receipt_count
        ),
        "post_open_planning_occ_admitted_header_seq": (
            None
            if planning_barrier is None
            else planning_barrier.header_seq
        ),
        "post_open_planning_occ_admitted_header_stamp_sec": (
            None
            if planning_barrier is None
            else planning_barrier.header_stamp_sec
        ),
        "post_open_planning_occ_fresh_source": (
            ""
            if planning_barrier is None
            else planning_barrier.planning_fresh_source
        ),
        "post_open_causal_costmap_baseline_receipt_count": (
            costmap_baseline_full_count
        ),
        "post_open_causal_costmap_baseline_update_receipt_count": (
            costmap_baseline_update_count
        ),
        # Stable primary trace keys are local incremental-update
        # receipt sequences; after planning OCC they are sampled
        # at that causal barrier instead of at action result.
        "baseline_global_costmap_seq": (
            costmap_baseline_update_count
        ),
        "observed_global_costmap_seq": current_update_count,
        "baseline_global_costmap_header_seq": (
            costmap_baseline_update_header_seq
        ),
        "observed_global_costmap_header_seq": current_update_header_seq,
        "baseline_global_costmap_full_seq": (
            costmap_baseline_full_count
        ),
        "observed_global_costmap_full_seq": current_full_count,
        "baseline_global_costmap_full_header_seq": (
            costmap_baseline_full_header_seq
        ),
        "observed_global_costmap_full_header_seq": (
            current_full_header_seq
        ),
        "costmap_wait_elapsed": elapsed_s,
        "costmap_fresh": False,
        "fresh_source": "",
    }
    if freshness.fresh:
        detail.update(post_open_costmap_fresh=True, costmap_fresh=True,
                      fresh_source=freshness.source)
        if freshness.direct_raw_costmap:
            detail["post_open_costmap_fast_path"] = True
    return detail
