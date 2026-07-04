from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
import time


CLUSTER_ACTIVE = "active"
CLUSTER_COVERED = "covered"
CLUSTER_FAILED = "failed"
CLUSTER_BLACKLISTED = "blacklisted"

SUBGOAL_IDLE = "idle"
SUBGOAL_SENT = "sent"
SUBGOAL_WAITING = "waiting"
SUBGOAL_REACHED = "reached"
SUBGOAL_REACHED_POSE_ONLY = "reached_pose_only"
SUBGOAL_FAILED = "failed"
SUBGOAL_STALLED = "stalled"


@dataclass
class BlockedGoalPoint:
    point: tuple[float, float]
    expire_time: float
    radius_m: float


@dataclass
class ClusterRecord:
    cluster_id: str
    status: str = CLUSTER_ACTIVE
    centroid_world: tuple[float, float] = (0.0, 0.0)
    subgoal_world: tuple[float, float] = (0.0, 0.0)
    last_seen: float = 0.0
    updated_at: float = 0.0
    failure_count: int = 0
    visit_count: int = 0
    blacklist_until: float = 0.0


@dataclass
class ActiveGoal:
    cluster_id: str
    point: tuple[float, float]
    sent_at: float
    last_progress_at: float
    last_robot_xy: tuple[float, float]
    goal_id: str = ""
    status: str = SUBGOAL_SENT


@dataclass
class ExplorerStateConfig:
    goal_reach_tolerance_m: float = 0.75
    goal_timeout_sec: float = 90.0
    stall_timeout_sec: float = 30.0
    stall_distance_m: float = 0.15
    blacklist_duration_sec: float = 25.0
    failed_cluster_retry_sec: float = 120.0
    frontier_match_distance_m: float = 1.0
    failed_point_blacklist_sec: float = 180.0
    failed_point_blacklist_radius_m: float = 1.25
    reached_point_blacklist_sec: float = 90.0
    reached_point_blacklist_radius_m: float = 0.75


@dataclass
class ExplorerState:
    config: ExplorerStateConfig = field(default_factory=ExplorerStateConfig)
    records: dict[str, ClusterRecord] = field(default_factory=dict)
    active_goal: ActiveGoal | None = None
    blocked_goal_points: list[BlockedGoalPoint] = field(default_factory=list)
    last_event: str = ""
    last_failure_reason: str = ""
    last_failure_source: str = ""
    last_replan_reason: str = ""
    last_replan_source: str = ""

    def update_seen_clusters(self, clusters, now: float | None = None) -> None:
        now = self._now(now)
        for cluster in clusters:
            record = self.records.get(cluster.cluster_id)
            if record is None:
                record = ClusterRecord(cluster_id=cluster.cluster_id)
                self.records[cluster.cluster_id] = record
            record.centroid_world = cluster.centroid_world
            record.subgoal_world = cluster.subgoal_world
            record.last_seen = now
            record.updated_at = now
            if record.status == CLUSTER_FAILED and now >= record.blacklist_until:
                record.status = CLUSTER_ACTIVE

    def is_cluster_available(self, cluster, now: float | None = None) -> bool:
        now = self._now(now)
        record = self.records.get(cluster.cluster_id)
        if record is None:
            return True
        if record.status == CLUSTER_COVERED:
            return False
        if record.status in (CLUSTER_FAILED, CLUSTER_BLACKLISTED) and now < record.blacklist_until:
            return False
        return True

    def start_goal(self, cluster, robot_xy: tuple[float, float], goal_id: str = "", now: float | None = None) -> ActiveGoal:
        now = self._now(now)
        self.active_goal = ActiveGoal(
            cluster_id=cluster.cluster_id,
            point=cluster.subgoal_world,
            sent_at=now,
            last_progress_at=now,
            last_robot_xy=robot_xy,
            goal_id=goal_id,
            status=SUBGOAL_SENT,
        )
        record = self._record_for(cluster.cluster_id)
        record.status = CLUSTER_ACTIVE
        record.centroid_world = cluster.centroid_world
        record.subgoal_world = cluster.subgoal_world
        record.visit_count += 1
        record.updated_at = now
        self.last_event = "goal_sent"
        return self.active_goal

    def clear_active_goal(self, status: str, now: float | None = None, event: str | None = None) -> None:
        now = self._now(now)
        if self.active_goal is not None:
            self.active_goal.status = status
        self.active_goal = None
        self.last_event = status if event is None else event

    def mark_active_reached(self, now: float | None = None) -> None:
        now = self._now(now)
        if self.active_goal is None:
            return
        record = self._record_for(self.active_goal.cluster_id)
        record.status = CLUSTER_COVERED
        record.updated_at = now
        self.clear_active_goal(SUBGOAL_REACHED, now)

    def mark_active_reached_pose_only(self, now: float | None = None) -> None:
        now = self._now(now)
        if self.active_goal is None:
            return
        self.block_goal_point(
            self.active_goal.point,
            duration_sec=self.config.reached_point_blacklist_sec,
            radius_m=self.config.reached_point_blacklist_radius_m,
            now=now,
        )
        record = self._record_for(self.active_goal.cluster_id)
        record.status = CLUSTER_ACTIVE
        record.updated_at = now
        self.clear_active_goal(SUBGOAL_REACHED_POSE_ONLY, now)

    def mark_active_failed(self, reason: str, now: float | None = None, source: str = "explorer") -> None:
        now = self._now(now)
        if self.active_goal is None:
            return
        record = self._record_for(self.active_goal.cluster_id)
        record.status = CLUSTER_FAILED
        record.failure_count += 1
        record.blacklist_until = now + self.config.failed_cluster_retry_sec
        record.updated_at = now
        self.block_goal_point(
            self.active_goal.point,
            duration_sec=self.config.failed_point_blacklist_sec,
            radius_m=self.config.failed_point_blacklist_radius_m,
            now=now,
        )
        self.last_failure_reason = reason
        self.last_failure_source = source
        self.last_replan_reason = reason
        self.last_replan_source = source
        self.clear_active_goal(SUBGOAL_FAILED, now, event=reason)

    def blacklist_cluster(self, cluster_id: str, now: float | None = None) -> None:
        now = self._now(now)
        record = self._record_for(cluster_id)
        record.status = CLUSTER_BLACKLISTED
        record.blacklist_until = now + self.config.blacklist_duration_sec
        record.updated_at = now
        self.last_event = "cluster_blacklisted"

    def update_goal_progress(self, robot_xy: tuple[float, float], now: float | None = None) -> str:
        now = self._now(now)
        goal = self.active_goal
        if goal is None:
            return SUBGOAL_IDLE
        goal.status = SUBGOAL_WAITING
        dist_to_goal = hypot(goal.point[0] - robot_xy[0], goal.point[1] - robot_xy[1])
        if dist_to_goal <= self.config.goal_reach_tolerance_m:
            return SUBGOAL_REACHED
        if now - goal.sent_at > self.config.goal_timeout_sec:
            self.mark_active_failed("goal_timeout", now, source="explorer")
            return SUBGOAL_FAILED
        moved = hypot(robot_xy[0] - goal.last_robot_xy[0], robot_xy[1] - goal.last_robot_xy[1])
        if moved >= self.config.stall_distance_m:
            goal.last_robot_xy = robot_xy
            goal.last_progress_at = now
        elif now - goal.last_progress_at > self.config.stall_timeout_sec:
            self.mark_active_failed("goal_stalled", now, source="explorer")
            return SUBGOAL_STALLED
        return SUBGOAL_WAITING

    def mark_active_covered_if_frontier_gone(self, has_frontier: bool, now: float | None = None) -> bool:
        if self.active_goal is None or has_frontier:
            return False
        self.mark_active_reached(now)
        self.last_event = "frontier_gone"
        return True

    def fail_active_if_goal_not_free(self, is_free: bool, now: float | None = None) -> bool:
        if self.active_goal is None or is_free:
            return False
        self.mark_active_failed("goal_not_free", now, source="explorer")
        return True

    def block_goal_point(
        self,
        point: tuple[float, float],
        duration_sec: float,
        radius_m: float | None = None,
        now: float | None = None,
    ) -> None:
        now = self._now(now)
        self._purge_blocked_goal_points(now)
        self.blocked_goal_points.append(
            BlockedGoalPoint(
                point=point,
                expire_time=now + duration_sec,
                radius_m=self.config.reached_point_blacklist_radius_m if radius_m is None else radius_m,
            )
        )

    def is_goal_point_blocked(self, point: tuple[float, float], now: float | None = None) -> bool:
        now = self._now(now)
        self._purge_blocked_goal_points(now)
        for item in self.blocked_goal_points:
            if hypot(point[0] - item.point[0], point[1] - item.point[1]) <= item.radius_m:
                return True
        return False

    def revisit_penalty(self, cluster) -> float:
        record = self.records.get(cluster.cluster_id)
        if record is None:
            return 0.0
        return min(1.0, 0.25 * float(record.visit_count))

    def failure_penalty(self, cluster) -> float:
        record = self.records.get(cluster.cluster_id)
        if record is None:
            return 0.0
        return min(1.0, 0.5 * float(record.failure_count))

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for record in self.records.values():
            counts[record.status] = counts.get(record.status, 0) + 1
        active = None
        if self.active_goal is not None:
            active = {
                "cluster_id": self.active_goal.cluster_id,
                "point": list(self.active_goal.point),
                "status": self.active_goal.status,
                "age_sec": max(0.0, time.time() - self.active_goal.sent_at),
            }
        return {
            "cluster_counts": counts,
            "active_goal": active,
            "blocked_goal_points": len(self.blocked_goal_points),
            "last_event": self.last_event,
            "last_failure_reason": self.last_failure_reason,
            "last_failure_source": self.last_failure_source,
            "last_replan_reason": self.last_replan_reason,
            "last_replan_source": self.last_replan_source,
        }

    def _record_for(self, cluster_id: str) -> ClusterRecord:
        record = self.records.get(cluster_id)
        if record is None:
            record = ClusterRecord(cluster_id=cluster_id)
            self.records[cluster_id] = record
        return record

    @staticmethod
    def _now(now: float | None = None) -> float:
        return float(time.time() if now is None else now)

    def _purge_blocked_goal_points(self, now: float) -> None:
        self.blocked_goal_points = [item for item in self.blocked_goal_points if item.expire_time > now]
