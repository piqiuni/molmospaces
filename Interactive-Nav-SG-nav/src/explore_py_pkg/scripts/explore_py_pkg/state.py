from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, pi
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
    yaw: float
    sent_at: float
    last_progress_at: float
    last_yaw_progress_at: float
    last_robot_xy: tuple[float, float]
    last_robot_yaw: float | None = None
    goal_id: str = ""
    status: str = SUBGOAL_SENT
    frontier_gone_count: int = 0


@dataclass
class ExplorerStateConfig:
    goal_reach_tolerance_m: float = 0.75
    goal_timeout_sec: float = 90.0
    stall_timeout_sec: float = 30.0
    stall_distance_m: float = 0.15
    min_goal_lifetime_sec: float = 8.0
    stall_yaw_progress_rad: float = 0.20
    rotation_stall_timeout_sec: float = 45.0
    blacklist_duration_sec: float = 25.0
    failed_cluster_retry_sec: float = 120.0
    failed_cluster_max_failures: int = 3
    frontier_match_distance_m: float = 1.0
    frontier_gone_confirm_ticks: int = 3
    frontier_gone_min_goal_age_sec: float = 8.0
    failed_point_soft_blacklist_sec: float = 45.0
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
    last_subgoal_world: tuple[float, float] | None = None

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
            if record.status == CLUSTER_COVERED:
                record.status = CLUSTER_ACTIVE
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

    def start_goal(
        self,
        cluster,
        robot_xy: tuple[float, float],
        robot_yaw: float | None = None,
        goal_id: str = "",
        now: float | None = None,
    ) -> ActiveGoal:
        now = self._now(now)
        self.active_goal = ActiveGoal(
            cluster_id=cluster.cluster_id,
            point=cluster.subgoal_world,
            yaw=float(getattr(cluster, "subgoal_yaw", 0.0)),
            sent_at=now,
            last_progress_at=now,
            last_yaw_progress_at=now,
            last_robot_xy=robot_xy,
            last_robot_yaw=robot_yaw,
            goal_id=goal_id,
            status=SUBGOAL_SENT,
        )
        record = self._record_for(cluster.cluster_id)
        record.status = CLUSTER_ACTIVE
        record.centroid_world = cluster.centroid_world
        record.subgoal_world = cluster.subgoal_world
        record.visit_count += 1
        record.updated_at = now
        self.last_subgoal_world = cluster.subgoal_world
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
        record.failure_count += 1
        hard_failure = record.failure_count >= max(1, self.config.failed_cluster_max_failures)
        if hard_failure:
            record.status = CLUSTER_FAILED
            record.blacklist_until = now + self.config.failed_cluster_retry_sec
        else:
            # A single failed viewpoint should not discard the whole frontier.
            # Keep the cluster active but penalized so another viewpoint can win.
            record.status = CLUSTER_ACTIVE
            record.blacklist_until = 0.0
        record.updated_at = now
        self.block_goal_point(
            self.active_goal.point,
            duration_sec=(
                self.config.failed_point_blacklist_sec
                if hard_failure
                else self.config.failed_point_soft_blacklist_sec
            ),
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

    def update_goal_progress(
        self,
        robot_xy: tuple[float, float],
        robot_yaw: float | None = None,
        now: float | None = None,
    ) -> str:
        now = self._now(now)
        goal = self.active_goal
        if goal is None:
            return SUBGOAL_IDLE
        goal.status = SUBGOAL_WAITING
        goal_age = now - goal.sent_at
        dist_to_goal = hypot(goal.point[0] - robot_xy[0], goal.point[1] - robot_xy[1])
        if dist_to_goal <= self.config.goal_reach_tolerance_m:
            if goal_age < self.config.min_goal_lifetime_sec:
                return SUBGOAL_WAITING
            return SUBGOAL_REACHED
        if goal_age > self.config.goal_timeout_sec:
            self.mark_active_failed("goal_timeout", now, source="explorer")
            return SUBGOAL_FAILED
        moved = hypot(robot_xy[0] - goal.last_robot_xy[0], robot_xy[1] - goal.last_robot_xy[1])
        if moved >= self.config.stall_distance_m:
            goal.last_robot_xy = robot_xy
            goal.last_progress_at = now
            goal.last_yaw_progress_at = now
            if robot_yaw is not None:
                goal.last_robot_yaw = robot_yaw
            self.last_event = "goal_translation_progress"
        elif robot_yaw is not None and goal.last_robot_yaw is not None:
            yaw_delta = self._angle_distance(robot_yaw, goal.last_robot_yaw)
            if yaw_delta >= self.config.stall_yaw_progress_rad:
                goal.last_robot_yaw = robot_yaw
                goal.last_yaw_progress_at = now
                self.last_event = "goal_rotating_to_align"
        elif robot_yaw is not None and goal.last_robot_yaw is None:
            goal.last_robot_yaw = robot_yaw
            goal.last_yaw_progress_at = now

        if goal_age < self.config.min_goal_lifetime_sec:
            return SUBGOAL_WAITING

        linear_stall = now - goal.last_progress_at > self.config.stall_timeout_sec
        yaw_stall = now - goal.last_yaw_progress_at > self.config.rotation_stall_timeout_sec
        if linear_stall and yaw_stall:
            self.mark_active_failed("goal_stalled", now, source="explorer")
            return SUBGOAL_STALLED
        return SUBGOAL_WAITING

    def mark_active_covered_if_frontier_gone(
        self,
        has_frontier: bool,
        now: float | None = None,
        confirm_ticks: int | None = None,
        min_goal_age_sec: float | None = None,
    ) -> bool:
        now = self._now(now)
        if self.active_goal is None:
            return False
        if has_frontier:
            self.active_goal.frontier_gone_count = 0
            return False
        self.active_goal.frontier_gone_count += 1
        self.last_event = "frontier_gone_pending"
        required_ticks = max(1, self.config.frontier_gone_confirm_ticks if confirm_ticks is None else confirm_ticks)
        required_age = (
            self.config.frontier_gone_min_goal_age_sec
            if min_goal_age_sec is None
            else max(0.0, min_goal_age_sec)
        )
        goal_age = now - self.active_goal.sent_at
        if self.active_goal.frontier_gone_count < required_ticks or goal_age < required_age:
            return False
        self.block_goal_point(
            self.active_goal.point,
            duration_sec=self.config.reached_point_blacklist_sec,
            radius_m=self.config.reached_point_blacklist_radius_m,
            now=now,
        )
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
                "yaw": self.active_goal.yaw,
                "status": self.active_goal.status,
                "age_sec": max(0.0, time.time() - self.active_goal.sent_at),
                "frontier_gone_count": self.active_goal.frontier_gone_count,
            }
        return {
            "cluster_counts": counts,
            "active_goal": active,
            "blocked_goal_points": len(self.blocked_goal_points),
            "last_subgoal_world": None if self.last_subgoal_world is None else list(self.last_subgoal_world),
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

    @staticmethod
    def _angle_distance(a: float, b: float) -> float:
        return abs((a - b + pi) % (2.0 * pi) - pi)
