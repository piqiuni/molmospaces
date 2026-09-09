"""Episode-safe lifecycle for the mandatory startup scan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StartupScanLifecycle:
    """Keep a scan instance stable while an episode id arrives late.

    Mapping can become ready before its graph has assigned an ``episode_id``.
    The first non-empty id therefore *binds* the active scan but must not reset
    it.  Only a later different non-empty id starts a new mandatory scan.
    """

    enabled: bool
    instance_index: int = 0
    bound_episode_id: str = ""
    state: str = ""
    failure_detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.state = "PENDING" if self.enabled else "DISABLED"
        if self.enabled:
            self.instance_index = max(1, int(self.instance_index))

    @property
    def candidate_id(self) -> str:
        return f"startup_scan:instance_{self.instance_index:06d}"

    def observe_episode(self, episode_id: object) -> bool:
        """Bind first id, or reset only when a genuinely new episode arrives."""

        if not self.enabled:
            return False
        incoming = str(episode_id or "")
        if not incoming:
            return False
        if not self.bound_episode_id:
            self.bound_episode_id = incoming
            return False
        if incoming == self.bound_episode_id:
            return False
        self.bound_episode_id = incoming
        self.instance_index += 1
        self.state = "PENDING"
        self.failure_detail = {}
        return True

    def record_feedback(
        self,
        candidate_id: object,
        status: object,
        detail: dict[str, Any] | None = None,
    ) -> bool:
        """Apply feedback only when it belongs to the active scan instance."""

        if not self.enabled or str(candidate_id or "") != self.candidate_id:
            return False
        normalized = str(status or "").upper()
        if normalized == "STARTED":
            self.state = "ACTIVE"
        elif normalized == "SUCCEEDED":
            self.state = "COMPLETE"
            self.failure_detail = {}
        elif normalized in {"FAILED", "CANCELED", "REJECTED"}:
            self.state = "FAILED"
            self.failure_detail = dict(detail or {})
            self.failure_detail.setdefault("status", normalized)
        return True

    def blocks_regular_candidates(self) -> bool:
        return self.enabled and self.state != "COMPLETE"

    def should_publish_scan(self, ready: bool) -> bool:
        return bool(
            self.enabled
            and ready
            and self.state in {"PENDING", "ACTIVE"}
        )
