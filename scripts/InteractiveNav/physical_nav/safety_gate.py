#!/usr/bin/env python3
"""Hard read-only gate used by the physical Phase-1 runtime."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class BlockedCommand:
    seq: int
    received_at: float
    command: dict[str, Any]
    reason: str = "physical_nav_phase1_read_only"


class ReadOnlySafetyGate:
    """Accepts intent for observability, never invokes an actuator callback."""

    def __init__(self, max_history: int = 100) -> None:
        self.max_history = max_history
        self._lock = threading.Lock()
        self._history: list[BlockedCommand] = []
        self._last_seq = 0

    @property
    def actuation_enabled(self) -> bool:
        return False

    def handle_intent(self, command: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._last_seq += 1
            item = BlockedCommand(self._last_seq, time.time(), dict(command))
            self._history.append(item)
            del self._history[:-self.max_history]
            return {
                "type": "read_only_blocked",
                "seq": item.seq,
                "received_at": item.received_at,
                "accepted": False,
                "reason": item.reason,
                "command": item.command,
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "actuation_enabled": False,
                "status": "READ_ONLY_BLOCKED",
                "last_seq": self._last_seq,
                "history": [vars(item) for item in self._history[-20:]],
            }
