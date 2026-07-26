from __future__ import annotations

import threading
import time


class LatestPriorityRequestQueue:
    def __init__(self, max_size: int) -> None:
        self.max_size = max(1, int(max_size))
        self._items: list[dict] = []
        self._condition = threading.Condition()
        self._closed = False

    @staticmethod
    def _priority_key(item: dict) -> tuple[float, int]:
        return (
            float(item.get("priority", 0.0)),
            -int(item.get("request_sequence", 0) or 0),
        )

    def put(self, item: dict) -> tuple[bool, dict | None]:
        payload = dict(item)
        with self._condition:
            if self._closed:
                return False, payload
            object_id = str(payload.get("object_id") or "")
            replaced = None
            for index, queued in enumerate(self._items):
                if object_id and str(queued.get("object_id") or "") == object_id:
                    replaced = self._items.pop(index)
                    self._items.append(payload)
                    self._condition.notify()
                    return True, replaced
            if len(self._items) < self.max_size:
                self._items.append(payload)
                self._condition.notify()
                return True, None
            weakest_index = min(
                range(len(self._items)), key=lambda index: self._priority_key(self._items[index])
            )
            weakest = self._items[weakest_index]
            if self._priority_key(payload) <= self._priority_key(weakest):
                return False, payload
            self._items[weakest_index] = payload
            self._condition.notify()
            return True, weakest

    def discard(self, object_id: str, request_sequence: int | None = None) -> list[dict]:
        object_id = str(object_id or "")
        if not object_id:
            return []
        with self._condition:
            kept = []
            removed = []
            for item in self._items:
                same_object = str(item.get("object_id") or "") == object_id
                same_sequence = request_sequence is None or int(
                    item.get("request_sequence", 0) or 0
                ) == int(request_sequence)
                if same_object and same_sequence:
                    removed.append(item)
                else:
                    kept.append(item)
            self._items = kept
            return removed

    def get(self, timeout_s: float) -> dict | None:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._condition:
            while not self._items and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
            if self._closed:
                return None
            best_index = max(
                range(len(self._items)), key=lambda index: self._priority_key(self._items[index])
            )
            return self._items.pop(best_index)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._items.clear()
            self._condition.notify_all()

    def __len__(self) -> int:
        with self._condition:
            return len(self._items)
