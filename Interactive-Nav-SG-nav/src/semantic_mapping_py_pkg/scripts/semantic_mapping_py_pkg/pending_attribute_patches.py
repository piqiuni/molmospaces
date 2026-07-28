from __future__ import annotations

import copy
from collections import OrderedDict


class PendingAttributePatchCache:
    """Bounded lifecycle-aware cache for patches that arrive before graph nodes."""

    def __init__(self, max_entries=128):
        self.max_entries = max(1, int(max_entries))
        self.episode_id = ""
        self.episode_generation = 0
        self.episode_active = True
        self._entries = OrderedDict()

    def __len__(self):
        return len(self._entries)

    @staticmethod
    def _generation(value):
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _sequence(patch):
        try:
            return max(0, int((patch or {}).get("request_sequence", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def clear(self):
        self._entries.clear()

    def update_lifecycle(
        self,
        *,
        episode_id=None,
        episode_generation=None,
        episode_active=None,
    ):
        next_generation = self._generation(
            self.episode_generation
            if episode_generation is None
            else episode_generation
        )
        if (
            self.episode_generation > 0
            and next_generation > 0
            and next_generation < self.episode_generation
        ):
            return False

        next_episode_id = (
            self.episode_id if episode_id is None else str(episode_id or "")
        )
        next_active = (
            self.episode_active if episode_active is None else bool(episode_active)
        )
        self.episode_id = next_episode_id
        self.episode_generation = next_generation
        self.episode_active = next_active
        if not next_active:
            self.clear()
            return True

        self._retain_current_or_future_entries()
        return True

    def observe_episode(self, episode_id):
        episode_id = str(episode_id or "")
        if not episode_id:
            return
        # GT reset messages and target lifecycle messages travel on separate
        # topics.  A new episode observed after the previous one was marked
        # inactive may therefore arrive first; allow that new identity to
        # reopen the cache while still keeping late messages for the old
        # episode blocked.
        if not self.episode_active and episode_id != self.episode_id:
            self.episode_active = True
        self.episode_id = episode_id
        self._retain_current_or_future_entries()

    def apply_or_store(
        self,
        graph_store,
        patch,
        *,
        episode_id="",
        episode_generation=0,
        stamp=None,
    ):
        if not self.episode_active or not isinstance(patch, dict):
            return False
        object_id = str(patch.get("object_id") or "")
        if not object_id:
            return False

        patch_episode_id = str(episode_id or patch.get("episode_id") or "")
        patch_generation = self._generation(
            episode_generation or patch.get("episode_generation")
        )
        if self._is_stale_lifecycle(patch_episode_id, patch_generation):
            return False

        entry = {
            "episode_id": patch_episode_id,
            "episode_generation": patch_generation,
            "object_id": object_id,
            "request_sequence": self._sequence(patch),
            "patch": copy.deepcopy(patch),
            "stamp": stamp,
        }
        if self._can_apply_now(graph_store, entry):
            return bool(graph_store.apply_attribute_patch(entry["patch"], stamp=stamp))
        self._put(entry)
        return False

    def replay(self, graph_store):
        changed = False
        for key, entry in list(self._entries.items()):
            if not self._is_current_entry(entry):
                continue
            if not self._can_apply_now(graph_store, entry):
                continue
            # Once the target node exists, the patch has been consumed even if
            # the graph store rejects it as an older request_sequence.
            applied = graph_store.apply_attribute_patch(
                entry["patch"], stamp=entry.get("stamp")
            )
            self._entries.pop(key, None)
            changed = bool(applied) or changed
        return changed

    def pending_keys(self):
        return list(self._entries.keys())

    def _put(self, entry):
        key = (
            int(entry["episode_generation"]),
            str(entry["episode_id"]),
            str(entry["object_id"]),
        )
        existing = self._entries.get(key)
        if existing is not None:
            if int(entry["request_sequence"]) < int(existing["request_sequence"]):
                return
            self._entries.pop(key, None)
        self._entries[key] = entry
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def _is_stale_lifecycle(self, episode_id, episode_generation):
        if (
            episode_generation > 0
            and self.episode_generation > 0
            and episode_generation < self.episode_generation
        ):
            return True
        if episode_generation > self.episode_generation:
            return False
        return bool(
            episode_id
            and self.episode_id
            and episode_id != self.episode_id
        )

    def _is_current_entry(self, entry):
        entry_generation = int(entry.get("episode_generation", 0) or 0)
        if (
            entry_generation > 0
            and self.episode_generation > 0
            and entry_generation != self.episode_generation
        ):
            return False
        entry_episode_id = str(entry.get("episode_id") or "")
        return not (
            entry_episode_id
            and self.episode_id
            and entry_episode_id != self.episode_id
        )

    def _retain_current_or_future_entries(self):
        retained = OrderedDict()
        for key, entry in self._entries.items():
            entry_generation = int(entry.get("episode_generation", 0) or 0)
            entry_episode_id = str(entry.get("episode_id") or "")
            stale_generation = bool(
                entry_generation > 0
                and self.episode_generation > 0
                and entry_generation < self.episode_generation
            )
            wrong_current_episode = bool(
                entry_generation <= self.episode_generation
                and entry_episode_id
                and self.episode_id
                and entry_episode_id != self.episode_id
            )
            if not stale_generation and not wrong_current_episode:
                retained[key] = entry
        self._entries = retained

    def _can_apply_now(self, graph_store, entry):
        if not self._is_current_entry(entry):
            return False
        graph_episode_id = str(getattr(graph_store, "episode_id", "") or "")
        entry_episode_id = str(entry.get("episode_id") or "")
        if entry_episode_id and graph_episode_id != entry_episode_id:
            return False
        return bool(graph_store.has_attribute_patch_target(entry["object_id"]))
