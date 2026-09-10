from __future__ import annotations

from typing import Any


_PUBLIC_COMMAND_FIELDS = (
    "command_id",
    "episode_id",
    "decision_id",
    "candidate_id",
    "event_id",
    "node_id",
    "object_id",
    "node_type",
    "action",
    "interaction_mode",
    "expected_state",
    "approach_goal_xyyaw",
)


def _validate_result_identity(result, command):
    # event_id belongs to the producer; a command event and result event may
    # legitimately differ. Target/attempt identity and requested action may not.
    for field in ("episode_id", "decision_id", "candidate_id", "node_id", "object_id", "action"):
        actual, expected = result.get(field), (command or {}).get(field)
        if actual in (None, "") or expected in (None, ""):
            continue
        actual, expected = str(actual), str(expected)
        if field == "action":
            actual, expected = actual.casefold(), expected.casefold()
        if actual != expected:
            raise ValueError(f"interaction_result_identity_conflict:{field}")


def merge_interaction_result_with_command(
    result: dict[str, Any], command: dict[str, Any] | None
) -> dict[str, Any]:
    """Fill public request metadata omitted by an opaque interaction result.

    The evaluator deliberately hides controller/joint internals.  Retaining the
    public command identifiers, requested postcondition and approach pose is
    sufficient for semantic state bookkeeping and post-open navigation.
    """

    _validate_result_identity(result or {}, command)
    merged = dict(result or {})
    for field in _PUBLIC_COMMAND_FIELDS:
        if merged.get(field) not in (None, "", []):
            continue
        value = (command or {}).get(field)
        if value not in (None, "", []):
            merged[field] = value
    return merged


def take_pending_interaction_command(
    pending_commands: dict[str, dict[str, Any]], result: dict[str, Any]
) -> dict[str, Any] | None:
    """Pop the matching command without accepting an explicit-ID replay.

    A result carrying ``command_id`` is an acknowledgement for one concrete
    command attempt.  Falling back to a node/object match when that ID is
    unknown lets a delayed result from an earlier episode mutate the current
    graph.  Legacy opaque results that carry no command ID retain the
    node/object fallback used by the simulator adapters.
    """

    key = str(result.get("command_id") or result.get("event_id") or "")
    if key:
        command = pending_commands.get(key)
        if command is not None:
            _validate_result_identity(result, command)
            return pending_commands.pop(key)
        # An explicit command id that is not pending is never resolved by a
        # weaker object/node heuristic.  This is the stale-replay guard.
        if str(result.get("command_id") or ""):
            return None
    node_id = str(result.get("node_id") or "")
    object_id = str(result.get("object_id") or result.get("instance_id") or "")
    for pending_key, command in reversed(list(pending_commands.items())):
        result_episode = str(result.get("episode_id") or "")
        command_episode = str(command.get("episode_id") or "")
        if result_episode and command_episode and result_episode != command_episode:
            continue
        if node_id and str(command.get("node_id") or "") == node_id:
            _validate_result_identity(result, command)
            return pending_commands.pop(pending_key)
        if object_id and str(command.get("object_id") or "") == object_id:
            _validate_result_identity(result, command)
            return pending_commands.pop(pending_key)
    return None
