from __future__ import annotations

from typing import Any


_PUBLIC_COMMAND_FIELDS = (
    "command_id",
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


def merge_interaction_result_with_command(
    result: dict[str, Any], command: dict[str, Any] | None
) -> dict[str, Any]:
    """Fill public request metadata omitted by an opaque interaction result.

    The evaluator deliberately hides controller/joint internals.  Retaining the
    public command identifiers, requested postcondition and approach pose is
    sufficient for semantic state bookkeeping and post-open navigation.
    """

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
    """Pop the matching command, preferring the evaluator's command ID."""

    key = str(result.get("command_id") or result.get("event_id") or "")
    if key:
        command = pending_commands.pop(key, None)
        if command is not None:
            return command
    node_id = str(result.get("node_id") or "")
    object_id = str(result.get("object_id") or result.get("instance_id") or "")
    for pending_key, command in reversed(list(pending_commands.items())):
        if node_id and str(command.get("node_id") or "") == node_id:
            return pending_commands.pop(pending_key)
        if object_id and str(command.get("object_id") or "") == object_id:
            return pending_commands.pop(pending_key)
    return None
