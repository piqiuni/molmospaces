from __future__ import annotations


class MoveBaseStatus:
    PENDING = 0
    ACTIVE = 1
    PREEMPTED = 2
    SUCCEEDED = 3
    ABORTED = 4
    REJECTED = 5
    PREEMPTING = 6
    RECALLING = 7
    RECALLED = 8
    LOST = 9


TERMINAL_SUCCESS = {MoveBaseStatus.SUCCEEDED}
TERMINAL_FAILURE = {
    MoveBaseStatus.ABORTED,
    MoveBaseStatus.REJECTED,
    MoveBaseStatus.LOST,
}
RUNNING = {
    MoveBaseStatus.PENDING,
    MoveBaseStatus.ACTIVE,
    MoveBaseStatus.PREEMPTING,
    MoveBaseStatus.RECALLING,
}


def latest_status(status_array):
    status_list = getattr(status_array, "status_list", None) or []
    if not status_list:
        return None
    return status_list[-1]


def status_name(status: int) -> str:
    names = {
        MoveBaseStatus.PENDING: "PENDING",
        MoveBaseStatus.ACTIVE: "ACTIVE",
        MoveBaseStatus.PREEMPTED: "PREEMPTED",
        MoveBaseStatus.SUCCEEDED: "SUCCEEDED",
        MoveBaseStatus.ABORTED: "ABORTED",
        MoveBaseStatus.REJECTED: "REJECTED",
        MoveBaseStatus.PREEMPTING: "PREEMPTING",
        MoveBaseStatus.RECALLING: "RECALLING",
        MoveBaseStatus.RECALLED: "RECALLED",
        MoveBaseStatus.LOST: "LOST",
    }
    return names.get(int(status), f"UNKNOWN_{status}")
