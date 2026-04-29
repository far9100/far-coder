"""In-session task tracking.

The active task list is bound by ``chat`` to the current ``Session.tasks``
list, so mutations made through this module are reflected in the persisted
session JSON automatically.
"""

from __future__ import annotations

import uuid
from datetime import datetime

VALID_STATUSES = ("pending", "in_progress", "completed")

_active: list[dict] = []


def bind(target_list: list[dict]) -> None:
    """Point the live task store at an external list.

    Subsequent create/update/list operations mutate ``target_list`` in
    place, so any other holder of the same reference (typically
    ``Session.tasks``) sees the updates.
    """
    global _active
    _active = target_list


def reset() -> None:
    """Clear the currently bound list."""
    _active.clear()


def move_into(target_list: list[dict]) -> None:
    """Move all currently-bound tasks into ``target_list`` and re-bind to it.

    Used when a session is created lazily after the first turn: any tasks
    the agent created on that turn are transferred from the interim list
    into the persisted ``Session.tasks`` so they are saved with the session.
    """
    global _active
    target_list[:] = _active
    _active = target_list


def create(content: str) -> dict:
    """Append a new pending task and return it."""
    if not content or not content.strip():
        raise ValueError("Task content cannot be empty")
    task = {
        "id": _new_id(),
        "content": content.strip(),
        "status": "pending",
        "created_at": _now(),
    }
    _active.append(task)
    return task


def update(task_id: str, status: str) -> dict:
    """Update an existing task's status. Raises if id or status is invalid."""
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{status}'. Use one of: {', '.join(VALID_STATUSES)}"
        )
    for t in _active:
        if t.get("id") == task_id:
            t["status"] = status
            return t
    raise KeyError(f"Task not found: {task_id}")


def list_all() -> list[dict]:
    """Return a shallow copy of the current task list."""
    return list(_active)


def has_active_tasks() -> bool:
    return bool(_active)


def _new_id() -> str:
    return uuid.uuid4().hex[:6]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
