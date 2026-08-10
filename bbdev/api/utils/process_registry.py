"""In-process registry for cancellable bbdev subprocesses."""

import os
import signal
import threading
from contextvars import ContextVar
from typing import Dict


_lock = threading.Lock()
_processes: Dict[str, int] = {}
_cancelled: set[str] = set()
_current_task_scope: ContextVar[str | None] = ContextVar("bbdev_task_scope", default=None)


def set_current_task_scope(scope: str) -> None:
    _current_task_scope.set(scope)


def current_task_scope() -> str | None:
    return _current_task_scope.get()


def register_process(scope: str, pid: int) -> None:
    with _lock:
        _processes[scope] = pid
        _cancelled.discard(scope)


def unregister_process(scope: str) -> None:
    with _lock:
        _processes.pop(scope, None)


def cancel_process(scope: str) -> bool:
    with _lock:
        pid = _processes.get(scope)
        _cancelled.add(scope)
    if pid is None:
        return False
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    return True


def force_cancel_process(scope: str, pid: int) -> bool:
    """Escalate a still-registered task process group after its grace period."""
    with _lock:
        if _processes.get(scope) != pid:
            return False
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return False
    return True


def process_pid(scope: str) -> int | None:
    with _lock:
        return _processes.get(scope)


def cancellation_requested(scope: str) -> bool:
    with _lock:
        return scope in _cancelled
