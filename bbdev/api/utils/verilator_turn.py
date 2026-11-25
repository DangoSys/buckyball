"""The host's turn for Verilator: a model takes many cores and its host
threads spin while they wait, so the simulations that measure (GVN
estimates, the tool bench, Verilator regressions of the timeline) take turns.

A waiter leaves a ticket in log/bench-verilator.queue/, named by when it
came and its process; the oldest ticket whose process lives goes first
(first come, first served: a bare flock is not), and then takes the flock of
log/bench-verilator.lock that every turn holds. A ticket's process is a
bbdev server that outlives the tasks it runs, so a waiter whose task was
cancelled leaves the queue itself.
"""

import contextlib
import fcntl
import os
import time
from pathlib import Path

from utils.process_registry import cancellation_requested, current_task_scope


@contextlib.contextmanager
def verilator_turn(repo: Path):
    queue = repo / "log/bench-verilator.queue"
    queue.mkdir(parents=True, exist_ok=True)
    ticket = queue / f"{time.time_ns()}-{os.getpid()}"
    ticket.touch()
    scope = current_task_scope()
    try:
        while True:
            if scope is not None and cancellation_requested(scope):
                raise RuntimeError("cancelled while waiting for the Verilator turn")
            live = []
            for waiting in sorted(queue.iterdir()):
                if Path(f"/proc/{waiting.name.split('-')[1]}").exists():
                    live.append(waiting)
                else:
                    waiting.unlink(missing_ok=True)
            if live[0] == ticket:
                break
            time.sleep(5)
        with open(repo / "log/bench-verilator.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield
    finally:
        ticket.unlink()
