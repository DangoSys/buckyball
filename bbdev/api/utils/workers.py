import json
import os
import subprocess
from pathlib import Path

from utils.port import HOST

SERVER_PORTS_FILE = "bbdev-server.json"


def server_ports_path(workflow_dir: str) -> str:
    return os.path.join(workflow_dir, "data", SERVER_PORTS_FILE)


def write_server_ports(workflow_dir: str, http_port: int, worker_port: int, bb_root: str) -> None:
    path = server_ports_path(workflow_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {
                "http_port": http_port,
                "worker_port": worker_port,
                "bb_root": os.path.realpath(bb_root),
            },
            f,
        )
        f.write("\n")
    os.replace(tmp, path)


def read_server_ports(workflow_dir: str) -> dict:
    path = server_ports_path(workflow_dir)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"invalid server ports file: {path}")
    for key in ("http_port", "worker_port", "bb_root"):
        if key not in data:
            raise RuntimeError(f"server ports file missing {key}: {path}")
    return data


def list_engine_workers(worker_port: int, timeout_s: float = 10.0) -> list:
    result = subprocess.run(
        [
            "iii",
            "trigger",
            "--function-id",
            "engine::workers::list",
            "--address",
            HOST,
            "--port",
            str(worker_port),
            "--timeout-ms",
            str(int(timeout_s * 1000)),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"engine::workers::list failed (rc={result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"engine::workers::list returned non-JSON: {result.stdout[:500]}") from e
    workers = payload.get("workers")
    if not isinstance(workers, list):
        raise RuntimeError(f"engine::workers::list missing workers list: {payload}")
    return workers


def _worker_pid(worker: dict) -> int | None:
    name = worker.get("name")
    if not isinstance(name, str) or ":" not in name:
        return None
    tail = name.rsplit(":", 1)[-1]
    if not tail.isdigit():
        return None
    return int(tail)


def _cwd(pid: int) -> str | None:
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except (FileNotFoundError, PermissionError, OSError):
        return None


def motia_worker_count() -> int:
    raw = os.environ.get("BBDEV_MOTIA_WORKERS", "1")
    try:
        n = int(raw)
    except ValueError as e:
        raise RuntimeError(f"BBDEV_MOTIA_WORKERS must be an int, got {raw!r}") from e
    if n < 1:
        raise RuntimeError(f"BBDEV_MOTIA_WORKERS must be >= 1, got {n}")
    return n


def classify_workspace_workers(worker_port: int, bb_root: str) -> tuple[list, list]:
    """Return (local, foreign) Motia workers attached to this iii."""
    root = os.path.realpath(bb_root)
    workers = list_engine_workers(worker_port)
    foreign = []
    local = []
    for worker in workers:
        funcs = worker.get("function_count") or 0
        runtime = worker.get("runtime")
        if funcs <= 0 and runtime not in ("python", "motia"):
            continue
        pid = _worker_pid(worker)
        if pid is None:
            foreign.append(worker)
            continue
        cwd = _cwd(pid)
        if cwd is None:
            foreign.append(worker)
            continue
        cwd_real = os.path.realpath(cwd)
        if Path(cwd_real).is_relative_to(root):
            local.append(worker)
        else:
            foreign.append(worker)
    return local, foreign


def assert_sole_workspace_workers(
    worker_port: int, bb_root: str, min_local: int | None = None
) -> int:
    """Refuse to run if any Motia worker outside this repo is connected to our iii.

    Shared machines often leave orphan Motia processes that reconnect to whatever
    process later binds the same worker port; iii then load-balances jobs into
    the foreign tree (e.g. /home/other/buckyball).

    Returns the number of local Motia workers when isolation checks pass.
    """
    root = os.path.realpath(bb_root)
    need = motia_worker_count() if min_local is None else min_local
    if need < 1:
        raise RuntimeError(f"min_local must be >= 1, got {need}")
    local, foreign = classify_workspace_workers(worker_port, bb_root)

    if foreign:
        detail = []
        for worker in foreign:
            pid = _worker_pid(worker)
            detail.append(
                {
                    "name": worker.get("name"),
                    "id": worker.get("id"),
                    "function_count": worker.get("function_count"),
                    "pid_cwd": _cwd(pid) if pid is not None else None,
                }
            )
        raise RuntimeError(
            "foreign Motia worker(s) connected to this iii engine; "
            f"refusing to run (worker_port={worker_port}, bb_root={root}): {detail}"
        )
    if len(local) < need:
        raise RuntimeError(
            f"need {need} Motia worker(s) from this workspace, found {len(local)} "
            f"(worker_port={worker_port}, bb_root={root})"
        )
    return len(local)
