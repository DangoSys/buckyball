import os
import signal
import shutil
import subprocess
import time


def _stat(pid: int):
  try:
    with open(f"/proc/{pid}/stat") as f:
      return f.read().split()
  except (FileNotFoundError, PermissionError):
    return None


def _exists(pid: int) -> bool:
  stat = _stat(pid)
  return stat is not None and stat[2] != "Z"


def _cmdline(pid: int) -> list[str]:
  try:
    with open(f"/proc/{pid}/cmdline", "rb") as f:
      raw = f.read()
  except (FileNotFoundError, PermissionError):
    return []
  return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def _cwd(pid: int) -> str | None:
  try:
    return os.readlink(f"/proc/{pid}/cwd")
  except (FileNotFoundError, PermissionError, OSError):
    return None


def _env(pid: int) -> dict[str, str]:
  try:
    with open(f"/proc/{pid}/environ", "rb") as f:
      raw = f.read()
  except (FileNotFoundError, PermissionError):
    return {}

  env = {}
  for item in raw.split(b"\0"):
    if not item or b"=" not in item:
      continue
    key, value = item.split(b"=", 1)
    env[key.decode(errors="replace")] = value.decode(errors="replace")
  return env


def _own(pid: int) -> bool:
  try:
    return os.stat(f"/proc/{pid}").st_uid == os.getuid()
  except (FileNotFoundError, PermissionError):
    return False


def _children(pid: int) -> list[int]:
  children = []
  try:
    entries = os.listdir("/proc")
  except FileNotFoundError:
    return children

  for entry in entries:
    if not entry.isdigit():
      continue
    child = int(entry)
    stat = _stat(child)
    if not stat:
      continue
    try:
      if int(stat[3]) == pid:
        children.append(child)
        children.extend(_children(child))
    except (IndexError, ValueError):
      continue
  return children


def _wait_gone(pids: set[int], timeout: float) -> set[int]:
  deadline = time.time() + timeout
  alive = set(pids)
  while alive and time.time() < deadline:
    alive = {pid for pid in alive if _exists(pid)}
    if alive:
      time.sleep(0.1)
  return {pid for pid in alive if _exists(pid)}


def _signal(pids: set[int], sig: signal.Signals):
  for pid in sorted(pids, reverse=True):
    if pid == os.getpid():
      continue
    try:
      os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
      pass


def kill_tree(pid: int, term_timeout: float = 5.0):
  targets = set(_children(pid))
  targets.add(pid)
  _signal(targets, signal.SIGTERM)
  alive = _wait_gone(targets, term_timeout)
  if alive:
    _signal(alive, signal.SIGKILL)
    _wait_gone(alive, 1.0)


def terminate_group(proc: subprocess.Popen, term_timeout: float = 5.0):
  if proc.poll() is not None:
    return

  try:
    os.killpg(proc.pid, signal.SIGTERM)
  except (ProcessLookupError, PermissionError, OSError):
    kill_tree(proc.pid, term_timeout=term_timeout)

  try:
    proc.wait(timeout=term_timeout)
    return
  except subprocess.TimeoutExpired:
    pass

  try:
    os.killpg(proc.pid, signal.SIGKILL)
  except (ProcessLookupError, PermissionError, OSError):
    kill_tree(proc.pid, term_timeout=1.0)

  try:
    proc.wait(timeout=1)
  except subprocess.TimeoutExpired:
    pass


def _is_workspace_server(pid: int, workflow_dir: str, worker_url: str | None = None) -> bool:
  repo_dir = os.path.dirname(workflow_dir)
  root_dir = os.path.dirname(repo_dir)
  if not _own(pid):
    return False

  cwd = _cwd(pid)
  if cwd not in (workflow_dir, repo_dir, root_dir):
    return False

  cmd = _cmdline(pid)
  if not cmd:
    return False

  exe = os.path.basename(cmd[0])
  is_bbdev = len(cmd) >= 2 and os.path.abspath(cmd[1]) == os.path.join(repo_dir, "bbdev")
  if is_bbdev and worker_url is None:
    return True

  if cwd != workflow_dir:
    return False
  if worker_url and _env(pid).get("III_URL") != worker_url:
    return False

  text = " ".join(cmd)
  return exe == "iii" or ("motia" in text and "dev" in cmd and "--dir" in cmd and "steps" in cmd)


def _workspace_pids(workflow_dir: str, worker_url: str | None = None) -> set[int]:
  pids = set()
  try:
    entries = os.listdir("/proc")
  except FileNotFoundError:
    return pids

  for entry in entries:
    if entry.isdigit() and _is_workspace_server(int(entry), workflow_dir, worker_url):
      pids.add(int(entry))
  return pids


def _port_pids(
  workflow_dir: str,
  port: int | None,
  worker_url: str | None = None,
) -> set[int]:
  if port is None:
    return set()

  result = subprocess.run(
    ["lsof", "-ti", f":{port}"],
    capture_output=True,
    text=True,
    check=False,
  )
  pids = set()
  for line in result.stdout.splitlines():
    try:
      pid = int(line)
      if _is_workspace_server(pid, workflow_dir, worker_url):
        pids.add(pid)
    except ValueError:
      pass
  return pids


def stop_workspace_servers(
  workflow_dir: str,
  port: int | None = None,
  worker_url: str | None = None,
) -> int:
  pids = _workspace_pids(workflow_dir, worker_url=worker_url)
  pids.update(_port_pids(workflow_dir, port, worker_url=worker_url))
  if not pids:
    return 0

  targets = set(pids)
  for pid in list(pids):
    targets.update(_children(pid))

  _signal(targets, signal.SIGTERM)
  alive = _wait_gone(targets, 5.0)
  if alive:
    _signal(alive, signal.SIGKILL)
    _wait_gone(alive, 1.0)
  return len(targets)


def clean_python_caches(workflow_dir: str):
  for rel in ("steps", "utils"):
    root = os.path.join(workflow_dir, rel)
    if not os.path.isdir(root):
      continue
    for dirpath, dirnames, _ in os.walk(root):
      if "__pycache__" not in dirnames:
        continue
      shutil.rmtree(os.path.join(dirpath, "__pycache__"), ignore_errors=True)
      dirnames.remove("__pycache__")
