"""Shared helpers for buckyball-dev MCP (bbdev HTTP lifecycle + polling)."""

from __future__ import annotations

import atexit
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

# bbdev/mcp -> repo root
REPO = Path(__file__).resolve().parents[2]
BBDEV = REPO / "bbdev" / "bbdev"
API = REPO / "bbdev" / "api"
MOTIA = API / ".venv" / "bin" / "motia"
LOG = REPO / "bbdev" / "server.log"
STATE_DIR = API / "data" / "state_store.db"
_proc: Optional[subprocess.Popen] = None
_port: Optional[int] = None
_log_fh = None
_submitted_trace_ids: set[str] = set()


def _log(msg: str) -> None:
    print(f"[buckyball-dev] {msg}", file=sys.stderr, flush=True)


def _fmt(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _err(msg: str) -> str:
    return _fmt({"success": False, "failure": True, "error": msg})


def _need(name: str, value: Optional[str]) -> Optional[str]:
    if value is None or not str(value).strip():
        return f"missing required parameter: {name}"
    return None


def _opt(params: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    for k, v in kwargs.items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        params[k] = v
    return params


def _preferred_http_port(lo: int = 5100, hi: int = 5500) -> int:
    size = hi - lo + 1
    return lo + (os.getuid() * 37) % size


def _free_port(lo: int = 5100, hi: int = 5500) -> int:
    preferred = _preferred_http_port(lo, hi)
    order = list(range(preferred, hi + 1)) + list(range(lo, preferred))
    for p in order:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", p))
                return p
        except OSError:
            continue
    raise RuntimeError(f"No available port in {lo}-{hi}")


def _assert_workspace_workers() -> None:
    sys.path.insert(0, str(API))
    from utils.workers import assert_sole_workspace_workers, read_server_ports

    ports = read_server_ports(str(API))
    assert_sole_workspace_workers(int(ports["worker_port"]), str(ports["bb_root"]))


def _http(
    method: str, url: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 30
):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return e.code, {"response_body": raw}


def _ready(port: int) -> bool:
    try:
        code, _ = _http("GET", f"http://127.0.0.1:{port}/compiler/build", timeout=2)
        return code != 404
    except Exception:
        return False


def _stop() -> None:
    global _proc, _port, _log_fh
    if _port is not None and BBDEV.is_file():
        subprocess.run(
            ["nix", "develop", "--command", str(BBDEV), "stop", "--server", "--port", str(_port)],
            cwd=str(BBDEV.parent),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    if _proc is not None:
        _proc.terminate()
        try:
            _proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None
    _port = None
    if _log_fh is not None:
        _log_fh.close()
        _log_fh = None


def _ensure() -> int:
    """Start `bbdev start --server` if needed. Returns HTTP port."""
    global _proc, _port, _log_fh

    if (
        _port is not None
        and _proc is not None
        and _proc.poll() is None
        and _ready(_port)
    ):
        _assert_workspace_workers()
        return _port
    if _proc is not None:
        _log(f"bbdev on port {_port} died; restarting")
        _stop()

    if not BBDEV.is_file():
        raise RuntimeError(f"missing bbdev CLI: {BBDEV}")
    if shutil.which("nix") is None:
        raise RuntimeError(
            "nix not found; MCP must start bbdev through the project development environment"
        )
    if not MOTIA.is_file():
        raise RuntimeError(
            f"missing {MOTIA}; install with: "
            f"cd {API} && uv venv .venv --python python3 --seed && "
            "uv pip install --python .venv/bin/python -r pyproject.toml"
        )

    port = _free_port()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    _log_fh = open(LOG, "a", encoding="utf-8")
    _proc = subprocess.Popen(
        ["nix", "develop", "--command", str(BBDEV), "start", "--server", "--port", str(port)],
        cwd=str(BBDEV.parent),
        stdout=_log_fh,
        stderr=_log_fh,
        start_new_session=True,
    )
    _port = port
    _log(f"starting bbdev on port {port} (log: {LOG})")

    for _ in range(120):
        if _proc.poll() is not None:
            tail = ""
            try:
                tail = LOG.read_text(encoding="utf-8", errors="replace")[-2000:]
            except OSError:
                pass
            _stop()
            raise RuntimeError(
                f"bbdev exited early; see {LOG}\n--- log tail ---\n{tail}"
            )
        if _ready(port):
            try:
                _assert_workspace_workers()
            except RuntimeError as e:
                msg = str(e)
                if (
                    ("need" in msg and "Motia worker" in msg)
                    or "engine::workers::list failed" in msg
                ):
                    time.sleep(1)
                    continue
                _stop()
                raise
            _log(f"bbdev ready on port {port}")
            return port
        time.sleep(1)

    _stop()
    raise RuntimeError(f"bbdev failed to start on port {port} within 120s; see {LOG}")


def _read_state(trace_id: str) -> Optional[Dict[str, Any]]:
    """Same as bbdev CLI: poll iii file state store (HTTP /result path_params are broken)."""
    path = STATE_DIR / f"{trace_id}.bin"
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        data, _ = json.JSONDecoder().raw_decode(raw)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        raise RuntimeError(f"state file root must be object: {path}")
    return data


def submit(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Submit an asynchronous bbdev task without waiting for completion."""
    port = _ensure()
    _assert_workspace_workers()
    base = f"http://127.0.0.1:{port}"
    _log(f"POST {endpoint} params={params}")

    status, submit = _http("POST", f"{base}{endpoint}", params, timeout=30)
    if status >= 400:
        return {
            "success": False,
            "failure": True,
            "status_code": status,
            "error": submit,
            "server_log": str(LOG),
            "port": port,
        }

    trace_id = submit.get("trace_id")
    if not trace_id:
        return {
            "success": False,
            "failure": True,
            "error": "no trace_id in submit response",
            "response": submit,
            "server_log": str(LOG),
            "port": port,
        }

    _submitted_trace_ids.add(trace_id)
    return {
        "accepted": True,
        "processing": True,
        "trace_id": trace_id,
        "port": port,
    }


def task_status(trace_id: str) -> Dict[str, Any]:
    """Read the terminal or in-progress state for a submitted bbdev task."""
    if not trace_id:
        raise ValueError("trace_id is required")

    state = _read_state(trace_id)
    if state is None:
        if trace_id not in _submitted_trace_ids:
            raise RuntimeError(f"unknown task trace_id: {trace_id}")
        return {
            "accepted": True,
            "processing": True,
            "queued": True,
            "trace_id": trace_id,
        }
    if "cancelled" in state:
        _submitted_trace_ids.discard(trace_id)
        body = state["cancelled"].get("body", state["cancelled"])
        if not isinstance(body, dict):
            raise RuntimeError(f"cancelled body must be object: {body!r}")
        body.setdefault("success", False)
        body.setdefault("failure", False)
        body.setdefault("cancelled", True)
        body.setdefault("processing", False)
        body.setdefault("trace_id", trace_id)
        return body
    if "success" in state:
        _submitted_trace_ids.discard(trace_id)
        out = state["success"].get("body", state["success"])
        if not isinstance(out, dict):
            raise RuntimeError(f"success body must be object: {out!r}")
        out.setdefault("success", True)
        out.setdefault("failure", False)
        out.setdefault("processing", False)
        out.setdefault("trace_id", trace_id)
        return out
    if "failure" in state:
        _submitted_trace_ids.discard(trace_id)
        body = state["failure"].get("body", state["failure"])
        return {
            "success": False,
            "failure": True,
            "processing": False,
            "trace_id": trace_id,
            "body": body,
            "server_log": str(LOG),
        }
    if "processing" in state:
        body = state["processing"]
        if body is True:
            out = {k: v for k, v in state.items() if k != "processing"}
        elif isinstance(body, dict):
            out = dict(body)
        else:
            raise RuntimeError(f"processing body must be object or true: {body!r}")
        out.setdefault("accepted", True)
        out["processing"] = True
        out.setdefault("trace_id", trace_id)
        return out
    raise RuntimeError(f"invalid task state for trace_id {trace_id}: {state!r}")


def _load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"TOML root must be a table: {path}")
    return data


def _balldomain_path(chip: str, balldomain: Optional[str]) -> Path:
    chip_root = REPO / "examples" / "chips" / chip
    if not chip_root.is_dir():
        raise FileNotFoundError(f"chip does not exist: {chip_root}")

    parse = REPO / "bazel" / "configparse"
    if str(parse) not in sys.path:
        sys.path.insert(0, str(parse))
    from chip_json import compiler_core, core_entry, load_topology

    data = load_topology(REPO, chip)
    core_name = compiler_core(data)
    if not core_name:
        raise ValueError(f"chip {chip} has no unique topology core")
    core = core_entry(data, core_name)

    core_root = REPO / "examples" / "cores" / core_name
    domains = core_root / "configs" / "balldomains"

    if balldomain is None:
        bd = core.get("balldomain")
        if not isinstance(bd, dict) or not isinstance(bd.get("_file"), str):
            raise ValueError(f"chip {chip} core {core_name} has no balldomain")
        path = (REPO / bd["_file"]).resolve()
    else:
        raw = Path(balldomain)
        if raw.is_absolute():
            path = raw
        elif balldomain.endswith(".toml"):
            candidates = [
                domains / raw.name,
                core_root / balldomain,
                chip_root / balldomain,
            ]
            path = next(
                (p.resolve() for p in candidates if p.is_file()),
                candidates[0].resolve(),
            )
        else:
            path = (domains / f"{balldomain}.toml").resolve()

    if not path.is_file():
        raise FileNotFoundError(f"balldomain toml does not exist: {path}")
    return path


def _validate(path: Path) -> Dict[str, Any]:
    cfg = _load_toml(path)
    mappings = cfg.get("ballIdMappings", [])
    if not isinstance(mappings, list):
        raise ValueError(f"{path}: ballIdMappings must be an array")
    isa = cfg.get("ballISA")
    if not isinstance(isa, list) or not isa:
        raise ValueError(f"{path}: missing or empty ballISA")
    for i, e in enumerate(isa):
        if not isinstance(e, dict):
            raise ValueError(f"{path}: ballISA[{i}] must be a table")
        if not isinstance(e.get("mnemonic"), str) or not e.get("mnemonic"):
            raise ValueError(f"{path}: ballISA[{i}].mnemonic invalid")
        if not isinstance(e.get("funct7"), int) or e.get("funct7") < 0:
            raise ValueError(f"{path}: ballISA[{i}].funct7 invalid")
        if not isinstance(e.get("bid"), int) or e.get("bid") < 0:
            raise ValueError(f"{path}: ballISA[{i}].bid invalid")

    ids = [m.get("ballId") for m in mappings]
    names = [m.get("ballName") for m in mappings]
    id_set = set(ids)

    missing_config = []
    bad_bw = []
    for m in mappings:
        name = m.get("ballName")
        if not m.get("ballClass"):
            bad_bw.append({"ballName": name, "error": "missing ballClass"})
        in_bw, out_bw = m.get("inBW"), m.get("outBW")
        if (
            not isinstance(in_bw, int)
            or in_bw <= 0
            or not isinstance(out_bw, int)
            or out_bw <= 0
        ):
            bad_bw.append({"ballName": name, "inBW": in_bw, "outBW": out_bw})
        cfg_rel = m.get("config")
        if not isinstance(cfg_rel, str) or not cfg_rel:
            missing_config.append(
                {"ballName": name, "config": cfg_rel, "error": "missing"}
            )
            continue
        cfg_path = (path.parent / cfg_rel).resolve()
        if not cfg_path.is_file():
            missing_config.append(
                {"ballName": name, "config": cfg_rel, "resolved": str(cfg_path)}
            )
            continue
        ball_cfg = _load_toml(cfg_path)
        ball = ball_cfg.get("ball")
        if not isinstance(ball, dict):
            missing_config.append(
                {"ballName": name, "config": cfg_rel, "error": "missing [ball]"}
            )

    funct7s = [e.get("funct7") for e in isa]
    mnemonics = [e.get("mnemonic") for e in isa]
    bids = [e.get("bid") for e in isa]

    orphan = sorted(id_set - set(bids))
    unknown = sorted(set(bids) - id_set)

    def dups(xs):
        return sorted(x for x in set(xs) if xs.count(x) > 1)

    checks = {
        "ballNum_matches_count": {
            "pass": cfg.get("ballNum") == len(mappings),
            "expected": len(mappings),
            "actual": cfg.get("ballNum"),
        },
        "ballId_strict_increment": {"pass": ids == list(range(len(ids))), "ids": ids},
        "ballId_no_duplicates": {
            "pass": len(ids) == len(set(ids)),
            "duplicates": dups(ids),
        },
        "ballName_no_duplicates": {
            "pass": len(names) == len(set(names)),
            "duplicates": dups(names),
        },
        "funct7_no_duplicates": {
            "pass": len(funct7s) == len(set(funct7s)),
            "duplicates": dups(funct7s),
        },
        "mnemonic_no_duplicates": {
            "pass": len(mnemonics) == len(set(mnemonics)),
            "duplicates": dups(mnemonics),
        },
        "isa_bid_in_mappings": {"pass": not unknown, "unknown_bids": unknown},
        "every_ball_has_isa": {"pass": not orphan, "ballIds_without_isa": orphan},
        "ball_config_files_exist": {
            "pass": not missing_config,
            "missing": missing_config,
        },
        "bandwidth_positive": {"pass": not bad_bw, "invalid": bad_bw},
    }

    id_to_isa: Dict[Any, list] = {}
    for e in isa:
        id_to_isa.setdefault(e.get("bid"), []).append(e)
    balls = [
        {
            "ballId": m.get("ballId"),
            "ballName": m.get("ballName"),
            "ballClass": m.get("ballClass"),
            "inBW": m.get("inBW"),
            "outBW": m.get("outBW"),
            "config": m.get("config"),
            "isa": [
                {"mnemonic": e.get("mnemonic"), "funct7": e.get("funct7")}
                for e in id_to_isa.get(m.get("ballId"), [])
            ],
        }
        for m in mappings
    ]

    return {
        "passed": all(c["pass"] for c in checks.values()),
        "chip_balldomain": str(path.relative_to(REPO)),
        "checks": checks,
        "balls": balls,
    }



atexit.register(_stop)


# Public aliases for tool modules
log = _log
fmt = _fmt
err = _err
need = _need
opt = _opt
balldomain_path = _balldomain_path
validate_toml = _validate
