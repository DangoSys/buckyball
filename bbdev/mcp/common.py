"""Shared helpers for buckyball-dev MCP (bbdev HTTP lifecycle + polling)."""

from __future__ import annotations

import atexit
import json
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


def _free_port(lo: int = 5100, hi: int = 5500) -> int:
    for p in range(lo, hi + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", p))
                return p
        except OSError:
            continue
    raise RuntimeError(f"No available port in {lo}-{hi}")


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
            [str(BBDEV), "stop", "--server", "--port", str(_port)],
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
        return _port
    if _proc is not None:
        _log(f"bbdev on port {_port} died; restarting")
        _stop()

    if not BBDEV.is_file():
        raise RuntimeError(f"missing bbdev CLI: {BBDEV}")
    if shutil.which("iii") is None:
        raise RuntimeError(
            "iii not found; run MCP via scripts/claude/run_mcp_server.sh"
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
        [str(BBDEV), "start", "--server", "--port", str(port)],
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


def _call(endpoint: str, params: Dict[str, Any], timeout: int = 600) -> Dict[str, Any]:
    port = _ensure()
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

    deadline = time.monotonic() + timeout
    beat = time.monotonic()
    stuck = 0
    while time.monotonic() < deadline:
        state = _read_state(trace_id)
        if state is None:
            time.sleep(2)
            continue
        if "success" in state:
            out = state["success"].get("body", state["success"])
            if not isinstance(out, dict):
                raise RuntimeError(f"success body must be object: {out!r}")
            out.setdefault("success", True)
            out.setdefault("trace_id", trace_id)
            out.setdefault("port", port)
            return out
        if "failure" in state:
            body = state["failure"].get("body", state["failure"])
            return {
                "success": False,
                "failure": True,
                "trace_id": trace_id,
                "port": port,
                "body": body,
                "server_log": str(LOG),
            }
        if state.get("processing") is True:
            stuck += 1
            if stuck >= 30:
                return {
                    "success": False,
                    "failure": True,
                    "error": "stuck in legacy processing state",
                    "trace_id": trace_id,
                    "port": port,
                    "server_log": str(LOG),
                }
        else:
            stuck = 0

        now = time.monotonic()
        if now - beat >= 30:
            _log(f"still running {endpoint} trace={trace_id}")
            beat = now
        time.sleep(2)

    return {
        "success": False,
        "failure": True,
        "error": f"timed out after {timeout}s waiting for {endpoint}",
        "trace_id": trace_id,
        "server_log": str(LOG),
        "port": port,
    }


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

    domains = chip_root / "configs" / "tiles" / "cores" / "balldomains"
    cores = chip_root / "configs" / "tiles" / "cores"

    if balldomain is None:
        default = cores / "default.toml"
        if not default.is_file():
            raise FileNotFoundError(f"missing {default}; pass balldomain= explicitly")
        rel = _load_toml(default).get("balldomain")
        if not isinstance(rel, str) or not rel:
            raise ValueError(f"cores/default.toml missing balldomain=: {default}")
        path = (cores / rel).resolve()
    else:
        raw = Path(balldomain)
        if raw.is_absolute():
            path = raw
        elif balldomain.endswith(".toml"):
            candidates = [
                domains / raw.name,
                cores / balldomain,
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
    isa = cfg.get("ballISA", [])
    if not isinstance(mappings, list) or not isinstance(isa, list):
        raise ValueError(f"{path}: ballIdMappings/ballISA must be arrays")

    ids = [m.get("ballId") for m in mappings]
    names = [m.get("ballName") for m in mappings]
    funct7s = [e.get("funct7") for e in isa]
    mnemonics = [e.get("mnemonic") for e in isa]
    bids = [e.get("bid") for e in isa]
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
        if cfg_rel is None:
            continue
        if not isinstance(cfg_rel, str) or not cfg_rel:
            missing_config.append(
                {"ballName": name, "config": cfg_rel, "error": "empty"}
            )
            continue
        cfg_path = (path.parent / cfg_rel).resolve()
        if not cfg_path.is_file():
            missing_config.append(
                {"ballName": name, "config": cfg_rel, "resolved": str(cfg_path)}
            )

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
call = _call
balldomain_path = _balldomain_path
validate_toml = _validate
