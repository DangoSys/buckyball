#!/usr/bin/env python3
"""MCP Server for Buckyball Claude Code workflow.

Provides:
- validate: static registration invariant checks
- bbdev_* tools: wrappers around bbdev HTTP API (server mode, auto-managed lifecycle)

Uses the official `mcp` Python SDK for protocol compatibility with both
Claude Code CLI and Cursor IDE.
"""

from __future__ import annotations

import atexit
import json
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
BBDEV_API_DIR = REPO_ROOT / "bbdev" / "api"
BBDEV_LOG_DIR = REPO_ROOT / "bbdev" / "api" / "steps"
BBDEV_SERVER_LOG = REPO_ROOT / "bbdev" / "server.log"

REGISTRATION_FILES = {
    "default_json": REPO_ROOT
    / "arch/src/main/scala/framework/balldomain/configs/default.json",
    "bus_register": REPO_ROOT
    / "arch/src/main/scala/examples/toy/balldomain/bbus/busRegister.scala",
    "disa": REPO_ROOT / "arch/src/main/scala/examples/toy/balldomain/DISA.scala",
    "domain_decoder": REPO_ROOT
    / "arch/src/main/scala/examples/toy/balldomain/DomainDecoder.scala",
}

# ---------------------------------------------------------------------------
# bbdev server lifecycle
# ---------------------------------------------------------------------------
_bbdev_proc: Optional[subprocess.Popen] = None
_bbdev_port: Optional[int] = None


def _find_available_port(start: int = 5200, end: int = 5500) -> int:
    for port in range(start, end + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available port in {start}-{end}")


def _ensure_bbdev_server() -> int:
    """Start bbdev server if not running. Returns port."""
    global _bbdev_proc, _bbdev_port

    if _bbdev_port is not None and _bbdev_proc is not None:
        if _bbdev_proc.poll() is None and _health_check(_bbdev_port):
            return _bbdev_port
        # Server died, clean up
        _stop_bbdev_server()

    # Clean AOF to prevent BullMQ replaying old events
    aof_dir = BBDEV_API_DIR / ".motia" / "appendonlydir"
    if aof_dir.exists():
        shutil.rmtree(aof_dir)

    port = _find_available_port()
    _log_file = open(BBDEV_SERVER_LOG, "a", encoding="utf-8")
    _bbdev_proc = subprocess.Popen(
        ["pnpm", "dev", "--port", str(port)],
        cwd=str(BBDEV_API_DIR),
        stdout=_log_file,
        stderr=_log_file,
    )
    _bbdev_port = port

    # Wait for server to be ready
    for _ in range(90):
        if _health_check(port):
            return port
        time.sleep(1)

    _stop_bbdev_server()
    raise RuntimeError(f"bbdev server failed to start on port {port} within 90s")


def _health_check(port: int) -> bool:
    try:
        import urllib.request

        req = urllib.request.Request(
            f"http://localhost:{port}",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _stop_bbdev_server():
    global _bbdev_proc, _bbdev_port
    if _bbdev_proc is not None:
        try:
            _bbdev_proc.terminate()
            _bbdev_proc.wait(timeout=5)
        except Exception:
            try:
                _bbdev_proc.kill()
            except Exception:
                pass
    _bbdev_proc = None
    _bbdev_port = None


atexit.register(_stop_bbdev_server)


def _bbdev_call(
    endpoint: str, params: Dict[str, Any], timeout: int = 600
) -> Dict[str, Any]:
    """Call bbdev HTTP API. Auto-starts server if needed."""
    port = _ensure_bbdev_server()
    url = f"http://localhost:{port}{endpoint}"

    data = json.dumps(params).encode("utf-8")
    import urllib.request

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8")
        except Exception:
            pass
        return {
            "success": False,
            "failure": True,
            "error": str(e),
            "status_code": e.code,
            "response_body": error_body,
            "server_log": str(BBDEV_SERVER_LOG),
        }
    except Exception as e:
        return {"success": False, "failure": True, "error": str(e)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_bitpat_values(text: str) -> List[int]:
    vals = []
    for m in re.finditer(r'BitPat\("b([01]+)"\)', text):
        vals.append(int(m.group(1), 2))
    return vals


def _extract_bus_register_names(text: str) -> List[str]:
    names = []
    for m in re.finditer(r'case\s+"(\w+)"', text):
        names.append(m.group(1))
    return names


def _extract_decoder_bids(text: str) -> List[int]:
    bids = []
    for m in re.finditer(r"(\d+)\.U,\s*rs2", text):
        bids.append(int(m.group(1)))
    return bids


def _fmt(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# MCP Server (using official SDK)
# ---------------------------------------------------------------------------
mcp = FastMCP("buckyball-dev")


@mcp.tool()
def validate() -> str:
    """Check 6 registration invariants: ballNum consistency, ballId strict increment, ballId no duplicates, funct7 no duplicates, busRegister matches default.json, decoder BIDs match default.json."""
    missing_files = []
    for name, path in REGISTRATION_FILES.items():
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        return f"ERROR: Missing registration files: {', '.join(missing_files)}"

    cfg = _read_json(REGISTRATION_FILES["default_json"])
    mappings = cfg.get("ballIdMappings", [])
    ids = [e.get("ballId") for e in mappings]
    names_from_json = [e.get("ballName") for e in mappings]

    disa_text = REGISTRATION_FILES["disa"].read_text(encoding="utf-8")
    funct7_values = _extract_bitpat_values(disa_text)

    bus_text = REGISTRATION_FILES["bus_register"].read_text(encoding="utf-8")
    bus_names = _extract_bus_register_names(bus_text)

    decoder_text = REGISTRATION_FILES["domain_decoder"].read_text(encoding="utf-8")
    decoder_bids = _extract_decoder_bids(decoder_text)

    checks = {
        "ballNum_matches_count": {
            "pass": cfg.get("ballNum") == len(mappings),
            "expected": len(mappings),
            "actual": cfg.get("ballNum"),
        },
        "ballId_strict_increment": {
            "pass": ids == list(range(len(ids))),
            "ids": ids,
        },
        "ballId_no_duplicates": {
            "pass": len(ids) == len(set(ids)),
            "duplicates": sorted(x for x in ids if ids.count(x) > 1),
        },
        "funct7_no_duplicates": {
            "pass": len(funct7_values) == len(set(funct7_values)),
            "duplicates": sorted(
                x for x in funct7_values if funct7_values.count(x) > 1
            ),
        },
        "busRegister_matches_json": {
            "pass": set(bus_names) == set(names_from_json),
            "in_json_not_bus": sorted(set(names_from_json) - set(bus_names)),
            "in_bus_not_json": sorted(set(bus_names) - set(names_from_json)),
        },
        "decoder_bids_match_json": {
            "pass": sorted(decoder_bids) == sorted(ids),
            "decoder_bids": sorted(decoder_bids),
            "json_ids": sorted(ids),
        },
    }

    all_passed = all(c["pass"] for c in checks.values())
    return _fmt({"passed": all_passed, "checks": checks})


@mcp.tool()
def bbdev_workload_build() -> str:
    """Compile CTest workloads (bb-tests). Calls bbdev POST /workload/build."""
    result = _bbdev_call("/workload/build", {}, timeout=120)
    return _fmt(result)


@mcp.tool()
def bbdev_verilator_run(
    binary: str,
    config: str = "sims.verilator.BuckyballToyVerilatorConfig",
    batch: bool = True,
    coverage: bool = False,
    jobs: Optional[int] = None,
) -> str:
    """Full verilator pipeline: clean -> verilog -> build -> sim. Calls bbdev POST /verilator/run."""
    api_params: Dict[str, Any] = {
        "binary": binary,
        "config": config,
        "batch": batch,
        "coverage": coverage,
    }
    if jobs is not None:
        api_params["jobs"] = jobs
    result = _bbdev_call("/verilator/run", api_params, timeout=1200)
    return _fmt(result)


@mcp.tool()
def bbdev_verilator_verilog(
    config: Optional[str] = None,
    balltype: Optional[str] = None,
) -> str:
    """Generate Verilog from Chisel. Supports --balltype for single Ball elaboration. Calls bbdev POST /verilator/verilog."""
    api_params: Dict[str, Any] = {}
    if config:
        api_params["config"] = config
    if balltype:
        api_params["balltype"] = balltype
    result = _bbdev_call("/verilator/verilog", api_params, timeout=600)
    return _fmt(result)


@mcp.tool()
def bbdev_verilator_build(
    jobs: int = 16,
    coverage: bool = False,
) -> str:
    """Build verilator simulation executable. Calls bbdev POST /verilator/build."""
    api_params: Dict[str, Any] = {"jobs": jobs}
    if coverage:
        api_params["coverage"] = True
    result = _bbdev_call("/verilator/build", api_params, timeout=600)
    return _fmt(result)


@mcp.tool()
def bbdev_verilator_sim(
    binary: str,
    batch: bool = True,
    coverage: bool = False,
) -> str:
    """Run verilator simulation (assumes already built). Calls bbdev POST /verilator/sim."""
    api_params: Dict[str, Any] = {
        "binary": binary,
        "batch": batch,
    }
    if coverage:
        api_params["coverage"] = True
    result = _bbdev_call("/verilator/sim", api_params, timeout=1200)
    return _fmt(result)


@mcp.tool()
def bbdev_sardine_run(
    workload: str = "ctest",
    coverage: bool = False,
) -> str:
    """Run sardine batch tests. Calls bbdev POST /sardine/run. With coverage=true, generates coverage report at bb-tests/sardine/reports/coverage/."""
    api_params: Dict[str, Any] = {"workload": workload}
    if coverage:
        api_params["coverage"] = True
    result = _bbdev_call("/sardine/run", api_params, timeout=1200)
    return _fmt(result)


@mcp.tool()
def bbdev_yosys_synth(
    top: Optional[str] = None,
    config: Optional[str] = None,
) -> str:
    """Run Yosys synthesis for area estimation + OpenSTA timing analysis. Generates hierarchy_report.txt, area_report.txt, and timing_report.txt in bbdev/api/steps/yosys/log/. Calls bbdev POST /yosys/synth."""
    api_params: Dict[str, Any] = {}
    if top:
        api_params["top"] = top
    if config:
        api_params["config"] = config
    result = _bbdev_call("/yosys/synth", api_params, timeout=600)
    return _fmt(result)


if __name__ == "__main__":
    mcp.run(transport="stdio")
