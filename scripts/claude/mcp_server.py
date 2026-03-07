#!/usr/bin/env python3
"""MCP Server for Buckyball Claude Code workflow.

Provides:
- validate: static registration invariant checks
- bbdev_* tools: wrappers around bbdev HTTP API (server mode, auto-managed lifecycle)
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
BBDEV_API_DIR = REPO_ROOT / "bbdev" / "api"

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


def _find_available_port(start: int = 5100, end: int = 5500) -> int:
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
    _bbdev_proc = subprocess.Popen(
        ["pnpm", "dev", "--port", str(port)],
        cwd=str(BBDEV_API_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
    except Exception as e:
        return {"success": False, "failure": True, "error": str(e)}


# ---------------------------------------------------------------------------
# MCP protocol helpers
# ---------------------------------------------------------------------------
def _write_message(payload: Dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    header = f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def _read_message() -> Optional[Dict[str, Any]]:
    content_length = None
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.decode("ascii", errors="ignore").strip()
        if not line:
            break
        if line.lower().startswith("content-length:"):
            content_length = int(line.split(":", 1)[1].strip())
    if content_length is None:
        return None
    body = sys.stdin.buffer.read(content_length)
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def _response(
    msg_id: Any, result: Any = None, error: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    return payload


def _ok(payload: Any) -> Dict[str, Any]:
    return {
        "content": [
            {"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}
        ],
        "isError": False,
    }


def _err(message: str) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "isError": True}


# ---------------------------------------------------------------------------
# validate tool
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


def handle_validate(params: Dict[str, Any]) -> Dict[str, Any]:
    missing_files = []
    for name, path in REGISTRATION_FILES.items():
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        return _err(f"Missing registration files: {', '.join(missing_files)}")

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
    return _ok({"passed": all_passed, "checks": checks})


# ---------------------------------------------------------------------------
# bbdev tool handlers
# ---------------------------------------------------------------------------
def handle_bbdev_workload_build(params: Dict[str, Any]) -> Dict[str, Any]:
    result = _bbdev_call("/workload/build", {}, timeout=120)
    return _ok(result)


def handle_bbdev_verilator_run(params: Dict[str, Any]) -> Dict[str, Any]:
    api_params = {
        "binary": params.get("binary", ""),
        "config": params.get("config", "sims.verilator.BuckyballToyVerilatorConfig"),
        "batch": params.get("batch", True),
        "coverage": params.get("coverage", False),
    }
    if params.get("jobs"):
        api_params["jobs"] = params["jobs"]
    result = _bbdev_call("/verilator/run", api_params, timeout=1200)
    return _ok(result)


def handle_bbdev_verilator_verilog(params: Dict[str, Any]) -> Dict[str, Any]:
    api_params = {}
    if params.get("config"):
        api_params["config"] = params["config"]
    if params.get("balltype"):
        api_params["balltype"] = params["balltype"]
    result = _bbdev_call("/verilator/verilog", api_params, timeout=600)
    return _ok(result)


def handle_bbdev_verilator_build(params: Dict[str, Any]) -> Dict[str, Any]:
    api_params = {"jobs": params.get("jobs", 16)}
    if params.get("coverage"):
        api_params["coverage"] = True
    result = _bbdev_call("/verilator/build", api_params, timeout=600)
    return _ok(result)


def handle_bbdev_verilator_sim(params: Dict[str, Any]) -> Dict[str, Any]:
    api_params = {
        "binary": params.get("binary", ""),
        "batch": params.get("batch", True),
    }
    if params.get("coverage"):
        api_params["coverage"] = True
    result = _bbdev_call("/verilator/sim", api_params, timeout=1200)
    return _ok(result)


def handle_bbdev_sardine_run(params: Dict[str, Any]) -> Dict[str, Any]:
    api_params = {"workload": params.get("workload", "ctest")}
    if params.get("coverage"):
        api_params["coverage"] = True
    result = _bbdev_call("/sardine/run", api_params, timeout=1200)
    return _ok(result)


def handle_bbdev_yosys_synth(params: Dict[str, Any]) -> Dict[str, Any]:
    api_params = {}
    if params.get("top"):
        api_params["top"] = params["top"]
    if params.get("config"):
        api_params["config"] = params["config"]
    result = _bbdev_call("/yosys/synth", api_params, timeout=600)
    return _ok(result)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
TOOLS = {
    "validate": {
        "description": "Check 6 registration invariants: ballNum consistency, ballId strict increment, "
        "ballId no duplicates, funct7 no duplicates, busRegister matches default.json, "
        "decoder BIDs match default.json.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "handler": handle_validate,
    },
    "bbdev_workload_build": {
        "description": "Compile CTest workloads (bb-tests). Calls bbdev POST /workload/build.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "handler": handle_bbdev_workload_build,
    },
    "bbdev_verilator_run": {
        "description": "Full verilator pipeline: clean -> verilog -> build -> sim. "
        "Calls bbdev POST /verilator/run.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "binary": {
                    "type": "string",
                    "description": "Test binary name (e.g. ctest_relu_test_singlecore-baremetal)",
                },
                "config": {
                    "type": "string",
                    "description": "Elaborate config class (default: sims.verilator.BuckyballToyVerilatorConfig)",
                },
                "batch": {
                    "type": "boolean",
                    "description": "Run in batch mode (default: true)",
                },
                "coverage": {
                    "type": "boolean",
                    "description": "Enable coverage collection (default: false)",
                },
                "jobs": {"type": "integer", "description": "Parallel build jobs"},
            },
            "required": ["binary"],
        },
        "handler": handle_bbdev_verilator_run,
    },
    "bbdev_verilator_verilog": {
        "description": "Generate Verilog from Chisel. Supports --balltype for single Ball elaboration. "
        "Calls bbdev POST /verilator/verilog.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "config": {"type": "string", "description": "Elaborate config class"},
                "balltype": {
                    "type": "string",
                    "description": "Single Ball type for standalone elaboration (e.g. reluball)",
                },
            },
        },
        "handler": handle_bbdev_verilator_verilog,
    },
    "bbdev_verilator_build": {
        "description": "Build verilator simulation executable. Calls bbdev POST /verilator/build.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "jobs": {
                    "type": "integer",
                    "description": "Parallel build jobs (default: 16)",
                },
                "coverage": {
                    "type": "boolean",
                    "description": "Build with coverage support",
                },
            },
        },
        "handler": handle_bbdev_verilator_build,
    },
    "bbdev_verilator_sim": {
        "description": "Run verilator simulation (assumes already built). Calls bbdev POST /verilator/sim.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "binary": {"type": "string", "description": "Test binary name"},
                "batch": {
                    "type": "boolean",
                    "description": "Batch mode (default: true)",
                },
                "coverage": {"type": "boolean", "description": "Enable coverage"},
            },
            "required": ["binary"],
        },
        "handler": handle_bbdev_verilator_sim,
    },
    "bbdev_sardine_run": {
        "description": "Run sardine batch tests. Calls bbdev POST /sardine/run. "
        "With coverage=true, generates coverage report at bb-tests/sardine/reports/coverage/.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workload": {
                    "type": "string",
                    "description": "Workload filter (default: ctest)",
                },
                "coverage": {
                    "type": "boolean",
                    "description": "Enable coverage collection and report",
                },
            },
        },
        "handler": handle_bbdev_sardine_run,
    },
    "bbdev_yosys_synth": {
        "description": "Run Yosys synthesis for area estimation + OpenSTA timing analysis. "
        "Generates hierarchy_report.txt, area_report.txt, and timing_report.txt "
        "in bbdev/api/steps/yosys/log/. Calls bbdev POST /yosys/synth.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "top": {
                    "type": "string",
                    "description": "Top module name (default: BuckyballAccelerator)",
                },
                "config": {"type": "string", "description": "Elaborate config class"},
            },
        },
        "handler": handle_bbdev_yosys_synth,
    },
}


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------
def serve() -> int:
    while True:
        msg = _read_message()
        if msg is None:
            return 0

        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params", {})

        try:
            if method == "initialize":
                _write_message(
                    _response(
                        msg_id,
                        {
                            "protocolVersion": "2024-11-05",
                            "serverInfo": {"name": "buckyball-dev", "version": "0.1.0"},
                            "capabilities": {"tools": {}},
                        },
                    )
                )
                continue

            if method == "notifications/initialized":
                continue

            if method == "tools/list":
                tools = [
                    {
                        "name": name,
                        "description": spec["description"],
                        "inputSchema": spec["inputSchema"],
                    }
                    for name, spec in TOOLS.items()
                ]
                _write_message(_response(msg_id, {"tools": tools}))
                continue

            if method == "tools/call":
                name = params.get("name")
                arguments = params.get("arguments", {})
                if name not in TOOLS:
                    _write_message(
                        _response(
                            msg_id,
                            error={"code": -32601, "message": f"Unknown tool: {name}"},
                        )
                    )
                    continue
                result = TOOLS[name]["handler"](arguments)
                _write_message(_response(msg_id, result))
                continue

            _write_message(
                _response(
                    msg_id,
                    error={"code": -32601, "message": f"Unknown method: {method}"},
                )
            )

        except Exception as exc:
            _write_message(
                _response(msg_id, error={"code": -32000, "message": str(exc)})
            )


if __name__ == "__main__":
    raise SystemExit(serve())
