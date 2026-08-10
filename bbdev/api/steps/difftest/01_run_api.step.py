from pathlib import Path

from motia import ApiRequest, ApiResponse, FlowContext, api

from utils.path import get_buckyball_path, get_verilator_build_dir


config = {
    "name": "difftest-run-api",
    "description": "Run Bank DiffTest on a selected backend with BEMU as the reference model",
    "flows": ["difftest"],
    "triggers": [api("POST", "/difftest/run")],
    "enqueues": ["difftest.run"],
}


def _required_string(body: dict, name: str) -> str | None:
    value = body.get(name)
    if not isinstance(value, str) or not value.strip() or value == "None":
        return None
    return value


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    backend = _required_string(body, "backend")
    if backend is None:
        return ApiResponse(
            status=400,
            body={
                "success": False,
                "error": "Missing required parameter(s): --backend",
            },
        )

    known_backends = ("verilator", "p2e")
    if backend not in known_backends:
        return ApiResponse(
            status=400,
            body={
                "success": False,
                "error": (
                    f"Unsupported DiffTest backend: {backend}; "
                    f"available backends: {', '.join(known_backends)}"
                ),
            },
        )
    if backend == "p2e":
        return ApiResponse(
            status=501,
            body={
                "success": False,
                "error": "DiffTest backend p2e is reserved but not implemented yet",
            },
        )

    required = {
        name: _required_string(body, name) for name in ("chip", "config", "binary")
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        missing_args = ", ".join("--" + name for name in missing)
        return ApiResponse(
            status=400,
            body={
                "success": False,
                "error": f"Missing required parameter(s): {missing_args}",
            },
        )

    unsupported = [name for name in ("rushB", "pk", "fast") if body.get(name)]
    if unsupported:
        unsupported_args = ", ".join("--" + name for name in unsupported)
        return ApiResponse(
            status=400,
            body={
                "success": False,
                "error": (
                    "Bank DiffTest only supports normal baremetal ELF mode; "
                    f"unsupported: {unsupported_args}"
                ),
            },
        )

    bbdir = get_buckyball_path()
    chip = required["chip"]
    manifest = Path(bbdir) / "examples" / "chips" / chip / "emu" / "Cargo.toml"
    if not manifest.is_file():
        return ApiResponse(
            status=400,
            body={"success": False, "error": f"Chip emulator manifest does not exist: {manifest}"},
        )

    data = {
        "backend": backend,
        **required,
        "jobs": body.get("jobs", 16),
        "vsrc_dir": get_verilator_build_dir(
            bbdir,
            required["config"],
            body.get("vsrc-dir") or body.get("output-dir"),
        ),
        "no_wave": body.get("no-wave", body.get("no_wave", False)),
        "itrace": body.get("itrace", False),
        "mtrace": body.get("mtrace", False),
        "pmctrace": body.get("pmctrace", False),
        "ctrace": body.get("ctrace", False),
        "banktrace": body.get("banktrace", False),
        "log_dir": body.get("log-dir") or body.get("log_dir"),
        "_trace_id": ctx.trace_id,
    }
    await ctx.enqueue({"topic": "difftest.run", "data": data})
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
