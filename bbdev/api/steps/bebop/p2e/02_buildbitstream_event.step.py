"""
bebop p2e buildbitstream event handler

Builds the P2E VVAC runtime case via bebop CLI:
  1. Resolve Verilog source directory (VSRC_PATH) from config
  2. Run bebop build p2e with rtl_dir and out_dir
  3. Validate generated bitstream and runtime artifacts
"""
import os
import shlex
import shutil
import sys
from datetime import datetime
from pathlib import Path

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import require_chip
from utils.path import bebop_cargo_env, get_buckyball_path, rtl_dir
from utils.p2e_timescale import normalize_p2e_timescale
from utils.stream_run import stream_run_logger_async
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "bebop-p2e-buildbitstream",
    "description": "Build Bebop P2E runtime case",
    "flows": ["bebop"],
    "triggers": [queue("bebop.p2e.buildbitstream")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    bebop_dir = f"{bbdir}/bebop"

    try:
        chip = require_chip(input_data)
    except ValueError as error:
        ctx.logger.error(str(error))
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "missing_chip"},
            trace_id=origin_tid,
        )
        return
    vsrc_dir = rtl_dir(bbdir, chip, "p2e", input_data.get("vsrc_dir"))
    normalize_p2e_timescale(vsrc_dir, ctx.logger)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    build_dir = (
        input_data.get("output_dir")
        or f"{bebop_dir}/build/{chip}-{timestamp}"
    )
    build_path = Path(build_dir).resolve()
    build_root = (Path(bebop_dir) / "build").resolve()
    if build_root not in build_path.parents:
        raise ValueError(f"P2E output_dir must be under {build_root}: {build_path}")
    if build_path.exists():
        shutil.rmtree(build_path)
    build_path.mkdir(parents=True)

    diff = bool(input_data.get("diff", False))
    manifest = Path(bbdir) / "examples" / "chips" / chip / "generated" / "bebop" / "Cargo.toml"
    features = ["p2e"]
    build_env = {**os.environ, **bebop_cargo_env(bbdir, chip)}
    if diff:
        features.append("bemu")
        build_env["CARGO_TARGET_DIR"] = os.path.join(bebop_dir, "target", f"{chip}-p2e-diff")
    build_cmd = shlex.join([
        "nix", "develop", "--ignore-env",
        "--keep-env-var", "HOME",
        "--keep-env-var", "ALL_PROXY",
        "--keep-env-var", "CARGO_TARGET_DIR",
        "-c", "cargo", "run", "--release",
        "--manifest-path", str(manifest),
        "--bin", "bebop",
        "--features", ",".join(features),
        "--", "build", "p2e",
        "--rtl-dir", vsrc_dir,
        "--out-dir", build_dir,
        *(["--diff"] if diff else []),
    ])
    ctx.logger.info("Building bebop p2e runtime case ...")
    build_result = await stream_run_logger_async(
        cmd=build_cmd,
        logger=ctx.logger,
        cwd=str(manifest.parent),
        stdout_prefix="bebop p2e build",
        stderr_prefix="bebop p2e build",
        env=build_env,
    )

    rtcfg_path = os.path.join(build_dir, "vvacDir", "runtimeDir", "rtcfg")
    libvctb_path = os.path.join(build_dir, "vvacDir", "runtimeDir", "lib", "lib_arm", "libvCtb.so")
    bitstream_path = os.path.join(build_dir, "fpgaCompDir", "bitstream.bit")
    runtime_path = os.path.join(build_dir, "bebop-p2e")
    extra_fields = {
        "task": "build",
        "vsrc_dir": vsrc_dir,
        "build_dir": build_dir,
        "rtcfg": rtcfg_path,
        "libvCtb": libvctb_path,
        "bitstream": bitstream_path,
        "runtime": runtime_path,
        "diff": diff,
        "timestamp": timestamp,
    }
    if input_data.get("from_regression_buildbitstream") and build_result.returncode == 0:
        extra_fields["bitstream"] = os.path.abspath(bitstream_path)

    await check_result(
        ctx,
        build_result.returncode,
        continue_run=False,
        extra_fields=extra_fields,
        trace_id=origin_tid,
    )
