"""
bebop verilator event handler

Runs bebop verilator simulation:
  1. Resolve binary path
  2. Resolve verilog source directory (VSRC_PATH)
  3. Run the built bebop binary with the new run/verilator CLI
"""
import json
import os
import shlex
import sys
from datetime import datetime

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from utils.path import get_buckyball_path, get_verilator_build_dir
from utils.stream_run import stream_run_logger
from utils.search_workload import search_workload
from utils.event_common import check_result, get_origin_trace_id
from build_marker import build_marker_path, read_build_marker

config = {
    "name": "bebop-verilator-sim",
    "description": "Run bebop verilator simulation",
    "flows": ["bebop"],
    "triggers": [queue("bebop.verilator.sim"), queue("bebop.verilator.run.sim")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    bebop_dir = f"{bbdir}/bebop"
    arch_dir = f"{bbdir}/arch"

    arch_config = input_data.get("config")
    if not isinstance(arch_config, str) or not arch_config or arch_config == "None":
        ctx.logger.error("Missing required parameter: config")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "missing_config"},
            trace_id=origin_tid,
        )
        return
    vsrc_dir = get_verilator_build_dir(bbdir, arch_config, input_data.get("vsrc_dir"))
    ctx.logger.info(f"Using verilog source directory: {vsrc_dir}")
    if not os.path.isdir(vsrc_dir):
        ctx.logger.error(f"VSRC_PATH does not exist: {vsrc_dir}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "vsrc_not_found", "vsrc_dir": vsrc_dir},
            trace_id=origin_tid,
        )
        return

    bebop_bin = f"{bebop_dir}/target/debug/bebop"
    if not os.path.isfile(bebop_bin):
        ctx.logger.error(f"bebop binary does not exist: {bebop_bin}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "bebop_binary_not_found", "binary": bebop_bin},
            trace_id=origin_tid,
        )
        return

    marker_path = build_marker_path(bebop_dir)
    try:
        marker = read_build_marker(bebop_dir)
    except FileNotFoundError:
        ctx.logger.error(f"bebop verilator build marker does not exist: {marker_path}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "build_marker_not_found", "marker": marker_path},
            trace_id=origin_tid,
        )
        return
    except (OSError, json.JSONDecodeError) as e:
        ctx.logger.error(f"failed to read bebop verilator build marker: {e}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "build_marker_read_failed", "marker": marker_path, "detail": str(e)},
            trace_id=origin_tid,
        )
        return

    expect_vsrc = os.path.abspath(vsrc_dir)
    expect_bin = os.path.abspath(bebop_bin)
    if (
        marker.get("config") != arch_config
        or marker.get("vsrc_dir") != expect_vsrc
        or marker.get("binary") != expect_bin
    ):
        ctx.logger.error(f"bebop verilator build marker mismatch: {marker}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={
                "error": "build_marker_mismatch",
                "marker": marker,
                "expected": {
                    "config": arch_config,
                    "vsrc_dir": expect_vsrc,
                    "binary": expect_bin,
                },
            },
            trace_id=origin_tid,
        )
        return

    binary_name = input_data.get("binary", "")
    binary_path = search_workload(f"{bbdir}/bb-tests/output/workloads/src", binary_name)
    if binary_path is None:
        ctx.logger.error(f"binary not found: {binary_name}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "binary_not_found", "binary": binary_name},
            trace_id=origin_tid,
        )
        return
    ctx.logger.info(f"binary_path: {binary_path}")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_dir = input_data.get("log_dir") or f"{arch_dir}/log/{timestamp}-{binary_name}"
    fst_dir = input_data.get("fst_dir") or f"{arch_dir}/waveform/{timestamp}-{binary_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(fst_dir, exist_ok=True)

    wave_arg = " --no-wave" if input_data.get("no-wave", False) or input_data.get("no_wave", False) else ""
    trace_args = ""
    for trace_name in ("itrace", "mtrace", "pmctrace", "ctrace", "banktrace"):
        if input_data.get(trace_name, False):
            trace_args += f" --{trace_name}"

    run_cmd = (
        f"{shlex.quote(bebop_bin)} run verilator "
        f"--elf={shlex.quote(binary_path)} "
        f"--log-dir={shlex.quote(log_dir)} "
        f"--fst-dir={shlex.quote(fst_dir)}"
        f"{wave_arg}"
        f"{trace_args}"
    )
    ctx.logger.info(f"Running bebop verilator: {run_cmd}")
    run_result = stream_run_logger(
        cmd=run_cmd,
        logger=ctx.logger,
        cwd=bebop_dir,
        stdout_prefix="bebop verilator",
        stderr_prefix="bebop verilator",
        env={
            **os.environ,
            "ARCH_CONFIG": arch_config,
            "VSRC_PATH": vsrc_dir,
        },
    )

    await check_result(
        ctx,
        run_result.returncode,
        continue_run=False,
        extra_fields={
            "task": "verilator",
            "binary": binary_path,
            "config": arch_config,
            "log_dir": log_dir,
            "fst_dir": fst_dir,
            "timestamp": timestamp,
        },
        trace_id=origin_tid,
    )
