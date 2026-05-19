import os
import subprocess
import sys
from pathlib import Path

from motia import FlowContext, queue

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "workload-tohex",
    "description": "convert all -baremetal ELF files under bb-tests/output/workloads/src to hex",
    "flows": ["workload"],
    "triggers": [queue("workload.tohex")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    script = f"{bbdir}/bbdev/api/steps/workload/scripts/elf2hex.py"
    search_root = Path(f"{bbdir}/bb-tests/output/workloads/src")

    ctx.logger.info("Search root", {"path": str(search_root)})

    if not search_root.is_dir():
        ctx.logger.error(
            "Workload output directory not found",
            {"path": str(search_root)},
        )
        await check_result(ctx, 1, continue_run=False, trace_id=origin_tid)
        return

    ctx.logger.info("Search root is a directory", {"path": str(search_root)})

    elf_files = sorted(p for p in search_root.rglob("*-baremetal") if p.is_file())

    ctx.logger.info("ELF files found", {"count": len(elf_files)})

    if not elf_files:
        ctx.logger.warn(
            "No -baremetal ELF files found",
            {"search_root": str(search_root)},
        )
        await check_result(ctx, 0, continue_run=False, trace_id=origin_tid)
        return

    ctx.logger.info(
        "Converting ELF files to hex",
        {"count": len(elf_files), "search_root": str(search_root)},
    )

    overall_rc = 0
    for elf in elf_files:
        command = f"python3 {script} {elf}"
        ctx.logger.info("Executing to-hex command", {"command": command})
        result = stream_run_logger(
            cmd=command,
            logger=ctx.logger,
            cwd=str(elf.parent),
            executable="bash",
            stdout_prefix=f"to-hex {elf.name}",
            stderr_prefix=f"to-hex {elf.name}",
        )
        if result.returncode != 0:
            overall_rc = result.returncode
            ctx.logger.error(
                "elf2hex failed",
                {"elf": str(elf), "returncode": result.returncode},
            )

    # ==================================================================================
    # Return conversion result
    # ==================================================================================
    success_result, failure_result = await check_result(
        ctx, overall_rc, continue_run=False, trace_id=origin_tid)

    # ==================================================================================
    #  finish workflow
    # ==================================================================================
    return
