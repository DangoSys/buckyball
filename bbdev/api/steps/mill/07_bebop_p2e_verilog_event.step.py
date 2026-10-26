import os
import shutil
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import require_design
from utils.path import get_buckyball_path
from utils.path import rtl_out
from steps.bebop.p2e.scripts.timescale import normalize_p2e_timescale
from utils.stream_run import stream_run_logger_async
from steps.mill.scripts.mill import elaborate_cmd
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "bebop-p2e-verilog",
    "description": "Generate P2E Verilog via mill",
    "flows": ["bebop"],
    "triggers": [queue("bebop.p2e.verilog")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    arch = os.path.join(bbdir, "arch")
    try:
        design = require_design(input_data)
        mill_config, build_dir = rtl_out(
            bbdir,
            design,
            "p2e",
            input_data.get("output_dir"),
            selection=input_data["target"],
        )
        shutil.rmtree(build_dir, ignore_errors=True)
        os.makedirs(build_dir)
        ctx.logger.info(f"Using mill config: {mill_config}")
        ctx.logger.info(f"Using build directory: {build_dir}")
        prefix = "bebop p2e verilog"
        command = elaborate_cmd(design, input_data["target"], "p2e", build_dir)
        if input_data.get("diff"):
            command += " --difftest"
        returncode = (
            await stream_run_logger_async(
                cmd=command,
                logger=ctx.logger,
                cwd=arch,
                stdout_prefix=prefix,
                stderr_prefix=prefix,
            )
        ).returncode
        if returncode == 0:
            normalize_p2e_timescale(build_dir, ctx.logger)
    except (ValueError, RuntimeError) as error:
        ctx.logger.error(str(error))
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "missing_design", "detail": str(error)},
            trace_id=origin_tid,
        )
        return
    ctx.logger.info(f"Using P2E design: {design}")
    ctx.logger.info(f"Using P2E output directory: {build_dir}")

    await check_result(
        ctx,
        returncode,
        continue_run=False,
        extra_fields={
            "task": "verilog",
            "design": design,
            "output_dir": build_dir,
            "top_module": "P2ETop",
        },
        trace_id=origin_tid,
    )
