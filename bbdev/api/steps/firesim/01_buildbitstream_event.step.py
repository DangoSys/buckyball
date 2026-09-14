import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path, sim_name
from utils.stream_run import stream_run_logger_async
from utils.event_common import check_result, get_origin_trace_id, require_chip

scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from firesim_env import setup_firesim_env
from chip_yaml import firesim_cmd, prepare_yamls

config = {
    "name": "firesim-buildbitstream",
    "description": "build bitstream",
    "flows": ["firesim"],
    "triggers": [queue("firesim.buildbitstream")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()

    try:
        chip = require_chip(input_data)
        target_config = sim_name(bbdir, chip, "firesim")
    except ValueError as error:
        ctx.logger.error(str(error))
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": str(error)},
            trace_id=origin_tid,
        )
        return

    yaml_dir = prepare_yamls(bbdir, chip, target_config)
    env = setup_firesim_env()

    au280_tcl = f"{bbdir}/thirdparty/firesim/platforms/xilinx_alveo_u280/cl_firesim/scripts/au280.tcl"
    if os.path.exists(au280_tcl):
        with open(au280_tcl, "r") as f:
            content = f.read()
        fixed_content = content.replace(
            "xilinx.com:au280:part0:1.2",
            "xilinx.com:au280:part0:1.0",
        )
        if content != fixed_content:
            with open(au280_tcl, "w") as f:
                f.write(fixed_content)
            ctx.logger.info("Fixed AU280 board_part version: 1.2 -> 1.0")

    ctx.logger.info(f"firesim buildbitstream chip={chip} TARGET_CONFIG={target_config}")
    result = await stream_run_logger_async(
        cmd=firesim_cmd("buildbitstream", yaml_dir),
        logger=ctx.logger,
        stdout_prefix="firesim buildbitstream",
        stderr_prefix="firesim buildbitstream",
        env=env,
    )

    await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields={"chip": chip, "target_config": target_config, "yaml_dir": yaml_dir},
        trace_id=origin_tid,
    )
