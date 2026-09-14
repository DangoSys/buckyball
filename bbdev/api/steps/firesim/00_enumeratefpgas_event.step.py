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
    "name": "firesim-enumeratefpgas",
    "description": "enumerate FPGAs",
    "flows": ["firesim"],
    "triggers": [queue("firesim.enumeratefpgas")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()

    try:
        chip = require_chip(input_data)
        target_config = sim_name(bbdir, chip, "firesim")
        yaml_dir = prepare_yamls(bbdir, chip, target_config)
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

    env = setup_firesim_env()
    ctx.logger.info(f"firesim enumeratefpgas chip={chip}")
    result = await stream_run_logger_async(
        cmd=firesim_cmd("enumeratefpgas", yaml_dir),
        logger=ctx.logger,
        stdout_prefix="firesim enumeratefpgas",
        stderr_prefix="firesim enumeratefpgas",
        env=env,
    )

    await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields={"chip": chip, "yaml_dir": yaml_dir},
        trace_id=origin_tid,
    )
