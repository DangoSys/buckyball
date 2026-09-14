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
from chip_yaml import firesim_cmd, prepare_yamls, recipe_name

config = {
    "name": "firesim-runworkload",
    "description": "run workload",
    "flows": ["firesim"],
    "triggers": [queue("firesim.runworkload")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()

    try:
        chip = require_chip(input_data)
        target_config = sim_name(bbdir, chip, "firesim")
        yaml_dir = prepare_yamls(bbdir, chip, target_config)
        hwdb = os.path.join(yaml_dir, "config_hwdb.yaml")
        text = open(hwdb, encoding="utf-8").read()
        if "bitstream_tar: null" in text or "bitstream_tar: file://" not in text:
            raise ValueError(
                f"no bitstream for recipe {recipe_name(chip)}; "
                f"run bbdev firesim --buildbitstream '--chip {chip}' first"
            )
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
    ctx.logger.info(f"firesim runworkload chip={chip} TARGET_CONFIG={target_config}")
    result = await stream_run_logger_async(
        cmd=firesim_cmd("runworkload", yaml_dir),
        logger=ctx.logger,
        stdout_prefix="firesim runworkload",
        stderr_prefix="firesim runworkload",
        env=env,
    )

    await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields={"chip": chip, "target_config": target_config, "yaml_dir": yaml_dir},
        trace_id=origin_tid,
    )
