import os
import shlex
import sys

from motia import FlowContext, queue

step_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.abspath(os.path.join(step_dir, "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
if step_dir not in sys.path:
    sys.path.insert(0, step_dir)

from utils.event_common import check_result, get_origin_trace_id
from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from scripts.uvm_common import checked_run_paths, default_test_name

config = {
    "name": "uvm-run",
    "description": "Build and run a Ball UVM simulation",
    "flows": ["uvm"],
    "triggers": [queue("uvm.run")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    info = {"task": "run", "ball": input_data.get("ball")}

    try:
        paths = checked_run_paths(bbdir, input_data)
    except Exception as e:
        ctx.logger.error(str(e))
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={**info, "error": str(e)},
            trace_id=origin_tid,
        )
        return

    info.update({
        "ball": paths["ball"],
        "verify_dir": paths["verify_dir"],
        "simv": paths["simv"],
        "dpi_lib": paths["dpi_lib"],
    })

    if not os.path.isfile(paths["simv"]):
        error = f"simv not found: {paths['simv']}. Run `bbdev uvm --build \"--config <cfg>\"` first."
        ctx.logger.error(error)
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={**info, "error": error},
            trace_id=origin_tid,
        )
        return

    test = input_data.get("test") or default_test_name(paths["ball"])
    if test is True:
        ctx.logger.error("Parameter --test requires a value")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={**info, "error": "invalid_test"},
            trace_id=origin_tid,
        )
        return

    script = (
        f"cd {shlex.quote(paths['verify_dir'])} && "
        f"{shlex.quote(paths['simv'])} "
        f"-sv_lib {shlex.quote(paths['dpi_lib'])} "
        f"+UVM_TESTNAME={shlex.quote(test)}"
    )
    cmd = f"nix develop {shlex.quote(paths['verify_env'])} --command bash -lc {shlex.quote(script)}"

    ctx.logger.info(f"Running UVM test {test} for ball {paths['ball']}")
    run_result = stream_run_logger(
        cmd=cmd,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="uvm run",
        stderr_prefix="uvm run",
    )

    await check_result(
        ctx,
        run_result.returncode,
        continue_run=False,
        extra_fields={**info, "test": test},
        trace_id=origin_tid,
    )
