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
from uvm_common import default_test_name, run_uvm_build, uvm_paths

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

    build_result, info = run_uvm_build(bbdir, input_data, ctx)
    if build_result.returncode != 0:
        await check_result(
            ctx,
            build_result.returncode,
            continue_run=False,
            extra_fields=info,
            trace_id=origin_tid,
        )
        return

    paths = uvm_paths(bbdir, input_data)
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
        f"{shlex.quote(paths['simv'])} +UVM_TESTNAME={shlex.quote(test)}"
    )
    cmd = f"nix develop {shlex.quote(paths['verify_env'])} --command zsh -lc {shlex.quote(script)}"

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
        extra_fields={**info, "task": "run", "test": test},
        trace_id=origin_tid,
    )
