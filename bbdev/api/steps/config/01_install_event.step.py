import os
import sys
from pathlib import Path

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import check_result, get_origin_trace_id, require_target
from utils.path import design_pb_path, get_buckyball_path

config = {
    "name": "config-install",
    "description": "install the design.pb of one config, or of every device's default config",
    "flows": ["config"],
    "triggers": [queue("config.install")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    if not set(input_data) & {"config", "device", "hart", "thread", "ball"}:
        # Every device, from its design's one <Design>DefaultConfig.toml.
        designs = Path(get_buckyball_path()) / "examples/designs"
        devices = sorted(p for p in designs.glob("*/devices/*") if p.is_dir())
        wrong = [
            f"{d.relative_to(designs)}: {len(list(d.glob('configs/*DefaultConfig.toml')))}"
            for d in devices
            if len(list(d.glob("configs/*DefaultConfig.toml"))) != 1
        ]
        if wrong:
            error = (
                "every device needs exactly one configs/*DefaultConfig.toml: "
                + ", ".join(wrong)
            )
            ctx.logger.error(error)
            await check_result(
                ctx,
                1,
                continue_run=False,
                extra_fields={"task": "install", "error": error},
                trace_id=origin_tid,
            )
            return
        installed = []
        for device in devices:
            target = {"device": device.name}
            selection = require_target(target, ("device",))
            installed.append(
                str(
                    design_pb_path(
                        get_buckyball_path(), target["design"], selection=selection
                    )
                )
            )
            ctx.logger.info(f"Wrote simulation protobuf: {installed[-1]}")
        await check_result(
            ctx,
            0,
            continue_run=False,
            extra_fields={"task": "install", "design_pb": installed},
            trace_id=origin_tid,
        )
        return
    try:
        selection = require_target(input_data, ("device", "hart", "thread", "ball"))
    except Exception as exc:
        ctx.logger.error(f"Config install failed: {exc}")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "install", "error": str(exc)},
            trace_id=origin_tid,
        )
        return
    pb = design_pb_path(get_buckyball_path(), input_data["design"], selection=selection)
    ctx.logger.info(f"Wrote simulation protobuf: {pb}")
    await check_result(
        ctx,
        0,
        continue_run=False,
        extra_fields={"task": "install", "design_pb": str(pb)},
        trace_id=origin_tid,
    )
