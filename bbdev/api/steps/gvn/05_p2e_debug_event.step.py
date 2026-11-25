import json
import subprocess
from datetime import datetime
from pathlib import Path

from motia import FlowContext, queue

from steps.gvn.scripts.fpga import fpga_bench
from utils.event_common import check_result, get_origin_trace_id
from utils.path import get_buckyball_path, log_dir

config = {
    "name": "gvn-p2e-debug",
    "description": "Check a model's GVN slices on FPGA against BEMU",
    "flows": ["gvn"],
    "triggers": [queue("gvn.p2e.debug")],
    "enqueues": ["gvn.bemu.debug", "bebop.p2e.runworkload"],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    trace = get_origin_trace_id(input_data, ctx)
    target = {key: input_data[key] for key in ("config", "device", "design", "target")}
    model = input_data["model"]
    directory = Path(
        log_dir(
            get_buckyball_path(),
            target["design"],
            "device",
            datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"),
            "gvn-p2e-debug",
            model,
            selection=target["target"],
        )
    )
    directory.mkdir(parents=True)
    estimate = directory / "fpga-debug.json"
    node = {"id": "fpga-gvn", "status": "running"}

    def write():
        temporary = estimate.with_suffix(".tmp")
        temporary.write_text(json.dumps(node, indent=2) + "\n")
        temporary.replace(estimate)

    try:
        node.update(
            await fpga_bench(ctx, target, node, input_data, [model], write, True)
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
        node.update(status="failure", error=str(error))
        for row in node.get("estimates", []):
            if row["status"] == "running":
                row.update(status="failure", error=str(error))
        ctx.logger.error(str(error))
    write()
    passed = node["status"] == "pass"
    if passed:
        ctx.logger.info(
            f"{model}: {node['estimates'][0]['passed']} distinct slices match BEMU on FPGA"
        )
    ctx.logger.info(f"FPGA GVN checks: {estimate}")
    await check_result(
        ctx,
        0 if passed else 1,
        continue_run=False,
        extra_fields={"checks": str(estimate)},
        trace_id=trace,
    )
