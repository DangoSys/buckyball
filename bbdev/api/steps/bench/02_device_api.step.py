import re

from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "bench-device-api",
    "description": "run one step of a device's evaluation timeline into a bbbench snapshot",
    "flows": ["bench"],
    "triggers": [api("POST", "/bench/device/:step")],
    "enqueues": ["bench.device"],
}

# The steps by the operations bench-device takes: nodes of the timeline.
nodes = {
    "workload": "workload",
    "bemu-workload": "bemu-workload",
    "model-build": "model-build",
    "bench-build": "bench-build",
    "verilator-workload": "verilator-workload",
    "bemu-model": "bemu-model",
    "bemu-bench": "bemu-bench",
    "bemu-accuracy": "bemu-accuracy",
    "verilator-gvn-perf": "verilator-gvn-perf",
    "verilator-gvn-debug": "verilator-gvn-debug",
    "uvm": "uvm",
    "dc-area": "dc-area",
    "power": "power",
    "fpga-workload": "fpga-workload",
    "fpga-gvn": "fpga-gvn",
    "fpga-model": "fpga-model",
    "buildbitstream": "fpga-bitstream",
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    body = request.body or {}
    step = request.path_params["step"]
    hardware = step in ("fpga-workload", "fpga-gvn", "fpga-model")
    fields = {"device"}
    allowed = fields | {"model", "bitstream", "fpga-location"} if hardware else fields
    if step == "buildbitstream":
        allowed = fields | {"diff"}
    if (
        step not in nodes
        or not fields <= set(body) <= allowed
        or not all(
            isinstance(v, bool) if k == "diff" else isinstance(v, str) and v
            for k, v in body.items()
        )
    ):
        return ApiResponse(
            status=400,
            body={
                "error": f"Invalid operation or arguments for {step}: expected {sorted(fields)}"
            },
        )
    if hardware and (
        body["device"] == "all"
        or (
            "fpga-location" in body
            and not re.fullmatch(r"[0-9]+\.[A-E]", body["fpga-location"])
        )
    ):
        return ApiResponse(
            status=400,
            body={"error": "Select one device and a board location such as 0.A"},
        )
    await ctx.enqueue(
        {
            "topic": "bench.device",
            "data": {
                **body,
                "step": nodes[step],
                "_trace_id": ctx.trace_id,
            },
        }
    )
    return ApiResponse(status=202, body={"trace_id": ctx.trace_id})
