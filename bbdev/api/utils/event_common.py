"""
Common utility functions for all event steps.
"""

import re

from utils.path import design_pb_path, get_buckyball_path
from utils.process_registry import set_current_task_scope


def require_design(data: dict) -> str:
    design = data.get("design")
    if not isinstance(design, str) or not design:
        raise ValueError("Missing required parameter: --design")
    extra = data.get("config")
    if isinstance(extra, str) and extra and extra != "None":
        raise ValueError(
            "do not pass --config; mill class comes from design config sims"
        )
    if not re.fullmatch(r"[A-Za-z0-9_-]+", design):
        raise ValueError(f"invalid design: {design}")
    return design


def require_target(data: dict, kinds=("device", "core")) -> str:
    """The selection of a request: device/<Config> for `--device <Config>`, a
    config registered by the design's sims config, or core/<entry>."""
    selected = [kind for kind in ("device", "core", "ip", "ball") if kind in data]
    if len(selected) != 1:
        raise ValueError(
            "Specify exactly one of --device <config>, --core <entry>, --ip <name>, --ball <name> after --design"
        )
    kind = selected[0]
    if kind not in kinds:
        raise ValueError(f"--{kind} is not supported by this operation")
    value = data[kind]
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"--{kind} requires a name")
    if kind != "device":
        return f"{kind}/{value}"
    design = require_design(data)
    pb = design_pb_path(get_buckyball_path(), design, selection=f"device/{value}")
    if not pb.is_file():
        raise ValueError(
            f"design {design!r} registers no device config {value!r}"
            " (run bbdev config --install after editing its sims config)"
        )
    return f"device/{value}"


def get_origin_trace_id(input_data, ctx):
    """Get the origin trace_id from input_data (passed by API step) or fall back to ctx.trace_id.

    iii 0.7+ assigns a new trace_id per handler invocation, so the API step's
    trace_id must be forwarded explicitly via input_data["_trace_id"].
    """
    trace_id = (
        input_data["_trace_id"]
        if isinstance(input_data, dict) and "_trace_id" in input_data
        else ctx.trace_id
    )
    set_current_task_scope(trace_id)
    return trace_id


async def check_result(
    ctx, returncode, continue_run=False, extra_fields=None, trace_id=None
):
    """
    Check returncode, create appropriate result objects and set state.

    Args:
        ctx: The flow context object
        returncode: The return code (int)
        continue_run: If True, set processing state instead of success/failure
        extra_fields: Optional dictionary of extra fields to include in result body
        trace_id: Optional trace_id to use as state scope (defaults to ctx.trace_id)

    Returns:
        tuple: (success_result, failure_result) - one will be None based on returncode and continue_run
    """
    extra_fields = extra_fields or {}
    scope = trace_id or ctx.trace_id

    if returncode != 0:
        failure_result = {
            "status": 500,
            "body": {
                "success": False,
                "failure": True,
                "processing": False,
                "returncode": returncode,
                **extra_fields,
            },
        }
        await ctx.state.set(scope, "failure", failure_result)
        return None, failure_result
    elif continue_run:
        await ctx.state.set(scope, "processing", {"processing": True, **extra_fields})
        return None, None
    else:
        success_result = {
            "status": 200,
            "body": {
                "success": True,
                "failure": False,
                "processing": False,
                "returncode": returncode,
                **extra_fields,
            },
        }
        await ctx.state.set(scope, "success", success_result)
        return success_result, None
