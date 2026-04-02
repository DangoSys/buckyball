"""
Common utility functions for all event steps.
"""


def get_origin_trace_id(input_data, ctx):
    """Get the origin trace_id from input_data (passed by API step) or fall back to ctx.trace_id.

    iii 0.7+ assigns a new trace_id per handler invocation, so the API step's
    trace_id must be forwarded explicitly via input_data["_trace_id"].
    """
    if isinstance(input_data, dict) and "_trace_id" in input_data:
        return input_data["_trace_id"]
    return ctx.trace_id


async def check_result(ctx, returncode, continue_run=False, extra_fields=None, trace_id=None):
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

    if continue_run:
        await ctx.state.set(scope, "processing", True)
        return None, None
    elif returncode != 0:
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
