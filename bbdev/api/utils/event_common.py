"""
Common utility functions for all event steps.
"""


async def check_result(context, returncode, continue_run=False, extra_fields=None):
    """
    Check returncode, create appropriate result objects and set state.

    Args:
        context: The event context object
        returncode: The return code (int)
        continue_run: If True, set processing state instead of success/failure
        extra_fields: Optional dictionary of extra fields to include in result body

    Returns:
        tuple: (success_result, failure_result) - one will be None based on returncode and continue_run
    """
    extra_fields = extra_fields or {}

    if continue_run:
        await context.state.set(context.trace_id, "processing", True)
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
        await context.state.set(context.trace_id, "failure", failure_result)
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
        await context.state.set(context.trace_id, "success", success_result)
        return success_result, None


# ==================================================================================
#  API waits for event return result
#
#  Expected return result format:
#  {
#    "status": 200/400/500,
#    "body": {
#      "success": true/false,
#      "failure": true/false,
#      "processing": true/false,
#      "return_code": 0,
#      other fields
#    }
#  }
#
#  Since the Motia framework wraps data in the data field, it needs to be unpacked
#       if isinstance(result, dict) and 'data' in result:
#          return result['data']
#       return result
# ==================================================================================


async def wait_for_result(context):
    """
    Check for task completion state (success or failure).
    Returns result if found, None if still processing.

    Args:
        context: The event context object

    Returns:
        dict or None: The result data if task completed, None if still processing
    """
    # Check for success result
    success_result = await context.state.get(context.trace_id, "success")
    if success_result and success_result.get("data"):
        # Filter out invalid null state
        if success_result == {"data": None} or (
            isinstance(success_result, dict)
            and success_result.get("data") is None
            and len(success_result) == 1
        ):
            await context.state.delete(context.trace_id, "success")
            return None
        context.logger.info("task completed")

        if isinstance(success_result, dict) and "data" in success_result:
            return success_result["data"]
        return success_result

    # Check for error status
    failure_result = await context.state.get(context.trace_id, "failure")
    if failure_result and failure_result.get("data"):
        context.logger.error("task failed", failure_result)

        if isinstance(failure_result, dict) and "data" in failure_result:
            return failure_result["data"]
        return failure_result

    return None
