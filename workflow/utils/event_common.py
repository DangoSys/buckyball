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
#  api 等待 event 的返回结果
#
#  期望返回结果是：
#  {
#    "status": 200/400/500,
#    "body": {
#      "success": true/false,
#      "failure": true/false,
#      "processing": true/false,
#      "return_code": 0,
#      其余字段
#    }
#  }
#
#  由于Motia框架会把数据包装在data字段中，所以需要解包
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
    # 检查成功结果
    success_result = await context.state.get(context.trace_id, "success")
    if success_result and success_result.get("data"):
        # 过滤无效的null状态
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

    # 检查错误状态
    failure_result = await context.state.get(context.trace_id, "failure")
    if failure_result and failure_result.get("data"):
        context.logger.error("task failed", failure_result)

        if isinstance(failure_result, dict) and "data" in failure_result:
            return failure_result["data"]
        return failure_result

    return None
