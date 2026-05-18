from motia import ApiRequest, ApiResponse, FlowContext, api

config = {
    "name": "result-query-api",
    "description": "Query task result by trace_id",
    "flows": ["common"],
    "triggers": [api("GET", "/result/{trace_id}")],
}


async def handler(request: ApiRequest, ctx: FlowContext) -> ApiResponse:
    """
    Query the result of a task by its trace_id.

    Returns:
        200: {"status": "success|failure|processing", "body": {...}}
        404: {"error": "No result found for trace_id"}
    """
    trace_id = request.path_params.get("trace_id", "")

    if not trace_id:
        return ApiResponse(
            status=400,
            body={"error": "trace_id is required"}
        )

    # Check for success state
    success_state = await ctx.state.get(trace_id, "success")
    if success_state:
        return ApiResponse(
            status=200,
            body={
                "status": "success",
                "body": success_state.get("body", success_state)
            }
        )

    # Check for failure state
    failure_state = await ctx.state.get(trace_id, "failure")
    if failure_state:
        return ApiResponse(
            status=200,
            body={
                "status": "failure",
                "body": failure_state.get("body", failure_state)
            }
        )

    # Check for processing state
    processing_state = await ctx.state.get(trace_id, "processing")
    if processing_state:
        return ApiResponse(
            status=200,
            body={
                "status": "processing",
                "body": {}
            }
        )

    # No state found
    return ApiResponse(
        status=404,
        body={"error": f"No result found for trace_id: {trace_id}"}
    )
