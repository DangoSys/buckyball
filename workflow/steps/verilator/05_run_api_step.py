import asyncio

config = {
  "type": "api",
  "name": "Verilator Complete Workflow", 
  "description": "trigger complete verilator workflow",
  "path": "/verilator/run",
  "method": "POST",
  "emits": ["verilator.run"],
  "flows": ["verilator"],
}

async def handler(req, context):
  body = req.get("body") or {}
  
  config = {
    "binary": body.get("binary", ""),
    "jobs": body.get("jobs", "16"),
    "batch": body.get("batch", False),
    "from_run_workflow": True
  }

  await context.emit({"topic": "verilator.run", "data": config})
  
# ==================================================================================
#  等待仿真结果
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
  while True:
    # 检查成功结果
    success_result = await context.state.get(context.trace_id, 'success')
    if success_result:
      # 过滤无效的null状态
      if success_result == {"data": None} or (isinstance(success_result, dict) and success_result.get('data') is None and len(success_result) == 1):
        await context.state.delete(context.trace_id, 'success')
        await asyncio.sleep(1)
        continue
      context.logger.info('仿真执行完成')

      if isinstance(success_result, dict) and 'data' in success_result:
        return success_result['data']
      return success_result
    
    # 检查错误状态
    failure_result = await context.state.get(context.trace_id, 'failure')
    if failure_result:
      context.logger.error('仿真执行失败', failure_result)

      if isinstance(failure_result, dict) and 'data' in failure_result:
        return failure_result['data']
      return failure_result
      
    await asyncio.sleep(1)
    