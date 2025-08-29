import asyncio

config = {
  "type": "api",
  "name": "Verilator Build",
  "description": "build verilator executable",
  "path": "/verilator/build",
  "method": "POST",
  "emits": ["verilator.build"],
  "flows": ["verilator"],
}

async def handler(req, context):
  body = req.get("body") or {}
  data = {"jobs": body.get("jobs", 16)}
  await context.emit({"topic": "verilator.build", "data": data})

# ==================================================================================
#  等待build结果
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
      context.logger.info('build success')
      return success_result  
    
    # 检查错误状态
    failure_result = await context.state.get(context.trace_id, 'failure')
    if failure_result:
      context.logger.error('build failed', failure_result)
      return failure_result  
