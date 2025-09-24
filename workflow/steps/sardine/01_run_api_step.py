import subprocess
import sys
import os
import asyncio

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if utils_path not in sys.path:
  sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path

config = {
  "type": "api",
  "name": "running sardine",
  "description": "running sardine",
  "path": "/sardine/run",
  "method": "POST",
  "emits": ["sardine.run"],
  "flows": ["sardine"],
}

async def handler(req, context):
  bbdir = get_buckyball_path()

  body = req.get("body") or {}
  
  data = {
    "workload": body.get("workload", "")
  }

  sardine_dir = f"{bbdir}/bb-tests/sardine"
  
  await context.emit({"topic": "sardine.run", "data": data})

# ==================================================================================
# 等待执行结果
# ==================================================================================
  while True:
    # 检查成功结果
    success_result = await context.state.get(context.trace_id, 'success')
    if success_result and success_result.get('data'):
      # print(f"DEBUG: API found success state: {success_result}")
      # 过滤无效的null状态
      if success_result == {"data": None} or (isinstance(success_result, dict) and success_result.get('data') is None and len(success_result) == 1):
        await context.state.delete(context.trace_id, 'success')
        await asyncio.sleep(1)
        continue
      context.logger.info('simulation completed')

      if isinstance(success_result, dict) and 'data' in success_result:
        return success_result['data']
      return success_result
    
    # 检查错误状态
    failure_result = await context.state.get(context.trace_id, 'failure')
    if failure_result and failure_result.get('data'):
      # print(f"DEBUG: API found failure state: {failure_result}")
      context.logger.error('simulation failed', failure_result)

      if isinstance(failure_result, dict) and 'data' in failure_result:
        return failure_result['data']
      return failure_result

    await asyncio.sleep(1)
