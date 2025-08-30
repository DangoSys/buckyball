# steps/github-doc-webhook.step.py  
import subprocess  
import asyncio  
from datetime import datetime  
import os
import sys

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if utils_path not in sys.path:
  sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path


config = {  
  'type': 'api',  
  'name': 'GitHub Doc Webhook',  
  'description': 'Receives GitHub push webhook and executes make doc command',  
  'path': '/doc',  
  'method': 'POST',  
  'emits': [''],  
  'flows': ['github-doc'],  
}  
  
async def handler(req, context):  
  bbdir = get_buckyball_path()
  context.logger.info('GitHub webhook received, starting doc deploy', {  
    'body': req.get('body'),  
    'trace_id': context.trace_id  
  })  

  await context.emit({"topic": "doc.deploy", "data": {}})
# ==================================================================================
# 等待执行结果
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
      context.logger.info('simulation completed')

      if isinstance(success_result, dict) and 'data' in success_result:
        return success_result['data']
      return success_result
    
    # 检查错误状态
    failure_result = await context.state.get(context.trace_id, 'failure')
    if failure_result:
      context.logger.error('simulation failed', failure_result)

      if isinstance(failure_result, dict) and 'data' in failure_result:
        return failure_result['data']
      return failure_result

    await asyncio.sleep(1)
