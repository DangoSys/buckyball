import asyncio
import sys
import os


config = {  
  "type": "api",  
  "name": "agent",  
  "description": "调用agent API进行流式对话",  
  "path": "/agent/chat",  
  "method": "POST",  
  "emits": ["agent.prompt"],  
  "bodySchema": {  
    "type": "object",  
    "properties": {  
      "message": {"type": "string"},  
      "model": {"type": "string", "default": "deepseek-chat"}  
    },  
    "required": ["message"]  
  },  
  "responseSchema": {  
    "200": {  
      "type": "object",  
      "properties": {  
        "traceId": {"type": "string"},  
        "status": {"type": "string"}  
      }  
    }  
  },  
  "flows": ["agent"]  
}  
  
async def handler(req, context):  
  context.logger.info('agent API - 接收到请求', {"body": req.get("body")})  
  message = req.get("body").get("message")  
  model = req.get("body").get("model", "deepseek-chat")  
    
  # 发送事件到处理步骤  
  await context.emit({  
    "topic": "agent.prompt",  
    "data": {  
      "message": message,  
      "model": model,  
      "traceId": context.trace_id  
    }  
  })  
    
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