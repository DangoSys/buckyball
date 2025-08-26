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
    
  return {  
    "status": 200,  
    "body": {  
      "traceId": context.trace_id,  
      "status": "processing"  
    }  
  }