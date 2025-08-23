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
  
  data = {
    "jobs": body.get("jobs", "16")
  }
  
  await context.emit({"topic": "verilator.build", "data": data})
  
  return {
    "status": 200,
    "body": {
      "message": "verilator build started",
      "trace_id": context.trace_id
    }
  }