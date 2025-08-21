config = {
  "type": "api",
  "name": "Verilator Verilog",
  "description": "generate verilog code",
  "path": "/verilator/verilog",
  "method": "POST",
  "emits": ["verilator.verilog"],
  "flows": ["verilator"],
}

async def handler(req, context):
  body = req.get("body") or {}
  
  data = {
    "jobs": body.get("jobs", 16)
  }
  
  await context.emit({"topic": "verilator.verilog", "data": data})
  
  return {
    "status": 200,
    "body": {
      "message": "verilator verilog generation started",
      "trace_id": context.trace_id
    }
  }