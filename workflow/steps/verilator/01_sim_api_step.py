config = {
  "type": "api",
  "name": "Verilator Sim",
  "description": "run verilator simulation",
  "path": "/verilator/sim",
  "method": "POST",
  "emits": ["verilator.sim"],
  "flows": ["verilator"],
}

async def handler(req, context):
  body = req.get("body") or {}
  
  binary = body.get("binary", "")
  if not binary:
    return {
      "status": 400,
      "body": {"error": "binary parameter is required"}
    }
  
  batch = body.get("batch", False)
  data = {"binary": binary, "batch": batch}
  
  await context.emit({"topic": "verilator.sim", "data": data})
  
  return {
    "status": 200,
    "body": {
      "message": "verilator simulation started",
      "trace_id": context.trace_id
    }
  }