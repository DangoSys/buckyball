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
  
  return {"status": 200, "body": { "message": "verilator complete workflow started", "trace_id": context.trace_id}}

