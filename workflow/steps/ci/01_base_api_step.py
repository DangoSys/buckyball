config = {
  "type": "api",
  "name": "CI Base Workflow", 
  "description": "trigger ci base workflow",
  "path": "/ci/base",
  "method": "POST",
  "emits": ["ci.base"],
  "flows": ["ci"],
}

async def handler(req, context):
  body = req.get("body") or {}
  
  config = {
    "binary": body.get("binary", ""),
    "jobs": body.get("jobs", "16"),
    "batch": body.get("batch", False),
    "from_run_workflow": True
  }
  
  await context.emit({"topic": "ci.base", "data": config})
  
  return {"status": 200, "body": { "message": "ci base workflow started", "trace_id": context.trace_id}}

