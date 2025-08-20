config = {
  "type": "api",
  "name": "verilator-api", 
  "description": "trigger verilator flow",
  "path": "/verilator",
  "method": "POST",
  "emits": ["build.start"],
  "flows": ["verilator"],
}

async def handler(req, context):
  body = req.get("body") or {}
  
  config = {
    "target": body.get("target", "run"),
    "binary": body.get("binary", ""),
    "clean": body.get("clean", True),
    "jobs": body.get("jobs", 16)
  }
  
  await context.emit({"topic": "build.start", "data": config})
  
  return {"status": 200, "body": {"message": f"build {config['target']} started"}}

