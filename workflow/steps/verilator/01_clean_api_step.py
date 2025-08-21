config = {
  "type": "api",
  "name": "Verilator Clean",
  "description": "clean build directory",
  "path": "/verilator/clean",
  "method": "POST",
  "emits": ["verilator.clean"],
  "flows": ["verilator"],
}

async def handler(req, context):
  await context.emit({"topic": "verilator.clean", "data": {}})
  
  return {
    "status": 200,
    "body": {
      "message": "verilator clean started",
      "trace_id": context.trace_id
    }
  }