config = {
  "type": "event",
  "name": "verilator error",
  "description": "handle verilator workflow errors",
  "subscribes": ["verilator.error"],
  "emits": [],
  "flows": ["verilator"],
}

async def handler(data, context):
  task = data.get("task", "unknown")
  error = data.get("error", "Unknown error")
  context.logger.error(f"Verilator {task} operation failed", {
    "task": task,
    "error": error,
    "data": data
  })