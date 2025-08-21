config = {
  "type": "event",
  "name": "verilator complete",
  "description": "handle verilator workflow completion",
  "subscribes": ["verilator.complete"],
  "emits": [],
  "flows": ["verilator"],
}

async def handler(data, context):
  task = data.get("task", "unknown")
  context.logger.info(f"Verilator {task} completed successfully", {"data": data})
