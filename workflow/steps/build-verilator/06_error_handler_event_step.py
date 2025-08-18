config = {
  "type": "event",
  "name": "build-error-handler",
  "description": "handle build errors and provide detailed error information",
  "subscribes": ["build.error"],
  "emits": ["build-verilator-finished"],
  "input": { 
    "type": "object", 
    "properties": {
      "debug": {"type": "boolean"},
      "workload": {"type": "string"},
      "clean_first": {"type": "boolean"},
      "target": {"type": "string"},
      "trace_id": {"type": "string"},
      "error": {"type": "string"},
      "step": {"type": "string"}
    }
  },
  "flows": ["build-verilator"],
}

async def handler(input_data, context):
  error = input_data.get("error", "Unknown error")
  step = input_data.get("step", "unknown")
  trace_id = input_data.get("trace_id", "")
  
  context.logger.error('Build process failed', {
    "step": step,
    "error": error,
    "trace_id": trace_id,
    "config": {
      "debug": input_data.get("debug"),
      "workload": input_data.get("workload"),
      "target": input_data.get("target")
    }
  })
  
  # 提供详细的错误信息和建议
  error_suggestions = {
    "clean": [
      "Check if build directory exists and has proper permissions",
      "Ensure git repository is accessible"
    ],
    "verilog": [
      "Check if mill is installed and accessible in PATH",
      "Verify Chisel/Scala code compiles correctly",
      "Check if required dependencies are available",
      "Try running 'mill -i __.compile' manually first"
    ],
    "verilator": [
      "Check if verilator is installed and accessible",
      "Verify C/C++ source files exist in expected locations",
      "Check if required libraries (dramsim, fesvr) are available",
      "Ensure include paths are correct"
    ],
    "simulation": [
      "Check if workload file exists",
      "Verify simulation arguments are correct",
      "Check available memory and disk space",
      "Consider reducing simulation complexity or timeout"
    ]
  }
  
  suggestions = error_suggestions.get(step, ["Contact support team for assistance"])
  
  # 发出最终的错误事件
  await context.emit({
    "topic": "build-verilator-finished",
    "data": {
      **input_data,
      "final_status": "error",
      "error_step": step,
      "error_message": error,
      "suggestions": suggestions
    }
  })
  
  context.logger.info('Error handling completed', {
    "trace_id": trace_id,
    "suggestions_provided": len(suggestions)
  })