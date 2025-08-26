import os
import sys

# 添加当前目录到路径
# current_dir = os.path.dirname(__file__)
# if current_dir not in sys.path:
#   sys.path.insert(0, current_dir)

from scripts.sim_runner import run_simulation

config = {
  "type": "api",
  "name": "Verilator Sim",
  "description": "run verilator simulation",
  "path": "/verilator/sim",
  "method": "POST",
  "emits": [""],
  "flows": ["verilator"],
}

async def handler(req, context):
  body = req.get("body") or {}
  
  binary = body.get("binary", "")
  if not binary:
    return {"status": 400, "body": {"error": "binary parameter is required"}}
  
  batch = body.get("batch", False)
  
  # context.logger.info('开始同步执行仿真', {"binary": binary, "batch": batch})
  
  # 同步函数
  result = run_simulation(binary, batch, context.logger)
  
  # context.logger.info('仿真执行完毕', {"result": result})
  
  if result["success"]:
    return {
      "status": 200,
      "body": {
        "message": "simulation completed successfully",
        "trace_id": context.trace_id,
        "result": result
      }
    }
  else:
    return {
      "status": 500,
      "body": {
        "error": "simulation failed",
        "trace_id": context.trace_id,
        "details": result
      }
    }