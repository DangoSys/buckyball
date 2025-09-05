import os
import subprocess
import sys

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if utils_path not in sys.path:
  sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger

config = {
  "type": "event",
  "name": "Marshal Launch", 
  "description": "launch marshal",
  "subscribes": ["marshal.launch"],
  "emits": ["marshal.launch", "marshal.complete", "marshal.error"],
  "flows": ["marshal"],
}

async def handler(data, context):
  bbdir = get_buckyball_path()
  script_dir = f"{bbdir}/workflow/steps/marshal/scripts"
# ==================================================================================
# 执行操作
# ==================================================================================  
  command = f"source {bbdir}/env.sh && ./marshal -v launch interactive.json" 
  result = stream_run_logger(cmd=command, logger=context.logger, cwd=script_dir)
  
# ==================================================================================
# 向API返回结果
# ==================================================================================  
  if result.returncode != 0:
    failure_result = {
      "status": 500,
      "body": {
        "success": False,
        "failure": True,
        "processing": False,
        "returncode": result.returncode,
      }
    }
    await context.state.set(context.trace_id, 'failure', failure_result)
  else:
    success_result = {
      "status": 200, 
      "body": {
        "success": True,
        "failure": False,
        "processing": False,
        "returncode": result.returncode,
      }
    }
    await context.state.set(context.trace_id, 'success', success_result)

# ==================================================================================
# 继续路由
# Routing to verilog or finish workflow
# For run workflow, continue to verilog; for standalone clean, complete
# ==================================================================================
  if result.returncode == 0:
    await context.emit({"topic": "marshal.complete", "data": {**data, "task": "marshal", "result": success_result}})
  else:
    await context.emit({"topic": "marshal.error", "data": {**data, "task": "marshal", "result": failure_result, "returncode": result.returncode}})

  return