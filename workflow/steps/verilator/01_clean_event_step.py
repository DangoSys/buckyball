import subprocess
import os
import sys

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if utils_path not in sys.path:
  sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger

config = {
  "type": "event", 
  "name": "make clean",
  "description": "clean build directory",
  "subscribes": ["verilator.run", "verilator.clean"],
  "emits": ["verilator.verilog", "verilator.complete", "verilator.error"],
  "flows": ["verilator"],
}

async def handler(data, context):
  bbdir = get_buckyball_path()
  build_dir = f"{bbdir}/arch/build"
# ==================================================================================
# 执行操作
# ==================================================================================
  command = f"rm -rf {build_dir}"
  result = stream_run_logger(cmd=command, logger=context.logger, cwd=bbdir)

# ==================================================================================
# 向API返回结果
# ==================================================================================
  if data.get("from_run_workflow"):
    await context.state.set(context.trace_id, 'processing', True)
  elif result.returncode != 0:
    failure_result = {
      "status": 500,
      "body": {
        "success": False,
        "failure": True,
        "processing": False,
        "returncode": result.returncode,
        # "stdout": result.stdout,
        # "stderr": result.stderr,
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
        # "stdout": result.stdout,
        # "stderr": result.stderr,
      }
    }
    await context.state.set(context.trace_id, 'success', success_result)

# ==================================================================================
# 继续路由
# Routing to verilog or finish workflow
# For run workflow, continue to verilog; for standalone clean, complete
# ==================================================================================
  if data.get("from_run_workflow"):
    await context.emit({"topic": "verilator.verilog", "data": {**data, "task": "run"}})
  elif result.returncode == 0:
    await context.emit({"topic": "verilator.complete", "data": {**data, "task": "clean", "result": success_result}})
  else:
    await context.emit({"topic": "verilator.error", "data": {**data, "task": "clean", "result": failure_result}})
  return
