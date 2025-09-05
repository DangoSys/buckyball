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
  "name": "Firesim Runworkload", 
  "description": "run workload",
  "subscribes": ["firesim.runworkload"],
  "emits": ["firesim.runworkload", "firesim.complete", "firesim.error"],
  "flows": ["firesim"],
}

async def handler(data, context):
  bbdir = get_buckyball_path()
  script_dir = f"{bbdir}/workflow/steps/firesim/scripts"
# ==================================================================================
# 执行操作
# ==================================================================================  
  command = f"source {bbdir}/env.sh && firesim runworkload " 
  command += f" -a {script_dir}/config_hwdb.yaml"
  command += f" -b {script_dir}/config_build.yaml"
  command += f" -r {script_dir}/config_build_recipes.yaml"
  command += f" -c {script_dir}/config_runtime.yaml"
  result = stream_run_logger(cmd=command, logger=context.logger)
  
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
    await context.emit({"topic": "firesim.complete", "data": {**data, "task": "firesim", "result": success_result}})
  else:
    await context.emit({"topic": "verilator.error", "data": {**data, "task": "verilog", "result": failure_result}})

  return