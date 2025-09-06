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
  "name": "Build Compiler", 
  "description": "build bitstream",
  "subscribes": ["compiler.build"],
  "emits": ["compiler.build", "compiler.complete", "compiler.error"],
  "flows": ["compiler"],
}

async def handler(data, context):
  bbdir = get_buckyball_path()
  script_dir = f"{bbdir}/workflow/steps/compiler/scripts"
  yaml_dir = f"{script_dir}/yaml"
# ==================================================================================
# 执行操作
# ==================================================================================
  command = f"source {bbdir}/env.sh && mkdir -p {bbdir}/compiler/build && cd {bbdir}/compiler/build && ninja -j{os.cpu_count()}"
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
    await context.emit({"topic": "compiler.complete", "data": {**data, "task": "compiler", "result": success_result}})
  else:
    await context.emit({"topic": "compiler.error", "data": {**data, "task": "compiler", "result": failure_result, "returncode": result.returncode}})

  return