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
  "name": "make verilog", 
  "description": "generate verilog code",
  "subscribes": ["verilator.verilog"],
  "emits": ["verilator.build", "verilator.complete", "verilator.error"],
  "flows": ["verilator"],
}

async def handler(data, context):
  bbdir = get_buckyball_path()
  build_dir = f"{bbdir}/arch/build"
  arch_dir = f"{bbdir}/arch"
# ==================================================================================
# 执行操作
# ==================================================================================
  command = f"cd {arch_dir} && mill -i __.test.runMain Elaborate " 
  command += f"--disable-annotation-unknown -strip-debug-info -O=debug "
  command += f"--split-verilog -o={build_dir}"
  result = stream_run_logger(cmd=command, logger=context.logger, cwd=bbdir)
  
  # Remove unwanted file  
  topname_file = f"{arch_dir}/TestHarness.sv"
  if os.path.exists(topname_file):
    os.remove(topname_file)

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
        "stdout": result.stdout,
        "stderr": result.stderr,
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
        "stdout": result.stdout,
        "stderr": result.stderr,
      }
    }
    await context.state.set(context.trace_id, 'success', success_result)

# ==================================================================================
# 继续路由
# Routing to verilog or finish workflow
# For run workflow, continue to verilog; for standalone clean, complete
# ==================================================================================
  if data.get("from_run_workflow"):
    await context.emit({"topic": "verilator.build", "data": {**data, "task": "run"}})
  elif result.returncode == 0:
    await context.emit({"topic": "verilator.complete", "data": {**data, "task": "verilog", "result": success_result}})
  else:
    await context.emit({"topic": "verilator.error", "data": {**data, "task": "verilog", "result": failure_result}})

  return