from contextlib import redirect_stdout
import os
from re import T
import subprocess
import sys
import time

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if utils_path not in sys.path:
  sys.path.insert(0, utils_path)


from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger

config = {
  "type": "event",
  "name": "build workload",
  "description": "build workload",
  "subscribes": ["workload.build"],
  "emits": [""],
  "flows": ["workload"],
}

async def handler(data, context):
  bbdir = get_buckyball_path()
  workload_dir = f"{bbdir}/bb-tests/workloads" 
  build_dir = f"{workload_dir}/build"
  
  # os.mkdir(f"{workload_dir}/build", exist_ok=True)
  subprocess.run(f"rm -rf {build_dir} && mkdir -p {build_dir}", shell=True)

  # command = f"source {bbdir}/env.sh && cd {workload_dir}/build && cmake .. && make build-all"  
  command = f"source {bbdir}/env.sh && cd {build_dir} && cmake -G Ninja .. && ninja build-all -j{os.cpu_count()}"  
  context.logger.info('Executing workload command', {  
    'command': command,  
    'cwd': build_dir  
  })  
  result = stream_run_logger(cmd=command, logger=context.logger, cwd=workload_dir, executable='bash', stdout_prefix="workload build", stderr_prefix="workload build")

# ==================================================================================
# 返回仿真结果
# ==================================================================================
  # 此处为run workflow的终点，status状态不再继续设为processing
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
#  finish workflow
# ==================================================================================
  return