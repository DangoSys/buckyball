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
  "name": "build functional simulator",
  "description": "build functional simulator",
  "subscribes": ["funcsim.build"],
  "emits": [""],
  "flows": ["funcsim"],
}

async def handler(data, context):
  bbdir = get_buckyball_path()
  funcsim_dir = f"{bbdir}/sims/func-sim" 
  build_dir = f"{funcsim_dir}/build"
  
  # os.mkdir(f"{workload_dir}/build", exist_ok=True)
  subprocess.run(f"rm -rf {build_dir} && mkdir -p {build_dir}", shell=True)

  riscv_dir = os.environ.get('RISCV')
  command = f"source {bbdir}/scripts/env-exit.sh && cd {build_dir} && ../configure --prefix={riscv_dir} && make -j$(nproc) && make install"
  context.logger.info('Executing funcsim command', {  
    'command': command,  
    'cwd': funcsim_dir  
  })  
  result = stream_run_logger(cmd=command, logger=context.logger, cwd=funcsim_dir, executable='bash', stdout_prefix="funcsim build", stderr_prefix="funcsim build")

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