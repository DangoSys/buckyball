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
  
# ==================================================================================
# 返回构建1结果
# ==================================================================================
  customext_dir = f"{funcsim_dir}/../customext/"
  if not os.path.exists(customext_dir):
    context.logger.error('customext directory not found', {'customext_dir': customext_dir})
  command = f"source {bbdir}/env.sh && cd {customext_dir} && mkdir -p build && cd build && cmake .. && make install"
  context.logger.info('Executing funcsim command', {  
    'command': command,  
    'cwd': customext_dir  
  })  
  result = stream_run_logger(cmd=command, logger=context.logger, cwd=customext_dir, executable='bash', stdout_prefix="build customext", stderr_prefix="build customext")


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


# ==================================================================================
# 返回构建2结果
# ==================================================================================
  build_dir = f"{funcsim_dir}/build"
  subprocess.run(f"rm -rf {build_dir} && mkdir -p {build_dir}", shell=True)
  riscv_dir = os.environ.get('RISCV')
  command = f"source {bbdir}/scripts/env-exit.sh && cd {build_dir} && ../configure --prefix={riscv_dir} && make -j$(nproc) && make install"
  context.logger.info('Executing funcsim command', {  
    'command': command,  
    'cwd': funcsim_dir  
  })  
  result = stream_run_logger(cmd=command, logger=context.logger, cwd=funcsim_dir, executable='bash', stdout_prefix="build funcsim", stderr_prefix="build funcsim")


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