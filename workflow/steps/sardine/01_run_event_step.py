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
  "name": "running sardine",
  "description": "running sardine",
  "subscribes": ["sardine.run"],
  "emits": [],
  "flows": ["sardine"],
}

async def handler(data, context):
  bbdir = get_buckyball_path()

  sardine_dir = f"{bbdir}/bb-tests/sardine"
  
  command = f"source {bbdir}/env.sh && python run_tests.py --allure -m \"({data.get('workload', '')})\""  
  context.logger.info('Executing sardine command', {  
    'command': command,  
    'cwd': sardine_dir  
  })  
  result = stream_run_logger(cmd=command, logger=context.logger, cwd=sardine_dir, executable='bash', stdout_prefix="sardine run", stderr_prefix="sardine run")

# ==================================================================================
# 返回仿真结果
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
#  finish workflow
# ==================================================================================
  return