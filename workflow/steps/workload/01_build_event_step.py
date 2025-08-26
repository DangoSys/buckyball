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
  
  command = f"source {bbdir}/env.sh && cd {workload_dir}/build && cmake ../ && make build-all"  
  context.logger.info('Executing workload command', {  
    'command': command,  
    'cwd': workload_dir  
  })  
  result = subprocess.run(command, cwd=workload_dir, shell=True)  
  
  if result.returncode != 0:
    context.logger.error('Workload build failed', {  
      'command': command,  
      'cwd': workload_dir  
    })
  else:
    context.logger.info('Workload build completed', {  
      'command': command,  
      'cwd': workload_dir  
    })
  
  return {
    "status": 200,
    "body": {
      "message": "workload build completed",
      "trace_id": context.trace_id
    }
  }