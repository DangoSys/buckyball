import subprocess
import sys
import os

utils_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/utils"
if utils_path not in sys.path:
  sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path

config = {
  "type": "api",
  "name": "running sardine",
  "description": "running sardine",
  "path": "/sardine/run",
  "method": "POST",
  "emits": ["sardine.run"],
  "flows": ["sardine"],
}

async def handler(req, context):
  bbdir = get_buckyball_path()

  body = req.get("body") or {}
  
  data = {
    "workload": body.get("workload", "")
  }

  sardine_dir = f"{bbdir}/bb-tests/sardine"
  
  # await context.emit({"topic": "sardine.run", "data": data})
  command = f"source {bbdir}/env.sh && python run_tests.py --allure -m \"({data.get('workload', '')})\""  
  context.logger.info('Executing sardine command', {  
    'command': command,  
    'cwd': sardine_dir  
  })  
  result = subprocess.run(command, cwd=sardine_dir, shell=True)  

  if result.returncode != 0:
    return {
      "status": 500,
      "body": {
        "message": "sardine run failed",
        "trace_id": context.trace_id
      }
    }
  else:
    return {
      "status": 200,
      "body": {
        "message": "sardine run completed",
        "trace_id": context.trace_id
      }
    }
  