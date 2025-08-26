import subprocess
import sys
import os

utils_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/utils"
if utils_path not in sys.path:
  sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path

config = {
  "type": "api",
  "name": "kill",
  "description": "kill task with trace ID",
  "path": "/kill",
  "method": "POST",
  "emits": ["kill"],
  "flows": ["kill"],
}

async def handler(req, context):
  bbdir = get_buckyball_path()

  body = req.get("body") or {}
  
  data = {
    "port": body.get("port", ""),
    "trace_id": body.get("trace_id", ""),
  }

  sardine_dir = f"{bbdir}/bb-tests/sardine"
  
  # command = f"source {bbdir}/env.sh && bbdev start --port 5999"
  # context.logger.info('Executing sardine command', {  
  #   'command': command,  
  #   'cwd': sardine_dir  
  # })  
      
  # result = subprocess.run(command, cwd=sardine_dir, shell=True)  


  
  return {
    "status": 200,
    "body": {
      "message": "sardine run started",
      "trace_id": context.trace_id
    }
  }