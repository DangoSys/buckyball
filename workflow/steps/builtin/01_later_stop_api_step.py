import subprocess
import sys
import os
import time

utils_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/utils"
if utils_path not in sys.path:
  sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path

config = {
  "type": "api",
  "name": "later stop",
  "description": "later stop",
  "path": "/later_stop",
  "method": "POST",
  "emits": ["later_stop.later_later_stop"],
  "flows": ["later_stop"]
}

async def handler(req, context):
  bbdir = get_buckyball_path()

  body = req.get("body") or {}
  
  data = {
    "port": body.get("port", ""),
  }

  await context.emit({"topic": "later_stop.later_later_stop", "data": data})

  port = data.get('port', "")
  command = f"source {bbdir}/env.sh && bbdev start --port {port}"
  result = subprocess.run(command, shell=True, text=True)  
  



  return {
    "status": 200,
    "body": {
      "message": "sardine run started",
      "trace_id": context.trace_id
    }
  }