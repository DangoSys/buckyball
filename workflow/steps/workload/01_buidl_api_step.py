import subprocess
import sys
import os

utils_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/utils"
if utils_path not in sys.path:
  sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path

config = {
  "type": "api",
  "name": "build workload",
  "description": "build workload",
  "path": "/workload/build",
  "method": "POST",
  "emits": ["workload.build"],
  "flows": ["workload"],
}

async def handler(req, context):
  bbdir = get_buckyball_path()

  body = req.get("body") or {}
  
  data = {
    "workload": body.get("workload", "")
  }

  workload_dir = f"{bbdir}/bb-tests/workload"
  
  await context.emit({"topic": "workload.build", "data": data})

  return {
    "status": 200,
    "body": {
      "message": "workload build started",
      "trace_id": context.trace_id
    }
  }