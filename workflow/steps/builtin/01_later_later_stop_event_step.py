import subprocess
import sys
import os
import time

utils_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/utils"
if utils_path not in sys.path:
  sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path

config = {
  "type": "event",
  "name": "later later stop",
  "description": "later later stop",
  "subscribes": ["later_stop.later_later_stop"],
  "emits": ["later_stop.later_later_later_stop"],
  "flows": ["later_stop"]
}

async def handler(data, context):
  bbdir = get_buckyball_path()

  port = data.get('port', "")

  time.sleep(10)
  command = f"source {bbdir}/env.sh && bbdev stop --port 5999"
  # context.logger.info('Executing sardine command', {  
  #   'command': command,  
  #   'cwd': sardine_dir  
  # })  
  result = subprocess.run(command, shell=True)  

  return {
    "status": 200,
    "body": {
      "message": "sardine run started",
      "trace_id": context.trace_id
    }
  }