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
  "name": "running sardine",
  "description": "running sardine",
  "subscribes": ["sardine.run"],
  "emits": ["sardine.complete", "sardine.error"],
  "flows": ["sardine"],
}

async def handler(data, context):
  bbdir = get_buckyball_path()

  sardine_dir = f"{bbdir}/bb-tests/sardine"
  time.sleep(20)
  # result = subprocess.run(f"python run_tests.py --allure -m \"({data.get('workload', '')})\"", cwd=sardine_dir, shell=True)
  
  command = f"source {bbdir}/env.sh && python run_tests.py --allure -m \"({data.get('workload', '')})\""  
  context.logger.info('Executing sardine command', {  
    'command': command,  
    'cwd': sardine_dir  
  })  
  
  result = subprocess.run(command, cwd=sardine_dir, shell=True)  
  
  command = f"source {bbdir}/env.sh && bbdev stop --port {data.get('port', 5400)}"  
  context.logger.info('Executing sardine command', {  
    'command': command,  
    'cwd': sardine_dir  
  }) 
  
  # if result.returncode != 0:
  #   await context.emit({"topic": "sardine.error", "data": {**data, "task": "run", "error": "sardine failed"}})
  #   subprocess.run(f"source {bbdir}/env.sh && bbdev stop --port 5100", shell=True)
  # else:
  #   await context.emit({"topic": "sardine.complete", "data": {**data, "task": "run"}})
  #   subprocess.run(f"source {bbdir}/env.sh && bbdev stop --port 5100", shell=True)
