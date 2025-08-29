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
  "name": "running doc deploy",
  "description": "running doc deploy",
  "subscribes": ["doc.deploy"],
  "emits": [],
  "flows": ["doc"],
}

async def handler(data, context):
  bbdir = get_buckyball_path()

  doc_dir = f"{bbdir}/docs/bb-note"
  
  command = f"source {bbdir}/env.sh && mdbook serve --open -p 3001"  
  context.logger.info('Executing doc deploy command', {  
    'command': command,  
    'cwd': doc_dir  
  })  
  result = subprocess.run(command, cwd=doc_dir, shell=True)  
  
  if result.returncode != 0:
    context.logger.error('Doc deploy failed', {  
      'command': command,  
      'cwd': doc_dir  
    })
  else:
    context.logger.info('Doc deploy completed', {  
      'command': command,  
      'cwd': doc_dir  
    })
  
  return {
    "status": 200,
    "body": {
      "message": "document deploy completed",
      "trace_id": context.trace_id
    }
  }
