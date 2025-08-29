# steps/github-doc-webhook.step.py  
import subprocess  
import asyncio  
from datetime import datetime  
import os
import sys

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if utils_path not in sys.path:
  sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path


config = {  
  'type': 'api',  
  'name': 'GitHub Doc Webhook',  
  'description': 'Receives GitHub push webhook and executes make doc command',  
  'path': '/doc',  
  'method': 'POST',  
  'emits': [''],  
  'flows': ['github-doc'],  
}  
  
async def handler(req, context):  
  context.logger.info('GitHub webhook received, starting doc deploy', {  
    'body': req.get('body'),  
    'trace_id': context.trace_id  
  })  
  
  bbdir = get_buckyball_path()

  doc_dir = f"{bbdir}/docs/bb-note"
  
  command = f"source {bbdir}/env.sh && mdbook serve --open -p 3001"  
  
  process = await asyncio.create_subprocess_exec(  
    command,  
    stdout=asyncio.subprocess.PIPE,  
    stderr=asyncio.subprocess.PIPE,  
    cwd=doc_dir  
  )  
      
  if process.returncode == 0:  
    context.logger.info('Doc deploy completed successfully', {  
      'command': command,  
      'cwd': doc_dir  
    })  

    return {  
      'status': 200,  
      'body': {  
        'message': 'Document deploy completed successfully',  
        'trace_id': context.trace_id  
      }  
    }  
  else:  
    context.logger.error('Doc deploy failed', {  
      'command': command,  
      'cwd': doc_dir  
    })  
    return {  
      'status': 500,  
      'body': {  
        'message': 'Document deploy failed',  
        'trace_id': context.trace_id  
      }  
    }  
  