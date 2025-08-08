import subprocess
import os

config = {
  "type": "api",
  "name": "Init Chipyard",
  "description": "Execute init-chipyard.sh script to initialize Chipyard environment",

  "path": "/install",
  "method": "POST",

  # This API Step emits events to topic `chipyard-initialized`
  "emits": ["chipyard-initialized"],

  # Expected request body for type checking and documentation
  "bodySchema": {
    "type": "object",
    "properties": { 
      "skip_setup": { "type": "boolean", "default": False }
    }
  },

  # Expected response body for type checking and documentation
  "responseSchema": {
    "200": {
      "type": "object",
      "properties": {
        "traceId": { "type": "string" },
        "message": { "type": "string" },
        "output": { "type": "string" }
      }
    }
  },

  # The flows this step belongs to, will be available in Workbench
  "flows": ["install"],
}

async def handler(req, context):
  context.logger.info('Executing init-chipyard.sh script')
  
  try:
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, 'init-chipyard.sh')
    
    context.logger.info(f'Running script: {script_path}')
    
    # Execute the init-chipyard.sh script with real-time output
    process = subprocess.Popen(
      [script_path],
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      cwd=script_dir,
      bufsize=1
    )
    
    # Collect real-time output
    output_lines = []
    context.logger.info('=== Script Output Start ===')
    
    for line in iter(process.stdout.readline, ''):
      line = line.rstrip()
      output_lines.append(line)
      context.logger.info(line)  # Real-time logging
      print(line, flush=True)  # Real-time console output
    
    process.stdout.close()
    return_code = process.wait()
    
    context.logger.info('=== Script Output End ===')
    
    full_output = '\n'.join(output_lines)
    
    if return_code == 0:
      context.logger.info('Chipyard initialization successful')
      
      # Emit success event
      await context.emit({
        "topic": 'chipyard-initialized',
        "data": { 
          "status": "success",
          "output": full_output
        },
      })
      
      return {
        "status": 200,
        "body": {
          "traceId": context.trace_id,
          "message": "✅ Chipyard initialized successfully",
          "output": full_output,
          "lines": len(output_lines)
        }
      }
    else:
      context.logger.error(f'Chipyard initialization failed')
      
      return {
        "status": 500,
        "body": {
          "traceId": context.trace_id,
          "message": "❌ Chipyard initialization failed",
          "error": full_output
        }
      }
      
  except Exception as e:
    context.logger.error(f'Error executing init-chipyard.sh: {str(e)}')
    
    return {
      "status": 500,
      "body": {
        "traceId": context.trace_id,
        "message": "Internal server error",
        "error": str(e)
      }
    }